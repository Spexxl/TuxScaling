use super::*;
use crate::recovery::LeaseCleanup;

fn suppress_unknown_swapchain_destroy(swapchain: vk::SwapchainKHR) -> bool {
    is_retired_swapchain(swapchain) || is_unknown_logical_swapchain(swapchain)
}

unsafe fn destroy_overlay(device_state: &DeviceState, overlay: OverlaySwapchain) {
    let _ = unsafe { device_state.device.device_wait_idle() };
    unsafe { overlay.destroy(&device_state.device) };
}

unsafe fn destroy_swapchain_inner(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let destroy = unsafe { device_downstream(device, c"vkDestroySwapchainKHR") };
    let state = swapchains()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&swapchain);
    if state.is_none() && suppress_unknown_swapchain_destroy(swapchain) {
        return;
    }
    let physical_swapchain = state
        .as_ref()
        .and_then(|state| state.lock().ok().map(|state| state.physical_handle))
        .unwrap_or(swapchain);
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if state.as_ref().is_some_and(|state| {
            state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .mapping
                .is_some()
        }) {
            retire_swapchain(swapchain);
        }
        let restore_surface = state.as_ref().and_then(|state| {
            let state = state.lock().unwrap_or_else(|e| e.into_inner());
            state.mapping.as_ref().map(|_| state.surface)
        });
        let device_state = devices()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&device)
            .cloned();
        if let Some(state) = state
            .and_then(|s| Arc::try_unwrap(s).ok())
            .map(|s| s.into_inner().unwrap_or_else(|e| e.into_inner()))
            && let Some(device_state) = device_state
        {
            let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
                destroy_overlay(&device_state, state.overlay)
            }));
        }
        if let Some(surface) = restore_surface {
            let another_virtual_swapchain = swapchains()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .values()
                .any(|state| {
                    let state = state.lock().unwrap_or_else(|e| e.into_inner());
                    state.surface == surface && state.mapping.is_some()
                });
            if !another_virtual_swapchain {
                restore_surface_window(surface);
            }
        }
    }));
    if let Some(proc) = destroy {
        let destroy_swapchain: vk::PFN_vkDestroySwapchainKHR = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_swapchain(device, physical_swapchain, allocation_callbacks) };
    }
}

pub(super) fn restore_surface_window(surface: vk::SurfaceKHR) {
    let lease = surfaces()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_mut(&surface)
        .and_then(|state| {
            state.logical_extent = None;
            state.logical_capabilities = None;
            state.negotiation = tuxscaling_display::PresentationNegotiation::direct();
            state.borderless_lease.take()
        });
    if let Some(lease) = lease {
        let mut cleanup = LeaseCleanup::acquired();
        if let Ok(display) = tuxscaling_display::X11Display::connect()
            && cleanup.restore_once()
        {
            let _ = display.restore(lease);
        }
        debug_assert!(cleanup.restore_count() <= 1);
    }
    for state in swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
    {
        if let Ok(mut state) = state.lock()
            && state.surface == surface
        {
            let has_contract = state.contract.is_some();
            if has_contract {
                let mut failed = tuxscaling_display::PresentationNegotiation::direct();
                failed.fail(tuxscaling_display::NegotiationFailure::OutputRecreation);
                state.negotiation = failed;
            } else {
                state.negotiation = tuxscaling_display::PresentationNegotiation::direct();
            }
            if let Some(contract) = state.contract.as_mut() {
                contract.set_state(tuxscaling_display::PresentationState::Failed);
            }
        }
    }
}

pub(super) unsafe extern "system" fn destroy_swapchain_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_swapchain_inner(device, swapchain, allocation_callbacks)
    }));
}

#[cfg(test)]
mod tests {
    use super::{restore_surface_window, suppress_unknown_swapchain_destroy};
    use crate::state::{X11Surface, surfaces};
    use ash::vk;
    use ash::vk::Handle;
    use tuxscaling_display::{
        BorderlessLease, Monitor, PresentationNegotiation, Rect, WindowSnapshot,
    };

    #[test]
    fn unknown_reserved_token_is_suppressed_before_downstream_destroy() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0abc);

        assert!(super::is_unknown_logical_swapchain(token));
        assert!(suppress_unknown_swapchain_destroy(token));
    }

    #[test]
    fn fail_open_cleanup_takes_lease_and_resets_negotiation() {
        let surface = vk::SurfaceKHR::from_raw(0xfeed);
        let mut negotiation = PresentationNegotiation::direct();
        negotiation.fail(tuxscaling_display::NegotiationFailure::DeadlineExpired);
        surfaces().lock().unwrap().insert(
            surface,
            X11Surface {
                window: 0,
                logical_extent: Some(vk::Extent2D {
                    width: 1,
                    height: 1,
                }),
                logical_capabilities: None,
                borderless_lease: Some(BorderlessLease {
                    window: 0,
                    original: WindowSnapshot {
                        rect: Rect::new(0, 0, 1, 1),
                        fullscreen: false,
                    },
                    monitor: Monitor::new(Rect::new(0, 0, 1, 1)),
                }),
                negotiation,
            },
        );

        restore_surface_window(surface);
        let state = surfaces().lock().unwrap().remove(&surface).unwrap();
        assert!(state.borderless_lease.is_none());
        assert!(state.logical_extent.is_none());
        assert_eq!(
            state.negotiation.public_state(),
            tuxscaling_display::PresentationState::Direct
        );
    }
}

unsafe fn destroy_device_inner(
    device: vk::Device,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let destroy = unsafe { device_downstream(device, c"vkDestroyDevice") };
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let device_state = devices()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&device);
        let overlays = {
            let mut swapchains = swapchains().lock().unwrap_or_else(|e| e.into_inner());
            let handles = swapchains
                .iter()
                .filter_map(|(handle, state)| {
                    let state = state.lock().unwrap_or_else(|e| e.into_inner());
                    if state.device != device {
                        return None;
                    }
                    if state.mapping.is_some() {
                        retire_swapchain(*handle);
                    }
                    Some(*handle)
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .filter_map(|handle| swapchains.remove(&handle))
                .collect::<Vec<_>>()
        };
        queues()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|_, state| state.device != device);
        if let Some(device_state) = device_state {
            let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
                let _ = device_state.device.device_wait_idle();
                for state in overlays {
                    if let Ok(state) = Arc::try_unwrap(state) {
                        let state = state.into_inner().unwrap_or_else(|e| e.into_inner());
                        let restore_surface = state.mapping.is_some().then_some(state.surface);
                        state.overlay.destroy(&device_state.device);
                        if let Some(surface) = restore_surface {
                            restore_surface_window(surface);
                        }
                    }
                }
            }));
        }
    }));
    if let Some(proc) = destroy {
        let destroy_device: vk::PFN_vkDestroyDevice = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_device(device, allocation_callbacks) };
    }
}

pub(super) unsafe extern "system" fn destroy_device(
    device: vk::Device,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_device_inner(device, allocation_callbacks)
    }));
}

unsafe fn destroy_instance_inner(
    instance: vk::Instance,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let destroy = unsafe { downstream(instance, c"vkDestroyInstance") };
    let _ = catch_unwind(AssertUnwindSafe(|| {
        instances()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&instance);
        crate::state::instance_api_versions()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&instance);
        crate::state::instance_dispatch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&instance);
    }));
    if let Some(proc) = destroy {
        let destroy_instance: vk::PFN_vkDestroyInstance = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_instance(instance, allocation_callbacks) };
    }
}

pub(super) unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_instance_inner(instance, allocation_callbacks)
    }));
}
