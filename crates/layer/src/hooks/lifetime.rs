use super::*;
use crate::recovery::LeaseCleanup;

fn suppress_unknown_swapchain_destroy(swapchain: vk::SwapchainKHR) -> bool {
    is_retired_swapchain(swapchain) || is_unknown_logical_swapchain(swapchain)
}

fn unique_physical_handles(
    current: vk::SwapchainKHR,
    retired: &[vk::SwapchainKHR],
) -> Vec<vk::SwapchainKHR> {
    let mut handles = Vec::with_capacity(retired.len() + 1);
    for handle in std::iter::once(current).chain(retired.iter().copied()) {
        if handle != vk::SwapchainKHR::null() && !handles.contains(&handle) {
            handles.push(handle);
        }
    }
    handles
}

struct SwapchainTeardown {
    physical_handles: Vec<vk::SwapchainKHR>,
    restore_surface: Option<vk::SurfaceKHR>,
    virtualized: bool,
    overlay: Option<OverlaySwapchain>,
}

fn take_swapchain_teardown(state: &Arc<Mutex<SwapchainState>>) -> SwapchainTeardown {
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    let virtualized = state.mapping.is_some();
    SwapchainTeardown {
        physical_handles: unique_physical_handles(
            state.physical_handle,
            &state.retired_physical_generations,
        ),
        restore_surface: virtualized.then_some(state.surface),
        virtualized,
        overlay: state.overlay.take(),
    }
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
    eprintln!(
        "TuxScaling evidence event=swapchain_destroy_request swapchain=0x{:x}",
        swapchain.as_raw(),
    );
    let destroy = unsafe { device_downstream(device, c"vkDestroySwapchainKHR") };
    let tracked_state = swapchains()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&swapchain)
        .cloned();
    if let Some(state) = &tracked_state {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        if state.lifecycle.blocks_frame_operations() {
            let _ = state.lifecycle.request_destroy();
            return;
        }
    }
    let state = swapchains()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&swapchain);
    if state.is_none() && suppress_unknown_swapchain_destroy(swapchain) {
        return;
    }
    let teardown = state
        .as_ref()
        .map(take_swapchain_teardown)
        .unwrap_or_else(|| SwapchainTeardown {
            physical_handles: vec![swapchain],
            restore_surface: None,
            virtualized: false,
            overlay: None,
        });
    if teardown.virtualized {
        retire_swapchain(swapchain);
    }
    drop(state);
    let device_state = devices()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&device)
        .cloned();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(device_state) = device_state.as_ref()
            && let Some(overlay) = teardown.overlay
        {
            let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
                destroy_overlay(device_state, overlay)
            }));
        }
        if let Some(surface) = teardown.restore_surface {
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
        for physical_swapchain in teardown.physical_handles {
            unsafe { destroy_swapchain(device, physical_swapchain, allocation_callbacks) };
        }
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
        let swapchains_to_destroy = {
            let mut swapchains = swapchains().lock().unwrap_or_else(|e| e.into_inner());
            let handles = swapchains
                .iter()
                .filter_map(|(handle, state)| {
                    if state.lock().unwrap_or_else(|e| e.into_inner()).device != device {
                        return None;
                    }
                    Some(*handle)
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .filter_map(|handle| {
                    let state = swapchains.remove(&handle)?;
                    let teardown = take_swapchain_teardown(&state);
                    if teardown.virtualized {
                        retire_swapchain(handle);
                    }
                    Some(teardown)
                })
                .collect::<Vec<_>>()
        };
        queues()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|_, state| state.device != device);
        let surfaces_to_restore = swapchains_to_destroy
            .iter()
            .filter_map(|teardown| teardown.restore_surface)
            .fold(Vec::new(), |mut surfaces, surface| {
                if !surfaces.contains(&surface) {
                    surfaces.push(surface);
                }
                surfaces
            });
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            if let Some(device_state) = device_state.as_ref() {
                let _ = device_state.device.device_wait_idle();
            }
            let destroy_swapchain = destroy.map(|proc| {
                std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkDestroySwapchainKHR>(
                    proc,
                )
            });
            for teardown in swapchains_to_destroy {
                if let Some(device_state) = device_state.as_ref()
                    && let Some(overlay) = teardown.overlay
                {
                    overlay.destroy(&device_state.device);
                }
                if let Some(destroy_swapchain) = destroy_swapchain {
                    for physical_swapchain in teardown.physical_handles {
                        destroy_swapchain(device, physical_swapchain, allocation_callbacks);
                    }
                }
            }
        }));
        for surface in surfaces_to_restore {
            let another_virtual_swapchain = swapchains()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .values()
                .any(|state| {
                    let state = state.lock().unwrap_or_else(|error| error.into_inner());
                    state.surface == surface && state.mapping.is_some()
                });
            if !another_virtual_swapchain {
                restore_surface_window(surface);
            }
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

#[cfg(test)]
mod tests {
    use super::{
        restore_surface_window, suppress_unknown_swapchain_destroy, unique_physical_handles,
    };
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

    #[test]
    fn retired_physical_generations_are_destroyed_once() {
        let current = vk::SwapchainKHR::from_raw(41);
        let retired = [
            vk::SwapchainKHR::from_raw(17),
            current,
            vk::SwapchainKHR::null(),
            vk::SwapchainKHR::from_raw(17),
        ];

        assert_eq!(
            unique_physical_handles(current, &retired),
            vec![current, retired[0]]
        );
    }
}
