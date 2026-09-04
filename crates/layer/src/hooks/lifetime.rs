use super::*;

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
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let state = swapchains()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&swapchain);
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
    }));
    if let Some(proc) = destroy {
        let destroy_swapchain: vk::PFN_vkDestroySwapchainKHR = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_swapchain(device, swapchain, allocation_callbacks) };
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
        let overlays = {
            let mut swapchains = swapchains().lock().unwrap_or_else(|e| e.into_inner());
            let handles = swapchains
                .iter()
                .filter_map(|(handle, state)| {
                    (state.lock().unwrap_or_else(|e| e.into_inner()).device == device)
                        .then_some(*handle)
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
                        state
                            .into_inner()
                            .unwrap_or_else(|e| e.into_inner())
                            .overlay
                            .destroy(&device_state.device);
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
