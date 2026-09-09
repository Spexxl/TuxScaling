use super::*;

unsafe fn create_xlib_surface_inner(
    instance: vk::Instance,
    create_info: *const vk::XlibSurfaceCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    let Some(proc) = (unsafe { downstream(instance, c"vkCreateXlibSurfaceKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create: vk::PFN_vkCreateXlibSurfaceKHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create(instance, create_info, allocation_callbacks, surface) };
    if result == vk::Result::SUCCESS && !create_info.is_null() && !surface.is_null() {
        surfaces()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                unsafe { *surface },
                X11Surface {
                    window: unsafe { (*create_info).window },
                    logical_extent: None,
                    logical_capabilities: None,
                    borderless_lease: None,
                    negotiation: tuxscaling_display::PresentationNegotiation::direct(),
                },
            );
    }
    result
}

pub(super) unsafe extern "system" fn create_xlib_surface_khr(
    instance: vk::Instance,
    create_info: *const vk::XlibSurfaceCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        create_xlib_surface_inner(instance, create_info, allocation_callbacks, surface)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn create_xcb_surface_inner(
    instance: vk::Instance,
    create_info: *const vk::XcbSurfaceCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    let Some(proc) = (unsafe { downstream(instance, c"vkCreateXcbSurfaceKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create: vk::PFN_vkCreateXcbSurfaceKHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create(instance, create_info, allocation_callbacks, surface) };
    if result == vk::Result::SUCCESS && !create_info.is_null() && !surface.is_null() {
        surfaces()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                unsafe { *surface },
                X11Surface {
                    window: unsafe { (*create_info).window as u64 },
                    logical_extent: None,
                    logical_capabilities: None,
                    borderless_lease: None,
                    negotiation: tuxscaling_display::PresentationNegotiation::direct(),
                },
            );
    }
    result
}

fn apply_logical_extent(
    capabilities: &mut vk::SurfaceCapabilitiesKHR,
    logical_extent: vk::Extent2D,
) {
    capabilities.current_extent = logical_extent;
    capabilities.min_image_extent = logical_extent;
    capabilities.max_image_extent = logical_extent;
}

fn logical_extent(surface: vk::SurfaceKHR) -> Option<vk::Extent2D> {
    surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .and_then(|surface| surface.logical_extent)
}

fn saved_capabilities(surface: vk::SurfaceKHR) -> Option<vk::SurfaceCapabilitiesKHR> {
    surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .and_then(|surface| surface.logical_capabilities)
}

unsafe fn instance_for_physical_device(
    physical_device: vk::PhysicalDevice,
) -> Option<ash::Instance> {
    instances()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .find(|instance| {
            unsafe { instance.enumerate_physical_devices() }
                .is_ok_and(|devices| devices.contains(&physical_device))
        })
        .cloned()
}

unsafe fn get_surface_capabilities_inner(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    capabilities: *mut vk::SurfaceCapabilitiesKHR,
) -> vk::Result {
    if capabilities.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(instance) = (unsafe { instance_for_physical_device(physical_device) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(proc) = (unsafe {
        downstream(
            instance.handle(),
            c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        )
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let get: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
        unsafe { std::mem::transmute(proc) };
    let result = unsafe { get(physical_device, surface, capabilities) };
    if result == vk::Result::SUCCESS
        && let Some(extent) = logical_extent(surface)
    {
        if let Some(saved) = saved_capabilities(surface) {
            unsafe { *capabilities = saved };
        } else {
            apply_logical_extent(unsafe { &mut *capabilities }, extent);
        }
    }
    result
}

pub(super) unsafe extern "system" fn get_physical_device_surface_capabilities_khr(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    capabilities: *mut vk::SurfaceCapabilitiesKHR,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        get_surface_capabilities_inner(physical_device, surface, capabilities)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn get_surface_capabilities2_inner(
    physical_device: vk::PhysicalDevice,
    surface_info: *const vk::PhysicalDeviceSurfaceInfo2KHR<'_>,
    capabilities: *mut vk::SurfaceCapabilities2KHR<'_>,
) -> vk::Result {
    if surface_info.is_null() || capabilities.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(instance) = (unsafe { instance_for_physical_device(physical_device) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(proc) = (unsafe {
        downstream(
            instance.handle(),
            c"vkGetPhysicalDeviceSurfaceCapabilities2KHR",
        )
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let get: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR =
        unsafe { std::mem::transmute(proc) };
    let result = unsafe { get(physical_device, surface_info, capabilities) };
    if result == vk::Result::SUCCESS
        && let Some(extent) = logical_extent(unsafe { (*surface_info).surface })
    {
        if let Some(saved) = saved_capabilities(unsafe { (*surface_info).surface }) {
            unsafe { (*capabilities).surface_capabilities = saved };
        } else {
            apply_logical_extent(unsafe { &mut (*capabilities).surface_capabilities }, extent);
        }
    }
    result
}

pub(super) unsafe extern "system" fn get_physical_device_surface_capabilities2_khr(
    physical_device: vk::PhysicalDevice,
    surface_info: *const vk::PhysicalDeviceSurfaceInfo2KHR<'_>,
    capabilities: *mut vk::SurfaceCapabilities2KHR<'_>,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        get_surface_capabilities2_inner(physical_device, surface_info, capabilities)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

pub(super) unsafe extern "system" fn create_xcb_surface_khr(
    instance: vk::Instance,
    create_info: *const vk::XcbSurfaceCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    surface: *mut vk::SurfaceKHR,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        create_xcb_surface_inner(instance, create_info, allocation_callbacks, surface)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn destroy_surface_inner(
    instance: vk::Instance,
    surface: vk::SurfaceKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let surface_state = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&surface);
    if let Some(surface_state) = surface_state
        && let Some(lease) = surface_state.borderless_lease
        && let Ok(display) = tuxscaling_display::X11Display::connect()
    {
        let _ = display.restore(lease);
    }
    let Some(proc) = (unsafe { downstream(instance, c"vkDestroySurfaceKHR") }) else {
        return;
    };
    let destroy: vk::PFN_vkDestroySurfaceKHR = unsafe { std::mem::transmute(proc) };
    unsafe { destroy(instance, surface, allocation_callbacks) };
}

pub(super) unsafe extern "system" fn destroy_surface_khr(
    instance: vk::Instance,
    surface: vk::SurfaceKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_surface_inner(instance, surface, allocation_callbacks)
    }));
}

#[cfg(test)]
mod tests {
    use super::apply_logical_extent;
    use ash::vk;

    #[test]
    fn logical_capabilities_lock_the_game_extent() {
        let mut capabilities = vk::SurfaceCapabilitiesKHR::default();
        let extent = vk::Extent2D {
            width: 1280,
            height: 720,
        };

        apply_logical_extent(&mut capabilities, extent);

        assert_eq!(capabilities.current_extent, extent);
        assert_eq!(capabilities.min_image_extent, extent);
        assert_eq!(capabilities.max_image_extent, extent);
    }
}
