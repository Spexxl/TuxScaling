use super::*;

unsafe fn get_swapchain_images_inner(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    image_count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    if image_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let virtual_images = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&swapchain)
        .and_then(|state| {
            state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .virtual_images
                .as_ref()
                .map(|images| images.iter().map(|image| image.handle).collect::<Vec<_>>())
        });
    if let Some(virtual_images) = virtual_images {
        let total = virtual_images.len() as u32;
        if images.is_null() {
            unsafe { *image_count = total };
            return vk::Result::SUCCESS;
        }
        let capacity = unsafe { *image_count }.min(total) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(virtual_images.as_ptr(), images, capacity);
            *image_count = capacity as u32;
        }
        return if capacity < total as usize {
            vk::Result::INCOMPLETE
        } else {
            vk::Result::SUCCESS
        };
    }
    let Some(proc) = (unsafe { device_downstream(device, c"vkGetSwapchainImagesKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let get: vk::PFN_vkGetSwapchainImagesKHR = unsafe { std::mem::transmute(proc) };
    unsafe { get(device, swapchain, image_count, images) }
}

pub(super) unsafe extern "system" fn get_swapchain_images_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    image_count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        get_swapchain_images_inner(device, swapchain, image_count, images)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn acquire_next_image_inner(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    image_index: *mut u32,
) -> vk::Result {
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImageKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let acquire: vk::PFN_vkAcquireNextImageKHR = unsafe { std::mem::transmute(proc) };
    unsafe { acquire(device, swapchain, timeout, semaphore, fence, image_index) }
}

pub(super) unsafe extern "system" fn acquire_next_image_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    image_index: *mut u32,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        acquire_next_image_inner(device, swapchain, timeout, semaphore, fence, image_index)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn acquire_next_image2_inner(
    device: vk::Device,
    info: *const vk::AcquireNextImageInfoKHR<'_>,
    image_index: *mut u32,
) -> vk::Result {
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImage2KHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let acquire: vk::PFN_vkAcquireNextImage2KHR = unsafe { std::mem::transmute(proc) };
    unsafe { acquire(device, info, image_index) }
}

pub(super) unsafe extern "system" fn acquire_next_image2_khr(
    device: vk::Device,
    info: *const vk::AcquireNextImageInfoKHR<'_>,
    image_index: *mut u32,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        acquire_next_image2_inner(device, info, image_index)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}
