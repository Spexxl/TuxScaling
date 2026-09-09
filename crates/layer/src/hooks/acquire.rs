use super::*;
use crate::mapping::Mapping;

fn reject_retired_swapchain(swapchain: vk::SwapchainKHR) -> Result<(), vk::Result> {
    if is_retired_swapchain(swapchain) || is_unknown_logical_swapchain(swapchain) {
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
    } else {
        Ok(())
    }
}

fn virtual_physical_swapchain(
    logical: vk::SwapchainKHR,
) -> Option<(Arc<Mutex<SwapchainState>>, vk::SwapchainKHR)> {
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&logical)
        .cloned()?;
    let physical = state
        .lock()
        .ok()
        .and_then(|state| state.mapping.as_ref().map(|_| state.physical_handle))?;
    Some((state, physical))
}

fn map_acquired_image(
    state: &Arc<Mutex<SwapchainState>>,
    image_index: *mut u32,
) -> Result<(), vk::Result> {
    if image_index.is_null() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    let Some(mapping) = state.mapping.as_mut() else {
        return Ok(());
    };
    let logical = map_acquire_result(mapping, vk::Result::SUCCESS, unsafe { *image_index })?
        .expect("successful acquire must produce a logical slot");
    unsafe { *image_index = logical };
    Ok(())
}

fn acquired(result: vk::Result) -> bool {
    matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR)
}

fn map_acquire_result(
    mapping: &mut Mapping,
    result: vk::Result,
    physical_index: u32,
) -> Result<Option<u32>, vk::Result> {
    if !acquired(result) {
        return Err(result);
    }
    mapping
        .acquire(physical_index)
        .map(Some)
        .map_err(|_| vk::Result::ERROR_OUT_OF_DATE_KHR)
}

unsafe fn copy_virtual_images(
    virtual_images: &[vk::Image],
    image_count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
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
    if capacity < total as usize {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}

unsafe fn get_swapchain_images_inner(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    image_count: *mut u32,
    images: *mut vk::Image,
) -> vk::Result {
    if let Err(error) = reject_retired_swapchain(swapchain) {
        return error;
    }
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
        return unsafe { copy_virtual_images(&virtual_images, image_count, images) };
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
    if let Err(error) = reject_retired_swapchain(swapchain) {
        return error;
    }
    let virtual_swapchain = virtual_physical_swapchain(swapchain);
    let physical_swapchain = virtual_swapchain
        .as_ref()
        .map_or(swapchain, |(_, physical)| *physical);
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImageKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let acquire: vk::PFN_vkAcquireNextImageKHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe {
        acquire(
            device,
            physical_swapchain,
            timeout,
            semaphore,
            fence,
            image_index,
        )
    };
    if acquired(result)
        && let Some((state, _)) = virtual_swapchain
        && let Err(error) = map_acquired_image(&state, image_index)
    {
        return error;
    }
    result
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
    if info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let logical_swapchain = unsafe { (*info).swapchain };
    if let Err(error) = reject_retired_swapchain(logical_swapchain) {
        return error;
    }
    let virtual_swapchain = virtual_physical_swapchain(logical_swapchain);
    let mut modified = unsafe { *info };
    if let Some((_, physical)) = &virtual_swapchain {
        modified.swapchain = *physical;
    }
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImage2KHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let acquire: vk::PFN_vkAcquireNextImage2KHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { acquire(device, &modified, image_index) };
    if acquired(result)
        && let Some((state, _)) = virtual_swapchain
        && let Err(error) = map_acquired_image(&state, image_index)
    {
        return error;
    }
    result
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

#[cfg(test)]
mod tests {
    use super::{copy_virtual_images, map_acquire_result, reject_retired_swapchain};
    use crate::state::retire_swapchain;
    use ash::vk;
    use ash::vk::Handle;

    fn images() -> [vk::Image; 3] {
        [
            vk::Image::from_raw(1),
            vk::Image::from_raw(2),
            vk::Image::from_raw(3),
        ]
    }

    #[test]
    fn reports_virtual_image_count_without_writing() {
        let source = images();
        let mut count = 0;

        let result = unsafe { copy_virtual_images(&source, &mut count, std::ptr::null_mut()) };

        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(count, source.len() as u32);
    }

    #[test]
    fn copies_all_virtual_images_when_capacity_is_sufficient() {
        let source = images();
        let mut count = source.len() as u32;
        let mut destination = [vk::Image::null(); 3];

        let result = unsafe { copy_virtual_images(&source, &mut count, destination.as_mut_ptr()) };

        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(count, source.len() as u32);
        assert_eq!(destination, source);
    }

    #[test]
    fn reports_incomplete_and_count_of_written_images() {
        let source = images();
        let mut count = 1;
        let mut destination = [vk::Image::null(); 1];

        let result = unsafe { copy_virtual_images(&source, &mut count, destination.as_mut_ptr()) };

        assert_eq!(result, vk::Result::INCOMPLETE);
        assert_eq!(count, 1);
        assert_eq!(destination[0], source[0]);
    }

    #[test]
    fn retired_logical_tokens_return_a_safe_error_without_driver_fallback() {
        use ash::vk::Handle;

        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0066);
        retire_swapchain(token);

        assert_eq!(
            reject_retired_swapchain(token),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        );
    }

    #[test]
    fn hook_acquire_present_round_trip_supports_distinct_indices_and_statuses() {
        use crate::mapping::Mapping;

        let mut mapping = Mapping::new(12, 2);
        let logical = map_acquire_result(&mut mapping, vk::Result::SUBOPTIMAL_KHR, 4)
            .unwrap()
            .unwrap();

        assert_eq!(logical, 0);
        assert_eq!(mapping.resolve(logical), Some(4));
        assert_eq!(mapping.present(logical), Ok(4));
        assert_eq!(
            map_acquire_result(&mut mapping, vk::Result::ERROR_OUT_OF_DATE_KHR, 1),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        );
    }
}
