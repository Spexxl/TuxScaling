use super::*;
#[cfg(test)]
use crate::mapping::Mapping;
use crate::mapping::ReleasePlan;
use std::time::Instant;

fn reject_retired_swapchain(swapchain: vk::SwapchainKHR) -> Result<(), vk::Result> {
    if is_retired_swapchain(swapchain) || is_unknown_logical_swapchain(swapchain) {
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
    } else {
        Ok(())
    }
}

type VirtualSwapchain = (Arc<Mutex<SwapchainState>>, vk::SwapchainKHR);

fn virtual_physical_swapchain(
    logical: vk::SwapchainKHR,
) -> Result<Option<VirtualSwapchain>, vk::Result> {
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&logical)
        .cloned();
    let Some(state) = state else {
        return Ok(None);
    };
    let state_guard = state.lock().unwrap_or_else(|error| error.into_inner());
    if state_guard.lifecycle.blocks_frame_operations() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let physical = state_guard
        .mapping
        .as_ref()
        .and_then(|_| {
            state_guard
                .contract
                .as_ref()
                .map(|contract| contract.handle())
        })
        .map(|_| state_guard.physical_handle);
    let Some(physical) = physical else {
        return Ok(None);
    };
    drop(state_guard);
    Ok(Some((state, physical)))
}

fn map_acquired_image(
    state: &Arc<Mutex<SwapchainState>>,
    logical_index: u32,
    image_index: *mut u32,
) -> Result<(), vk::Result> {
    if image_index.is_null() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    if state.lifecycle.blocks_frame_operations() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let Some(mapping) = state.mapping.as_mut() else {
        return Ok(());
    };
    mapping
        .bind_reserved(logical_index, unsafe { *image_index })
        .map_err(|_| vk::Result::ERROR_OUT_OF_DATE_KHR)?;
    unsafe { *image_index = logical_index };
    Ok(())
}

fn reserve_image_slot(state: &Arc<Mutex<SwapchainState>>) -> Result<Option<u32>, vk::Result> {
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    if state.lifecycle.blocks_frame_operations() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let Some(mapping) = state.mapping.as_mut() else {
        return Ok(None);
    };
    mapping
        .reserve_slot()
        .map(Some)
        .map_err(|_| vk::Result::ERROR_OUT_OF_DATE_KHR)
}

fn cancel_reserved_image(state: &Arc<Mutex<SwapchainState>>, logical_index: Option<u32>) {
    let Some(logical_index) = logical_index else {
        return;
    };
    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(mapping) = state.mapping.as_mut() {
        mapping.cancel_reservation(logical_index);
    }
}

unsafe fn release_acquired_physical_image(
    device: vk::Device,
    physical_swapchain: vk::SwapchainKHR,
    physical_index: u32,
) -> bool {
    let Some(release) =
        (unsafe { downstream_release_proc(device, c"vkReleaseSwapchainImagesEXT") })
    else {
        return false;
    };
    let indices = [physical_index];
    let info = vk::ReleaseSwapchainImagesInfoEXT::default()
        .swapchain(physical_swapchain)
        .image_indices(&indices);
    unsafe { release(device, &info) == vk::Result::SUCCESS }
}

unsafe fn downstream_release_proc(
    device: vk::Device,
    preferred_name: &CStr,
) -> Option<vk::PFN_vkReleaseSwapchainImagesEXT> {
    let alternate_name = if preferred_name.to_bytes() == b"vkReleaseSwapchainImagesKHR" {
        c"vkReleaseSwapchainImagesEXT"
    } else {
        c"vkReleaseSwapchainImagesKHR"
    };
    let proc = unsafe { device_downstream(device, preferred_name) }
        .or_else(|| unsafe { device_downstream(device, alternate_name) })?;
    Some(unsafe {
        std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkReleaseSwapchainImagesEXT>(
            proc,
        )
    })
}

fn release_info_with_plan<'a>(
    original: &vk::ReleaseSwapchainImagesInfoEXT<'a>,
    physical_swapchain: vk::SwapchainKHR,
    plan: &ReleasePlan,
) -> vk::ReleaseSwapchainImagesInfoEXT<'a> {
    let mut modified = *original;
    modified.swapchain = physical_swapchain;
    modified.image_index_count = plan.physical_indices().len() as u32;
    modified.p_image_indices = plan.physical_indices().as_ptr();
    modified
}

struct ReleaseTranslation {
    state: Arc<Mutex<SwapchainState>>,
    game_surface: vk::SurfaceKHR,
    physical_swapchain: vk::SwapchainKHR,
    plan: ReleasePlan,
}

enum ReleaseTarget {
    Direct,
    Virtual(ReleaseTranslation),
}

fn prepare_release_target(
    info: &vk::ReleaseSwapchainImagesInfoEXT<'_>,
) -> Result<ReleaseTarget, vk::Result> {
    let logical_swapchain = info.swapchain;
    if is_retired_swapchain(logical_swapchain) {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&logical_swapchain)
        .cloned();
    let Some(state) = state else {
        return if is_unknown_logical_swapchain(logical_swapchain) {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        } else {
            Ok(ReleaseTarget::Direct)
        };
    };
    let state_guard = state.lock().unwrap_or_else(|error| error.into_inner());
    if state_guard.lifecycle.blocks_frame_operations() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let Some(mapping) = state_guard.mapping.as_ref() else {
        return if LogicalSwapchainHandle::is_reserved(logical_swapchain.as_raw()) {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        } else {
            Ok(ReleaseTarget::Direct)
        };
    };
    if info.image_index_count != 0 && info.p_image_indices.is_null() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let logical_indices = if info.image_index_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(info.p_image_indices, info.image_index_count as usize) }
    };
    let plan = mapping
        .plan_release(logical_indices)
        .map_err(|_| vk::Result::ERROR_OUT_OF_DATE_KHR)?;
    Ok(ReleaseTarget::Virtual(ReleaseTranslation {
        state: state.clone(),
        game_surface: state_guard.game_surface,
        physical_swapchain: state_guard.physical_handle,
        plan,
    }))
}

unsafe fn release_swapchain_images_inner(
    device: vk::Device,
    info: *const vk::ReleaseSwapchainImagesInfoEXT<'_>,
    preferred_name: &CStr,
) -> vk::Result {
    if info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let original = unsafe { &*info };
    let target = match prepare_release_target(original) {
        Ok(target) => target,
        Err(error) => return error,
    };
    let mut modified = *original;
    let virtual_translation = match &target {
        ReleaseTarget::Direct => None,
        ReleaseTarget::Virtual(translation) => {
            modified =
                release_info_with_plan(original, translation.physical_swapchain, &translation.plan);
            Some(translation)
        }
    };
    let Some(release) = (unsafe { downstream_release_proc(device, preferred_name) }) else {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let result = unsafe { release(device, &modified) };
    let Some(translation) = virtual_translation else {
        return result;
    };
    if result != vk::Result::SUCCESS {
        return result;
    }
    let (committed, idle, report) = {
        let mut state = translation
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(mapping) = state.mapping.as_mut() else {
            return vk::Result::ERROR_OUT_OF_DATE_KHR;
        };
        let (committed, idle) = {
            let committed = mapping.commit_release(&translation.plan);
            (committed, committed && mapping.is_idle())
        };
        let report = committed && !state.maintenance_release_reported;
        if report {
            state.maintenance_release_reported = true;
        }
        (committed, idle, report)
    };
    if !committed {
        eprintln!("TuxScaling evidence event=maintenance1_release_rejected reason=stale_mapping");
        return vk::Result::ERROR_OUT_OF_DATE_KHR;
    }
    if report {
        eprintln!(
            "TuxScaling evidence event=maintenance1_release logical_count={} physical_count={} result=SUCCESS",
            original.image_index_count,
            translation.plan.physical_indices().len(),
        );
    }
    if idle
        && let Some(device_state) = devices()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&device)
            .cloned()
    {
        unsafe {
            super::creation::observe_surface_negotiation(
                &device_state,
                translation.game_surface,
                Instant::now(),
            );
        }
    }
    result
}

pub(super) unsafe extern "system" fn release_swapchain_images(
    device: vk::Device,
    info: *const vk::ReleaseSwapchainImagesInfoEXT<'_>,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        release_swapchain_images_inner(device, info, c"vkReleaseSwapchainImagesEXT")
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

pub(super) unsafe extern "system" fn release_swapchain_images_khr(
    device: vk::Device,
    info: *const vk::ReleaseSwapchainImagesInfoEXT<'_>,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        release_swapchain_images_inner(device, info, c"vkReleaseSwapchainImagesKHR")
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

fn acquired(result: vk::Result) -> bool {
    matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR)
}

#[cfg(test)]
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
    let tracked_state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&swapchain)
        .cloned();
    let virtual_images = if let Some(state) = tracked_state {
        let state = state.lock().unwrap_or_else(|error| error.into_inner());
        if state.lifecycle.blocks_frame_operations() {
            return vk::Result::ERROR_OUT_OF_DATE_KHR;
        }
        state.contract.as_ref().map(|contract| {
            let snapshot = contract.logical_snapshot();
            debug_assert_eq!(contract.logical_images(), snapshot.images());
            snapshot.images().to_vec()
        })
    } else {
        None
    };
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
    let virtual_swapchain = match virtual_physical_swapchain(swapchain) {
        Ok(virtual_swapchain) => virtual_swapchain,
        Err(error) => return error,
    };
    let physical_swapchain = virtual_swapchain
        .as_ref()
        .map_or(swapchain, |(_, physical)| *physical);
    if image_index.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let reserved_logical = if let Some((state, _)) = virtual_swapchain.as_ref() {
        match reserve_image_slot(state) {
            Ok(reserved) => reserved,
            Err(error) => return error,
        }
    } else {
        None
    };
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImageKHR") }) else {
        if let Some((state, _)) = virtual_swapchain.as_ref() {
            cancel_reserved_image(state, reserved_logical);
        }
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
    if !acquired(result) {
        eprintln!(
            "TuxScaling evidence event=acquire_result swapchain=0x{:x} result={result:?}",
            swapchain.as_raw(),
        );
    }
    if let Some((state, _)) = virtual_swapchain {
        if acquired(result) {
            if let Some(logical_index) = reserved_logical
                && let Err(error) = map_acquired_image(&state, logical_index, image_index)
            {
                cancel_reserved_image(&state, Some(logical_index));
                let physical_index = unsafe { *image_index };
                if !unsafe {
                    release_acquired_physical_image(device, physical_swapchain, physical_index)
                } {
                    eprintln!(
                        "TuxScaling: failed to bind acquired physical image; downstream WSI did not expose a safe release path"
                    );
                }
                return error;
            }
        } else {
            cancel_reserved_image(&state, reserved_logical);
        }
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
    if image_index.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let logical_swapchain = unsafe { (*info).swapchain };
    if let Err(error) = reject_retired_swapchain(logical_swapchain) {
        return error;
    }
    let virtual_swapchain = match virtual_physical_swapchain(logical_swapchain) {
        Ok(virtual_swapchain) => virtual_swapchain,
        Err(error) => return error,
    };
    let reserved_logical = if let Some((state, _)) = virtual_swapchain.as_ref() {
        match reserve_image_slot(state) {
            Ok(reserved) => reserved,
            Err(error) => return error,
        }
    } else {
        None
    };
    let mut modified = unsafe { *info };
    if let Some((_, physical)) = &virtual_swapchain {
        modified.swapchain = *physical;
    }
    let Some(proc) = (unsafe { device_downstream(device, c"vkAcquireNextImage2KHR") }) else {
        if let Some((state, _)) = virtual_swapchain.as_ref() {
            cancel_reserved_image(state, reserved_logical);
        }
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let acquire: vk::PFN_vkAcquireNextImage2KHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { acquire(device, &modified, image_index) };
    if let Some((state, _)) = virtual_swapchain {
        if acquired(result) {
            if let Some(logical_index) = reserved_logical
                && let Err(error) = map_acquired_image(&state, logical_index, image_index)
            {
                cancel_reserved_image(&state, Some(logical_index));
                let physical_index = unsafe { *image_index };
                if !unsafe {
                    release_acquired_physical_image(device, modified.swapchain, physical_index)
                } {
                    eprintln!(
                        "TuxScaling: failed to bind acquired physical image; downstream WSI did not expose a safe release path"
                    );
                }
                return error;
            }
        } else {
            cancel_reserved_image(&state, reserved_logical);
        }
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
    use super::{
        copy_virtual_images, map_acquire_result, prepare_release_target, reject_retired_swapchain,
        release_info_with_plan,
    };
    use crate::mapping::Mapping;
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

    #[test]
    fn unequal_logical_count_rejects_before_downstream_acquire() {
        use crate::mapping::Mapping;

        let mut mapping = Mapping::new(3, 1);
        assert_eq!(mapping.acquire(2), Ok(0));

        let mut downstream_acquires = 0;
        let reservation = mapping.reserve_slot();
        if reservation.is_ok() {
            downstream_acquires += 1;
        }

        assert_eq!(
            reservation,
            Err(crate::mapping::AcquireError::NoLogicalSlot)
        );
        assert_eq!(downstream_acquires, 0);
    }

    #[test]
    fn release_hook_recorder_receives_physical_handle_and_indices_in_order() {
        let mut mapping = Mapping::new(14, 3);
        let first = mapping.acquire(4).unwrap();
        let second = mapping.acquire(1).unwrap();
        let logical_indices = [second, first];
        let plan = mapping.plan_release(&logical_indices).unwrap();
        let logical_swapchain = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0041);
        let physical_swapchain = vk::SwapchainKHR::from_raw(0x0042);
        let original = vk::ReleaseSwapchainImagesInfoEXT::default()
            .swapchain(logical_swapchain)
            .image_indices(&logical_indices);

        let modified = release_info_with_plan(&original, physical_swapchain, &plan);
        let recorded_indices = unsafe {
            std::slice::from_raw_parts(
                modified.p_image_indices,
                modified.image_index_count as usize,
            )
        };

        assert_eq!(modified.swapchain, physical_swapchain);
        assert_eq!(recorded_indices, &[1, 4]);
        assert_eq!(mapping.resolve(first), Some(4));
        assert_eq!(mapping.resolve(second), Some(1));
    }

    #[test]
    fn invalid_tagged_release_is_rejected_before_downstream_lookup() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0043);
        let info = vk::ReleaseSwapchainImagesInfoEXT::default().swapchain(token);

        assert!(matches!(
            prepare_release_target(&info),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        ));
    }
}
