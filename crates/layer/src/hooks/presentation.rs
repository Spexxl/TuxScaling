use super::present_chain::{MappedPresentRegions, PresentChain};
use super::*;
use crate::hooks::present_id::PresentIdReservation;
use crate::mapping::{LogicalSwapchainHandle, rewrite_present_array};
use crate::recovery::{FrameBinding, PresentationPath};
use ash::vk::Handle;
use std::time::Instant;

struct PresentTranslation {
    swapchains: Vec<vk::SwapchainKHR>,
    image_indices: Vec<u32>,
    releases: Vec<(Arc<Mutex<SwapchainState>>, u32)>,
    present_targets: Vec<Option<PresentIdTarget>>,
    present_ids: Vec<PresentIdReservationRecord>,
}

struct PresentIdTarget {
    state: Arc<Mutex<SwapchainState>>,
    generation: u64,
    physical: vk::SwapchainKHR,
}

struct PresentIdReservationRecord {
    state: Arc<Mutex<SwapchainState>>,
    reservation: PresentIdReservation,
}

unsafe fn translate_present(
    info: &vk::PresentInfoKHR<'_>,
) -> Result<Option<PresentTranslation>, vk::Result> {
    if info.swapchain_count == 0 {
        return Ok(None);
    }
    if info.p_swapchains.is_null() || info.p_image_indices.is_null() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let swapchain_handles =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) };
    let image_indices =
        unsafe { std::slice::from_raw_parts(info.p_image_indices, info.swapchain_count as usize) };
    let states = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut translated = PresentTranslation {
        swapchains: Vec::with_capacity(swapchain_handles.len()),
        image_indices: Vec::with_capacity(image_indices.len()),
        releases: Vec::new(),
        present_targets: Vec::with_capacity(swapchain_handles.len()),
        present_ids: Vec::new(),
    };
    let mut changed = false;
    for (logical_handle, logical_index) in swapchain_handles.iter().zip(image_indices) {
        if is_retired_swapchain(*logical_handle) {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        let state = states.get(logical_handle).cloned();
        let Some(state) = state else {
            if is_retired_swapchain(*logical_handle)
                || LogicalSwapchainHandle::is_reserved(logical_handle.as_raw())
            {
                return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            translated.swapchains.push(*logical_handle);
            translated.image_indices.push(*logical_index);
            translated.present_targets.push(None);
            continue;
        };
        let state_guard = state.lock().unwrap_or_else(|error| error.into_inner());
        if state_guard.lifecycle.blocks_frame_operations() {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        let Some(mapping) = state_guard.mapping.as_ref() else {
            translated.swapchains.push(*logical_handle);
            translated.image_indices.push(*logical_index);
            translated.present_targets.push(None);
            continue;
        };
        let (physical_swapchain, physical_index) =
            rewrite_present_array(mapping, state_guard.physical_handle, *logical_index)?;
        translated.swapchains.push(physical_swapchain);
        translated.image_indices.push(physical_index);
        translated.releases.push((state.clone(), *logical_index));
        translated.present_targets.push(Some(PresentIdTarget {
            state: state.clone(),
            generation: state_guard.generation,
            physical: state_guard.physical_handle,
        }));
        changed = true;
    }
    Ok(changed.then_some(translated))
}

unsafe fn stage_present_ids(
    info: &vk::PresentInfoKHR<'_>,
    chain: &PresentChain<'_>,
    translation: &mut PresentTranslation,
) -> Result<(), vk::Result> {
    let ids = unsafe { chain.ids_slice() };
    if ids.is_empty() {
        return Ok(());
    }
    if ids.len() != translation.present_targets.len() || info.swapchain_count as usize != ids.len()
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut staged = Vec::new();
    for (target, id) in translation.present_targets.iter().zip(ids) {
        let Some(target) = target else {
            continue;
        };
        let mut state = target
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.lifecycle.blocks_frame_operations()
            || state.generation != target.generation
            || state.physical_handle != target.physical
        {
            drop(state);
            rollback_present_ids(&staged);
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        let reservation = match state.present_ids.stage(
            target.generation,
            target.physical,
            std::slice::from_ref(id),
        ) {
            Ok(reservation) => reservation,
            Err(_) => {
                drop(state);
                rollback_present_ids(&staged);
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
        };
        drop(state);
        staged.push(PresentIdReservationRecord {
            state: target.state.clone(),
            reservation,
        });
    }
    translation.present_ids = staged;
    Ok(())
}

fn commit_present_ids(translation: &mut PresentTranslation) {
    for record in translation.present_ids.drain(..) {
        let mut state = record
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _ = state.present_ids.commit(record.reservation);
    }
}

fn rollback_present_ids(records: &[PresentIdReservationRecord]) {
    for record in records {
        let mut state = record
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _ = state.present_ids.rollback(record.reservation.clone());
    }
}

fn rollback_translation_present_ids(translation: &mut PresentTranslation) {
    rollback_present_ids(&translation.present_ids);
    translation.present_ids.clear();
}

fn release_presented_images(translation: &PresentTranslation) {
    for (state, logical_index) in &translation.releases {
        if let Ok(mut state) = state.lock()
            && let Some(mapping) = state.mapping.as_mut()
        {
            let _ = mapping.present(*logical_index);
        }
    }
}

fn should_retry_native_publication(result: vk::Result) -> bool {
    result == vk::Result::SUCCESS
}

fn present_committed(result: vk::Result) -> bool {
    matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR)
}

fn with_overlay_wait<R>(
    mut modified: vk::PresentInfoKHR<'_>,
    overlay_wait: Option<vk::Semaphore>,
    invoke: impl FnOnce(&vk::PresentInfoKHR<'_>) -> R,
) -> R {
    let wait_storage = overlay_wait.map(|semaphore| [semaphore]);
    if let Some(wait_storage) = wait_storage.as_ref() {
        modified.wait_semaphore_count = 1;
        modified.p_wait_semaphores = wait_storage.as_ptr();
    }
    invoke(&modified)
}

fn report_maintenance_present(info: &vk::PresentInfoKHR<'_>, chain: &PresentChain<'_>) {
    let fence_count = chain
        .fences
        .as_ref()
        .map_or(0, |fences| fences.swapchain_count);
    let mode_count = chain
        .modes
        .as_ref()
        .map_or(0, |modes| modes.swapchain_count);
    if fence_count == 0 && mode_count == 0 {
        return;
    }
    let presented = if info.swapchain_count == 0 || info.p_swapchains.is_null() {
        return;
    } else {
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) }
    };
    let states = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for swapchain in presented {
        let Some(state) = states.get(swapchain) else {
            continue;
        };
        let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
        if state.mapping.is_some() && !state.maintenance_present_reported {
            state.maintenance_present_reported = true;
            eprintln!(
                "TuxScaling evidence event=maintenance1_present fences={} modes={} virtual=1",
                fence_count, mode_count,
            );
            break;
        }
    }
}

unsafe fn observe_presented_surfaces(queue_state: QueueState, info: &vk::PresentInfoKHR<'_>) {
    if info.swapchain_count == 0 || info.p_swapchains.is_null() {
        return;
    }
    let handles =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) };
    let surfaces = {
        let states = swapchains()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        handles
            .iter()
            .filter_map(|handle| states.get(handle))
            .filter_map(|state| {
                state.lock().ok().and_then(|state| {
                    (!state.lifecycle.blocks_frame_operations()).then_some(state.game_surface)
                })
            })
            .collect::<Vec<_>>()
    };
    let Some(device_state) = devices()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&queue_state.device)
        .cloned()
    else {
        return;
    };
    for surface in surfaces {
        unsafe {
            super::creation::observe_surface_negotiation(&device_state, surface, Instant::now());
        }
    }
}

unsafe fn uses_virtual_output(info: &vk::PresentInfoKHR<'_>) -> bool {
    if info.swapchain_count == 0 || info.p_swapchains.is_null() {
        return false;
    }
    let presented =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) };
    let states = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    presented.iter().any(|swapchain| {
        states.get(swapchain).is_some_and(|state| {
            let state = state.lock().unwrap_or_else(|error| error.into_inner());
            !state.lifecycle.blocks_frame_operations() && state.contract.is_some()
        })
    })
}

unsafe fn map_present_regions(
    info: &vk::PresentInfoKHR<'_>,
    chain: &PresentChain<'_>,
) -> Option<MappedPresentRegions> {
    chain.regions.as_ref()?;
    if info.swapchain_count == 0 || info.p_swapchains.is_null() {
        return Some(MappedPresentRegions::default());
    }
    let source_regions = unsafe { chain.regions_slice() };
    let presented =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) };
    if source_regions.len() != presented.len() {
        return None;
    }
    let states = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mapped = presented
        .iter()
        .zip(source_regions)
        .map(|(swapchain, region)| {
            if region.rectangle_count == 0 {
                return Vec::new();
            }
            let rectangles = unsafe {
                std::slice::from_raw_parts(region.p_rectangles, region.rectangle_count as usize)
            };
            let state = states.get(swapchain).cloned();
            let virtual_output = state.as_ref().is_some_and(|state| {
                state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .virtual_images
                    .is_some()
            });
            rectangles
                .iter()
                .map(|rectangle| {
                    if virtual_output {
                        state.as_ref().and_then(|state| state.lock().ok()).map_or(
                            *rectangle,
                            |state| {
                                state.overlay.as_ref().map_or(*rectangle, |overlay| {
                                    let mapped = overlay.map_damage_rect(vk::Rect2D {
                                        offset: rectangle.offset,
                                        extent: rectangle.extent,
                                    });
                                    vk::RectLayerKHR {
                                        offset: mapped.offset,
                                        extent: mapped.extent,
                                        layer: rectangle.layer,
                                    }
                                })
                            },
                        )
                    } else {
                        *rectangle
                    }
                })
                .collect()
        })
        .collect();
    Some(MappedPresentRegions::from_rectangles(mapped))
}

unsafe fn submit_overlay(
    queue: vk::Queue,
    queue_state: QueueState,
    info: &vk::PresentInfoKHR<'_>,
    handoff: &mut Option<vk::Semaphore>,
) -> Option<vk::Semaphore> {
    if info.swapchain_count == 0
        || info.p_swapchains.is_null()
        || info.p_image_indices.is_null()
        || (info.wait_semaphore_count != 0 && info.p_wait_semaphores.is_null())
    {
        return None;
    }
    let device_state = devices()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&queue_state.device)
        .cloned()?;
    if !queue_state.processing_allowed
        || !device_state
            .queue_families
            .get(queue_state.family_index as usize)
            .is_some_and(|family| {
                family
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
    {
        return None;
    }
    let presented_swapchains =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) };
    let image_indices =
        unsafe { std::slice::from_raw_parts(info.p_image_indices, info.swapchain_count as usize) };
    let states = swapchains().lock().unwrap_or_else(|e| e.into_inner());
    let tracked = presented_swapchains
        .iter()
        .zip(image_indices)
        .filter_map(|(swapchain, image_index)| {
            let state = states.get(swapchain).cloned()?;
            let physical_index = state
                .lock()
                .ok()
                .and_then(|state| {
                    state
                        .mapping
                        .as_ref()
                        .and_then(|mapping| mapping.resolve(*image_index))
                })
                .unwrap_or(*image_index);
            Some((state, *image_index, physical_index))
        })
        .collect::<Vec<_>>();
    drop(states);
    let mut prepared = Vec::with_capacity(tracked.len());
    for (state, logical_index, physical_index) in tracked {
        let mut swapchain_state = state.lock().unwrap_or_else(|e| e.into_inner());
        if swapchain_state.device != queue_state.device {
            continue;
        }
        if swapchain_state.lifecycle.blocks_frame_operations() {
            continue;
        }
        let has_contract = swapchain_state.contract.is_some();
        let temporal_allowed = swapchain_state
            .contract
            .as_ref()
            .is_some_and(|contract| contract.presentation_path() == PresentationPath::Temporal);
        if let Some(contract) = swapchain_state.contract.as_ref()
            && contract
                .bind(FrameBinding::new(logical_index, physical_index))
                .is_err()
        {
            continue;
        }
        let Some(overlay) = swapchain_state.overlay.as_mut() else {
            continue;
        };
        let prepared_frame = if has_contract && !temporal_allowed {
            unsafe {
                overlay.prepare_spatial_fallback(
                    queue,
                    queue_state.family_index,
                    logical_index,
                    physical_index,
                    vk::Result::NOT_READY,
                )
            }
        } else {
            unsafe {
                overlay.prepare_frame(
                    &device_state.device,
                    queue,
                    queue_state.family_index,
                    logical_index,
                    physical_index,
                )
            }
        };
        match prepared_frame {
            Ok(frame) => prepared.push((state.clone(), frame)),
            Err(error) => match unsafe {
                overlay.prepare_spatial_fallback(
                    queue,
                    queue_state.family_index,
                    logical_index,
                    physical_index,
                    error,
                )
            } {
                Ok(frame) => prepared.push((state.clone(), frame)),
                Err(fallback_error) => {
                    eprintln!(
                        "TuxScaling: spatial fallback failed ({fallback_error:?}); bypassing overlay"
                    );
                    overlay.disable();
                }
            },
        }
    }
    if prepared.is_empty() {
        return None;
    }
    let wait_semaphores = if info.wait_semaphore_count == 0 {
        &[]
    } else {
        unsafe {
            std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
        }
    };
    let mut previous = None;
    for (state, frame) in prepared {
        let previous_wait;
        let waits = if let Some(semaphore) = previous {
            previous_wait = [semaphore];
            &previous_wait
        } else {
            wait_semaphores
        };
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&frame.command_buffer))
            .signal_semaphores(std::slice::from_ref(&frame.render_complete));
        if unsafe { device_state.device.reset_fences(&[frame.fence]) }.is_err()
            || unsafe {
                device_state.device.queue_submit(
                    queue,
                    std::slice::from_ref(&submit_info),
                    frame.fence,
                )
            }
            .is_err()
        {
            if let Ok(mut state) = state.lock()
                && let Some(overlay) = state.overlay.as_mut()
            {
                overlay.disable();
            }
            break;
        }
        let (overlay_report, input_route_report, maintenance_report) = if let Ok(mut state) =
            state.lock()
            && let Some(overlay) = state.overlay.as_mut()
        {
            overlay.submitted();
            let overlay_report = state.mapping.is_some() && !state.maintenance_overlay_reported;
            let input_route_report = state.present_surface.is_some() && !state.input_route_reported;
            let maintenance_report = device_state.wsi.maintenance1.enabled && overlay_report;
            if overlay_report {
                state.maintenance_overlay_reported = true;
            }
            if input_route_report {
                state.input_route_reported = true;
            }
            (overlay_report, input_route_report, maintenance_report)
        } else {
            (false, false, false)
        };
        if overlay_report {
            eprintln!("TuxScaling evidence event=overlay_submitted virtual=1");
        }
        if input_route_report && let Ok(state) = state.lock() {
            eprintln!(
                "TuxScaling evidence event=input_route_active game_surface=0x{:x} presenter_surface=0x{:x} mode=absolute bars=reject_motion_clamp_button",
                state.game_surface.as_raw(),
                state.present_surface.map_or(0, |surface| surface.as_raw()),
            );
        }
        if maintenance_report {
            eprintln!("TuxScaling evidence event=overlay_submitted maintenance1=1");
        }
        previous = Some(frame.render_complete);
    }
    *handoff = previous;
    previous
}

unsafe fn queue_present_inner(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    let Some(queue_state) = queues()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&queue)
        .copied()
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(proc) = (unsafe { device_downstream(queue_state.device, c"vkQueuePresentKHR") })
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let present: vk::PFN_vkQueuePresentKHR = unsafe { std::mem::transmute(proc) };
    if present_info.is_null() {
        return unsafe { present(queue, present_info) };
    }
    let info = unsafe { &*present_info };
    unsafe { observe_presented_surfaces(queue_state, info) };
    let translation = unsafe { translate_present(info) };
    let mut translation = match translation {
        Ok(translation) => translation,
        Err(error) => return error,
    };
    let present_chain = if unsafe { uses_virtual_output(info) } {
        match unsafe { PresentChain::parse(info.p_next, info.swapchain_count) } {
            Ok(chain) => Some(chain),
            Err(error) => {
                eprintln!(
                    "TuxScaling evidence event=maintenance1_present_rejected reason={}",
                    error.reason()
                );
                return vk::Result::ERROR_FEATURE_NOT_PRESENT;
            }
        }
    } else {
        None
    };
    if let (Some(translation), Some(chain)) = (translation.as_mut(), present_chain.as_ref())
        && let Err(error) = unsafe { stage_present_ids(info, chain, translation) }
    {
        return error;
    }
    let overlay_complete = crate::handoff::handoff(|handoff| unsafe {
        let _ = submit_overlay(queue, queue_state, info, handoff);
    });
    if overlay_complete.is_some() || translation.is_some() {
        let mut modified = *info;
        if let Some(translation) = &translation {
            modified.p_swapchains = translation.swapchains.as_ptr();
            modified.p_image_indices = translation.image_indices.as_ptr();
        }
        let mapped_regions = present_chain
            .as_ref()
            .and_then(|chain| unsafe { map_present_regions(info, chain) });
        let result = with_overlay_wait(modified, overlay_complete, |modified| {
            if let Some(chain) = present_chain.as_ref() {
                chain.with_rebuilt_chain(*modified, mapped_regions.as_ref(), |modified| unsafe {
                    present(queue, modified)
                })
            } else {
                unsafe { present(queue, modified) }
            }
        });
        if result != vk::Result::SUCCESS && result != vk::Result::SUBOPTIMAL_KHR {
            eprintln!(
                "TuxScaling evidence event=present_result result={result:?} swapchains={}",
                info.swapchain_count,
            );
        }
        if present_committed(result) {
            if let Some(translation) = translation.as_mut() {
                commit_present_ids(translation);
                release_presented_images(translation);
            }
            if should_retry_native_publication(result) {
                // The pre-present observation intentionally cannot replace a
                // generation while the submitted logical slot is still mapped.
                // Retry after releasing it so the next generation sees an idle
                // mapping without delaying or changing the WSI result. A
                // suboptimal result is returned unchanged so the application
                // can recreate before native publication is attempted.
                unsafe { observe_presented_surfaces(queue_state, info) };
            }
            if let Some(chain) = present_chain.as_ref() {
                report_maintenance_present(info, chain);
            }
        }
        if !present_committed(result)
            && let Some(translation) = translation.as_mut()
        {
            rollback_translation_present_ids(translation);
        }
        if result != vk::Result::SUCCESS && !info.p_swapchains.is_null() {
            let presented = unsafe {
                std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize)
            };
            let states = swapchains().lock().unwrap_or_else(|e| e.into_inner());
            for swapchain in presented {
                if let Some(state) = states.get(swapchain).cloned()
                    && let Ok(mut state) = state.lock()
                    && let Some(overlay) = state.overlay.as_mut()
                {
                    overlay.presentation_failed();
                }
            }
        }
        return result;
    }
    unsafe { present(queue, present_info) }
}

pub(super) unsafe extern "system" fn queue_present_khr(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        queue_present_inner(queue, present_info)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

#[cfg(test)]
mod tests {
    use super::{
        present_committed, should_retry_native_publication, translate_present, with_overlay_wait,
    };
    use crate::state::retire_swapchain;
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn present_hook_rejects_a_retired_logical_token_without_translation_fallback() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0077);
        retire_swapchain(token);
        let swapchains = [token];
        let indices = [0];
        let info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices);

        assert!(matches!(
            unsafe { translate_present(&info) },
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        ));
    }

    #[test]
    fn retries_native_publication_after_a_successful_present_release() {
        assert!(should_retry_native_publication(vk::Result::SUCCESS));
        assert!(!should_retry_native_publication(vk::Result::SUBOPTIMAL_KHR));
        assert!(!should_retry_native_publication(
            vk::Result::ERROR_OUT_OF_DATE_KHR
        ));
    }

    #[test]
    fn defers_native_publication_after_a_suboptimal_present() {
        assert!(!should_retry_native_publication(vk::Result::SUBOPTIMAL_KHR));
        assert!(present_committed(vk::Result::SUBOPTIMAL_KHR));
        assert!(present_committed(vk::Result::SUCCESS));
        assert!(!present_committed(vk::Result::ERROR_OUT_OF_DATE_KHR));
    }

    #[test]
    fn overlay_wait_semaphore_remains_valid_for_downstream_callback() {
        let semaphore = vk::Semaphore::from_raw(0xcf00_0000_00cf);
        let info = vk::PresentInfoKHR::default();

        let observed = with_overlay_wait(info, Some(semaphore), |modified| unsafe {
            assert_eq!(modified.wait_semaphore_count, 1);
            assert!(!modified.p_wait_semaphores.is_null());
            *modified.p_wait_semaphores
        });

        assert_eq!(observed, semaphore);
    }
}
