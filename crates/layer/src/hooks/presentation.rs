use super::*;

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
            state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .virtual_images
                .is_some()
        })
    })
}

unsafe fn has_only_incremental_present(next: *const std::ffi::c_void) -> bool {
    if next.is_null() {
        return false;
    }
    let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
    header.s_type == vk::StructureType::PRESENT_REGIONS_KHR && header.p_next.is_null()
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
            states
                .get(swapchain)
                .cloned()
                .map(|state| (state, *image_index))
        })
        .collect::<Vec<_>>();
    drop(states);
    let mut prepared = Vec::with_capacity(tracked.len());
    for (state, image_index) in tracked {
        let mut swapchain_state = state.lock().unwrap_or_else(|e| e.into_inner());
        if swapchain_state.device != queue_state.device {
            continue;
        }
        match unsafe {
            swapchain_state.overlay.prepare_frame(
                &device_state.device,
                queue,
                queue_state.family_index,
                image_index,
            )
        } {
            Ok(frame) => prepared.push((state.clone(), frame)),
            Err(_) => {
                swapchain_state.overlay.disable();
            }
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
            state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .overlay
                .disable();
            break;
        }
        state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .overlay
            .submitted();
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
    let overlay_complete = crate::handoff::handoff(|handoff| unsafe {
        let _ = submit_overlay(queue, queue_state, info, handoff);
    });
    if let Some(render_complete) = overlay_complete {
        let mut modified = *info;
        modified.wait_semaphore_count = 1;
        modified.p_wait_semaphores = &render_complete;
        if unsafe { uses_virtual_output(info) }
            && unsafe { has_only_incremental_present(info.p_next) }
        {
            modified.p_next = std::ptr::null();
        }
        let result = unsafe { present(queue, &modified) };
        if result != vk::Result::SUCCESS && !info.p_swapchains.is_null() {
            let presented = unsafe {
                std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize)
            };
            let states = swapchains().lock().unwrap_or_else(|e| e.into_inner());
            for swapchain in presented {
                if let Some(state) = states.get(swapchain).cloned() {
                    state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .overlay
                        .presentation_failed();
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
