use super::*;

unsafe fn submit_overlay(
    queue: vk::Queue,
    queue_state: QueueState,
    info: &vk::PresentInfoKHR<'_>,
    handoff: &mut Option<vk::Semaphore>,
) -> Option<vk::Semaphore> {
    if info.swapchain_count != 1
        || !info.p_next.is_null()
        || info.p_swapchains.is_null()
        || info.p_image_indices.is_null()
        || (info.wait_semaphore_count != 0 && info.p_wait_semaphores.is_null())
    {
        return None;
    }
    let swapchain = unsafe { *info.p_swapchains };
    let image_index = unsafe { *info.p_image_indices };
    let device_state = devices()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&queue_state.device)
        .cloned()?;
    let state = swapchains()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&swapchain)
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
    let mut swapchain_state = state.lock().unwrap_or_else(|e| e.into_inner());
    if swapchain_state.device != queue_state.device {
        return None;
    }
    let prepared = unsafe {
        swapchain_state.overlay.prepare_frame(
            &device_state.device,
            queue,
            queue_state.family_index,
            image_index,
        )
    };
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(_) => {
            swapchain_state.overlay.disable();
            return None;
        }
    };
    let wait_semaphores = if info.wait_semaphore_count == 0 {
        &[]
    } else {
        unsafe {
            std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
        }
    };
    let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; wait_semaphores.len()];
    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(std::slice::from_ref(&prepared.command_buffer))
        .signal_semaphores(std::slice::from_ref(&prepared.render_complete));
    if unsafe { device_state.device.reset_fences(&[prepared.fence]) }.is_err() {
        swapchain_state.overlay.disable();
        return None;
    }
    let result = unsafe {
        device_state
            .device
            .queue_submit(queue, std::slice::from_ref(&submit_info), prepared.fence)
    };
    if result.is_ok() {
        *handoff = Some(prepared.render_complete);
        swapchain_state.overlay.submitted();
        Some(prepared.render_complete)
    } else {
        swapchain_state.overlay.disable();
        None
    }
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
        let result = unsafe { present(queue, &modified) };
        if result != vk::Result::SUCCESS
            && let Some(state) = swapchains()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&unsafe { *info.p_swapchains })
                .cloned()
        {
            state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .overlay
                .presentation_failed();
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
