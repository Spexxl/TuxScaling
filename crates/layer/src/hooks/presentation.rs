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

unsafe fn find_present_regions(
    mut next: *const std::ffi::c_void,
) -> Option<vk::PresentRegionsKHR<'static>> {
    let mut regions = None;
    while !next.is_null() {
        let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
        if header.s_type == vk::StructureType::PRESENT_REGIONS_KHR {
            regions = Some(unsafe { *next.cast::<vk::PresentRegionsKHR<'static>>() });
        } else if !matches!(
            header.s_type,
            vk::StructureType::PRESENT_ID_KHR | vk::StructureType::PRESENT_TIMES_INFO_GOOGLE
        ) {
            return None;
        }
        next = header.p_next.cast();
    }
    regions
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
            Err(error) => match unsafe {
                swapchain_state.overlay.prepare_spatial_fallback(
                    queue,
                    queue_state.family_index,
                    image_index,
                    error,
                )
            } {
                Ok(frame) => prepared.push((state.clone(), frame)),
                Err(fallback_error) => {
                    eprintln!(
                        "TuxScaling: spatial fallback failed ({fallback_error:?}); bypassing overlay"
                    );
                    swapchain_state.overlay.disable();
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
        let mut mapped_rectangles = Vec::<Vec<vk::RectLayerKHR>>::new();
        let mapped_regions: Vec<vk::PresentRegionKHR<'_>>;
        let mut mapped_present_regions = vk::PresentRegionsKHR::default();
        let mut present_id = None;
        let mut present_times = None;
        let mut chain = info.p_next;
        while !chain.is_null() {
            let header = unsafe { &*chain.cast::<vk::BaseInStructure<'_>>() };
            match header.s_type {
                vk::StructureType::PRESENT_ID_KHR => {
                    present_id = Some(unsafe { *chain.cast::<vk::PresentIdKHR<'_>>() })
                }
                vk::StructureType::PRESENT_TIMES_INFO_GOOGLE => {
                    present_times = Some(unsafe { *chain.cast::<vk::PresentTimesInfoGOOGLE<'_>>() })
                }
                _ => {}
            }
            chain = header.p_next.cast();
        }
        if unsafe { uses_virtual_output(info) }
            && let Some(source) = unsafe { find_present_regions(info.p_next) }
        {
            let valid = source.swapchain_count == info.swapchain_count
                && (source.swapchain_count == 0 || !source.p_regions.is_null());
            if valid {
                let source_regions = unsafe {
                    std::slice::from_raw_parts(source.p_regions, source.swapchain_count as usize)
                };
                let presented = unsafe {
                    std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize)
                };
                let states = swapchains()
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                for (swapchain, region) in presented.iter().zip(source_regions) {
                    let state = states.get(swapchain).cloned();
                    let virtual_output = state.as_ref().is_some_and(|state| {
                        state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .virtual_images
                            .is_some()
                    });
                    let region_rectangles = if region.rectangle_count == 0 {
                        Vec::new()
                    } else if region.p_rectangles.is_null() {
                        mapped_rectangles.clear();
                        break;
                    } else {
                        let rectangles = unsafe {
                            std::slice::from_raw_parts(
                                region.p_rectangles,
                                region.rectangle_count as usize,
                            )
                        };
                        rectangles
                            .iter()
                            .map(|rectangle| {
                                if virtual_output {
                                    state.as_ref().and_then(|state| state.lock().ok()).map_or(
                                        *rectangle,
                                        |state| {
                                            let mapped =
                                                state.overlay.map_damage_rect(vk::Rect2D {
                                                    offset: rectangle.offset,
                                                    extent: rectangle.extent,
                                                });
                                            vk::RectLayerKHR {
                                                offset: mapped.offset,
                                                extent: mapped.extent,
                                                layer: rectangle.layer,
                                            }
                                        },
                                    )
                                } else {
                                    *rectangle
                                }
                            })
                            .collect()
                    };
                    mapped_rectangles.push(region_rectangles);
                }
                if mapped_rectangles.len() == info.swapchain_count as usize {
                    mapped_regions = mapped_rectangles
                        .iter()
                        .map(|rectangles| vk::PresentRegionKHR {
                            rectangle_count: rectangles.len() as u32,
                            p_rectangles: rectangles.as_ptr(),
                            _marker: std::marker::PhantomData,
                        })
                        .collect();
                    mapped_present_regions.swapchain_count = mapped_regions.len() as u32;
                    mapped_present_regions.p_regions = mapped_regions.as_ptr();
                    {
                        let mut next = std::ptr::null();
                        if let Some(times) = &mut present_times {
                            times.p_next = next;
                            next = (times as *const vk::PresentTimesInfoGOOGLE<'_>).cast();
                        }
                        if let Some(id) = &mut present_id {
                            id.p_next = next;
                            next = (id as *const vk::PresentIdKHR<'_>).cast();
                        }
                        mapped_present_regions.p_next = next;
                    }
                    modified.p_next =
                        (&mapped_present_regions as *const vk::PresentRegionsKHR<'_>).cast();
                } else {
                    modified.p_next = std::ptr::null();
                }
            } else {
                modified.p_next = std::ptr::null();
            }
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
