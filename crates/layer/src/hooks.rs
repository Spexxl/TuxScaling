use ash::vk;
use std::{
    ffi::{CStr, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
};
use tuxscaling_overlay_vulkan::{OverlaySwapchain, SwapchainInfo};

use super::{
    loader::{LAYER_LINK_INFO, LayerCreateInfo, LayerLink, next_gipa},
    state::{DeviceState, QueueState, SwapchainState, devices, instances, queues, swapchains},
};

unsafe fn find_layer_link(
    mut next: *const c_void,
    expected_type: vk::StructureType,
) -> Option<*mut LayerLink> {
    while !next.is_null() {
        let info = next.cast::<LayerCreateInfo>();
        if unsafe { (*info).s_type } == expected_type
            && unsafe { (*info).function } == LAYER_LINK_INFO
        {
            return Some(unsafe { (*info).data.layer_info });
        }
        next = unsafe { (*info).p_next };
    }
    None
}

unsafe fn downstream(instance: vk::Instance, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_instance_proc_addr = *next_gipa().lock().expect("layer state lock poisoned");
    let get_instance_proc_addr = get_instance_proc_addr?;
    unsafe { get_instance_proc_addr(instance, name.as_ptr()) }
}

unsafe fn device_downstream(device: vk::Device, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_device_proc_addr = devices()
        .lock()
        .expect("device map lock poisoned")
        .get(&device)?
        .get_device_proc_addr;
    unsafe { get_device_proc_addr(device, name.as_ptr()) }
}

unsafe fn create_instance_inner(
    create_info: *const vk::InstanceCreateInfo<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    instance: *mut vk::Instance,
) -> vk::Result {
    if create_info.is_null() || instance.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(link) = (unsafe {
        find_layer_link(
            (*create_info).p_next,
            vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
        )
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let get_instance_proc_addr = unsafe { (*link).get_instance_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*link).next = next };
    *next_gipa().lock().expect("layer state lock poisoned") = Some(get_instance_proc_addr);

    let Some(proc) = (unsafe { downstream(vk::Instance::null(), c"vkCreateInstance") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_instance: vk::PFN_vkCreateInstance = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create_instance(create_info, allocation_callbacks, instance) };
    if result == vk::Result::SUCCESS {
        let ash_instance = unsafe {
            ash::Instance::load(
                &ash::StaticFn {
                    get_instance_proc_addr,
                },
                *instance,
            )
        };
        instances()
            .lock()
            .expect("instance map lock poisoned")
            .insert(unsafe { *instance }, ash_instance);
    }
    result
}

unsafe extern "system" fn create_instance(
    create_info: *const vk::InstanceCreateInfo<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    instance: *mut vk::Instance,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        create_instance_inner(create_info, allocation_callbacks, instance)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn create_device_inner(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    if create_info.is_null() || device.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(link) = (unsafe {
        find_layer_link(
            (*create_info).p_next,
            vk::StructureType::LOADER_DEVICE_CREATE_INFO,
        )
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let get_device_proc_addr = unsafe { (*link).get_device_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*link).next = next };
    let Some(proc) = (unsafe { downstream(vk::Instance::null(), c"vkCreateDevice") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_device: vk::PFN_vkCreateDevice = unsafe { std::mem::transmute(proc) };
    let result =
        unsafe { create_device(physical_device, create_info, allocation_callbacks, device) };
    if result != vk::Result::SUCCESS {
        return result;
    }
    let Some(instance) = instances()
        .lock()
        .expect("instance map lock poisoned")
        .values()
        .next()
        .cloned()
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let ash_device = unsafe {
        ash::Device::load_with(
            |command| {
                get_device_proc_addr(*device, command.as_ptr())
                    .map_or(std::ptr::null(), |function| function as *const c_void)
            },
            *device,
        )
    };
    devices().lock().expect("device map lock poisoned").insert(
        unsafe { *device },
        DeviceState {
            get_device_proc_addr,
            physical_device,
            instance,
            device: ash_device,
        },
    );
    result
}

unsafe extern "system" fn create_device(
    physical_device: vk::PhysicalDevice,
    create_info: *const vk::DeviceCreateInfo<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        create_device_inner(physical_device, create_info, allocation_callbacks, device)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn register_queue(device: vk::Device, family_index: u32, queue: *mut vk::Queue) {
    if !queue.is_null() {
        queues().lock().expect("queue map lock poisoned").insert(
            unsafe { *queue },
            QueueState {
                device,
                family_index,
            },
        );
    }
}

unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    queue: *mut vk::Queue,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let Some(proc) = device_downstream(device, c"vkGetDeviceQueue") else {
            return;
        };
        let get_queue: vk::PFN_vkGetDeviceQueue = std::mem::transmute(proc);
        get_queue(device, queue_family_index, queue_index, queue);
        register_queue(device, queue_family_index, queue);
    }));
}

unsafe extern "system" fn get_device_queue2(
    device: vk::Device,
    queue_info: *const vk::DeviceQueueInfo2<'_>,
    queue: *mut vk::Queue,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let Some(proc) = device_downstream(device, c"vkGetDeviceQueue2") else {
            return;
        };
        let get_queue: vk::PFN_vkGetDeviceQueue2 = std::mem::transmute(proc);
        get_queue(device, queue_info, queue);
        if !queue_info.is_null() {
            register_queue(device, (*queue_info).queue_family_index, queue);
        }
    }));
}

unsafe fn create_swapchain_inner(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    if create_info.is_null() || swapchain.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(proc) = (unsafe { device_downstream(device, c"vkCreateSwapchainKHR") }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_swapchain: vk::PFN_vkCreateSwapchainKHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create_swapchain(device, create_info, allocation_callbacks, swapchain) };
    if result != vk::Result::SUCCESS {
        return result;
    }
    let Some(device_state) = devices()
        .lock()
        .expect("device map lock poisoned")
        .get(&device)
        .cloned()
    else {
        return result;
    };
    let handle = unsafe { *swapchain };
    let info = SwapchainInfo {
        format: unsafe { (*create_info).image_format },
        extent: unsafe { (*create_info).image_extent },
    };
    let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
    let Ok(images) = (unsafe { loader.get_swapchain_images(handle) }) else {
        return result;
    };
    let overlay = unsafe {
        OverlaySwapchain::new(
            &device_state.instance,
            device_state.physical_device,
            &device_state.device,
            info,
            images,
        )
    };
    if let Ok(overlay) = overlay {
        swapchains()
            .lock()
            .expect("swapchain map lock poisoned")
            .insert(handle, SwapchainState { device, overlay });
    }
    result
}

unsafe extern "system" fn create_swapchain_khr(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        create_swapchain_inner(device, create_info, allocation_callbacks, swapchain)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
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
    let destroy = unsafe { device_downstream(device, c"vkDestroySwapchainKHR") };
    let state = swapchains()
        .lock()
        .expect("swapchain map lock poisoned")
        .remove(&swapchain);
    if let Some(state) = state
        && let Some(device_state) = devices()
            .lock()
            .expect("device map lock poisoned")
            .get(&state.device)
            .cloned()
    {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            destroy_overlay(&device_state, state.overlay)
        }));
    }
    if let Some(proc) = destroy {
        let destroy_swapchain: vk::PFN_vkDestroySwapchainKHR = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_swapchain(device, swapchain, allocation_callbacks) };
    }
}

unsafe extern "system" fn destroy_swapchain_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_swapchain_inner(device, swapchain, allocation_callbacks)
    }));
}

unsafe fn submit_overlay(
    queue: vk::Queue,
    queue_state: QueueState,
    info: &vk::PresentInfoKHR<'_>,
) -> Option<vk::Semaphore> {
    if info.swapchain_count != 1
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
        .expect("device map lock poisoned")
        .get(&queue_state.device)
        .cloned()?;
    let mut swapchains = swapchains().lock().expect("swapchain map lock poisoned");
    let swapchain_state = swapchains.get_mut(&swapchain)?;
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
    let wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; wait_semaphores.len()];
    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(std::slice::from_ref(&prepared.command_buffer))
        .signal_semaphores(std::slice::from_ref(&prepared.render_complete));
    let result = unsafe {
        device_state
            .device
            .queue_submit(queue, std::slice::from_ref(&submit_info), prepared.fence)
    };
    if result.is_ok() {
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
        .expect("queue map lock poisoned")
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
    let overlay_complete = unsafe { submit_overlay(queue, queue_state, info) };
    if let Some(render_complete) = overlay_complete {
        let mut modified = *info;
        modified.wait_semaphore_count = 1;
        modified.p_wait_semaphores = &render_complete;
        return unsafe { present(queue, &modified) };
    }
    unsafe { present(queue, present_info) }
}

unsafe extern "system" fn queue_present_khr(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        queue_present_inner(queue, present_info)
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

unsafe fn destroy_device_inner(
    device: vk::Device,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let destroy = unsafe { device_downstream(device, c"vkDestroyDevice") };
    let device_state = devices()
        .lock()
        .expect("device map lock poisoned")
        .remove(&device);
    let overlays = {
        let mut swapchains = swapchains().lock().expect("swapchain map lock poisoned");
        let handles = swapchains
            .iter()
            .filter_map(|(handle, state)| (state.device == device).then_some(*handle))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .filter_map(|handle| swapchains.remove(&handle))
            .collect::<Vec<_>>()
    };
    queues()
        .lock()
        .expect("queue map lock poisoned")
        .retain(|_, state| state.device != device);
    if let Some(device_state) = device_state {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            let _ = device_state.device.device_wait_idle();
            for state in overlays {
                state.overlay.destroy(&device_state.device);
            }
        }));
    }
    if let Some(proc) = destroy {
        let destroy_device: vk::PFN_vkDestroyDevice = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_device(device, allocation_callbacks) };
    }
}

unsafe extern "system" fn destroy_device(
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
    instances()
        .lock()
        .expect("instance map lock poisoned")
        .remove(&instance);
    if let Some(proc) = destroy {
        let destroy_instance: vk::PFN_vkDestroyInstance = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_instance(instance, allocation_callbacks) };
    }
}

unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        destroy_instance_inner(instance, allocation_callbacks)
    }));
}

pub(crate) unsafe fn get_instance_proc_addr_inner(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) };
    match name.to_bytes() {
        b"vkCreateInstance" => unsafe {
            std::mem::transmute::<vk::PFN_vkCreateInstance, vk::PFN_vkVoidFunction>(
                create_instance as vk::PFN_vkCreateInstance,
            )
        },
        b"vkCreateDevice" => unsafe {
            std::mem::transmute::<vk::PFN_vkCreateDevice, vk::PFN_vkVoidFunction>(
                create_device as vk::PFN_vkCreateDevice,
            )
        },
        b"vkDestroyInstance" => unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyInstance, vk::PFN_vkVoidFunction>(
                destroy_instance as vk::PFN_vkDestroyInstance,
            )
        },
        _ => unsafe { downstream(instance, name) },
    }
}

pub(crate) unsafe fn get_device_proc_addr_inner(
    device: vk::Device,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) };
    match name.to_bytes() {
        b"vkGetDeviceQueue" => unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceQueue, vk::PFN_vkVoidFunction>(
                get_device_queue as vk::PFN_vkGetDeviceQueue,
            )
        },
        b"vkGetDeviceQueue2" => unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceQueue2, vk::PFN_vkVoidFunction>(
                get_device_queue2 as vk::PFN_vkGetDeviceQueue2,
            )
        },
        b"vkCreateSwapchainKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkCreateSwapchainKHR, vk::PFN_vkVoidFunction>(
                create_swapchain_khr as vk::PFN_vkCreateSwapchainKHR,
            )
        },
        b"vkDestroySwapchainKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkDestroySwapchainKHR, vk::PFN_vkVoidFunction>(
                destroy_swapchain_khr as vk::PFN_vkDestroySwapchainKHR,
            )
        },
        b"vkDestroyDevice" => unsafe {
            std::mem::transmute::<vk::PFN_vkDestroyDevice, vk::PFN_vkVoidFunction>(
                destroy_device as vk::PFN_vkDestroyDevice,
            )
        },
        b"vkQueuePresentKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkQueuePresentKHR, vk::PFN_vkVoidFunction>(
                queue_present_khr as vk::PFN_vkQueuePresentKHR,
            )
        },
        _ => unsafe { device_downstream(device, name) },
    }
}
