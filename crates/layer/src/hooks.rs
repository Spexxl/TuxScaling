use ash::vk;
use std::{
    ffi::{CStr, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
};
use tuxscaling_overlay_vulkan::SwapchainInfo;
use tuxscaling_runtime::SwapchainRuntime as OverlaySwapchain;

use super::{
    loader::{LAYER_LINK_INFO, LayerCreateInfo, next_gipa},
    state::{DeviceState, QueueState, SwapchainState, devices, instances, queues, swapchains},
};

mod creation;
mod lifetime;
mod presentation;
use creation::{
    create_device, create_instance, create_swapchain_khr, get_device_queue, get_device_queue2,
};
use lifetime::{destroy_device, destroy_instance, destroy_swapchain_khr};
use presentation::queue_present_khr;

unsafe fn downstream(instance: vk::Instance, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_instance_proc_addr = if instance == vk::Instance::null() {
        *next_gipa().lock().unwrap_or_else(|e| e.into_inner())
    } else {
        crate::state::instance_dispatch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&instance)
            .copied()
    };
    let get_instance_proc_addr = get_instance_proc_addr?;
    unsafe { get_instance_proc_addr(instance, name.as_ptr()) }
}

unsafe fn device_downstream(device: vk::Device, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_device_proc_addr = devices()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&device)?
        .get_device_proc_addr;
    unsafe { get_device_proc_addr(device, name.as_ptr()) }
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
        b"vkGetDeviceQueue"
        | b"vkGetDeviceQueue2"
        | b"vkCreateSwapchainKHR"
        | b"vkDestroySwapchainKHR"
        | b"vkDestroyDevice"
        | b"vkQueuePresentKHR" => unsafe {
            get_device_proc_addr_inner(vk::Device::null(), name.as_ptr())
        },
        b"vkGetInstanceProcAddr" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, unsafe extern "system" fn()>(
                super::layer_vkGetInstanceProcAddr,
            )
        }),
        b"vkGetDeviceProcAddr" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                super::layer_vkGetDeviceProcAddr,
            )
        }),
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
