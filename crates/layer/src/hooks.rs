use ash::vk;
use std::{
    ffi::{CStr, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
};
use tuxscaling_overlay_vulkan::SwapchainInfo;
use tuxscaling_runtime::{
    SwapchainImages, SwapchainRuntime as OverlaySwapchain, SwapchainRuntimeCreateInfo,
};

use super::{
    loader::{
        DeviceLayerLink, InstanceLayerLink, LAYER_LINK_INFO, LayerCreateInfo, next_gipa, next_gpdpa,
    },
    state::{
        DeviceState, QueueState, SwapchainState, X11Surface, devices, instance_api_versions,
        instances, is_retired_swapchain, queues, retire_swapchain, surfaces, swapchains,
    },
};

mod acquire;
mod creation;
mod lifetime;
mod presentation;
mod surface;
use acquire::{acquire_next_image_khr, acquire_next_image2_khr, get_swapchain_images_khr};
use creation::{
    create_device, create_instance, create_swapchain_khr, get_device_queue, get_device_queue2,
};
use lifetime::{destroy_device, destroy_instance, destroy_swapchain_khr};
use presentation::queue_present_khr;
use surface::{
    create_xcb_surface_khr, create_xlib_surface_khr, destroy_surface_khr,
    get_physical_device_surface_capabilities_khr, get_physical_device_surface_capabilities2_khr,
};

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

unsafe fn enumerate_device_extension_properties_inner(
    physical_device: vk::PhysicalDevice,
    layer_name: *const i8,
    property_count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if property_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let Some(instance) = (unsafe { instance_for_physical_device(physical_device) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(proc) =
        (unsafe { downstream(instance.handle(), c"vkEnumerateDeviceExtensionProperties") })
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let enumerate: vk::PFN_vkEnumerateDeviceExtensionProperties =
        unsafe { std::mem::transmute(proc) };
    unsafe { enumerate(physical_device, layer_name, property_count, properties) }
}

unsafe extern "system" fn enumerate_device_extension_properties(
    physical_device: vk::PhysicalDevice,
    layer_name: *const i8,
    property_count: *mut u32,
    properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        enumerate_device_extension_properties_inner(
            physical_device,
            layer_name,
            property_count,
            properties,
        )
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
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
        b"vkCreateXlibSurfaceKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkCreateXlibSurfaceKHR, vk::PFN_vkVoidFunction>(
                create_xlib_surface_khr as vk::PFN_vkCreateXlibSurfaceKHR,
            )
        },
        b"vkCreateXcbSurfaceKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkCreateXcbSurfaceKHR, vk::PFN_vkVoidFunction>(
                create_xcb_surface_khr as vk::PFN_vkCreateXcbSurfaceKHR,
            )
        },
        b"vkDestroySurfaceKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkDestroySurfaceKHR, vk::PFN_vkVoidFunction>(
                destroy_surface_khr as vk::PFN_vkDestroySurfaceKHR,
            )
        },
        b"vkEnumerateDeviceExtensionProperties" => unsafe {
            std::mem::transmute::<
                vk::PFN_vkEnumerateDeviceExtensionProperties,
                vk::PFN_vkVoidFunction,
            >(enumerate_device_extension_properties)
        },
        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" => unsafe {
            std::mem::transmute::<
                vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
                vk::PFN_vkVoidFunction,
            >(
                get_physical_device_surface_capabilities_khr
                    as vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
            )
        },
        b"vkGetPhysicalDeviceSurfaceCapabilities2KHR" => unsafe {
            std::mem::transmute::<
                vk::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR,
                vk::PFN_vkVoidFunction,
            >(
                get_physical_device_surface_capabilities2_khr
                    as vk::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR,
            )
        },
        b"vkGetDeviceQueue"
        | b"vkGetDeviceQueue2"
        | b"vkCreateSwapchainKHR"
        | b"vkDestroySwapchainKHR"
        | b"vkGetSwapchainImagesKHR"
        | b"vkAcquireNextImageKHR"
        | b"vkAcquireNextImage2KHR"
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

pub(crate) unsafe fn get_physical_device_proc_addr_inner(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) };
    if name.to_bytes() == b"vkEnumerateDeviceExtensionProperties" {
        return unsafe {
            std::mem::transmute::<
                vk::PFN_vkEnumerateDeviceExtensionProperties,
                vk::PFN_vkVoidFunction,
            >(enumerate_device_extension_properties)
        };
    }
    let get_physical_device_proc_addr = *next_gpdpa().lock().unwrap_or_else(|e| e.into_inner());
    let get_physical_device_proc_addr = get_physical_device_proc_addr?;
    unsafe { get_physical_device_proc_addr(instance, name.as_ptr()) }
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
        b"vkGetSwapchainImagesKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkGetSwapchainImagesKHR, vk::PFN_vkVoidFunction>(
                get_swapchain_images_khr as vk::PFN_vkGetSwapchainImagesKHR,
            )
        },
        b"vkAcquireNextImageKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkAcquireNextImageKHR, vk::PFN_vkVoidFunction>(
                acquire_next_image_khr as vk::PFN_vkAcquireNextImageKHR,
            )
        },
        b"vkAcquireNextImage2KHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkAcquireNextImage2KHR, vk::PFN_vkVoidFunction>(
                acquire_next_image2_khr as vk::PFN_vkAcquireNextImage2KHR,
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
