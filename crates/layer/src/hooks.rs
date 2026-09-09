use ash::vk;
use ash::vk::Handle;
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
        instances, is_retired_swapchain, is_unknown_logical_swapchain, queues, retire_swapchain,
        surfaces, swapchains,
    },
};
use crate::mapping::LogicalSwapchainHandle;

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

fn is_swapchain_related_proc(name: &CStr) -> bool {
    let bytes = name.to_bytes();
    bytes
        .windows(b"Swapchain".len())
        .any(|window| window == b"Swapchain")
        || bytes
            .windows(b"Present".len())
            .any(|window| window == b"Present")
}

// Only extensions whose swapchain-facing behavior is translated by this
// layer are allowlisted.  A conservative deny-by-default policy is important
// here: vendor extensions frequently carry a VkSwapchainKHR without putting
// "Swapchain" or "Present" in every command name.
const VIRTUALIZATION_SAFE_EXTENSIONS: &[&[u8]] = &[
    b"VK_KHR_swapchain",
    b"VK_KHR_incremental_present",
    b"VK_EXT_swapchain_colorspace",
];

pub(crate) fn virtualization_extension_safe(create_info: &vk::DeviceCreateInfo<'_>) -> bool {
    if create_info.enabled_extension_count == 0 || create_info.pp_enabled_extension_names.is_null()
    {
        return true;
    }
    let names = unsafe {
        std::slice::from_raw_parts(
            create_info.pp_enabled_extension_names,
            create_info.enabled_extension_count as usize,
        )
    };
    names.iter().all(|name| {
        if name.is_null() {
            return false;
        }
        let name = unsafe { CStr::from_ptr(*name) }.to_bytes();
        VIRTUALIZATION_SAFE_EXTENSIONS
            .iter()
            .any(|extension| name == *extension)
    })
}

fn device_allows_virtualization(device: vk::Device) -> bool {
    devices()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&device)
        .is_none_or(|state| state.virtualization_extension_safe)
}

unsafe fn extension_proc_or_downstream(
    _device: vk::Device,
    _name: &CStr,
    layer_proc: vk::PFN_vkVoidFunction,
) -> vk::PFN_vkVoidFunction {
    // Keep the layer wrapper for every device. It forwards untagged direct
    // swapchains itself, while rejecting virtual/unknown tagged tokens before
    // an ICD can see them.
    layer_proc
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
        | b"vkReleaseSwapchainImagesEXT"
        | b"vkGetSwapchainStatusKHR"
        | b"vkWaitForPresentKHR"
        | b"vkGetPastPresentationTimingGOOGLE"
        | b"vkGetRefreshCycleDurationGOOGLE"
        | b"vkGetSwapchainCounterEXT"
        | b"vkSetHdrMetadataEXT"
        | b"vkAcquireFullScreenExclusiveModeEXT"
        | b"vkReleaseFullScreenExclusiveModeEXT"
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
        b"vkReleaseSwapchainImagesEXT" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<vk::PFN_vkReleaseSwapchainImagesEXT, vk::PFN_vkVoidFunction>(
                    reject_release_swapchain_images_ext as vk::PFN_vkReleaseSwapchainImagesEXT,
                ),
            )
        },
        b"vkGetSwapchainStatusKHR" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<vk::PFN_vkGetSwapchainStatusKHR, vk::PFN_vkVoidFunction>(
                    reject_get_swapchain_status_khr as vk::PFN_vkGetSwapchainStatusKHR,
                ),
            )
        },
        b"vkWaitForPresentKHR" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<vk::PFN_vkWaitForPresentKHR, vk::PFN_vkVoidFunction>(
                    reject_wait_for_present_khr as vk::PFN_vkWaitForPresentKHR,
                ),
            )
        },
        b"vkGetPastPresentationTimingGOOGLE" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<
                    vk::PFN_vkGetPastPresentationTimingGOOGLE,
                    vk::PFN_vkVoidFunction,
                >(
                    reject_past_presentation_timing_google
                        as vk::PFN_vkGetPastPresentationTimingGOOGLE,
                ),
            )
        },
        b"vkGetRefreshCycleDurationGOOGLE" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<
                    vk::PFN_vkGetRefreshCycleDurationGOOGLE,
                    vk::PFN_vkVoidFunction,
                >(
                    reject_refresh_cycle_duration_google as vk::PFN_vkGetRefreshCycleDurationGOOGLE
                ),
            )
        },
        b"vkGetSwapchainCounterEXT" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<vk::PFN_vkGetSwapchainCounterEXT, vk::PFN_vkVoidFunction>(
                    reject_swapchain_counter_ext as vk::PFN_vkGetSwapchainCounterEXT,
                ),
            )
        },
        b"vkSetHdrMetadataEXT" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<vk::PFN_vkSetHdrMetadataEXT, vk::PFN_vkVoidFunction>(
                    reject_set_hdr_metadata_ext as vk::PFN_vkSetHdrMetadataEXT,
                ),
            )
        },
        b"vkAcquireFullScreenExclusiveModeEXT" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<
                    vk::PFN_vkAcquireFullScreenExclusiveModeEXT,
                    vk::PFN_vkVoidFunction,
                >(
                    reject_acquire_full_screen_exclusive_mode_ext
                        as vk::PFN_vkAcquireFullScreenExclusiveModeEXT,
                ),
            )
        },
        b"vkReleaseFullScreenExclusiveModeEXT" => unsafe {
            extension_proc_or_downstream(
                device,
                name,
                std::mem::transmute::<
                    vk::PFN_vkReleaseFullScreenExclusiveModeEXT,
                    vk::PFN_vkVoidFunction,
                >(
                    reject_release_full_screen_exclusive_mode_ext
                        as vk::PFN_vkReleaseFullScreenExclusiveModeEXT,
                ),
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
        _ if is_swapchain_related_proc(name) => {
            if device != vk::Device::null() && device_allows_virtualization(device) {
                None
            } else {
                unsafe { device_downstream(device, name) }
            }
        }
        _ => unsafe { device_downstream(device, name) },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SwapchainTokenKind {
    Direct,
    Virtual,
    UnknownTagged,
}

fn swapchain_token_kind(swapchain: vk::SwapchainKHR) -> SwapchainTokenKind {
    let tracked = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&swapchain)
        .cloned();
    if tracked.is_some_and(|state| {
        state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .mapping
            .is_some()
    }) {
        SwapchainTokenKind::Virtual
    } else if is_retired_swapchain(swapchain)
        || LogicalSwapchainHandle::is_reserved(swapchain.as_raw())
    {
        SwapchainTokenKind::UnknownTagged
    } else {
        SwapchainTokenKind::Direct
    }
}

fn reject_extension_token(swapchain: vk::SwapchainKHR) -> Option<vk::Result> {
    match swapchain_token_kind(swapchain) {
        SwapchainTokenKind::Direct => None,
        SwapchainTokenKind::Virtual => Some(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
        SwapchainTokenKind::UnknownTagged => Some(vk::Result::ERROR_OUT_OF_DATE_KHR),
    }
}

unsafe fn downstream_result(
    device: vk::Device,
    name: &CStr,
    fallback: vk::Result,
    invoke: impl FnOnce(unsafe extern "system" fn()) -> vk::Result,
) -> vk::Result {
    let Some(proc) = (unsafe { device_downstream(device, name) }) else {
        return fallback;
    };
    invoke(proc)
}

unsafe extern "system" fn reject_release_swapchain_images_ext(
    device: vk::Device,
    info: *const vk::ReleaseSwapchainImagesInfoEXT<'_>,
) -> vk::Result {
    if info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let swapchain = unsafe { (*info).swapchain };
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkReleaseSwapchainImagesEXT",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let release: vk::PFN_vkReleaseSwapchainImagesEXT = std::mem::transmute(proc);
                release(device, info)
            },
        )
    }
}

unsafe extern "system" fn reject_get_swapchain_status_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkGetSwapchainStatusKHR",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let get_status: vk::PFN_vkGetSwapchainStatusKHR = std::mem::transmute(proc);
                get_status(device, swapchain)
            },
        )
    }
}

unsafe extern "system" fn reject_wait_for_present_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    present_id: u64,
    timeout: u64,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkWaitForPresentKHR",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let wait: vk::PFN_vkWaitForPresentKHR = std::mem::transmute(proc);
                wait(device, swapchain, present_id, timeout)
            },
        )
    }
}

unsafe extern "system" fn reject_past_presentation_timing_google(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    count: *mut u32,
    timings: *mut vk::PastPresentationTimingGOOGLE,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkGetPastPresentationTimingGOOGLE",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let get: vk::PFN_vkGetPastPresentationTimingGOOGLE = std::mem::transmute(proc);
                get(device, swapchain, count, timings)
            },
        )
    }
}

unsafe extern "system" fn reject_refresh_cycle_duration_google(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    properties: *mut vk::RefreshCycleDurationGOOGLE,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkGetRefreshCycleDurationGOOGLE",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let get: vk::PFN_vkGetRefreshCycleDurationGOOGLE = std::mem::transmute(proc);
                get(device, swapchain, properties)
            },
        )
    }
}

unsafe extern "system" fn reject_swapchain_counter_ext(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    counter: vk::SurfaceCounterFlagsEXT,
    value: *mut u64,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkGetSwapchainCounterEXT",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let get: vk::PFN_vkGetSwapchainCounterEXT = std::mem::transmute(proc);
                get(device, swapchain, counter, value)
            },
        )
    }
}

unsafe extern "system" fn reject_acquire_full_screen_exclusive_mode_ext(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkAcquireFullScreenExclusiveModeEXT",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let acquire: vk::PFN_vkAcquireFullScreenExclusiveModeEXT =
                    std::mem::transmute(proc);
                acquire(device, swapchain)
            },
        )
    }
}

unsafe extern "system" fn reject_release_full_screen_exclusive_mode_ext(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> vk::Result {
    if let Some(error) = reject_extension_token(swapchain) {
        return error;
    }
    unsafe {
        downstream_result(
            device,
            c"vkReleaseFullScreenExclusiveModeEXT",
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            |proc| {
                let release: vk::PFN_vkReleaseFullScreenExclusiveModeEXT =
                    std::mem::transmute(proc);
                release(device, swapchain)
            },
        )
    }
}

unsafe extern "system" fn reject_set_hdr_metadata_ext(
    device: vk::Device,
    swapchain_count: u32,
    swapchains_ptr: *const vk::SwapchainKHR,
    metadata: *const vk::HdrMetadataEXT<'_>,
) {
    if swapchain_count != 0 && swapchains_ptr.is_null() {
        return;
    }
    let swapchains_slice =
        unsafe { std::slice::from_raw_parts(swapchains_ptr, swapchain_count as usize) };
    if swapchains_slice
        .iter()
        .any(|swapchain| reject_extension_token(*swapchain).is_some())
    {
        return;
    }
    let Some(proc) = (unsafe { device_downstream(device, c"vkSetHdrMetadataEXT") }) else {
        return;
    };
    let set: vk::PFN_vkSetHdrMetadataEXT = unsafe { std::mem::transmute(proc) };
    unsafe { set(device, swapchain_count, swapchains_ptr, metadata) };
}

#[cfg(test)]
mod tests {
    use super::{
        get_device_proc_addr_inner, is_swapchain_related_proc,
        reject_acquire_full_screen_exclusive_mode_ext,
        reject_release_full_screen_exclusive_mode_ext, virtualization_extension_safe,
    };
    use ash::vk;
    use ash::vk::Handle;
    use std::ffi::CStr;

    #[test]
    fn unknown_swapchain_proc_names_are_not_forwarded_to_downstream() {
        let name = CStr::from_bytes_with_nul(b"vkFutureSwapchainCommandKHR\0").unwrap();

        assert!(is_swapchain_related_proc(name));
        assert!(
            unsafe { get_device_proc_addr_inner(ash::vk::Device::null(), name.as_ptr()) }.is_none()
        );
    }

    #[test]
    fn known_swapchain_extension_proc_names_have_safe_layer_dispatch() {
        let names = [
            c"vkWaitForPresentKHR",
            c"vkGetPastPresentationTimingGOOGLE",
            c"vkGetRefreshCycleDurationGOOGLE",
            c"vkGetSwapchainCounterEXT",
            c"vkGetSwapchainStatusKHR",
            c"vkReleaseSwapchainImagesEXT",
            c"vkSetHdrMetadataEXT",
            c"vkAcquireFullScreenExclusiveModeEXT",
            c"vkReleaseFullScreenExclusiveModeEXT",
        ];

        for name in names {
            assert!(
                unsafe { get_device_proc_addr_inner(ash::vk::Device::null(), name.as_ptr()) }
                    .is_some()
            );
        }
    }

    #[test]
    fn full_screen_exclusive_dispatch_rejects_unknown_tagged_tokens() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0fed);

        assert_eq!(
            unsafe { reject_acquire_full_screen_exclusive_mode_ext(vk::Device::null(), token) },
            vk::Result::ERROR_OUT_OF_DATE_KHR
        );
        assert_eq!(
            unsafe { reject_release_full_screen_exclusive_mode_ext(vk::Device::null(), token) },
            vk::Result::ERROR_OUT_OF_DATE_KHR
        );
    }

    #[test]
    fn unsupported_swapchain_extensions_disable_virtualization_before_promotion() {
        let extension_names = [vk::EXT_FULL_SCREEN_EXCLUSIVE_NAME.as_ptr()];
        let info = vk::DeviceCreateInfo::default().enabled_extension_names(&extension_names);

        assert!(!virtualization_extension_safe(&info));

        let safe_info = vk::DeviceCreateInfo::default();
        assert!(virtualization_extension_safe(&safe_info));
    }

    #[test]
    fn vendor_swapchain_commands_disable_virtualization_before_promotion() {
        let extension_name = std::ffi::CString::new("VK_NV_present_barrier").unwrap();
        let extension_names = [extension_name.as_ptr()];
        let info = vk::DeviceCreateInfo::default().enabled_extension_names(&extension_names);

        assert!(!virtualization_extension_safe(&info));
    }
}
