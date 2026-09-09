mod handoff;
mod hooks;
mod loader;
mod mapping;
mod state;

use ash::vk;
use loader::LOADER_INTERFACE_VERSION;
pub use loader::NegotiateLayerInterface;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub const CRATE_NAME: &str = "tuxscaling-layer";

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "system" fn layer_vkGetInstanceProcAddr(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        hooks::get_instance_proc_addr_inner(instance, name)
    }))
    .unwrap_or(None)
}

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "system" fn layer_vkGetDeviceProcAddr(
    device: vk::Device,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        hooks::get_device_proc_addr_inner(device, name)
    }))
    .unwrap_or(None)
}

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "system" fn layer_vkGetPhysicalDeviceProcAddr(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        hooks::get_physical_device_proc_addr_inner(instance, name)
    }))
    .unwrap_or(None)
}

unsafe fn negotiate_inner(version: *mut NegotiateLayerInterface) -> vk::Result {
    if version.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let version = unsafe { &mut *version };
    if version.s_type != 1 || version.interface_version < LOADER_INTERFACE_VERSION {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    version.interface_version = LOADER_INTERFACE_VERSION;
    version.get_instance_proc_addr = layer_vkGetInstanceProcAddr;
    version.get_device_proc_addr = layer_vkGetDeviceProcAddr;
    version.get_physical_device_proc_addr = Some(layer_vkGetPhysicalDeviceProcAddr);
    vk::Result::SUCCESS
}

#[unsafe(no_mangle)]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    version: *mut NegotiateLayerInterface,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe { negotiate_inner(version) }))
        .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}
