use ash::vk;
use std::{
    ffi::c_void,
    sync::{Mutex, OnceLock},
};

pub(crate) const LAYER_LINK_INFO: i32 = 0;
pub(crate) const LOADER_INTERFACE_VERSION: u32 = 2;

#[repr(C)]
pub(crate) struct LayerLink {
    pub(crate) next: *mut Self,
    pub(crate) get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub(crate) get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
pub(crate) union LayerCreateInfoData {
    pub(crate) layer_info: *mut LayerLink,
    pub(crate) _set_loader_data: *const c_void,
}

#[repr(C)]
pub(crate) struct LayerCreateInfo {
    pub(crate) s_type: vk::StructureType,
    pub(crate) p_next: *const c_void,
    pub(crate) function: i32,
    pub(crate) data: LayerCreateInfoData,
}

#[repr(C)]
pub struct NegotiateLayerInterface {
    pub(crate) s_type: vk::StructureType,
    pub(crate) p_next: *const c_void,
    pub(crate) interface_version: u32,
    pub(crate) get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    pub(crate) get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub(crate) get_physical_device_proc_addr: vk::PFN_vkVoidFunction,
}

static NEXT_GIPA: OnceLock<Mutex<Option<vk::PFN_vkGetInstanceProcAddr>>> = OnceLock::new();

pub(crate) fn next_gipa() -> &'static Mutex<Option<vk::PFN_vkGetInstanceProcAddr>> {
    NEXT_GIPA.get_or_init(|| Mutex::new(None))
}
