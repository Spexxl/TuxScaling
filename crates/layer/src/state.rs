use ash::vk;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use tuxscaling_overlay_vulkan::OverlaySwapchain;

pub(crate) struct SwapchainState {
    pub(crate) device: vk::Device,
    pub(crate) overlay: OverlaySwapchain,
}

#[derive(Clone, Copy)]
pub(crate) struct QueueState {
    pub(crate) device: vk::Device,
    pub(crate) family_index: u32,
}

#[derive(Clone)]
pub(crate) struct DeviceState {
    pub(crate) get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) instance: ash::Instance,
    pub(crate) device: ash::Device,
}

static INSTANCES: OnceLock<Mutex<HashMap<vk::Instance, ash::Instance>>> = OnceLock::new();
static DEVICES: OnceLock<Mutex<HashMap<vk::Device, DeviceState>>> = OnceLock::new();
static QUEUES: OnceLock<Mutex<HashMap<vk::Queue, QueueState>>> = OnceLock::new();
static SWAPCHAINS: OnceLock<Mutex<HashMap<vk::SwapchainKHR, SwapchainState>>> = OnceLock::new();

pub(crate) fn instances() -> &'static Mutex<HashMap<vk::Instance, ash::Instance>> {
    INSTANCES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn devices() -> &'static Mutex<HashMap<vk::Device, DeviceState>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn queues() -> &'static Mutex<HashMap<vk::Queue, QueueState>> {
    QUEUES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn swapchains() -> &'static Mutex<HashMap<vk::SwapchainKHR, SwapchainState>> {
    SWAPCHAINS.get_or_init(|| Mutex::new(HashMap::new()))
}
