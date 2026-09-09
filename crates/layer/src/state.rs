use ash::{vk, vk::Handle};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use tuxscaling_runtime::{SetLoaderData, SwapchainRuntime as OverlaySwapchain};
use tuxscaling_vulkan::Image;

use crate::mapping::{LogicalSwapchainHandle, Mapping};

#[derive(Clone, Copy)]
pub(crate) struct X11Surface {
    pub(crate) window: u64,
    pub(crate) logical_extent: Option<vk::Extent2D>,
    pub(crate) logical_capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    pub(crate) borderless_lease: Option<tuxscaling_display::BorderlessLease>,
    pub(crate) negotiation: tuxscaling_display::PresentationNegotiation,
}

pub(crate) struct SwapchainState {
    pub(crate) device: vk::Device,
    pub(crate) surface: vk::SurfaceKHR,
    /// The application-visible key remains stable even if a later task
    /// replaces the downstream WSI generation.
    pub(crate) logical_handle: vk::SwapchainKHR,
    pub(crate) physical_handle: vk::SwapchainKHR,
    pub(crate) mapping: Option<Mapping>,
    pub(crate) negotiation: tuxscaling_display::PresentationNegotiation,
    pub(crate) overlay: OverlaySwapchain,
    pub(crate) virtual_images: Option<Vec<Image>>,
}

#[derive(Clone, Copy)]
pub(crate) struct QueueState {
    pub(crate) device: vk::Device,
    pub(crate) family_index: u32,
    pub(crate) processing_allowed: bool,
}

#[derive(Clone)]
pub(crate) struct DeviceState {
    pub(crate) overlay_supported: bool,
    /// Virtualization is disabled when an enabled swapchain extension has no
    /// complete logical-to-physical translation in this Task 2 layer.
    pub(crate) virtualization_extension_safe: bool,
    pub(crate) vulkan_api_version: u32,
    pub(crate) queue_families: Vec<vk::QueueFamilyProperties>,
    pub(crate) set_loader_data: Option<SetLoaderData>,
    pub(crate) get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) instance: ash::Instance,
    pub(crate) device: ash::Device,
}

static INSTANCES: OnceLock<Mutex<HashMap<vk::Instance, ash::Instance>>> = OnceLock::new();
static INSTANCE_API_VERSIONS: OnceLock<Mutex<HashMap<vk::Instance, u32>>> = OnceLock::new();
static DEVICES: OnceLock<Mutex<HashMap<vk::Device, DeviceState>>> = OnceLock::new();
static QUEUES: OnceLock<Mutex<HashMap<vk::Queue, QueueState>>> = OnceLock::new();
static SWAPCHAINS: OnceLock<Mutex<HashMap<vk::SwapchainKHR, Arc<Mutex<SwapchainState>>>>> =
    OnceLock::new();
static RETIRED_SWAPCHAINS: OnceLock<Mutex<std::collections::HashSet<vk::SwapchainKHR>>> =
    OnceLock::new();
static SURFACES: OnceLock<Mutex<HashMap<vk::SurfaceKHR, X11Surface>>> = OnceLock::new();

pub(crate) fn instances() -> &'static Mutex<HashMap<vk::Instance, ash::Instance>> {
    INSTANCES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn instance_api_versions() -> &'static Mutex<HashMap<vk::Instance, u32>> {
    INSTANCE_API_VERSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn devices() -> &'static Mutex<HashMap<vk::Device, DeviceState>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn queues() -> &'static Mutex<HashMap<vk::Queue, QueueState>> {
    QUEUES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn swapchains() -> &'static Mutex<HashMap<vk::SwapchainKHR, Arc<Mutex<SwapchainState>>>>
{
    SWAPCHAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::{is_retired_swapchain, retire_swapchain};
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn retired_logical_tokens_are_explicitly_rejected() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0055);

        assert!(!is_retired_swapchain(token));
        retire_swapchain(token);
        assert!(is_retired_swapchain(token));
    }
}

pub(crate) fn retire_swapchain(swapchain: vk::SwapchainKHR) {
    RETIRED_SWAPCHAINS
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(swapchain);
}

pub(crate) fn is_retired_swapchain(swapchain: vk::SwapchainKHR) -> bool {
    RETIRED_SWAPCHAINS
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .contains(&swapchain)
}

pub(crate) fn is_unknown_logical_swapchain(swapchain: vk::SwapchainKHR) -> bool {
    LogicalSwapchainHandle::is_reserved(swapchain.as_raw())
        && !swapchains()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(&swapchain)
}

pub(crate) fn surfaces() -> &'static Mutex<HashMap<vk::SurfaceKHR, X11Surface>> {
    SURFACES.get_or_init(|| Mutex::new(HashMap::new()))
}
pub(crate) fn instance_dispatch()
-> &'static Mutex<HashMap<vk::Instance, vk::PFN_vkGetInstanceProcAddr>> {
    static DISPATCH: OnceLock<Mutex<HashMap<vk::Instance, vk::PFN_vkGetInstanceProcAddr>>> =
        OnceLock::new();
    DISPATCH.get_or_init(|| Mutex::new(HashMap::new()))
}
