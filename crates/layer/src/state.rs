use ash::{vk, vk::Handle};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};
use tuxscaling_runtime::{SetLoaderData, SwapchainRuntime as OverlaySwapchain};
use tuxscaling_vulkan::Image;

use crate::hooks::maintenance::{
    MaintenanceCreateError, SwapchainMaintenanceTemplate, supported_swapchain_flags,
};
use crate::hooks::wsi_compatibility::DeviceWsiCapabilities;
use crate::mapping::{LogicalSwapchainHandle, Mapping};
use crate::recovery::{LogicalSwapchainContract, ReconfigurationLifecycle};

#[derive(Clone)]
pub(crate) struct SwapchainTemplate {
    pub(crate) flags: vk::SwapchainCreateFlagsKHR,
    pub(crate) min_image_count: u32,
    pub(crate) image_format: vk::Format,
    pub(crate) image_color_space: vk::ColorSpaceKHR,
    pub(crate) image_array_layers: u32,
    pub(crate) image_usage: vk::ImageUsageFlags,
    pub(crate) image_sharing_mode: vk::SharingMode,
    pub(crate) queue_family_indices: Vec<u32>,
    pub(crate) pre_transform: vk::SurfaceTransformFlagsKHR,
    pub(crate) composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub(crate) present_mode: vk::PresentModeKHR,
    pub(crate) clipped: vk::Bool32,
    pub(crate) maintenance: SwapchainMaintenanceTemplate,
}

impl SwapchainTemplate {
    pub(crate) fn from_create_info(
        info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<Self, MaintenanceCreateError> {
        if !supported_swapchain_flags(info.flags) {
            return Err(MaintenanceCreateError::UnsupportedFlags);
        }
        let maintenance = unsafe { SwapchainMaintenanceTemplate::parse(info.p_next) }?;
        let queue_family_indices = if info.image_sharing_mode == vk::SharingMode::CONCURRENT
            && !info.p_queue_family_indices.is_null()
        {
            unsafe {
                std::slice::from_raw_parts(
                    info.p_queue_family_indices,
                    info.queue_family_index_count as usize,
                )
                .to_vec()
            }
        } else {
            Vec::new()
        };
        Ok(Self {
            flags: info.flags,
            min_image_count: info.min_image_count,
            image_format: info.image_format,
            image_color_space: info.image_color_space,
            image_array_layers: info.image_array_layers,
            image_usage: info.image_usage,
            image_sharing_mode: info.image_sharing_mode,
            queue_family_indices,
            pre_transform: info.pre_transform,
            composite_alpha: info.composite_alpha,
            present_mode: info.present_mode,
            clipped: info.clipped,
            maintenance,
        })
    }

    pub(crate) fn with_create_info<R>(
        &self,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
        old_swapchain: vk::SwapchainKHR,
        invoke: impl FnOnce(&vk::SwapchainCreateInfoKHR<'_>) -> R,
    ) -> R {
        let mut scaling = self.maintenance.scaling.map(|scaling| {
            vk::SwapchainPresentScalingCreateInfoEXT::default()
                .scaling_behavior(scaling.behavior)
                .present_gravity_x(scaling.gravity_x)
                .present_gravity_y(scaling.gravity_y)
        });
        let mut modes =
            self.maintenance.present_modes.as_ref().map(|modes| {
                vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(modes)
            });
        let mut next = std::ptr::null();
        if let Some(scaling) = scaling.as_mut() {
            scaling.p_next = next;
            next = (scaling as *mut vk::SwapchainPresentScalingCreateInfoEXT<'_>).cast();
        }
        if let Some(modes) = modes.as_mut() {
            modes.p_next = next;
            next = (modes as *mut vk::SwapchainPresentModesCreateInfoEXT<'_>).cast();
        }
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .flags(self.flags)
            .surface(surface)
            .min_image_count(self.min_image_count)
            .image_format(self.image_format)
            .image_color_space(self.image_color_space)
            .image_extent(extent)
            .image_array_layers(self.image_array_layers)
            .image_usage(self.image_usage)
            .image_sharing_mode(self.image_sharing_mode)
            .queue_family_indices(&self.queue_family_indices)
            .pre_transform(self.pre_transform)
            .composite_alpha(self.composite_alpha)
            .present_mode(self.present_mode)
            .clipped(self.clipped != 0)
            .old_swapchain(old_swapchain);
        info.p_next = next;
        invoke(&info)
    }
}

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
    pub(crate) overlay: Option<OverlaySwapchain>,
    pub(crate) virtual_images: Option<Vec<Image>>,
    pub(crate) physical_images: Vec<vk::Image>,
    pub(crate) retired_physical_generations: Vec<vk::SwapchainKHR>,
    pub(crate) maintenance_overlay_reported: bool,
    pub(crate) maintenance_present_reported: bool,
    pub(crate) maintenance_release_reported: bool,
    pub(crate) generation: u64,
    pub(crate) template: Option<SwapchainTemplate>,
    pub(crate) contract: Option<LogicalSwapchainContract>,
    pub(crate) lifecycle: ReconfigurationLifecycle,
}

#[derive(Clone, Copy)]
pub(crate) struct QueueState {
    pub(crate) device: vk::Device,
    pub(crate) family_index: u32,
    pub(crate) processing_allowed: bool,
}

#[derive(Clone)]
pub(crate) struct DeviceState {
    pub(crate) wsi: DeviceWsiCapabilities,
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
static RECREATING_SURFACES: OnceLock<Mutex<std::collections::HashSet<vk::SurfaceKHR>>> =
    OnceLock::new();

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

pub(crate) fn is_reconfiguring_swapchain(swapchain: vk::SwapchainKHR) -> bool {
    swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&swapchain)
        .and_then(|state| state.lock().ok())
        .is_some_and(|state| state.lifecycle.blocks_frame_operations())
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

pub(crate) fn begin_surface_recreation(surface: vk::SurfaceKHR) -> bool {
    RECREATING_SURFACES
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(surface)
}

pub(crate) fn end_surface_recreation(surface: vk::SurfaceKHR) {
    RECREATING_SURFACES
        .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&surface);
}

pub(crate) fn instance_dispatch()
-> &'static Mutex<HashMap<vk::Instance, vk::PFN_vkGetInstanceProcAddr>> {
    static DISPATCH: OnceLock<Mutex<HashMap<vk::Instance, vk::PFN_vkGetInstanceProcAddr>>> =
        OnceLock::new();
    DISPATCH.get_or_init(|| Mutex::new(HashMap::new()))
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
