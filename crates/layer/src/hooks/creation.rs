use super::swapchain_create::{
    SwapchainCompatibilityError, SwapchainCreateChain, supported_swapchain_flags,
};
use super::*;
use crate::mapping::{LogicalSwapchainHandle, Mapping, OldSwapchain};
use crate::recovery::{LogicalSwapchainContract, PhysicalGeneration, ReconfigurationTicket};
use ash::vk::Handle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tuxscaling_display::{Extent, PresentationNegotiation, PresentationState, SurfaceExtent};

fn virtual_swapchain_supported(info: &vk::SwapchainCreateInfoKHR<'_>) -> bool {
    if !supported_swapchain_flags(info.flags)
        || info.image_array_layers != 1
        || (info.image_sharing_mode == vk::SharingMode::CONCURRENT
            && (info.queue_family_index_count < 2 || info.p_queue_family_indices.is_null()))
    {
        return false;
    }
    unsafe { SwapchainCreateChain::from_create_info(info) }.is_ok()
}

#[cfg(test)]
fn logical_image_count(requested: u32, physical: usize) -> usize {
    let requested = requested as usize;
    if requested != 0 { requested } else { physical }
}

fn preflight_logical_image_count(requested: u32) -> Option<usize> {
    let count = requested as usize;
    (count != 0).then_some(count)
}

#[derive(Clone)]
pub(crate) struct SwapchainVirtualizationPlan {
    pub(crate) template: crate::state::SwapchainTemplate,
    pub(crate) image_flags: vk::ImageCreateFlags,
    pub(crate) view_formats: Vec<vk::Format>,
}

fn preflight_swapchain_virtualization(
    wsi: &crate::hooks::wsi_compatibility::DeviceWsiCapabilities,
    info: &vk::SwapchainCreateInfoKHR<'_>,
    capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    format_features: vk::FormatFeatureFlags,
) -> Result<SwapchainVirtualizationPlan, SwapchainCompatibilityError> {
    if let Some(incompatible) = wsi.incompatible.as_ref() {
        return Err(SwapchainCompatibilityError::from_wsi(incompatible));
    }
    if info.image_array_layers != 1
        || (info.image_sharing_mode == vk::SharingMode::CONCURRENT
            && (info.queue_family_index_count < 2 || info.p_queue_family_indices.is_null()))
    {
        return Err(SwapchainCompatibilityError::UnsupportedImageContract(
            vk::Result::ERROR_FEATURE_NOT_PRESENT,
        ));
    }
    if !supported_swapchain_flags(info.flags) {
        return Err(SwapchainCompatibilityError::UnsupportedFlags {
            bits: info.flags.as_raw(),
        });
    }
    if info
        .flags
        .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
        && !wsi.mutable_format
    {
        return Err(SwapchainCompatibilityError::UnsupportedImageContract(
            vk::Result::ERROR_EXTENSION_NOT_PRESENT,
        ));
    }
    if !tuxscaling_capture::supported_format(info.image_format, info.image_color_space)
        || matches!(
            info.present_mode,
            vk::PresentModeKHR::SHARED_DEMAND_REFRESH
                | vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH
        )
    {
        return Err(SwapchainCompatibilityError::UnsupportedImageContract(
            vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
        ));
    }
    let capabilities =
        capabilities.ok_or(SwapchainCompatibilityError::UnsupportedImageContract(
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ))?;
    let required_usage = vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::STORAGE;
    if !capabilities.supported_usage_flags.contains(required_usage)
        || !format_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST)
        || preflight_logical_image_count(info.min_image_count).is_none()
    {
        return Err(SwapchainCompatibilityError::UnsupportedImageContract(
            vk::Result::ERROR_FEATURE_NOT_PRESENT,
        ));
    }
    let template = crate::state::SwapchainTemplate::from_create_info(info)?;
    let image_flags = if info
        .flags
        .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
    {
        vk::ImageCreateFlags::MUTABLE_FORMAT
    } else {
        vk::ImageCreateFlags::empty()
    };
    let view_formats = template
        .create_chain
        .view_formats
        .clone()
        .unwrap_or_default();
    Ok(SwapchainVirtualizationPlan {
        template,
        image_flags,
        view_formats,
    })
}

unsafe fn allocate_logical_images(
    state: &DeviceState,
    info: &vk::SwapchainCreateInfoKHR<'_>,
    plan: &SwapchainVirtualizationPlan,
    count: usize,
) -> Result<Vec<tuxscaling_vulkan::Image>, vk::Result> {
    let memory = unsafe {
        state
            .instance
            .get_physical_device_memory_properties(state.physical_device)
    };
    let usage = info.image_usage
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::SAMPLED;
    let queue_families = if info.image_sharing_mode == vk::SharingMode::CONCURRENT {
        unsafe {
            std::slice::from_raw_parts(
                info.p_queue_family_indices,
                info.queue_family_index_count as usize,
            )
        }
    } else {
        &[]
    };
    let options = tuxscaling_vulkan::ImageCreateOptions {
        image_flags: plan.image_flags,
        view_formats: &plan.view_formats,
        queue_family_indices: queue_families,
    };
    (0..count)
        .map(|_| unsafe {
            tuxscaling_vulkan::Image::with_options(
                &state.device,
                &memory,
                info.image_extent,
                info.image_format,
                usage,
                &options,
            )
        })
        .collect()
}

fn publish_negotiation(
    surface_handle: vk::SurfaceKHR,
    negotiation: tuxscaling_display::PresentationNegotiation,
) {
    if let Some(surface) = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_mut(&surface_handle)
    {
        surface.negotiation = negotiation;
    }
    let states = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for (logical_handle, swapchain) in states.iter() {
        if is_retired_swapchain(*logical_handle) {
            continue;
        }
        if let Ok(mut swapchain) = swapchain.lock()
            && swapchain.surface == surface_handle
        {
            if negotiation.public_state() == tuxscaling_display::PresentationState::Direct
                && swapchain
                    .contract
                    .as_ref()
                    .is_some_and(LogicalSwapchainContract::is_failed)
            {
                continue;
            }
            swapchain.negotiation = negotiation;
            if let Some(contract) = swapchain.contract.as_mut() {
                contract.set_state(negotiation.public_state());
            }
        }
    }
}

fn translate_old_swapchain(logical: vk::SwapchainKHR) -> Result<vk::SwapchainKHR, vk::Result> {
    if logical == vk::SwapchainKHR::null() {
        return Ok(logical);
    }
    let handles = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&logical)
        .and_then(|state| {
            state
                .lock()
                .ok()
                .map(|state| (state.logical_handle, state.physical_handle))
        });
    if handles.is_none() && (is_retired_swapchain(logical) || is_unknown_logical_swapchain(logical))
    {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    Ok(handles
        .and_then(|(logical, physical)| {
            OldSwapchain::translate(Some(logical.as_raw()), logical.as_raw(), physical.as_raw())
        })
        .map(vk::SwapchainKHR::from_raw)
        .unwrap_or(logical))
}

#[derive(Clone)]
struct ActiveLogicalSwapchain {
    surface: vk::SurfaceKHR,
    contract: LogicalSwapchainContract,
    negotiation: PresentationNegotiation,
}

fn active_logical_swapchain(logical: vk::SwapchainKHR) -> Option<ActiveLogicalSwapchain> {
    if logical == vk::SwapchainKHR::null() {
        return None;
    }
    swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&logical)
        .and_then(|state| state.lock().ok())
        .and_then(|state| {
            state
                .contract
                .clone()
                .map(|contract| ActiveLogicalSwapchain {
                    surface: state.surface,
                    contract,
                    negotiation: state.negotiation,
                })
        })
}

struct SurfaceRecreationGuard(vk::SurfaceKHR);

impl SurfaceRecreationGuard {
    fn try_new(surface: vk::SurfaceKHR) -> Option<Self> {
        crate::state::begin_surface_recreation(surface).then_some(Self(surface))
    }
}

impl Drop for SurfaceRecreationGuard {
    fn drop(&mut self) {
        crate::state::end_surface_recreation(self.0);
    }
}

fn translate_recreation_create_info(
    logical_info: &vk::SwapchainCreateInfoKHR<'_>,
    old_contract: &LogicalSwapchainContract,
    downstream_extent: SurfaceExtent,
) -> Result<(vk::Extent2D, vk::SwapchainKHR), vk::Result> {
    if logical_info.old_swapchain != old_contract.handle() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let physical_extent = match downstream_extent {
        SurfaceExtent::Fixed(extent) => vk::Extent2D {
            width: extent.width,
            height: extent.height,
        },
        SurfaceExtent::Range { minimum, maximum } => {
            let current = old_contract.generation().extent();
            if surface_extent_accepts(downstream_extent, current) {
                current
            } else {
                vk::Extent2D {
                    width: logical_info
                        .image_extent
                        .width
                        .clamp(minimum.width, maximum.width),
                    height: logical_info
                        .image_extent
                        .height
                        .clamp(minimum.height, maximum.height),
                }
            }
        }
    };
    if physical_extent.width == 0
        || physical_extent.height == 0
        || !surface_extent_accepts(downstream_extent, physical_extent)
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    // Keep the previous physical generation active.  Passing it as
    // `oldSwapchain` would retire it in the driver and make present-wait
    // queries for IDs submitted to that generation invalid.
    Ok((physical_extent, vk::SwapchainKHR::null()))
}

#[derive(Clone, Copy)]
struct PhysicalCreateTarget {
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    old_swapchain: vk::SwapchainKHR,
}

unsafe fn create_physical_swapchain(
    create: vk::PFN_vkCreateSwapchainKHR,
    device: vk::Device,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    template: Option<&crate::state::SwapchainTemplate>,
    fallback_info: &vk::SwapchainCreateInfoKHR<'_>,
    target: PhysicalCreateTarget,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    if let Some(template) = template {
        template.with_create_info(
            target.surface,
            target.extent,
            target.old_swapchain,
            |create_info| unsafe { create(device, create_info, allocation_callbacks, swapchain) },
        )
    } else {
        unsafe { create(device, fallback_info, allocation_callbacks, swapchain) }
    }
}

fn allocate_logical_swapchain(physical: vk::SwapchainKHR) -> Option<vk::SwapchainKHR> {
    static NEXT_LOGICAL_SWAPCHAIN: AtomicU64 = AtomicU64::new(1);

    // A tagged token is safe only while the downstream ICD has not produced a
    // handle in that namespace.  Refuse virtualization on an observed clash
    // rather than ever exposing a physical handle as a logical identity.
    if LogicalSwapchainHandle::is_reserved(physical.as_raw()) {
        return None;
    }
    for _ in 0..1024 {
        let counter = NEXT_LOGICAL_SWAPCHAIN.fetch_add(1, Ordering::Relaxed);
        let candidate =
            vk::SwapchainKHR::from_raw(LogicalSwapchainHandle::from_counter(counter).raw());
        let collision = swapchains()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .any(|(logical, state)| {
                *logical == candidate
                    || state
                        .lock()
                        .ok()
                        .is_some_and(|state| state.physical_handle == candidate)
            });
        if !collision {
            return Some(candidate);
        }
    }
    None
}

unsafe fn downstream_surface_capabilities(
    state: &DeviceState,
    surface: vk::SurfaceKHR,
) -> Option<vk::SurfaceCapabilitiesKHR> {
    let proc = unsafe {
        downstream(
            state.instance.handle(),
            c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        )
    }?;
    let get: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
        unsafe { std::mem::transmute(proc) };
    let mut capabilities = vk::SurfaceCapabilitiesKHR::default();
    (unsafe { get(state.physical_device, surface, &mut capabilities) } == vk::Result::SUCCESS)
        .then_some(capabilities)
}

fn surface_extent(capabilities: vk::SurfaceCapabilitiesKHR) -> SurfaceExtent {
    if capabilities.current_extent.width != u32::MAX {
        SurfaceExtent::fixed(Extent::new(
            capabilities.current_extent.width,
            capabilities.current_extent.height,
        ))
    } else {
        SurfaceExtent::Range {
            minimum: Extent::new(
                capabilities.min_image_extent.width,
                capabilities.min_image_extent.height,
            ),
            maximum: Extent::new(
                capabilities.max_image_extent.width,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn initial_physical_extent(
    requested: vk::Extent2D,
    surface_extent: SurfaceExtent,
    virtual_eligible: bool,
) -> vk::Extent2D {
    if !virtual_eligible {
        return requested;
    }
    match surface_extent {
        SurfaceExtent::Fixed(extent) => vk::Extent2D {
            width: extent.width,
            height: extent.height,
        },
        SurfaceExtent::Range { minimum, maximum } => vk::Extent2D {
            width: requested.width.clamp(minimum.width, maximum.width),
            height: requested.height.clamp(minimum.height, maximum.height),
        },
    }
}

#[cfg(test)]
const fn native_generation_failure_can_restore(destroy_requested: bool) -> bool {
    !destroy_requested
}

#[derive(Clone, Copy)]
struct InitialOutputTarget {
    window: u64,
    monitor: tuxscaling_display::Monitor,
}

fn native_output_target(
    surface: vk::SurfaceKHR,
    preflight_eligible: bool,
) -> Option<InitialOutputTarget> {
    if !preflight_eligible {
        return None;
    }
    let config = if let Ok(path) = std::env::var("TUXSCALING_CONFIG") {
        let source = std::fs::read_to_string(path).ok()?;
        tuxscaling_config::Config::parse(&source).ok()?
    } else {
        tuxscaling_config::Config::default()
    };
    if !matches!(
        config.output_resolution,
        tuxscaling_config::OutputResolution::Native
    ) {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} reason=config_not_native",
            surface.as_raw(),
        );
        return None;
    }
    // Pending Wine/Proton Win32 surfaces retry their process-window
    // association here, when the X11 window may exist even though it did not
    // at surface-creation time. Other unknown surfaces stay unresolved.
    let Some(window) = super::surface::surface_window(surface) else {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} reason=no_window",
            surface.as_raw(),
        );
        return None;
    };
    let Ok(display) = tuxscaling_display::X11Display::connect() else {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} reason=display_unavailable",
            surface.as_raw(),
        );
        return None;
    };
    let Ok(described) = display.describe_window(window) else {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} window={} reason=window_unavailable",
            surface.as_raw(),
            window,
        );
        return None;
    };
    let Ok(monitor) = display.monitor_for_window(window) else {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} window={} rect={}x{}+{}+{} fullscreen={} reason=target_unavailable",
            surface.as_raw(),
            window,
            described.rect.width,
            described.rect.height,
            described.rect.x,
            described.rect.y,
            described.fullscreen,
        );
        return None;
    };
    let target_extent = monitor.rect.extent();
    if !target_extent.is_valid() {
        eprintln!(
            "TuxScaling evidence event=borderless_skipped surface=0x{:x} window={} reason=invalid_target",
            surface.as_raw(),
            window,
        );
        return None;
    }
    Some(InitialOutputTarget { window, monitor })
}

fn has_live_virtual_swapchain(surface: vk::SurfaceKHR) -> bool {
    swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .any(|state| {
            state
                .lock()
                .is_ok_and(|state| state.surface == surface && state.mapping.is_some())
        })
}

/// Drops a stale logical capability override so capability queries report the
/// truthful downstream state. Only used when no virtual swapchain is alive on
/// the surface; the window and lease are left untouched.
fn clear_surface_logical_override(surface: vk::SurfaceKHR) {
    if let Some(state) = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_mut(&surface)
    {
        state.logical_extent = None;
        state.logical_capabilities = None;
        state.negotiation = tuxscaling_display::PresentationNegotiation::direct();
    }
}

pub(super) unsafe fn observe_surface_negotiation(
    device_state: &DeviceState,
    surface: vk::SurfaceKHR,
    now: Instant,
) -> tuxscaling_display::PresentationState {
    let _ = super::surface::refresh_pending_win32_surface(surface);
    let Some(snapshot) = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .copied()
    else {
        return tuxscaling_display::PresentationState::Direct;
    };
    if snapshot.window == super::surface::PENDING_WIN32_WINDOW {
        return snapshot.negotiation.public_state();
    }
    let Ok(display) = tuxscaling_display::X11Display::connect() else {
        return snapshot.negotiation.public_state();
    };
    let Ok(window) = display.describe_window(snapshot.window) else {
        return snapshot.negotiation.public_state();
    };
    let Some(capabilities) = (unsafe { downstream_surface_capabilities(device_state, surface) })
    else {
        return snapshot.negotiation.public_state();
    };
    let mut negotiation = snapshot.negotiation;
    let _ = negotiation.observe(
        window.rect,
        window.fullscreen,
        surface_extent(capabilities),
        now,
    );
    if negotiation.output_recreation_ready() {
        eprintln!(
            "TuxScaling evidence event=native_publication_attempt window={}x{}+{}+{} fullscreen={} surface={:?}",
            window.rect.width,
            window.rect.height,
            window.rect.x,
            window.rect.y,
            window.fullscreen,
            surface_extent(capabilities),
        );
    }
    let mut state = negotiation.public_state();
    if negotiation.failure().is_some() {
        super::lifetime::restore_surface_window(surface);
        negotiation = tuxscaling_display::PresentationNegotiation::direct();
        state = tuxscaling_display::PresentationState::Direct;
    }
    publish_negotiation(surface, negotiation);
    if negotiation.output_recreation_ready()
        && unsafe { publish_native_generation(device_state, surface) }
    {
        state = tuxscaling_display::PresentationState::Virtualized;
    }
    state
}

struct NativeGenerationSnapshot {
    state: Arc<Mutex<SwapchainState>>,
    ticket: ReconfigurationTicket,
    old_physical: vk::SwapchainKHR,
    old_physical_images: Vec<vk::Image>,
    old_runtime_info: SwapchainInfo,
    template: crate::state::SwapchainTemplate,
    contract: LogicalSwapchainContract,
    negotiation: PresentationNegotiation,
    hdr_metadata: Option<crate::hooks::swapchain_metadata::OwnedHdrMetadata>,
}

fn begin_native_generation(
    state: Arc<Mutex<SwapchainState>>,
    surface: vk::SurfaceKHR,
    observed: tuxscaling_display::X11Window,
    surface_extent: SurfaceExtent,
) -> Option<(NativeGenerationSnapshot, OverlaySwapchain)> {
    let mut guard = state.lock().unwrap_or_else(|error| error.into_inner());
    if guard.surface != surface
        || !guard
            .mapping
            .as_ref()
            .is_some_and(crate::mapping::Mapping::is_idle)
        || guard.lifecycle.blocks_frame_operations()
        || !guard.negotiation.native_observation_is_current(
            observed.rect,
            observed.fullscreen,
            surface_extent,
            Instant::now(),
        )
        || guard
            .contract
            .as_ref()
            .is_none_or(|contract| contract.generation().id() != guard.generation)
    {
        return None;
    }
    let template = guard.template.clone()?;
    let contract = guard.contract.clone()?;
    let ticket = ReconfigurationTicket::new(guard.logical_handle, surface, guard.generation);
    if !guard.lifecycle.begin(ticket) {
        return None;
    }
    let Some(runtime) = guard.overlay.take() else {
        let _ = guard.lifecycle.abort(ticket);
        return None;
    };
    let old_physical = guard.physical_handle;
    let old_physical_images = guard.physical_images.clone();
    let old_runtime_info = SwapchainInfo {
        format: template.image_format,
        color_space: template.image_color_space,
        extent: contract.generation().extent(),
    };
    let negotiation = guard.negotiation;
    let hdr_metadata = guard.hdr_metadata;
    drop(guard);
    let snapshot = NativeGenerationSnapshot {
        state,
        ticket,
        old_physical,
        old_physical_images,
        old_runtime_info,
        template,
        contract,
        negotiation,
        hdr_metadata,
    };
    Some((snapshot, runtime))
}

unsafe fn destroy_abandoned_generation(
    device_state: &DeviceState,
    snapshot: &NativeGenerationSnapshot,
    runtime: OverlaySwapchain,
    new_physical: vk::SwapchainKHR,
    loader: &ash::khr::swapchain::Device,
) {
    let removed = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .remove(&snapshot.ticket.logical_handle());
    let owns_old_physical = removed.is_some();
    if owns_old_physical {
        retire_swapchain(snapshot.ticket.logical_handle());
    }
    let _ = unsafe { device_state.device.device_wait_idle() };
    unsafe { runtime.destroy(&device_state.device) };
    if new_physical != vk::SwapchainKHR::null() {
        unsafe { loader.destroy_swapchain(new_physical, None) };
    }
    if owns_old_physical {
        unsafe { loader.destroy_swapchain(snapshot.old_physical, None) };
    }
    let another_virtual_swapchain = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .any(|state| {
            let state = state.lock().unwrap_or_else(|error| error.into_inner());
            state.surface == snapshot.ticket.surface() && state.mapping.is_some()
        });
    if owns_old_physical && !another_virtual_swapchain {
        super::lifetime::restore_surface_window(snapshot.ticket.surface());
    }
}

unsafe fn destroy_temporary_generation(
    device_state: &DeviceState,
    snapshot: &NativeGenerationSnapshot,
    new_physical: vk::SwapchainKHR,
    loader: &ash::khr::swapchain::Device,
) {
    if new_physical == vk::SwapchainKHR::null() || new_physical == snapshot.old_physical {
        return;
    }
    let _ = unsafe { device_state.device.device_wait_idle() };
    unsafe { loader.destroy_swapchain(new_physical, None) };
}

unsafe fn finish_native_generation_failure(
    device_state: &DeviceState,
    snapshot: NativeGenerationSnapshot,
    runtime: OverlaySwapchain,
    new_physical: vk::SwapchainKHR,
    loader: &ash::khr::swapchain::Device,
    runtime_reconfigured: bool,
) {
    let mut runtime = Some(runtime);
    let destroy_requested = {
        let mut guard = snapshot
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let destroy_requested = guard.lifecycle.destroy_requested();
        if destroy_requested {
            let _ = guard.lifecycle.take_destroy_request(snapshot.ticket);
        }
        destroy_requested
    };
    if destroy_requested {
        unsafe {
            destroy_abandoned_generation(
                device_state,
                &snapshot,
                runtime
                    .take()
                    .expect("destroyed transaction owns the runtime"),
                new_physical,
                loader,
            )
        };
        return;
    }

    if runtime_reconfigured
        && unsafe {
            runtime
                .as_mut()
                .expect("reconfiguration owns the runtime")
                .restore_output(
                    snapshot.old_runtime_info,
                    snapshot.old_physical_images.clone(),
                )
        }
        .is_err()
    {
        unsafe {
            destroy_abandoned_generation(
                device_state,
                &snapshot,
                runtime.take().expect("failed restoration owns the runtime"),
                new_physical,
                loader,
            )
        };
        return;
    }

    let restored = {
        let mut guard = snapshot
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if guard.lifecycle.destroy_requested() {
            let _ = guard.lifecycle.take_destroy_request(snapshot.ticket);
            false
        } else if guard.lifecycle.can_publish(
            snapshot.ticket,
            guard.logical_handle,
            guard.surface,
            guard.generation,
            is_retired_swapchain(guard.logical_handle),
        ) {
            guard.overlay = runtime.take();
            let _ = guard.lifecycle.abort(snapshot.ticket);
            true
        } else {
            false
        }
    };
    if restored {
        unsafe { destroy_temporary_generation(device_state, &snapshot, new_physical, loader) };
    } else {
        unsafe {
            destroy_abandoned_generation(
                device_state,
                &snapshot,
                runtime
                    .take()
                    .expect("abandoned transaction owns the runtime"),
                new_physical,
                loader,
            )
        };
    }
}

/// Replaces only the downstream WSI generation after exact X11 and Vulkan
/// confirmation.  The state marker and moved runtime form a two-phase
/// transaction: no external call observes a held swapchain mutex, and no
/// frame operation can use a partially replaced generation.
pub(super) unsafe fn publish_native_generation(
    device_state: &DeviceState,
    surface: vk::SurfaceKHR,
) -> bool {
    let Some(_recreation_guard) = SurfaceRecreationGuard::try_new(surface) else {
        return false;
    };
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .find(|(logical_handle, state)| {
            !is_retired_swapchain(**logical_handle)
                && state
                    .lock()
                    .ok()
                    .is_some_and(|state| state.surface == surface && state.mapping.is_some())
        })
        .map(|(_, state)| state.clone());
    let Some(state) = state else {
        eprintln!(
            "TuxScaling evidence event=native_publication_skipped reason=no_active_swapchain"
        );
        return false;
    };
    let display = match tuxscaling_display::X11Display::connect() {
        Ok(display) => display,
        Err(_) => return false,
    };
    let window = match surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .copied()
    {
        Some(surface) => surface.window,
        None => return false,
    };
    let observed = match display.describe_window(window) {
        Ok(observed) => observed,
        Err(_) => return false,
    };
    let capabilities = match unsafe { downstream_surface_capabilities(device_state, surface) } {
        Some(capabilities) => capabilities,
        None => return false,
    };
    let surface_extent = surface_extent(capabilities);
    let Some((snapshot, mut runtime)) =
        begin_native_generation(state, surface, observed, surface_extent)
    else {
        eprintln!("TuxScaling evidence event=native_publication_skipped reason=stale_or_busy");
        return false;
    };
    let target_extent = vk::Extent2D {
        width: observed.rect.width,
        height: observed.rect.height,
    };
    let mut published_negotiation = snapshot.negotiation;
    if !published_negotiation.output_recreated(
        observed.rect,
        observed.fullscreen,
        surface_extent,
        Instant::now(),
    ) || !surface_extent_accepts(surface_extent, target_extent)
    {
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                vk::SwapchainKHR::null(),
                &ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device),
                false,
            )
        };
        return false;
    }
    let Some(proc) =
        (unsafe { device_downstream(device_state.device.handle(), c"vkCreateSwapchainKHR") })
    else {
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                vk::SwapchainKHR::null(),
                &ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device),
                false,
            )
        };
        return false;
    };
    let create: vk::PFN_vkCreateSwapchainKHR = unsafe { std::mem::transmute(proc) };
    let mut new_physical = vk::SwapchainKHR::null();
    let result = snapshot.template.with_create_info(
        surface,
        target_extent,
        vk::SwapchainKHR::null(),
        |create_info| unsafe {
            create(
                device_state.device.handle(),
                create_info,
                std::ptr::null(),
                &mut new_physical,
            )
        },
    );
    let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
    if result != vk::Result::SUCCESS || new_physical == vk::SwapchainKHR::null() {
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                new_physical,
                &loader,
                false,
            )
        };
        return false;
    }
    let new_images = match unsafe { loader.get_swapchain_images(new_physical) } {
        Ok(images) => images,
        Err(_) => {
            unsafe {
                finish_native_generation_failure(
                    device_state,
                    snapshot,
                    runtime,
                    new_physical,
                    &loader,
                    false,
                )
            };
            return false;
        }
    };
    if let Some(metadata) = snapshot.hdr_metadata
        && !unsafe {
            super::apply_hdr_metadata(device_state.device.handle(), new_physical, metadata)
        }
    {
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                new_physical,
                &loader,
                false,
            )
        };
        return false;
    }
    let info = SwapchainInfo {
        format: snapshot.template.image_format,
        color_space: snapshot.template.image_color_space,
        extent: target_extent,
    };
    if unsafe { runtime.reconfigure_output(info, new_images.clone()) }.is_err() {
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                new_physical,
                &loader,
                false,
            )
        };
        return false;
    }

    let mut runtime = Some(runtime);
    let published = {
        let mut guard = snapshot
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prepared = match (guard.mapping.clone(), guard.contract.clone()) {
            (Some(mut mapping), Some(mut contract)) => {
                let next_generation = snapshot.ticket.generation().saturating_add(1);
                let valid = guard.lifecycle.can_publish(
                    snapshot.ticket,
                    guard.logical_handle,
                    guard.surface,
                    guard.generation,
                    is_retired_swapchain(guard.logical_handle),
                );
                valid
                    && mapping.replace_generation(next_generation)
                    && contract
                        .replace_generation(
                            PhysicalGeneration::new(
                                next_generation,
                                new_physical,
                                target_extent,
                                new_images.len(),
                            ),
                            published_negotiation.public_state(),
                        )
                        .is_ok()
            }
            _ => false,
        };
        let next_generation = snapshot.ticket.generation().saturating_add(1);
        let logical_handle = guard.logical_handle;
        let current_surface = guard.surface;
        let current_generation = guard.generation;
        let retired = is_retired_swapchain(logical_handle);
        let published = prepared
            && guard.lifecycle.publish(
                snapshot.ticket,
                logical_handle,
                current_surface,
                current_generation,
                retired,
            );
        if !published {
            false
        } else {
            let mut mapping = guard.mapping.take().expect("mapping was prepared");
            let mut contract = guard.contract.take().expect("contract was prepared");
            let _ = mapping.replace_generation(next_generation);
            let _ = contract.replace_generation(
                PhysicalGeneration::new(
                    next_generation,
                    new_physical,
                    target_extent,
                    new_images.len(),
                ),
                published_negotiation.public_state(),
            );
            contract.set_state(published_negotiation.public_state());
            guard.mapping = Some(mapping);
            guard.contract = Some(contract);
            guard.negotiation = published_negotiation;
            guard.physical_handle = new_physical;
            guard.physical_images = new_images.clone();
            guard.maintenance_overlay_reported = false;
            guard.maintenance_present_reported = false;
            guard.maintenance_release_reported = false;
            guard.generation = next_generation;
            if snapshot.old_physical != vk::SwapchainKHR::null()
                && !guard
                    .retired_physical_generations
                    .contains(&snapshot.old_physical)
            {
                guard
                    .retired_physical_generations
                    .push(snapshot.old_physical);
            }
            guard.overlay = runtime.take();
            true
        }
    };
    if !published {
        let runtime = runtime.expect("failed publication retains the runtime");
        unsafe {
            finish_native_generation_failure(
                device_state,
                snapshot,
                runtime,
                new_physical,
                &loader,
                true,
            )
        };
        return false;
    }
    let logical_extent = snapshot.contract.game_extent();
    publish_negotiation(surface, published_negotiation);
    eprintln!(
        "TuxScaling evidence event=virtual_swapchain_active logical_handle=0x{:x} logical={}x{} physical={}x{} generation={}",
        snapshot.ticket.logical_handle().as_raw(),
        logical_extent.width,
        logical_extent.height,
        target_extent.width,
        target_extent.height,
        snapshot.ticket.generation().saturating_add(1),
    );
    true
}

fn surface_extent_accepts(surface: SurfaceExtent, target: vk::Extent2D) -> bool {
    match surface {
        SurfaceExtent::Fixed(extent) => {
            extent.width == target.width && extent.height == target.height
        }
        SurfaceExtent::Range { minimum, maximum } => {
            target.width >= minimum.width
                && target.width <= maximum.width
                && target.height >= minimum.height
                && target.height <= maximum.height
        }
    }
}

fn logical_capabilities_for_extent(
    existing: Option<vk::SurfaceCapabilitiesKHR>,
    fallback: Option<vk::SurfaceCapabilitiesKHR>,
    logical_extent: vk::Extent2D,
) -> Option<vk::SurfaceCapabilitiesKHR> {
    let mut capabilities = existing.or(fallback)?;
    capabilities.current_extent = logical_extent;
    capabilities.min_image_extent = logical_extent;
    capabilities.max_image_extent = logical_extent;
    Some(capabilities)
}

fn physical_swapchain_info(info: &vk::SwapchainCreateInfoKHR<'_>) -> SwapchainInfo {
    SwapchainInfo {
        format: info.image_format,
        color_space: info.image_color_space,
        extent: info.image_extent,
    }
}

fn temporal_enabled_for_logical_creation(
    virtual_eligible: bool,
    previous_state: Option<PresentationState>,
) -> bool {
    !virtual_eligible || previous_state == Some(PresentationState::Virtualized)
}

unsafe fn find_device_callback(
    mut next: *const c_void,
) -> Option<tuxscaling_runtime::SetLoaderData> {
    while !next.is_null() {
        let info = next.cast::<LayerCreateInfo>();
        if unsafe {
            (*info).s_type == vk::StructureType::LOADER_DEVICE_CREATE_INFO && (*info).function == 1
        } {
            let callback = unsafe { (*info).data._set_loader_data };
            return if callback.is_null() {
                None
            } else {
                Some(unsafe {
                    std::mem::transmute::<*const c_void, tuxscaling_runtime::SetLoaderData>(
                        callback,
                    )
                })
            };
        }
        next = unsafe { (*info).p_next };
    }
    None
}

unsafe fn find_layer_link(
    mut next: *const c_void,
    expected_type: vk::StructureType,
) -> Option<*mut LayerCreateInfo> {
    while !next.is_null() {
        let info = next.cast::<LayerCreateInfo>();
        if unsafe { (*info).s_type } == expected_type
            && unsafe { (*info).function } == LAYER_LINK_INFO
        {
            return Some(info.cast_mut());
        }
        next = unsafe { (*info).p_next };
    }
    None
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
    let chain = link;
    let link = unsafe { (*chain).data.layer_info.cast::<InstanceLayerLink>() };
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let get_instance_proc_addr = unsafe { (*link).get_instance_proc_addr };
    *next_gpdpa().lock().unwrap_or_else(|e| e.into_inner()) =
        unsafe { (*link).get_physical_device_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*chain).data.layer_info = next.cast() };
    *next_gipa().lock().unwrap_or_else(|e| e.into_inner()) = Some(get_instance_proc_addr);

    let Some(proc) =
        (unsafe { get_instance_proc_addr(vk::Instance::null(), c"vkCreateInstance".as_ptr()) })
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_instance: vk::PFN_vkCreateInstance = unsafe { std::mem::transmute(proc) };
    let requested_api_version = unsafe {
        (*create_info)
            .p_application_info
            .as_ref()
            .map(|application| application.api_version)
            .filter(|version| *version != 0)
            .unwrap_or(vk::API_VERSION_1_0)
    };
    let mut supported_api_version = vk::API_VERSION_1_0;
    if let Some(proc) = unsafe {
        get_instance_proc_addr(vk::Instance::null(), c"vkEnumerateInstanceVersion".as_ptr())
    } {
        let enumerate_instance_version: vk::PFN_vkEnumerateInstanceVersion =
            unsafe { std::mem::transmute(proc) };
        let _ = unsafe { enumerate_instance_version(&mut supported_api_version) };
    }
    let vulkan_api_version = if requested_api_version < vk::API_VERSION_1_2
        && supported_api_version >= vk::API_VERSION_1_2
    {
        vk::API_VERSION_1_2
    } else {
        requested_api_version
    };
    let mut modified_application_info = unsafe {
        (*create_info)
            .p_application_info
            .as_ref()
            .copied()
            .unwrap_or_default()
    };
    if modified_application_info.s_type == vk::StructureType::APPLICATION_INFO {
        modified_application_info.api_version = vulkan_api_version;
    } else {
        modified_application_info = vk::ApplicationInfo::default().api_version(vulkan_api_version);
    }
    let mut modified_create_info = unsafe { *create_info };
    modified_create_info.p_application_info = &modified_application_info;
    let result = unsafe { create_instance(&modified_create_info, allocation_callbacks, instance) };
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if result == vk::Result::SUCCESS {
            crate::state::instance_dispatch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(unsafe { *instance }, get_instance_proc_addr);
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
                .unwrap_or_else(|e| e.into_inner())
                .insert(unsafe { *instance }, ash_instance);
            instance_api_versions()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(unsafe { *instance }, vulkan_api_version);
        }
        result
    }));
    result
}
pub(super) unsafe extern "system" fn create_instance(
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
    let chain = link;
    let link = unsafe { (*chain).data.layer_info.cast::<DeviceLayerLink>() };
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let candidates = instances()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let Some(instance) = candidates.into_iter().find(|instance| {
        unsafe { instance.enumerate_physical_devices() }
            .is_ok_and(|handles| handles.contains(&physical_device))
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let vulkan_api_version = instance_api_versions()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&instance.handle())
        .copied()
        .unwrap_or(vk::API_VERSION_1_0);
    let get_device_proc_addr = unsafe { (*link).get_device_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*chain).data.layer_info = next.cast() };
    let Some(proc) = (unsafe {
        ((*link).get_instance_proc_addr)(instance.handle(), c"vkCreateDevice".as_ptr())
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_device: vk::PFN_vkCreateDevice = unsafe { std::mem::transmute(proc) };
    let physical_features = unsafe { instance.get_physical_device_features(physical_device) };
    let physical_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let mut modified_info = unsafe { *create_info };
    let original_p_next = unsafe { (*create_info).p_next };
    let original_p_enabled_features = unsafe { (*create_info).p_enabled_features };
    let mut enabled_features = unsafe { (*create_info).p_enabled_features.as_ref() }
        .copied()
        .unwrap_or_default();
    let mut features2_override = vk::PhysicalDeviceFeatures2::default();
    let mut vulkan12_override = vk::PhysicalDeviceVulkan12Features::default();
    let mut supported_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    if physical_properties.api_version >= vk::API_VERSION_1_2 {
        let mut supported_features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut supported_vulkan12 as *mut _ as *mut c_void,
            ..Default::default()
        };
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut supported_features2)
        };
    }
    let mut next = unsafe { (*create_info).p_next.cast::<vk::BaseInStructure<'_>>() };
    let mut has_features2 = false;
    let mut existing_vulkan12: *mut vk::PhysicalDeviceVulkan12Features<'static> =
        std::ptr::null_mut();
    while !next.is_null() {
        match unsafe { (*next).s_type } {
            vk::StructureType::PHYSICAL_DEVICE_FEATURES_2 => has_features2 = true,
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                existing_vulkan12 = next.cast_mut().cast();
            }
            _ => {}
        }
        next = unsafe { (*next).p_next.cast() };
    }
    let enable_storage_write = physical_features.shader_storage_image_write_without_format != 0;
    let enable_shader_int16 = physical_features.shader_int16 != 0;
    let enable_shader_float16 = supported_vulkan12.shader_float16 != 0;
    let mut use_features2_override = false;
    if !has_features2 && (enable_storage_write || enable_shader_int16 || enable_shader_float16) {
        features2_override.features = enabled_features;
        if enable_storage_write {
            features2_override
                .features
                .shader_storage_image_write_without_format = vk::TRUE;
        }
        if enable_shader_int16 {
            features2_override.features.shader_int16 = vk::TRUE;
        }
        if enable_shader_float16 {
            if existing_vulkan12.is_null() {
                vulkan12_override.shader_float16 = vk::TRUE;
                vulkan12_override.p_next = original_p_next as *mut c_void;
                features2_override.p_next = &mut vulkan12_override as *mut _ as *mut c_void;
            } else {
                unsafe {
                    (*existing_vulkan12).shader_float16 = vk::TRUE;
                }
                features2_override.p_next = original_p_next as *mut c_void;
            }
        } else {
            features2_override.p_next = original_p_next as *mut c_void;
        }
        modified_info.p_enabled_features = std::ptr::null();
        modified_info.p_next = &features2_override as *const _ as *const c_void;
        use_features2_override = true;
    }
    if !use_features2_override && !original_p_enabled_features.is_null() {
        if enable_storage_write {
            enabled_features.shader_storage_image_write_without_format = vk::TRUE;
        }
        if enable_shader_int16 {
            enabled_features.shader_int16 = vk::TRUE;
        }
        modified_info.p_enabled_features = &enabled_features;
    }
    let result = unsafe {
        create_device(
            physical_device,
            &modified_info,
            allocation_callbacks,
            device,
        )
    };
    if result != vk::Result::SUCCESS {
        return result;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let ash_device = unsafe {
            ash::Device::load_with(
                |command| {
                    if command.to_bytes() == b"vkAllocateCommandBuffers" {
                        return allocate_internal_commands as *const c_void;
                    }
                    get_device_proc_addr(*device, command.as_ptr())
                        .map_or(std::ptr::null(), |function| function as *const c_void)
                },
                *device,
            )
        };
        let wsi = unsafe {
            super::wsi_compatibility::DeviceWsiCapabilities::from_create_info(
                &*create_info,
                vulkan_api_version,
            )
        };
        if wsi.maintenance1.enabled {
            let flavor = match wsi.maintenance1.flavor {
                Some(super::maintenance::Maintenance1Flavor::Ext) => "ext",
                Some(super::maintenance::Maintenance1Flavor::Khr) => "khr",
                None => "unknown",
            };
            eprintln!("TuxScaling evidence event=maintenance1_device enabled=1 flavor={flavor}");
        }
        if let Some(incompatible) = &wsi.incompatible {
            eprintln!(
                "TuxScaling evidence event=virtualization_preflight result=direct reason=incompatible_wsi_extension extension={}",
                String::from_utf8_lossy(&incompatible.name),
            );
        }
        devices().lock().unwrap_or_else(|e| e.into_inner()).insert(
            unsafe { *device },
            DeviceState {
                wsi,
                vulkan_api_version,
                queue_families: unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                },
                get_device_proc_addr,
                set_loader_data: unsafe { find_device_callback((*create_info).p_next) },
                physical_device,
                instance,
                device: ash_device,
            },
        );
        result
    }));
    result
}
unsafe extern "system" fn allocate_internal_commands(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    commands: *mut vk::CommandBuffer,
) -> vk::Result {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        let state = devices()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&device)
            .cloned();
        let Some(state) = state else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(proc) = (state.get_device_proc_addr)(device, c"vkAllocateCommandBuffers".as_ptr())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let allocate: vk::PFN_vkAllocateCommandBuffers = std::mem::transmute(proc);
        let result = allocate(device, info, commands);
        if result == vk::Result::SUCCESS
            && let Some(set) = state.set_loader_data
        {
            use ash::vk::Handle;
            for i in 0..(*info).command_buffer_count as usize {
                let result = set(device, (*commands.add(i)).as_raw() as *mut c_void);
                if result != vk::Result::SUCCESS {
                    return result;
                }
            }
        }
        result
    }))
    .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

pub(super) unsafe extern "system" fn create_device(
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

unsafe fn register_queue(
    device: vk::Device,
    family_index: u32,
    queue: *mut vk::Queue,
    flags: vk::DeviceQueueCreateFlags,
) {
    if !queue.is_null() {
        queues().lock().unwrap_or_else(|e| e.into_inner()).insert(
            unsafe { *queue },
            QueueState {
                device,
                family_index,
                processing_allowed: flags.is_empty(),
            },
        );
    }
}

pub(super) unsafe extern "system" fn get_device_queue(
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
        register_queue(
            device,
            queue_family_index,
            queue,
            vk::DeviceQueueCreateFlags::empty(),
        );
    }));
}

pub(super) unsafe extern "system" fn get_device_queue2(
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
            register_queue(
                device,
                (*queue_info).queue_family_index,
                queue,
                (*queue_info).flags,
            );
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
    let state = devices()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&device)
        .cloned();
    let original = unsafe { &*create_info };
    eprintln!(
        "TuxScaling evidence event=swapchain_create_request old_swapchain=0x{:x} surface=0x{:x} extent={}x{}",
        original.old_swapchain.as_raw(),
        original.surface.as_raw(),
        original.image_extent.width,
        original.image_extent.height,
    );
    if is_retired_swapchain(original.old_swapchain) {
        return vk::Result::ERROR_OUT_OF_DATE_KHR;
    }
    if is_reconfiguring_swapchain(original.old_swapchain) {
        return vk::Result::ERROR_OUT_OF_DATE_KHR;
    }
    let old_logical = active_logical_swapchain(original.old_swapchain);
    if old_logical
        .as_ref()
        .is_some_and(|old| old.surface != original.surface)
    {
        return vk::Result::ERROR_OUT_OF_DATE_KHR;
    }
    let _recreation_guard = if old_logical.is_some() {
        let Some(guard) = SurfaceRecreationGuard::try_new(original.surface) else {
            return vk::Result::ERROR_OUT_OF_DATE_KHR;
        };
        Some(guard)
    } else {
        None
    };
    let old_swapchain = match translate_old_swapchain(original.old_swapchain) {
        Ok(swapchain) => swapchain,
        Err(error) => return error,
    };
    let mut driver_original = *original;
    driver_original.old_swapchain = old_swapchain;
    let mut modified = driver_original;
    let logical_capabilities = state.as_ref().and_then(|state| unsafe {
        let get: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
            std::mem::transmute(downstream(
                state.instance.handle(),
                c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            )?);
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        (get(state.physical_device, original.surface, &mut caps) == vk::Result::SUCCESS)
            .then_some(caps)
    });
    let needed = vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::STORAGE;
    let mut capture_enabled = false;
    if let Some(state) = &state
        && logical_capabilities.is_some_and(|caps| caps.supported_usage_flags.contains(needed))
    {
        let features = unsafe {
            state
                .instance
                .get_physical_device_format_properties(state.physical_device, original.image_format)
        }
        .optimal_tiling_features;
        if features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST)
        {
            modified.image_usage |= needed;
            capture_enabled = true;
        }
    }
    let format_features = state
        .as_ref()
        .map_or(vk::FormatFeatureFlags::empty(), |state| {
            unsafe {
                state.instance.get_physical_device_format_properties(
                    state.physical_device,
                    original.image_format,
                )
            }
            .optimal_tiling_features
        });
    let mut virtualization_plan = state.as_ref().and_then(|state| {
        match preflight_swapchain_virtualization(
            &state.wsi,
            &modified,
            logical_capabilities,
            format_features,
        ) {
            Ok(plan) => Some(plan),
            Err(error) => {
                eprintln!(
                    "TuxScaling evidence event=virtualization_preflight result=direct reason={}",
                    error.reason()
                );
                None
            }
        }
    });
    let mut virtual_preflight_eligible = virtualization_plan.is_some();
    // Refresh a pending Wine/Proton Win32 association before snapshotting so
    // the first swapchain creation after the X11 window appears can virtualize.
    let _ = super::surface::refresh_pending_win32_surface(original.surface);
    let surface_snapshot = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&original.surface)
        .copied();
    let native_target = native_output_target(original.surface, virtual_preflight_eligible);
    let initial_target = if old_logical.is_some() {
        None
    } else {
        native_target.filter(|target| {
            let extent = target.monitor.rect.extent();
            extent.width != original.image_extent.width
                || extent.height != original.image_extent.height
        })
    };
    // A native request on a promoted surface means the application adopted the
    // promoted window as its own resolution. There is nothing to upscale, so
    // no logical predecessor is inferred: the request below goes direct, and
    // the stale-override guard afterwards keeps capability queries truthful.
    // The lease is kept, so a later smaller request promotes idempotently and
    // virtualizes again without disturbing the window.
    let virtual_intent = old_logical.is_some() || initial_target.is_some();
    let mut preallocated_virtual_images = None;
    if let Some(plan) = virtualization_plan.as_ref() {
        if let Some(count) = preflight_logical_image_count(original.min_image_count) {
            match unsafe {
                allocate_logical_images(
                    state.as_ref().expect("eligible state must exist"),
                    original,
                    plan,
                    count,
                )
            } {
                Ok(images) => preallocated_virtual_images = Some(images),
                Err(error) => {
                    let error = SwapchainCompatibilityError::UnsupportedImageContract(error);
                    eprintln!(
                        "TuxScaling evidence event=virtualization_preflight result=direct reason={}",
                        error.reason()
                    );
                    virtualization_plan = None;
                    virtual_preflight_eligible = false;
                }
            }
        } else {
            virtualization_plan = None;
            virtual_preflight_eligible = false;
        }
    }
    if !virtual_preflight_eligible {
        super::lifetime::restore_surface_window(original.surface);
    }
    if virtualization_plan.is_some() && !virtual_intent && old_logical.is_none() {
        // Eligible but intentionally direct with no virtual successor: drop a
        // stale logical override so later capability queries stay truthful,
        // unless a virtual swapchain is still alive on this surface.
        if !has_live_virtual_swapchain(original.surface) {
            clear_surface_logical_override(original.surface);
        }
    }
    if virtual_intent
        && virtual_preflight_eligible
        && let Some(capabilities) = logical_capabilities
    {
        modified.image_extent =
            initial_physical_extent(original.image_extent, surface_extent(capabilities), true);
    }
    // Probe X11 only.  The physical swapchain is intentionally created at
    // the extent accepted by the surface now; promotion happens only after
    // the logical token and images have been installed below.
    if old_logical.is_some() && !virtual_preflight_eligible {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    if let Some(target) = initial_target {
        eprintln!(
            "TuxScaling evidence event=borderless_target extent={}x{} origin={}+{}",
            target.monitor.rect.width,
            target.monitor.rect.height,
            target.monitor.rect.x,
            target.monitor.rect.y,
        );
    }
    if let Some(old) = old_logical.as_ref() {
        let Some(device_state) = state.as_ref() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(capabilities) =
            (unsafe { downstream_surface_capabilities(device_state, original.surface) })
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(_plan) = virtualization_plan.as_ref() else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let (physical_extent, physical_old_swapchain) = match translate_recreation_create_info(
            original,
            &old.contract,
            surface_extent(capabilities),
        ) {
            Ok(info) => info,
            Err(error) => return error,
        };
        modified.image_extent = physical_extent;
        modified.old_swapchain = physical_old_swapchain;
        eprintln!(
            "TuxScaling evidence event=logical_recreation_translated old_logical=0x{:x} old_physical=0x{:x} downstream_extent={}x{}",
            old.contract.handle().as_raw(),
            old.contract.generation().handle().as_raw(),
            physical_extent.width,
            physical_extent.height,
        );
    }
    let physical_extent = modified.image_extent;
    let physical_old_swapchain = modified.old_swapchain;
    let mut result = unsafe {
        create_physical_swapchain(
            create_swapchain,
            device,
            allocation_callbacks,
            virtualization_plan.as_ref().map(|plan| &plan.template),
            &modified,
            PhysicalCreateTarget {
                surface: original.surface,
                extent: physical_extent,
                old_swapchain: physical_old_swapchain,
            },
            swapchain,
        )
    };
    if result != vk::Result::SUCCESS
        && capture_enabled
        && modified.image_usage != original.image_usage
    {
        capture_enabled = false;
        modified.image_usage = original.image_usage;
        let retry_plan = if virtualization_plan.is_some() {
            state.as_ref().and_then(|state| {
                match preflight_swapchain_virtualization(
                    &state.wsi,
                    &modified,
                    logical_capabilities,
                    format_features,
                ) {
                    Ok(plan) => Some(plan),
                    Err(error) => {
                        eprintln!(
                            "TuxScaling evidence event=virtualization_preflight result=direct reason={}",
                            error.reason()
                        );
                        None
                    }
                }
            })
        } else {
            None
        };
        if virtualization_plan.is_some() && retry_plan.is_none() {
            result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
        } else if result != vk::Result::ERROR_FEATURE_NOT_PRESENT {
            result = unsafe {
                create_physical_swapchain(
                    create_swapchain,
                    device,
                    allocation_callbacks,
                    retry_plan.as_ref().map(|plan| &plan.template),
                    &modified,
                    PhysicalCreateTarget {
                        surface: original.surface,
                        extent: physical_extent,
                        old_swapchain: physical_old_swapchain,
                    },
                    swapchain,
                )
            };
            if result == vk::Result::SUCCESS {
                virtualization_plan = retry_plan;
            }
        }
    }
    if result != vk::Result::SUCCESS {
        super::lifetime::restore_surface_window(original.surface);
        return result;
    }
    let post_result = catch_unwind(AssertUnwindSafe(|| {
        let Some(device_state) = devices()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&device)
            .cloned()
        else {
            return result;
        };
        if !modified
            .image_usage
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || original.image_array_layers != 1
            || !supported_swapchain_flags(original.flags)
        {
            return result;
        }
        let handle = unsafe { *swapchain };
        eprintln!(
            "TuxScaling swapchain: format={:?} color_space={:?} flags={:?} capture={capture_enabled} usage={:?}",
            original.image_format, original.image_color_space, original.flags, modified.image_usage
        );
        let info = physical_swapchain_info(&modified);
        let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
        let output_images = match unsafe { loader.get_swapchain_images(handle) } {
            Ok(images) => images,
            Err(_) => return result,
        };
        let mut virtual_eligible = virtual_intent
            && virtual_preflight_eligible
            && virtual_swapchain_supported(original)
            && preallocated_virtual_images.is_some();
        // Allocate both logical identity and logical images before the X11
        // promotion call.  The token is installed on the first create path,
        // so the application never gets a physical identity during pending.
        let logical_handle = if virtual_eligible {
            match allocate_logical_swapchain(handle) {
                Some(logical) => logical,
                None => {
                    virtual_eligible = false;
                    handle
                }
            }
        } else {
            handle
        };
        let virtual_images = if virtual_eligible {
            preallocated_virtual_images.take()
        } else {
            None
        };
        let game_images = virtual_images.as_ref().map_or_else(
            || output_images.clone(),
            |images| images.iter().map(|image| image.handle).collect(),
        );
        let game_extent = if virtual_images.is_some() {
            original.image_extent
        } else {
            info.extent
        };
        let window = surfaces()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&original.surface)
            .map(|surface| surface.window)
            .filter(|window| *window != super::surface::PENDING_WIN32_WINDOW);
        let persisted_negotiation = surface_snapshot.map(|surface| surface.negotiation);
        let monitor = initial_target.map(|target| {
            let rect = target.monitor.rect;
            [rect.x, rect.y, rect.width as i32, rect.height as i32]
        });
        let overlay = unsafe {
            OverlaySwapchain::new(
                &device_state.instance,
                device_state.physical_device,
                &device_state.device,
                SwapchainRuntimeCreateInfo {
                    info,
                    images: SwapchainImages {
                        game_images,
                        game_extent,
                        output_images: output_images.clone(),
                    },
                    capture_enabled,
                    vulkan_api_version: device_state.vulkan_api_version,
                    window,
                    fullscreen: window
                        .and_then(|window| {
                            tuxscaling_display::X11Display::connect()
                                .ok()?
                                .target_for_window(window)
                                .ok()
                        })
                        .is_some_and(|target| target.is_fullscreen()),
                    monitor,
                    temporal_enabled: temporal_enabled_for_logical_creation(
                        virtual_eligible,
                        old_logical
                            .as_ref()
                            .map(|old| old.negotiation.public_state()),
                    ),
                },
                device_state.set_loader_data,
            )
        };
        let Ok(overlay) = overlay else {
            eprintln!("TuxScaling: overlay disabled for swapchain");
            return result;
        };
        let template = virtual_eligible.then(|| {
            virtualization_plan
                .as_ref()
                .expect("virtual preflight owns a valid template")
                .template
                .clone()
        });
        let mut negotiation = if let Some(old) = old_logical.as_ref() {
            old.negotiation
        } else if virtual_eligible {
            let mut negotiation =
                persisted_negotiation.unwrap_or_else(PresentationNegotiation::direct);
            if let Some(target) = initial_target
                && negotiation.public_state() == PresentationState::Direct
            {
                let _ = negotiation.request_borderless(target.monitor.rect, Instant::now());
            }
            negotiation
        } else {
            persisted_negotiation.unwrap_or_else(PresentationNegotiation::direct)
        };
        let prior_logical_capabilities = surfaces()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&original.surface)
            .and_then(|surface| surface.logical_capabilities);
        let advertised_logical_capabilities = if virtual_eligible {
            logical_capabilities_for_extent(
                prior_logical_capabilities,
                logical_capabilities,
                original.image_extent,
            )
        } else {
            logical_capabilities
        };
        let contract = virtual_images.as_ref().and_then(|images| {
            LogicalSwapchainContract::new(
                logical_handle,
                images.iter().map(|image| image.handle).collect(),
                original.image_extent,
                PhysicalGeneration::new(0, handle, info.extent, output_images.len()),
                negotiation.public_state(),
            )
            .ok()
        });
        let state = Arc::new(Mutex::new(SwapchainState {
            device,
            surface: original.surface,
            logical_handle,
            physical_handle: handle,
            mapping: virtual_images
                .as_ref()
                .map(|images| Mapping::new(0, images.len())),
            negotiation,
            overlay: Some(overlay),
            virtual_images,
            physical_images: output_images.clone(),
            retired_physical_generations: Vec::new(),
            maintenance_overlay_reported: false,
            maintenance_present_reported: false,
            maintenance_release_reported: false,
            generation: 0,
            template,
            contract,
            hdr_metadata: None,
            present_ids: crate::hooks::present_id::PresentIdHistory::new(),
            lifecycle: crate::recovery::ReconfigurationLifecycle::new(),
        }));
        swapchains()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(logical_handle, state);
        if old_logical.is_some() {
            // The old application token remains in the table until the
            // application's destroy call can retire its physical resources,
            // but it is no longer a valid input to any translated command.
            retire_swapchain(original.old_swapchain);
        }
        eprintln!(
            "TuxScaling evidence event=logical_swapchain_created logical_handle=0x{:x} physical_handle=0x{:x} logical={}x{} physical={}x{} virtual={} mutable_format={} view_formats={} negotiation={}",
            logical_handle.as_raw(),
            handle.as_raw(),
            original.image_extent.width,
            original.image_extent.height,
            info.extent.width,
            info.extent.height,
            u8::from(virtual_eligible),
            u8::from(
                virtual_eligible
                    && virtualization_plan.as_ref().is_some_and(|plan| {
                        plan.image_flags
                            .contains(vk::ImageCreateFlags::MUTABLE_FORMAT)
                    }),
            ),
            if virtual_eligible {
                virtualization_plan
                    .as_ref()
                    .map_or(0, |plan| plan.view_formats.len())
            } else {
                0
            },
            if virtual_eligible {
                "negotiating"
            } else {
                "direct"
            },
        );
        if virtual_eligible {
            if let Some(target) = initial_target {
                let display = match tuxscaling_display::X11Display::connect() {
                    Ok(display) => display,
                    Err(_) => {
                        if let Some(state) = swapchains()
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .remove(&logical_handle)
                            && let Ok(state) = Arc::try_unwrap(state)
                        {
                            let overlay = state
                                .into_inner()
                                .unwrap_or_else(|error| error.into_inner())
                                .overlay;
                            if let Some(overlay) = overlay {
                                unsafe { overlay.destroy(&device_state.device) };
                            }
                        }
                        return result;
                    }
                };
                let existing_lease = surface_snapshot
                    .and_then(|surface| surface.borderless_lease)
                    .filter(|lease| {
                        lease.window == target.window && lease.monitor == target.monitor
                    });
                let lease = match existing_lease
                    .map(Ok)
                    .unwrap_or_else(|| display.promote_borderless(target.window))
                {
                    Ok(lease) => lease,
                    Err(_) => {
                        if let Some(state) = swapchains()
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .remove(&logical_handle)
                            && let Ok(state) = Arc::try_unwrap(state)
                        {
                            let overlay = state
                                .into_inner()
                                .unwrap_or_else(|error| error.into_inner())
                                .overlay;
                            if let Some(overlay) = overlay {
                                unsafe { overlay.destroy(&device_state.device) };
                            }
                        }
                        return result;
                    }
                };
                let state = swapchains()
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .get(&logical_handle)
                    .cloned()
                    .expect("initial virtual state is installed before promotion");
                negotiation = {
                    let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
                    let _ = state.negotiation.borderless_requested(Instant::now());
                    state.negotiation
                };
                {
                    let mut surfaces = surfaces().lock().unwrap_or_else(|error| error.into_inner());
                    if let Some(surface) = surfaces.get_mut(&original.surface) {
                        surface.logical_extent = Some(original.image_extent);
                        surface.logical_capabilities = advertised_logical_capabilities;
                        surface.negotiation = negotiation;
                        surface.borderless_lease = Some(lease);
                    }
                }
                unsafe { *swapchain = logical_handle };
                publish_negotiation(original.surface, negotiation);
                eprintln!(
                    "TuxScaling evidence event=virtual_swapchain_negotiating logical_handle=0x{:x} logical={}x{} physical={}x{}",
                    logical_handle.as_raw(),
                    original.image_extent.width,
                    original.image_extent.height,
                    info.extent.width,
                    info.extent.height,
                );
            } else {
                {
                    let mut surfaces = surfaces().lock().unwrap_or_else(|error| error.into_inner());
                    if let Some(surface) = surfaces.get_mut(&original.surface) {
                        surface.logical_extent = Some(original.image_extent);
                        surface.logical_capabilities = advertised_logical_capabilities;
                        surface.negotiation = negotiation;
                    }
                }
                unsafe { *swapchain = logical_handle };
                publish_negotiation(original.surface, negotiation);
                eprintln!(
                    "TuxScaling evidence event=logical_swapchain_recreated logical_handle=0x{:x} logical={}x{} physical={}x{}",
                    logical_handle.as_raw(),
                    original.image_extent.width,
                    original.image_extent.height,
                    info.extent.width,
                    info.extent.height,
                );
            }
        }
        result
    }));
    if post_result.is_err() {
        super::lifetime::restore_surface_window(original.surface);
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    post_result.unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}
pub(super) unsafe extern "system" fn create_swapchain_khr(
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

#[cfg(test)]
mod tests {
    use super::{
        initial_physical_extent, logical_image_count, native_generation_failure_can_restore,
        preflight_swapchain_virtualization, temporal_enabled_for_logical_creation,
        translate_old_swapchain, translate_recreation_create_info, virtual_swapchain_supported,
    };
    use crate::hooks::swapchain_create::SwapchainCompatibilityError;
    use crate::recovery::{
        LogicalSwapchainContract, PhysicalGeneration, ReconfigurationLifecycle,
        ReconfigurationTicket,
    };
    use crate::state::SwapchainTemplate;
    use crate::state::retire_swapchain;
    use ash::vk;
    use ash::vk::Handle;
    use std::time::{Duration, Instant};
    use tuxscaling_display::{
        Extent, PresentationNegotiation, PresentationState, Rect, SurfaceExtent,
    };

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    fn virtualization_capabilities() -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            current_extent: extent(3440, 1440),
            min_image_extent: extent(1, 1),
            max_image_extent: extent(8192, 8192),
            supported_usage_flags: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        }
    }

    fn with_mutable_create_info<R>(
        formats: &[vk::Format],
        invoke: impl FnOnce(&vk::SwapchainCreateInfoKHR<'_>) -> R,
    ) -> R {
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(formats);
        let info = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
            .min_image_count(2)
            .image_format(formats[0])
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent(1280, 720))
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut list);
        invoke(&info)
    }

    fn compatible_wsi() -> crate::hooks::wsi_compatibility::DeviceWsiCapabilities {
        crate::hooks::wsi_compatibility::DeviceWsiCapabilities {
            mutable_format: true,
            ..Default::default()
        }
    }

    #[test]
    fn broad_unrelated_extensions_and_valid_mutable_chain_publish_a_plan() {
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_KHR_swapchain_mutable_format".as_ptr(),
            c"VK_EXT_memory_budget".as_ptr(),
        ];
        let device_info = vk::DeviceCreateInfo::default().enabled_extension_names(&names);
        let wsi = unsafe {
            crate::hooks::wsi_compatibility::DeviceWsiCapabilities::from_create_info(
                &device_info,
                vk::API_VERSION_1_2,
            )
        };
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let plan = with_mutable_create_info(&formats, |info| {
            preflight_swapchain_virtualization(
                &wsi,
                info,
                Some(virtualization_capabilities()),
                vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST,
            )
        })
        .unwrap();

        assert_eq!(plan.view_formats, formats);
        assert_eq!(plan.image_flags, vk::ImageCreateFlags::MUTABLE_FORMAT);
        assert_eq!(plan.template.image_format, formats[0]);
    }

    #[test]
    fn incompatible_wsi_is_rejected_before_logical_reservation() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let wsi = crate::hooks::wsi_compatibility::DeviceWsiCapabilities {
            incompatible: Some(crate::hooks::wsi_compatibility::IncompatibleWsiExtension {
                name: b"VK_KHR_shared_presentable_image".to_vec(),
                reason: "shared presentable image semantics are not translated",
            }),
            ..compatible_wsi()
        };

        assert!(matches!(
            with_mutable_create_info(&formats, |info| {
                preflight_swapchain_virtualization(
                    &wsi,
                    info,
                    Some(virtualization_capabilities()),
                    vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST,
                )
            }),
            Err(SwapchainCompatibilityError::IncompatibleWsiExtension { name, reason })
                if name == b"VK_KHR_shared_presentable_image" && reason
                    == "shared presentable image semantics are not translated"
        ));
    }

    #[test]
    fn initial_and_replacement_generations_share_the_owned_contract() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let plan = with_mutable_create_info(&formats, |info| {
            preflight_swapchain_virtualization(
                &compatible_wsi(),
                info,
                Some(virtualization_capabilities()),
                vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST,
            )
        })
        .unwrap();
        let mut observed = Vec::new();
        for (extent, old_swapchain) in [
            (extent(1280, 720), vk::SwapchainKHR::null()),
            (extent(3440, 1440), vk::SwapchainKHR::from_raw(41)),
        ] {
            plan.template.with_create_info(
                vk::SurfaceKHR::from_raw(7),
                extent,
                old_swapchain,
                |info| {
                    observed.push((
                        info.flags,
                        info.image_format,
                        info.image_color_space,
                        info.image_usage,
                        info.image_sharing_mode,
                        info.p_next,
                        info.image_extent,
                        info.old_swapchain,
                    ));
                },
            );
        }

        assert_eq!(observed[0].0, observed[1].0);
        assert_eq!(observed[0].1, observed[1].1);
        assert_eq!(observed[0].2, observed[1].2);
        assert_eq!(observed[0].3, observed[1].3);
        assert_eq!(observed[0].4, observed[1].4);
        assert_ne!(observed[0].5, std::ptr::null());
        assert_ne!(observed[1].5, std::ptr::null());
        assert_eq!(observed[0].6, extent(1280, 720));
        assert_eq!(observed[1].6, extent(3440, 1440));
        assert_eq!(observed[0].7, vk::SwapchainKHR::null());
        assert_eq!(observed[1].7, vk::SwapchainKHR::from_raw(41));
    }

    #[test]
    fn failed_image_contract_preflight_returns_before_a_logical_handle_exists() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let result = with_mutable_create_info(&formats, |info| {
            preflight_swapchain_virtualization(
                &compatible_wsi(),
                info,
                Some(virtualization_capabilities()),
                vk::FormatFeatureFlags::empty(),
            )
        });

        assert!(matches!(
            result,
            Err(SwapchainCompatibilityError::UnsupportedImageContract(
                vk::Result::ERROR_FEATURE_NOT_PRESENT,
            ))
        ));
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mutable_format_virtual_swapchains() {
        let info = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT);

        assert!(!virtual_swapchain_supported(&info));
    }

    #[test]
    fn rejects_protected_swapchains() {
        let info =
            vk::SwapchainCreateInfoKHR::default().flags(vk::SwapchainCreateFlagsKHR::PROTECTED);

        assert!(!virtual_swapchain_supported(&info));
    }

    #[test]
    fn rejects_device_group_swapchains() {
        let mut group = vk::DeviceGroupSwapchainCreateInfoKHR::default();
        let info = vk::SwapchainCreateInfoKHR::default().push_next(&mut group);

        assert!(!virtual_swapchain_supported(&info));
    }

    #[test]
    fn accepts_exclusive_single_layer_swapchains() {
        let info = vk::SwapchainCreateInfoKHR::default()
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE);

        assert!(virtual_swapchain_supported(&info));
    }

    #[test]
    fn layer_gate_requires_exact_observed_native_output_before_recreation() {
        let target = Rect::new(-1920, 0, 1920, 1080);
        let now = Instant::now();
        let mut negotiation = PresentationNegotiation::direct();

        assert!(negotiation.request_borderless(target, now));
        assert!(negotiation.borderless_requested(now));
        assert!(!negotiation.observe(
            Rect::new(-1920, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            now,
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            target,
            false,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            Rect::new(-1920, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            now + Duration::from_millis(2),
        ));
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(3),
        ));
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(4),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(5),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn retired_old_swapchain_is_rejected_instead_of_forwarded() {
        let token = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0088);
        retire_swapchain(token);

        assert_eq!(
            translate_old_swapchain(token),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
        );
    }

    #[test]
    fn creation_keeps_logical_image_count_independent_from_physical_count() {
        assert_eq!(logical_image_count(2, 3), 2);
        assert_eq!(logical_image_count(4, 2), 4);
        assert_eq!(logical_image_count(0, 3), 3);
    }

    #[test]
    fn initial_virtual_generation_uses_fixed_surface_extent_for_physical_swapchain() {
        let requested = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        let native = Extent::new(3440, 1440);

        assert_eq!(
            initial_physical_extent(requested, SurfaceExtent::fixed(native), true),
            vk::Extent2D {
                width: 3440,
                height: 1440,
            }
        );
        assert_eq!(
            initial_physical_extent(requested, SurfaceExtent::fixed(native), false),
            requested
        );
    }

    #[test]
    fn stale_logical_override_is_dropped_for_truthful_direct_fallback() {
        use crate::state::surfaces;

        let surface = vk::SurfaceKHR::from_raw(0xad07);
        surfaces().lock().unwrap().insert(
            surface,
            crate::state::X11Surface {
                window: 7,
                logical_extent: Some(extent(1280, 720)),
                logical_capabilities: Some(vk::SurfaceCapabilitiesKHR::default()),
                borderless_lease: None,
                negotiation: PresentationNegotiation::direct(),
            },
        );

        super::clear_surface_logical_override(surface);

        let state = surfaces().lock().unwrap().remove(&surface).unwrap();
        assert!(state.logical_extent.is_none());
        assert!(state.logical_capabilities.is_none());
        assert_eq!(state.negotiation.public_state(), PresentationState::Direct);
    }

    #[test]
    fn native_generation_failure_restores_unless_destroy_was_requested() {
        assert!(native_generation_failure_can_restore(false));
        assert!(!native_generation_failure_can_restore(true));
    }

    #[test]
    fn recreation_hook_translates_physical_extent_and_preserves_new_logical_contract() {
        let surface = vk::SurfaceKHR::from_raw(0x9001);
        let old_logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0101);
        let new_logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0102);
        let logical_images = vec![vk::Image::from_raw(0x101), vk::Image::from_raw(0x102)];
        let new_logical_images = vec![vk::Image::from_raw(0x201), vk::Image::from_raw(0x202)];
        let old_physical = vk::SwapchainKHR::from_raw(0x301);
        let new_physical = vk::SwapchainKHR::from_raw(0x302);
        let logical_extent = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        let native_extent = tuxscaling_display::Extent::new(3440, 1440);
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(logical_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old_logical);
        let old_contract = LogicalSwapchainContract::new(
            old_logical,
            logical_images,
            logical_extent,
            PhysicalGeneration::new(0, old_physical, logical_extent, 2),
            PresentationState::Virtualized,
        )
        .unwrap();
        let _template = SwapchainTemplate::from_create_info(&info).unwrap();

        let (recorded_extent, recorded_old_swapchain) = translate_recreation_create_info(
            &info,
            &old_contract,
            SurfaceExtent::fixed(native_extent),
        )
        .unwrap();

        let downstream_recorder = [(recorded_extent, recorded_old_swapchain)];
        let (recorded_extent, recorded_old_swapchain) = downstream_recorder[0];
        let new_contract = LogicalSwapchainContract::new(
            new_logical,
            new_logical_images.clone(),
            logical_extent,
            PhysicalGeneration::new(1, new_physical, recorded_extent, 3),
            PresentationState::Virtualized,
        )
        .unwrap();

        assert_eq!(
            recorded_extent,
            vk::Extent2D {
                width: 3440,
                height: 1440,
            }
        );
        assert_eq!(recorded_old_swapchain, vk::SwapchainKHR::null());
        assert_eq!(new_contract.handle(), new_logical);
        assert_eq!(new_contract.logical_images(), new_logical_images.as_slice());
        assert_eq!(new_contract.game_extent(), logical_extent);
    }

    #[test]
    fn recreation_accepts_a_changed_logical_extent() {
        let surface = vk::SurfaceKHR::from_raw(0x9002);
        let old_logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0111);
        let old_physical = vk::SwapchainKHR::from_raw(0x311);
        let old_extent = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        let new_extent = vk::Extent2D {
            width: 960,
            height: 540,
        };
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(new_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old_logical);
        let old_contract = LogicalSwapchainContract::new(
            old_logical,
            vec![vk::Image::from_raw(0x111)],
            old_extent,
            PhysicalGeneration::new(0, old_physical, old_extent, 2),
            PresentationState::Virtualized,
        )
        .unwrap();
        let _template = SwapchainTemplate::from_create_info(&info).unwrap();

        let (physical_extent, physical_old_swapchain) = translate_recreation_create_info(
            &info,
            &old_contract,
            SurfaceExtent::fixed(Extent::new(3440, 1440)),
        )
        .unwrap();

        assert_eq!(physical_extent.width, 3440);
        assert_eq!(physical_extent.height, 1440);
        assert_eq!(physical_old_swapchain, vk::SwapchainKHR::null());
    }

    #[test]
    fn active_logical_recreation_keeps_temporal_processing_enabled() {
        assert!(!temporal_enabled_for_logical_creation(
            true,
            Some(PresentationState::Negotiating)
        ));
        assert!(temporal_enabled_for_logical_creation(
            true,
            Some(PresentationState::Virtualized)
        ));
        assert!(temporal_enabled_for_logical_creation(false, None));
    }

    #[test]
    fn native_generation_hook_reentrant_downstream_callback_can_query_layer_registry() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0201);
        let surface = vk::SurfaceKHR::from_raw(0x9201);
        let ticket = ReconfigurationTicket::new(logical, surface, 1);
        let registry = std::sync::Arc::new(std::sync::Mutex::new(ReconfigurationLifecycle::new()));
        assert!(registry.lock().unwrap().begin(ticket));

        let callback_registry = registry.clone();
        let downstream_callback =
            std::thread::spawn(move || callback_registry.lock().unwrap().blocks_frame_operations());

        assert!(downstream_callback.join().unwrap());
        assert!(registry.lock().unwrap().abort(ticket));
    }
}
