use super::*;
use crate::mapping::{LogicalSwapchainHandle, Mapping, OldSwapchain};
use crate::recovery::{LogicalSwapchainContract, PhysicalGeneration};
use ash::vk::Handle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tuxscaling_display::{Extent, PresentationNegotiation, SurfaceExtent};

fn virtual_swapchain_supported(info: &vk::SwapchainCreateInfoKHR<'_>) -> bool {
    if !info.flags.is_empty()
        || info.image_array_layers != 1
        || (info.image_sharing_mode == vk::SharingMode::CONCURRENT
            && (info.queue_family_index_count < 2 || info.p_queue_family_indices.is_null()))
    {
        return false;
    }
    // The physical-generation template deliberately owns only the core
    // create parameters.  Unknown pNext chains therefore fail open instead
    // of being silently dropped during an internal generation replacement.
    info.p_next.is_null()
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

unsafe fn allocate_logical_images(
    state: &DeviceState,
    info: &vk::SwapchainCreateInfoKHR<'_>,
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
    (0..count)
        .map(|_| unsafe {
            tuxscaling_vulkan::Image::with_sharing(
                &state.device,
                &memory,
                info.image_extent,
                info.image_format,
                usage,
                queue_families,
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
    for swapchain in states.values() {
        if let Ok(mut swapchain) = swapchain.lock()
            && swapchain.surface == surface_handle
        {
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

#[derive(Clone, Copy)]
struct InitialOutputTarget {
    window: u64,
    monitor: tuxscaling_display::Monitor,
}

fn initial_output_target(
    surface: vk::SurfaceKHR,
    game_extent: vk::Extent2D,
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
        return None;
    }
    let window = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .copied()?
        .window;
    let display = tuxscaling_display::X11Display::connect().ok()?;
    let target = display.target_for_window(window).ok()?;
    let target_extent = target.monitor.rect.extent();
    (target_extent.is_valid()
        && target_extent != Extent::new(game_extent.width, game_extent.height))
    .then_some(InitialOutputTarget {
        window,
        monitor: target.monitor,
    })
}

pub(super) unsafe fn observe_surface_negotiation(
    device_state: &DeviceState,
    surface: vk::SurfaceKHR,
    now: Instant,
) -> tuxscaling_display::PresentationState {
    let Some(snapshot) = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .copied()
    else {
        return tuxscaling_display::PresentationState::Direct;
    };
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

/// Replaces only the downstream WSI generation after exact X11 and Vulkan
/// confirmation.  The logical handle, images, extent, and mapping stay in
/// the state entry throughout this operation.
pub(super) unsafe fn publish_native_generation(
    device_state: &DeviceState,
    surface: vk::SurfaceKHR,
) -> bool {
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .find(|state| {
            state
                .lock()
                .ok()
                .is_some_and(|state| state.surface == surface && state.mapping.is_some())
        })
        .cloned();
    let Some(state) = state else {
        return false;
    };
    let display = match tuxscaling_display::X11Display::connect() {
        Ok(display) => display,
        Err(_) => {
            super::lifetime::restore_surface_window(surface);
            return false;
        }
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
        Err(_) => {
            super::lifetime::restore_surface_window(surface);
            return false;
        }
    };
    let capabilities = match unsafe { downstream_surface_capabilities(device_state, surface) } {
        Some(capabilities) => capabilities,
        None => {
            super::lifetime::restore_surface_window(surface);
            return false;
        }
    };
    let surface_extent = surface_extent(capabilities);
    let mut guard = state.lock().unwrap_or_else(|error| error.into_inner());
    if !guard.negotiation.native_observation_is_current(
        observed.rect,
        observed.fullscreen,
        surface_extent,
        Instant::now(),
    ) {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    if !guard
        .mapping
        .as_ref()
        .is_some_and(crate::mapping::Mapping::is_idle)
    {
        return false;
    }
    let Some(template) = guard.template.as_ref() else {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    };
    let target_extent = vk::Extent2D {
        width: observed.rect.width,
        height: observed.rect.height,
    };
    if !surface_extent_accepts(surface_extent, target_extent) {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    let Some(proc) =
        (unsafe { device_downstream(device_state.device.handle(), c"vkCreateSwapchainKHR") })
    else {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    };
    let create: vk::PFN_vkCreateSwapchainKHR = unsafe { std::mem::transmute(proc) };
    let old_physical = guard.physical_handle;
    let contract_generation = guard.contract.as_ref().map(|contract| {
        debug_assert!(contract.game_extent().width > 0 && contract.game_extent().height > 0);
        contract.generation()
    });
    let next_generation = contract_generation
        .map_or(guard.generation, |generation| generation.id())
        .saturating_add(1);
    let mut published_negotiation = guard.negotiation;
    if !published_negotiation.output_recreated(
        observed.rect,
        observed.fullscreen,
        surface_extent,
        Instant::now(),
    ) {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    let create_info = template.create_info(surface, target_extent, old_physical);
    let mut new_physical = vk::SwapchainKHR::null();
    let result = unsafe {
        create(
            device_state.device.handle(),
            &create_info,
            std::ptr::null(),
            &mut new_physical,
        )
    };
    if result != vk::Result::SUCCESS || new_physical == vk::SwapchainKHR::null() {
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
    let new_images = match unsafe { loader.get_swapchain_images(new_physical) } {
        Ok(images) => images,
        Err(_) => {
            unsafe { loader.destroy_swapchain(new_physical, None) };
            drop(guard);
            super::lifetime::restore_surface_window(surface);
            return false;
        }
    };
    let info = SwapchainInfo {
        format: template.image_format,
        color_space: template.image_color_space,
        extent: target_extent,
    };
    if unsafe { guard.overlay.reconfigure_output(info, new_images.clone()) }.is_err() {
        unsafe { loader.destroy_swapchain(new_physical, None) };
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    let Some(mapping) = guard.mapping.as_mut() else {
        unsafe { loader.destroy_swapchain(new_physical, None) };
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    };
    if !mapping.replace_generation(next_generation) {
        unsafe { loader.destroy_swapchain(new_physical, None) };
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    if let Some(contract) = guard.contract.as_mut()
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
            .is_err()
    {
        unsafe { loader.destroy_swapchain(new_physical, None) };
        drop(guard);
        super::lifetime::restore_surface_window(surface);
        return false;
    }
    guard.negotiation = published_negotiation;
    if let Some(contract) = guard.contract.as_mut() {
        contract.set_state(published_negotiation.public_state());
    }
    guard.physical_handle = new_physical;
    guard.physical_images = new_images;
    guard.generation = next_generation;
    drop(guard);
    publish_negotiation(surface, published_negotiation);
    unsafe { loader.destroy_swapchain(old_physical, None) };
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

unsafe fn has_present_fences(mut next: *const c_void) -> bool {
    while !next.is_null() {
        let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
        if header.s_type == vk::StructureType::PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT
        {
            return unsafe {
                (*next.cast::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'_>>())
                    .swapchain_maintenance1
                    != 0
            };
        }
        next = header.p_next.cast();
    }
    false
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
        devices().lock().unwrap_or_else(|e| e.into_inner()).insert(
            unsafe { *device },
            DeviceState {
                overlay_supported: !unsafe { has_present_fences((*create_info).p_next) },
                virtualization_extension_safe: super::virtualization_extension_safe(unsafe {
                    &*create_info
                }),
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
    let mut virtual_preflight_eligible = state.as_ref().is_some_and(|state| {
        state.overlay_supported
            && state.virtualization_extension_safe
            && virtual_swapchain_supported(original)
            && tuxscaling_capture::supported_format(
                original.image_format,
                original.image_color_space,
            )
            && original.image_array_layers == 1
            && original.flags.is_empty()
            && !matches!(
                original.present_mode,
                vk::PresentModeKHR::SHARED_DEMAND_REFRESH
                    | vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH
            )
            && logical_capabilities.is_some_and(|caps| {
                caps.supported_usage_flags.contains(needed)
                    && unsafe {
                        state.instance.get_physical_device_format_properties(
                            state.physical_device,
                            original.image_format,
                        )
                    }
                    .optimal_tiling_features
                    .contains(
                        vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST,
                    )
            })
    });
    let mut preallocated_virtual_images = None;
    if virtual_preflight_eligible {
        if let Some(count) = preflight_logical_image_count(original.min_image_count) {
            match unsafe {
                allocate_logical_images(
                    state.as_ref().expect("eligible state must exist"),
                    original,
                    count,
                )
            } {
                Ok(images) => preallocated_virtual_images = Some(images),
                Err(error) => {
                    eprintln!("TuxScaling: logical image preflight failed: {error:?}");
                    virtual_preflight_eligible = false;
                }
            }
        } else {
            virtual_preflight_eligible = false;
        }
    }
    if !virtual_preflight_eligible {
        super::lifetime::restore_surface_window(original.surface);
    }
    // Probe X11 only.  The physical swapchain is intentionally created at
    // the extent accepted by the surface now; promotion happens only after
    // the logical token and images have been installed below.
    let initial_target = initial_output_target(
        original.surface,
        original.image_extent,
        virtual_preflight_eligible,
    );
    let mut capture_enabled = false;
    if let Some(state) = &state
        && state.overlay_supported
        && tuxscaling_capture::supported_format(original.image_format, original.image_color_space)
        && original.image_array_layers == 1
        && original.flags.is_empty()
        && !matches!(
            original.present_mode,
            vk::PresentModeKHR::SHARED_DEMAND_REFRESH
                | vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH
        )
        && let Some(proc) = unsafe {
            downstream(
                state.instance.handle(),
                c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            )
        }
    {
        let get: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
            unsafe { std::mem::transmute(proc) };
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        if unsafe { get(state.physical_device, original.surface, &mut caps) } == vk::Result::SUCCESS
            && caps.supported_usage_flags.contains(needed)
        {
            let features = unsafe {
                state.instance.get_physical_device_format_properties(
                    state.physical_device,
                    original.image_format,
                )
            }
            .optimal_tiling_features;
            if features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::BLIT_DST)
            {
                modified.image_usage |= needed;
                capture_enabled = true;
            }
        }
    }
    let mut result =
        unsafe { create_swapchain(device, &modified, allocation_callbacks, swapchain) };
    if result != vk::Result::SUCCESS
        && capture_enabled
        && modified.image_usage != original.image_usage
    {
        capture_enabled = false;
        modified = driver_original;
        result =
            unsafe { create_swapchain(device, &driver_original, allocation_callbacks, swapchain) };
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
        if !device_state.overlay_supported {
            eprintln!("TuxScaling bypass: application-managed presentation fences");
            return result;
        }
        if !modified
            .image_usage
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            || original.image_array_layers != 1
            || !original.flags.is_empty()
        {
            return result;
        }
        let handle = unsafe { *swapchain };
        eprintln!(
            "TuxScaling swapchain: format={:?} color_space={:?} flags={:?} capture={capture_enabled} usage={:?}",
            original.image_format, original.image_color_space, original.flags, modified.image_usage
        );
        let info = SwapchainInfo {
            format: modified.image_format,
            color_space: modified.image_color_space,
            extent: original.image_extent,
        };
        let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
        let output_images = match unsafe { loader.get_swapchain_images(handle) } {
            Ok(images) => images,
            Err(_) => return result,
        };
        let mut virtual_eligible = initial_target.is_some()
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
            .map(|surface| surface.window);
        let persisted_negotiation = surfaces()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&original.surface)
            .map(|surface| surface.negotiation);
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
                    temporal_enabled: !virtual_eligible,
                },
                device_state.set_loader_data,
            )
        };
        let Ok(overlay) = overlay else {
            eprintln!("TuxScaling: overlay disabled for swapchain");
            return result;
        };
        let template =
            virtual_eligible.then(|| crate::state::SwapchainTemplate::from_create_info(&modified));
        let mut negotiation = if virtual_eligible {
            initial_target.map_or_else(PresentationNegotiation::direct, |_| {
                let mut negotiation = PresentationNegotiation::direct();
                let _ = negotiation
                    .request_borderless(initial_target.unwrap().monitor.rect, Instant::now());
                negotiation
            })
        } else {
            persisted_negotiation.unwrap_or_else(PresentationNegotiation::direct)
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
            overlay,
            virtual_images,
            physical_images: output_images.clone(),
            generation: 0,
            template,
            contract,
        }));
        swapchains()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(logical_handle, state);
        if virtual_eligible {
            let target = initial_target.expect("virtual target checked above");
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
                        unsafe { overlay.destroy(&device_state.device) };
                    }
                    return result;
                }
            };
            let lease = match display.promote_borderless(target.window) {
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
                        unsafe { overlay.destroy(&device_state.device) };
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
                    surface.logical_capabilities = logical_capabilities;
                    surface.negotiation = negotiation;
                    surface.borderless_lease = Some(lease);
                }
            }
            unsafe { *swapchain = logical_handle };
            publish_negotiation(original.surface, negotiation);
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
    use super::{logical_image_count, translate_old_swapchain, virtual_swapchain_supported};
    use crate::state::retire_swapchain;
    use ash::vk;
    use ash::vk::Handle;
    use std::time::{Duration, Instant};
    use tuxscaling_display::{
        Extent, PresentationNegotiation, PresentationState, Rect, SurfaceExtent,
    };

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
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(2),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(3),
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
}
