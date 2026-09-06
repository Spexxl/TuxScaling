use super::*;

#[derive(Clone, Copy)]
struct ResizedWindow {
    window: u64,
    original: tuxscaling_display::Rect,
}

fn virtual_swapchain_supported(info: &vk::SwapchainCreateInfoKHR<'_>) -> bool {
    if !info.flags.is_empty()
        || info.image_array_layers != 1
        || info.image_sharing_mode != vk::SharingMode::EXCLUSIVE
        || info.queue_family_index_count != 0
    {
        return false;
    }
    let mut next = info.p_next.cast::<vk::BaseInStructure<'_>>();
    while !next.is_null() {
        let structure_type = unsafe { (*next).s_type };
        if structure_type == vk::StructureType::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR {
            return false;
        }
        next = unsafe { (*next).p_next.cast() };
    }
    true
}

fn clear_surface_virtualization(surface: vk::SurfaceKHR) {
    if let Some(state) = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_mut(&surface)
    {
        state.logical_extent = None;
        state.original_window = None;
    }
}

impl ResizedWindow {
    fn restore(self) {
        if let Ok(display) = tuxscaling_display::X11Display::connect() {
            let _ =
                display.resize_window(self.window, tuxscaling_display::Monitor::new(self.original));
        }
    }
}

struct DirectSwapchain<'a> {
    create_swapchain: vk::PFN_vkCreateSwapchainKHR,
    loader: &'a ash::khr::swapchain::Device,
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR<'a>,
    allocation_callbacks: *const vk::AllocationCallbacks<'a>,
    swapchain: *mut vk::SwapchainKHR,
    surface: vk::SurfaceKHR,
}

impl DirectSwapchain<'_> {
    unsafe fn recreate(
        self,
        resized_window: Option<ResizedWindow>,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        if let Some(window) = resized_window {
            window.restore();
        }
        clear_surface_virtualization(self.surface);
        unsafe {
            self.loader
                .destroy_swapchain(*self.swapchain, self.allocation_callbacks.as_ref())
        };
        let result = unsafe {
            (self.create_swapchain)(
                self.device,
                self.create_info,
                self.allocation_callbacks,
                self.swapchain,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(result);
        }
        unsafe { self.loader.get_swapchain_images(*self.swapchain) }.inspect_err(|_| unsafe {
            self.loader
                .destroy_swapchain(*self.swapchain, self.allocation_callbacks.as_ref())
        })
    }
}

fn virtual_output_extent(
    surface: vk::SurfaceKHR,
    game_extent: vk::Extent2D,
) -> Option<(vk::Extent2D, ResizedWindow)> {
    let config = if let Ok(path) = std::env::var("TUXSCALING_CONFIG") {
        let source = std::fs::read_to_string(path).ok()?;
        tuxscaling_config::Config::parse(&source).ok()?
    } else {
        tuxscaling_config::Config::default()
    };
    let target = match config.output_resolution {
        tuxscaling_config::OutputResolution::Swapchain => return None,
        tuxscaling_config::OutputResolution::Native => None,
        tuxscaling_config::OutputResolution::Fixed { width, height } => {
            Some(vk::Extent2D { width, height })
        }
    };
    let surface = surfaces()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&surface)
        .copied()?;
    let display = tuxscaling_display::X11Display::connect().ok()?;
    let target_info = display.target_for_window(surface.window).ok()?;
    if !target_info.window.fullscreen && target_info.window.rect != target_info.monitor.rect {
        return None;
    }
    let target = target.unwrap_or(vk::Extent2D {
        width: target_info.monitor.rect.width,
        height: target_info.monitor.rect.height,
    });
    if target == game_extent || target.width == 0 || target.height == 0 {
        return None;
    }
    display
        .resize_window(
            surface.window,
            tuxscaling_display::Monitor::new(tuxscaling_display::Rect::new(
                target_info.monitor.rect.x,
                target_info.monitor.rect.y,
                target.width,
                target.height,
            )),
        )
        .ok()?;
    Some((
        target,
        ResizedWindow {
            window: surface.window,
            original: target_info.window.rect,
        },
    ))
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
    let link = unsafe { (*chain).data.layer_info };
    if link.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let get_instance_proc_addr = unsafe { (*link).get_instance_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*chain).data.layer_info = next };
    *next_gipa().lock().unwrap_or_else(|e| e.into_inner()) = Some(get_instance_proc_addr);

    let Some(proc) =
        (unsafe { get_instance_proc_addr(vk::Instance::null(), c"vkCreateInstance".as_ptr()) })
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_instance: vk::PFN_vkCreateInstance = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create_instance(create_info, allocation_callbacks, instance) };
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
    let link = unsafe { (*chain).data.layer_info };
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
    let get_device_proc_addr = unsafe { (*link).get_device_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*chain).data.layer_info = next };
    let Some(proc) = (unsafe {
        ((*link).get_instance_proc_addr)(instance.handle(), c"vkCreateDevice".as_ptr())
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_device: vk::PFN_vkCreateDevice = unsafe { std::mem::transmute(proc) };
    let result =
        unsafe { create_device(physical_device, create_info, allocation_callbacks, device) };
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
    let mut modified = *original;
    let mut resized = virtual_output_extent(original.surface, original.image_extent);
    if let Some((extent, _)) = resized {
        modified.image_extent = extent;
    }
    let needed = vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::COLOR_ATTACHMENT;
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
    if resized.is_some() && !capture_enabled {
        resized.take().unwrap().1.restore();
        modified = *original;
    }
    let mut result =
        unsafe { create_swapchain(device, &modified, allocation_callbacks, swapchain) };
    if result != vk::Result::SUCCESS {
        if let Some((_, resized_window)) = resized.take() {
            resized_window.restore();
            modified = *original;
            capture_enabled = false;
            result =
                unsafe { create_swapchain(device, create_info, allocation_callbacks, swapchain) };
        } else if capture_enabled && modified.image_usage != original.image_usage {
            capture_enabled = false;
            modified = *original;
            result =
                unsafe { create_swapchain(device, create_info, allocation_callbacks, swapchain) };
        }
    }
    if result != vk::Result::SUCCESS {
        return result;
    }
    let recovery_window = resized.map(|(_, window)| window);
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
        let mut handle = unsafe { *swapchain };
        eprintln!(
            "TuxScaling swapchain: format={:?} color_space={:?} flags={:?} capture={capture_enabled} usage={:?}",
            original.image_format, original.image_color_space, original.flags, modified.image_usage
        );
        let mut info = SwapchainInfo {
            format: modified.image_format,
            extent: modified.image_extent,
        };
        let loader = ash::khr::swapchain::Device::new(&device_state.instance, &device_state.device);
        let direct = || DirectSwapchain {
            create_swapchain,
            loader: &loader,
            device,
            create_info,
            allocation_callbacks,
            swapchain,
            surface: original.surface,
        };
        let mut resized_window = resized.map(|(_, window)| window);
        let mut output_images = match unsafe { loader.get_swapchain_images(handle) } {
            Ok(images) => images,
            Err(_) if resized_window.is_some() => {
                match unsafe { direct().recreate(resized_window.take()) } {
                    Ok(images) => {
                        handle = unsafe { *swapchain };
                        capture_enabled = false;
                        info = SwapchainInfo {
                            format: original.image_format,
                            extent: original.image_extent,
                        };
                        images
                    }
                    Err(error) => return error,
                }
            }
            Err(_) => return result,
        };
        let mut virtual_images = None;
        let mut virtual_window = None;
        if let Some(resized_window) = resized_window.take() {
            let virtual_eligible = virtual_swapchain_supported(original);
            let virtual_result = if virtual_eligible {
                let memory = unsafe {
                    device_state
                        .instance
                        .get_physical_device_memory_properties(device_state.physical_device)
                };
                let usage = original.image_usage | needed | vk::ImageUsageFlags::SAMPLED;
                output_images
                    .iter()
                    .map(|_| unsafe {
                        tuxscaling_vulkan::Image::new(
                            &device_state.device,
                            &memory,
                            original.image_extent,
                            original.image_format,
                            usage,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            };
            match virtual_result {
                Ok(images) => {
                    surfaces()
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .entry(original.surface)
                        .and_modify(|surface| {
                            surface.logical_extent = Some(original.image_extent);
                            if surface.original_window.is_none() {
                                surface.original_window = Some(resized_window.original);
                            }
                        });
                    virtual_images = Some(images);
                    virtual_window = Some(resized_window);
                }
                Err(error) => {
                    eprintln!("TuxScaling: virtual swapchain bypassed: {error:?}");
                    let recreated = unsafe { direct().recreate(Some(resized_window)) };
                    let Ok(images) = recreated else {
                        return recreated
                            .err()
                            .unwrap_or(vk::Result::ERROR_INITIALIZATION_FAILED);
                    };
                    output_images = images;
                    handle = unsafe { *swapchain };
                    info = SwapchainInfo {
                        format: original.image_format,
                        extent: original.image_extent,
                    };
                    capture_enabled = false;
                }
            }
        }
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
        let was_virtual = virtual_window.is_some();
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
                        output_images,
                    },
                    capture_enabled,
                    window,
                },
                device_state.set_loader_data,
            )
        };
        match overlay {
            Ok(overlay) => {
                swapchains()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(
                        handle,
                        Arc::new(Mutex::new(SwapchainState {
                            device,
                            surface: original.surface,
                            overlay,
                            virtual_images,
                        })),
                    );
            }
            Err(error) if was_virtual => {
                eprintln!(
                    "TuxScaling: overlay initialization failed; restoring direct swapchain: {error:?}"
                );
                let recreated = unsafe { direct().recreate(virtual_window) };
                if let Err(error) = recreated {
                    return error;
                }
            }
            Err(error) => {
                eprintln!("TuxScaling: overlay disabled for swapchain: {error:?}");
            }
        }
        result
    }));
    if post_result.is_err() {
        if let Some(window) = recovery_window {
            window.restore();
            clear_surface_virtualization(original.surface);
        }
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
    use super::virtual_swapchain_supported;
    use ash::vk;

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
}
