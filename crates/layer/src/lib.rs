use ash::vk;
use egui_ash_renderer::{Options as RendererOptions, Renderer};
use std::{
    collections::HashMap,
    ffi::{CStr, c_void},
    sync::{Mutex, OnceLock},
};

pub const CRATE_NAME: &str = "tuxscaling-layer";

const LAYER_LINK_INFO: i32 = 0;
const LOADER_INTERFACE_VERSION: u32 = 2;

#[repr(C)]
struct LayerLink {
    next: *mut Self,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
union LayerCreateInfoData {
    layer_info: *mut LayerLink,
    _set_loader_data: *const c_void,
}

#[repr(C)]
struct LayerCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: i32,
    data: LayerCreateInfoData,
}

#[repr(C)]
pub struct NegotiateLayerInterface {
    s_type: vk::StructureType,
    p_next: *const c_void,
    interface_version: u32,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    get_physical_device_proc_addr: vk::PFN_vkVoidFunction,
}

static NEXT_GIPA: OnceLock<Mutex<Option<vk::PFN_vkGetInstanceProcAddr>>> = OnceLock::new();
static INSTANCES: OnceLock<Mutex<HashMap<vk::Instance, ash::Instance>>> = OnceLock::new();
static DEVICES: OnceLock<Mutex<HashMap<vk::Device, DeviceState>>> = OnceLock::new();
static QUEUES: OnceLock<Mutex<HashMap<vk::Queue, QueueState>>> = OnceLock::new();
static SWAPCHAINS: OnceLock<Mutex<HashMap<vk::SwapchainKHR, SwapchainState>>> = OnceLock::new();

#[derive(Clone, Copy)]
struct SwapchainMeta {
    format: vk::Format,
    extent: vk::Extent2D,
}

struct SwapchainState {
    meta: SwapchainMeta,
    images: Vec<vk::Image>,
    _image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    renderer: Renderer,
    context: egui::Context,
    command_pool: Option<vk::CommandPool>,
    command_buffer: Option<vk::CommandBuffer>,
    render_complete: Option<vk::Semaphore>,
}

#[derive(Clone, Copy)]
struct QueueState {
    device: vk::Device,
    family_index: u32,
}

#[derive(Clone)]
struct DeviceState {
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    physical_device: vk::PhysicalDevice,
    instance: ash::Instance,
    device: ash::Device,
}

fn next_gipa() -> &'static Mutex<Option<vk::PFN_vkGetInstanceProcAddr>> {
    NEXT_GIPA.get_or_init(|| Mutex::new(None))
}

fn instances() -> &'static Mutex<HashMap<vk::Instance, ash::Instance>> {
    INSTANCES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn devices() -> &'static Mutex<HashMap<vk::Device, DeviceState>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn queues() -> &'static Mutex<HashMap<vk::Queue, QueueState>> {
    QUEUES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn swapchains() -> &'static Mutex<HashMap<vk::SwapchainKHR, SwapchainState>> {
    SWAPCHAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn is_srgb_swapchain_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SRGB
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::B8G8R8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

unsafe fn create_overlay_swapchain(
    state: &DeviceState,
    meta: SwapchainMeta,
    images: Vec<vk::Image>,
) -> Result<SwapchainState, vk::Result> {
    let attachment = vk::AttachmentDescription::default()
        .format(meta.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_reference = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_reference));
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass));
    let render_pass = unsafe { state.device.create_render_pass(&render_pass_info, None) }?;

    let mut image_views = Vec::with_capacity(images.len());
    let mut framebuffers = Vec::with_capacity(images.len());
    for image in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(meta.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { state.device.create_image_view(&view_info, None) }?;
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(meta.extent.width)
            .height(meta.extent.height)
            .layers(1);
        let framebuffer = unsafe { state.device.create_framebuffer(&framebuffer_info, None) }?;
        image_views.push(view);
        framebuffers.push(framebuffer);
    }

    let renderer = Renderer::with_default_allocator(
        &state.instance,
        state.physical_device,
        state.device.clone(),
        render_pass,
        RendererOptions {
            srgb_framebuffer: is_srgb_swapchain_format(meta.format),
            ..Default::default()
        },
    )
    .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    Ok(SwapchainState {
        meta,
        images,
        _image_views: image_views,
        render_pass,
        framebuffers,
        renderer,
        context: egui::Context::default(),
        command_pool: None,
        command_buffer: None,
        render_complete: None,
    })
}

unsafe fn initialize_overlay_commands(
    state: &DeviceState,
    queue_family_index: u32,
    swapchain: &mut SwapchainState,
) -> Result<(), vk::Result> {
    if swapchain.command_pool.is_some() {
        return Ok(());
    }
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { state.device.create_command_pool(&pool_info, None) }?;
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { state.device.allocate_command_buffers(&allocate_info) }?[0];
    let render_complete = unsafe {
        state
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
    }?;
    swapchain.command_pool = Some(command_pool);
    swapchain.command_buffer = Some(command_buffer);
    swapchain.render_complete = Some(render_complete);
    Ok(())
}

unsafe fn record_overlay(
    device: &DeviceState,
    queue: vk::Queue,
    swapchain: &mut SwapchainState,
    image_index: u32,
) -> Result<(), vk::Result> {
    let command_pool = swapchain
        .command_pool
        .expect("overlay commands initialized");
    let command_buffer = swapchain
        .command_buffer
        .expect("overlay commands initialized");
    let framebuffer = *swapchain
        .framebuffers
        .get(image_index as usize)
        .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
    let image = *swapchain
        .images
        .get(image_index as usize)
        .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

    unsafe { device.device.queue_wait_idle(queue) }?;
    unsafe {
        device
            .device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
    }?;

    let frame = tuxscaling_overlay::render_smoke_frame(
        &swapchain.context,
        [swapchain.meta.extent.width, swapchain.meta.extent.height],
        1.0,
    );
    swapchain
        .renderer
        .set_textures(queue, command_pool, &frame.textures_delta.set)
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .device
            .begin_command_buffer(command_buffer, &begin_info)
    }?;

    let to_color = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        device.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_color),
        );
    }
    let clear_values: [vk::ClearValue; 0] = [];
    let render_pass_info = vk::RenderPassBeginInfo::default()
        .render_pass(swapchain.render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.meta.extent,
        })
        .clear_values(&clear_values);
    unsafe {
        device.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );
    }
    swapchain
        .renderer
        .cmd_draw(
            command_buffer,
            swapchain.meta.extent,
            frame.pixels_per_point,
            &frame.primitives,
        )
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
    unsafe { device.device.cmd_end_render_pass(command_buffer) };

    let to_present = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(image)
        .subresource_range(to_color.subresource_range);
    unsafe {
        device.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_present),
        );
        device.device.end_command_buffer(command_buffer)?;
    }
    Ok(())
}

unsafe fn find_layer_link(
    mut next: *const c_void,
    expected_type: vk::StructureType,
) -> Option<*mut LayerLink> {
    while !next.is_null() {
        let info = next.cast::<LayerCreateInfo>();
        if unsafe { (*info).s_type } == expected_type
            && unsafe { (*info).function } == LAYER_LINK_INFO
        {
            return Some(unsafe { (*info).data.layer_info });
        }
        next = unsafe { (*info).p_next };
    }
    None
}

unsafe fn device_downstream(device: vk::Device, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_device_proc_addr = devices()
        .lock()
        .expect("device map lock poisoned")
        .get(&device)?
        .get_device_proc_addr;
    unsafe { get_device_proc_addr(device, name.as_ptr()) }
}

unsafe fn downstream(instance: vk::Instance, name: &CStr) -> vk::PFN_vkVoidFunction {
    let get_instance_proc_addr = *next_gipa().lock().expect("layer state lock poisoned");
    let get_instance_proc_addr = get_instance_proc_addr?;
    unsafe { get_instance_proc_addr(instance, name.as_ptr()) }
}

unsafe extern "system" fn create_instance(
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
    let get_instance_proc_addr = unsafe { (*link).get_instance_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*link).next = next };
    *next_gipa().lock().expect("layer state lock poisoned") = Some(get_instance_proc_addr);

    let name = c"vkCreateInstance";
    let Some(proc) = (unsafe { downstream(vk::Instance::null(), name) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_instance: vk::PFN_vkCreateInstance = unsafe { std::mem::transmute(proc) };
    eprintln!("tuxscaling: vkCreateInstance");
    let result = unsafe { create_instance(create_info, allocation_callbacks, instance) };
    if result == vk::Result::SUCCESS {
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
            .expect("instance map lock poisoned")
            .insert(unsafe { *instance }, ash_instance);
    }
    result
}

unsafe extern "system" fn create_device(
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
    let get_device_proc_addr = unsafe { (*link).get_device_proc_addr };
    let next = unsafe { (*link).next };
    unsafe { (*link).next = next };
    let name = c"vkCreateDevice";
    let Some(proc) = (unsafe { downstream(vk::Instance::null(), name) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_device: vk::PFN_vkCreateDevice = unsafe { std::mem::transmute(proc) };
    eprintln!("tuxscaling: vkCreateDevice");
    let result =
        unsafe { create_device(physical_device, create_info, allocation_callbacks, device) };
    if result == vk::Result::SUCCESS {
        let Some(instance) = instances()
            .lock()
            .expect("instance map lock poisoned")
            .values()
            .next()
            .cloned()
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        eprintln!("tuxscaling: loading ash device dispatch");
        let ash_device = unsafe {
            ash::Device::load_with(
                |command| {
                    get_device_proc_addr(*device, command.as_ptr())
                        .map_or(std::ptr::null(), |function| function as *const c_void)
                },
                *device,
            )
        };
        eprintln!("tuxscaling: ash device dispatch loaded");
        devices().lock().expect("device map lock poisoned").insert(
            unsafe { *device },
            DeviceState {
                get_device_proc_addr,
                physical_device,
                instance,
                device: ash_device,
            },
        );
        eprintln!("tuxscaling: device state registered");
    }
    result
}

unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    queue: *mut vk::Queue,
) {
    let name = c"vkGetDeviceQueue";
    if let Some(proc) = unsafe { device_downstream(device, name) } {
        let get_queue: vk::PFN_vkGetDeviceQueue = unsafe { std::mem::transmute(proc) };
        unsafe { get_queue(device, queue_family_index, queue_index, queue) };
        if !queue.is_null() {
            queues().lock().expect("queue map lock poisoned").insert(
                unsafe { *queue },
                QueueState {
                    device,
                    family_index: queue_family_index,
                },
            );
        }
    }
}

unsafe extern "system" fn create_swapchain_khr(
    device: vk::Device,
    create_info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
    swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    if create_info.is_null() || swapchain.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let name = c"vkCreateSwapchainKHR";
    let Some(proc) = (unsafe { device_downstream(device, name) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let create_swapchain: vk::PFN_vkCreateSwapchainKHR = unsafe { std::mem::transmute(proc) };
    let result = unsafe { create_swapchain(device, create_info, allocation_callbacks, swapchain) };
    if result == vk::Result::SUCCESS {
        let meta = SwapchainMeta {
            format: unsafe { (*create_info).image_format },
            extent: unsafe { (*create_info).image_extent },
        };
        let handle = unsafe { *swapchain };
        let state = devices()
            .lock()
            .expect("device map lock poisoned")
            .get(&device)
            .cloned();
        if let Some(state) = state {
            let loader = ash::khr::swapchain::Device::new(&state.instance, &state.device);
            match unsafe { loader.get_swapchain_images(handle) } {
                Ok(images) => match unsafe { create_overlay_swapchain(&state, meta, images) } {
                    Ok(resources) => {
                        eprintln!("tuxscaling: egui renderer initialized");
                        swapchains()
                            .lock()
                            .expect("swapchain map lock poisoned")
                            .insert(handle, resources);
                    }
                    Err(error) => eprintln!("tuxscaling: egui renderer init failed: {error:?}"),
                },
                Err(error) => eprintln!("tuxscaling: get swapchain images failed: {error:?}"),
            }
        }
        eprintln!(
            "tuxscaling: vkCreateSwapchainKHR format={:?} extent={}x{}",
            meta.format, meta.extent.width, meta.extent.height
        );
    }
    result
}

unsafe extern "system" fn destroy_swapchain_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    allocation_callbacks: *const vk::AllocationCallbacks<'_>,
) {
    let name = c"vkDestroySwapchainKHR";
    let destroy = unsafe { device_downstream(device, name) };
    swapchains()
        .lock()
        .expect("swapchain map lock poisoned")
        .remove(&swapchain);
    if let Some(proc) = destroy {
        let destroy_swapchain: vk::PFN_vkDestroySwapchainKHR = unsafe { std::mem::transmute(proc) };
        unsafe { destroy_swapchain(device, swapchain, allocation_callbacks) };
    }
}

unsafe extern "system" fn queue_present_khr(
    queue: vk::Queue,
    present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    let Some(queue_state) = queues()
        .lock()
        .expect("queue map lock poisoned")
        .get(&queue)
        .copied()
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let device = queue_state.device;
    let mut overlay_complete = None;
    if !present_info.is_null() {
        let info = unsafe { &*present_info };
        if info.swapchain_count == 1
            && !info.p_swapchains.is_null()
            && !info.p_image_indices.is_null()
        {
            let swapchain = unsafe { *info.p_swapchains };
            let image_index = unsafe { *info.p_image_indices };
            let device_state = devices()
                .lock()
                .expect("device map lock poisoned")
                .get(&device)
                .cloned();
            if let Some(device_state) = device_state
                && let Some(swapchain_state) = swapchains()
                    .lock()
                    .expect("swapchain map lock poisoned")
                    .get_mut(&swapchain)
            {
                let needs_command_resources = swapchain_state.command_pool.is_none();
                match unsafe {
                    initialize_overlay_commands(
                        &device_state,
                        queue_state.family_index,
                        swapchain_state,
                    )
                } {
                    Ok(()) => match unsafe {
                        record_overlay(&device_state, queue, swapchain_state, image_index)
                    } {
                        Ok(()) => {
                            let wait_semaphores = if info.wait_semaphore_count == 0 {
                                &[]
                            } else {
                                unsafe {
                                    std::slice::from_raw_parts(
                                        info.p_wait_semaphores,
                                        info.wait_semaphore_count as usize,
                                    )
                                }
                            };
                            let wait_stages = vec![
                                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                                wait_semaphores.len()
                            ];
                            let command_buffer = swapchain_state
                                .command_buffer
                                .expect("overlay commands initialized");
                            let render_complete = swapchain_state
                                .render_complete
                                .expect("overlay commands initialized");
                            let submit_info = vk::SubmitInfo::default()
                                .wait_semaphores(wait_semaphores)
                                .wait_dst_stage_mask(&wait_stages)
                                .command_buffers(std::slice::from_ref(&command_buffer))
                                .signal_semaphores(std::slice::from_ref(&render_complete));
                            match unsafe {
                                device_state.device.queue_submit(
                                    queue,
                                    std::slice::from_ref(&submit_info),
                                    vk::Fence::null(),
                                )
                            } {
                                Ok(()) => {
                                    overlay_complete = Some(render_complete);
                                    if needs_command_resources {
                                        eprintln!("tuxscaling: egui command resources initialized");
                                    }
                                }
                                Err(error) => {
                                    eprintln!("tuxscaling: egui submit failed: {error:?}")
                                }
                            }
                        }
                        Err(error) => eprintln!("tuxscaling: egui draw failed: {error:?}"),
                    },
                    Err(error) => {
                        eprintln!("tuxscaling: egui command resources failed: {error:?}")
                    }
                }
            }
        }
    }
    let name = c"vkQueuePresentKHR";
    let Some(proc) = (unsafe { device_downstream(device, name) }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let present: vk::PFN_vkQueuePresentKHR = unsafe { std::mem::transmute(proc) };
    if let Some(render_complete) = overlay_complete {
        let mut modified = unsafe { *present_info };
        modified.wait_semaphore_count = 1;
        modified.p_wait_semaphores = &render_complete;
        return unsafe { present(queue, &modified) };
    }
    unsafe { present(queue, present_info) }
}

unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) };
    if name.to_bytes() == b"vkCreateInstance" {
        return unsafe {
            std::mem::transmute::<vk::PFN_vkCreateInstance, vk::PFN_vkVoidFunction>(
                create_instance as vk::PFN_vkCreateInstance,
            )
        };
    }
    if name.to_bytes() == b"vkCreateDevice" {
        return unsafe {
            std::mem::transmute::<vk::PFN_vkCreateDevice, vk::PFN_vkVoidFunction>(
                create_device as vk::PFN_vkCreateDevice,
            )
        };
    }
    unsafe { downstream(instance, name) }
}

unsafe extern "system" fn get_device_proc_addr(
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
        b"vkQueuePresentKHR" => unsafe {
            std::mem::transmute::<vk::PFN_vkQueuePresentKHR, vk::PFN_vkVoidFunction>(
                queue_present_khr as vk::PFN_vkQueuePresentKHR,
            )
        },
        _ => unsafe { device_downstream(device, name) },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn layer_vkGetInstanceProcAddr(
    instance: vk::Instance,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    unsafe { get_instance_proc_addr(instance, name) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn layer_vkGetDeviceProcAddr(
    device: vk::Device,
    name: *const i8,
) -> vk::PFN_vkVoidFunction {
    unsafe { get_device_proc_addr(device, name) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    version: *mut NegotiateLayerInterface,
) -> vk::Result {
    if version.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let version = unsafe { &mut *version };
    if version.s_type != vk::StructureType::LOADER_INSTANCE_CREATE_INFO
        || version.interface_version < LOADER_INTERFACE_VERSION
    {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    version.interface_version = LOADER_INTERFACE_VERSION;
    version.get_instance_proc_addr = get_instance_proc_addr;
    version.get_device_proc_addr = get_device_proc_addr;
    version.get_physical_device_proc_addr = None;
    eprintln!("tuxscaling: Vulkan layer negotiated");
    vk::Result::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_linear_output_for_float_swapchains() {
        assert!(!is_srgb_swapchain_format(vk::Format::R16G16B16A16_SFLOAT));
    }

    #[test]
    fn uses_linear_output_for_srgb_swapchains() {
        assert!(is_srgb_swapchain_format(vk::Format::B8G8R8A8_SRGB));
    }
}
