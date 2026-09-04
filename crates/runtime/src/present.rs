#![allow(clippy::missing_safety_doc)]
use ash::vk;
use std::time::Instant;
use tuxscaling_capture::Capture;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_overlay::FrameDiagnostics;
use tuxscaling_overlay_vulkan::{OverlayRenderer, SwapchainInfo};
use tuxscaling_temporal::History;
use tuxscaling_vulkan::{image_barrier, memory_barrier};

pub type SetLoaderData = unsafe extern "system" fn(vk::Device, *mut std::ffi::c_void) -> vk::Result;

#[derive(Clone, Copy)]
pub struct FrameSubmission {
    pub command_buffer: vk::CommandBuffer,
    pub render_complete: vk::Semaphore,
    pub fence: vk::Fence,
}
struct Slot {
    command: vk::CommandBuffer,
    fence: vk::Fence,
    semaphore: vk::Semaphore,
}
pub struct SwapchainRuntime {
    device: ash::Device,
    info: SwapchainInfo,
    images: Vec<vk::Image>,
    overlay: Option<OverlayRenderer>,
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    queue: Option<vk::Queue>,
    enabled: bool,
    set_loader_data: Option<SetLoaderData>,
    capture: Option<Capture>,
    motion: Option<MotionEstimator>,
    history: History,
    start: Instant,
    pending_time: std::time::Duration,
    mode: u32,
    queries: vk::QueryPool,
    query_ready: Vec<bool>,
    timestamp_period: f32,
    timings: Vec<[f32; 3]>,
    diagnostics: FrameDiagnostics,
}
impl SwapchainRuntime {
    pub unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        info: SwapchainInfo,
        images: Vec<vk::Image>,
        set_loader_data: Option<SetLoaderData>,
        capture_enabled: bool,
    ) -> Result<Self, vk::Result> {
        let config = if let Ok(path) = std::env::var("TUXSCALING_CONFIG") {
            let source = std::fs::read_to_string(path).map_err(|error| {
                eprintln!("TuxScaling configuration: {error}");
                vk::Result::ERROR_INITIALIZATION_FAILED
            })?;
            tuxscaling_config::Config::parse(&source).map_err(|error| {
                eprintln!("TuxScaling configuration: {error}");
                vk::Result::ERROR_INITIALIZATION_FAILED
            })?
        } else {
            tuxscaling_config::Config::default()
        };
        if !config.enabled {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let overlay = unsafe { OverlayRenderer::new(instance, physical, device, info, &images) }?;
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let capture = if capture_enabled {
            Some(unsafe { Capture::new(device, &memory, info.extent, info.format) }?)
        } else {
            None
        };
        let mut motion = if let Some(capture) = &capture {
            Some(unsafe {
                MotionEstimator::new(
                    device,
                    &memory,
                    info.extent,
                    capture.color.view,
                    matches!(
                        info.format,
                        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
                    ),
                )
            }?)
        } else {
            None
        };
        if let Some(motion) = &mut motion {
            motion.cut_thresholds = [
                config.scene_distance_threshold,
                config.scene_consistency_threshold,
            ];
        }
        let mode = match std::env::var("TUXSCALING_VIEW").as_deref() {
            Ok("luminance") => 1,
            Ok("motion") => 2,
            Ok("confidence") => 3,
            Ok("original") => 0,
            Err(_) => config.debug_view as u32,
            Ok(other) => {
                eprintln!("TuxScaling: unsupported TUXSCALING_VIEW={other}");
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        };
        let image_count = images.len();
        Ok(Self {
            device: device.clone(),
            info,
            images,
            overlay: Some(overlay),
            pool: vk::CommandPool::null(),
            slots: Vec::new(),
            queue: None,
            enabled: true,
            set_loader_data,
            capture,
            motion,
            history: History::default(),
            start: Instant::now(),
            pending_time: std::time::Duration::ZERO,
            mode,
            queries: vk::QueryPool::null(),
            query_ready: vec![false; image_count],
            timestamp_period: if properties.limits.timestamp_compute_and_graphics != 0 {
                properties.limits.timestamp_period
            } else {
                0.0
            },
            timings: Vec::new(),
            diagnostics: FrameDiagnostics {
                state: if capture_enabled {
                    "Capture ready"
                } else {
                    "Bypass: capture unsupported"
                }
                .into(),
                mode: ["Original", "Luminance", "Motion", "Confidence"][mode as usize].into(),
                ..Default::default()
            },
        })
    }
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    unsafe fn initialize(&mut self, queue: vk::Queue, family: u32) -> Result<(), vk::Result> {
        if let Some(selected) = self.queue {
            return if selected == queue {
                Ok(())
            } else {
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            };
        }
        self.queue = Some(queue);
        self.pool = unsafe {
            self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        if self.timestamp_period > 0.0 {
            self.queries = unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(self.images.len() as u32 * 4),
                    None,
                )
            }?;
        }
        let commands = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.images.len() as u32),
            )
        }?;
        for command in commands {
            if let Some(set) = self.set_loader_data {
                use ash::vk::Handle;
                let result = unsafe {
                    set(
                        self.device.handle(),
                        command.as_raw() as *mut std::ffi::c_void,
                    )
                };
                if result != vk::Result::SUCCESS {
                    return Err(result);
                }
            }
            self.slots.push(Slot {
                command,
                fence: vk::Fence::null(),
                semaphore: vk::Semaphore::null(),
            });
            let slot = self.slots.last_mut().unwrap();
            slot.fence = unsafe {
                self.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }?;
            slot.semaphore = unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }?;
        }
        Ok(())
    }
    pub unsafe fn prepare_frame(
        &mut self,
        _device: &ash::Device,
        queue: vk::Queue,
        family: u32,
        index: u32,
    ) -> Result<FrameSubmission, vk::Result> {
        if !self.enabled {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        unsafe { self.initialize(queue, family) }?;
        let index = index as usize;
        let slot = self
            .slots
            .get(index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        unsafe { self.device.wait_for_fences(&[slot.fence], true, u64::MAX) }?;
        if self.query_ready[index] && self.queries != vk::QueryPool::null() {
            let mut times = [0u64; 4];
            if unsafe {
                self.device.get_query_pool_results(
                    self.queries,
                    index as u32 * 4,
                    &mut times,
                    vk::QueryResultFlags::TYPE_64,
                )
            }
            .is_ok()
            {
                let ms = [0, 1, 2].map(|i| {
                    times[i + 1].wrapping_sub(times[i]) as f32 * self.timestamp_period / 1_000_000.0
                });
                self.diagnostics.capture_ms = ms[0];
                self.diagnostics.motion_ms = ms[1];
                self.diagnostics.overlay_ms = ms[2];
                if self.history.frame_id > 30 {
                    if self.timings.len() == 18000 {
                        self.timings.remove(0);
                    }
                    self.timings.push(ms);
                }
            }
        }
        self.pending_time = self.start.elapsed();
        let valid = self.history.valid(self.pending_time);
        if self.motion.is_some() {
            self.diagnostics.state = if valid {
                "Estimated motion"
            } else {
                "History reset"
            }
            .into();
        }
        let frame =
            self.overlay
                .as_mut()
                .unwrap()
                .prepare(queue, self.pool, index, &self.diagnostics)?;
        unsafe {
            self.device
                .reset_command_buffer(slot.command, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                slot.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if self.queries != vk::QueryPool::null() {
                self.device
                    .cmd_reset_query_pool(slot.command, self.queries, index as u32 * 4, 4);
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    index as u32 * 4,
                );
            }
            if let Some(capture) = &mut self.capture {
                capture.record(&self.device, slot.command, self.images[index]);
            } else {
                image_barrier(
                    &self.device,
                    slot.command,
                    self.images[index],
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            if self.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    self.queries,
                    index as u32 * 4 + 1,
                );
            }
            if let Some(motion) = &mut self.motion {
                motion.record(slot.command, self.history.write_index(), valid, self.mode);
                if self.mode != 0 {
                    image_barrier(
                        &self.device,
                        slot.command,
                        motion.visualization.handle,
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    );
                    image_barrier(
                        &self.device,
                        slot.command,
                        self.images[index],
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );
                    let layers = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1);
                    let offsets = [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: self.info.extent.width as i32,
                            y: self.info.extent.height as i32,
                            z: 1,
                        },
                    ];
                    let region = vk::ImageBlit::default()
                        .src_subresource(layers)
                        .dst_subresource(layers)
                        .src_offsets(offsets)
                        .dst_offsets(offsets);
                    self.device.cmd_blit_image(
                        slot.command,
                        motion.visualization.handle,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.images[index],
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                        vk::Filter::NEAREST,
                    );
                    image_barrier(
                        &self.device,
                        slot.command,
                        motion.visualization.handle,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    );
                    image_barrier(
                        &self.device,
                        slot.command,
                        self.images[index],
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    );
                }
                memory_barrier(&self.device, slot.command);
            }
            if self.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    self.queries,
                    index as u32 * 4 + 2,
                );
            }
            self.overlay
                .as_mut()
                .unwrap()
                .record(slot.command, index, &frame)?;
            image_barrier(
                &self.device,
                slot.command,
                self.images[index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            if self.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    index as u32 * 4 + 3,
                );
            }
            self.device.end_command_buffer(slot.command)?;
        }
        self.query_ready[index] = true;
        Ok(FrameSubmission {
            command_buffer: slot.command,
            fence: slot.fence,
            render_complete: slot.semaphore,
        })
    }
    pub fn submitted(&mut self) {
        self.history.commit(self.pending_time);
        self.diagnostics.frame_id = self.history.frame_id;
    }
    pub fn presentation_failed(&mut self) {
        self.history.reset();
    }
    pub unsafe fn destroy(self, _device: &ash::Device) {
        drop(self);
    }
}
impl Drop for SwapchainRuntime {
    fn drop(&mut self) {
        if !self.timings.is_empty() {
            let mut summary = String::new();
            for (axis, name) in ["capture", "flow", "overlay"].iter().enumerate() {
                let mut values = self.timings.iter().map(|v| v[axis]).collect::<Vec<_>>();
                values.sort_by(f32::total_cmp);
                summary.push_str(&format!(
                    " {name}: median={:.3} p95={:.3} ms",
                    values[values.len() / 2],
                    values[(values.len() - 1) * 95 / 100]
                ));
            }
            eprintln!(
                "TuxScaling {}x{} GPU samples={}{}",
                self.info.extent.width,
                self.info.extent.height,
                self.timings.len(),
                summary
            );
        }
        drop(self.motion.take());
        drop(self.capture.take());
        drop(self.overlay.take());
        unsafe {
            for slot in &self.slots {
                self.device.destroy_semaphore(slot.semaphore, None);
                self.device.destroy_fence(slot.fence, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_query_pool(self.queries, None);
        }
    }
}
