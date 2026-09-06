#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_capture::Capture;
use tuxscaling_motion::MotionQuality;
use tuxscaling_overlay::FrameDiagnostics;
use tuxscaling_overlay_vulkan::{OverlayRenderer, SwapchainInfo};
use tuxscaling_temporal::{FrameExtent, GuidanceReset};
use tuxscaling_upscaler::{ResolutionPlan, content_viewport};
use tuxscaling_vulkan::{image_barrier, memory_barrier};

#[path = "pipeline.rs"]
mod pipeline;
pub use pipeline::{FrameTimingState, TemporalPipeline, TemporalPipelineDescriptor, TimingState};

pub type SetLoaderData = unsafe extern "system" fn(vk::Device, *mut std::ffi::c_void) -> vk::Result;

#[derive(Clone, Copy)]
pub struct FrameSubmission {
    pub command_buffer: vk::CommandBuffer,
    pub render_complete: vk::Semaphore,
    pub fence: vk::Fence,
}

#[derive(Clone)]
pub struct SwapchainImages {
    pub game_images: Vec<vk::Image>,
    pub game_extent: vk::Extent2D,
    pub output_images: Vec<vk::Image>,
}

impl SwapchainImages {
    pub fn direct(images: Vec<vk::Image>, extent: vk::Extent2D) -> Self {
        Self {
            game_images: images.clone(),
            game_extent: extent,
            output_images: images,
        }
    }

    fn is_valid(&self) -> bool {
        !self.game_images.is_empty() && self.game_images.len() == self.output_images.len()
    }
}

pub struct SwapchainRuntimeCreateInfo {
    pub info: SwapchainInfo,
    pub images: SwapchainImages,
    pub capture_enabled: bool,
    pub window: Option<u64>,
    pub fullscreen: bool,
}

struct Slot {
    command: vk::CommandBuffer,
    fence: vk::Fence,
    semaphore: vk::Semaphore,
}
const GPU_PHASES: usize = 13;
const GPU_TIMESTAMPS: usize = 16;

fn mode_name(mode: u32) -> &'static str {
    [
        "Original",
        "Luminance",
        "Motion",
        "Confidence",
        "Reconstructed",
        "History",
        "Reactive",
        "Disocclusion",
        "Depth",
        "Composition",
        "Exposure",
    ]
    .get(mode as usize)
    .copied()
    .unwrap_or("Original")
}

fn debug_mode_id(name: &str) -> Option<u32> {
    Some(match name {
        "original" => 0,
        "luminance" => 1,
        "motion" => 2,
        "confidence" => 3,
        "reconstructed" => 4,
        "history" => 5,
        "reactive" => 6,
        "disocclusion" => 7,
        "depth" => 8,
        "composition" => 9,
        "exposure" => 10,
        _ => return None,
    })
}

fn motion_quality(quality: tuxscaling_config::MotionQuality) -> MotionQuality {
    match quality {
        tuxscaling_config::MotionQuality::Ultra => MotionQuality::Ultra,
        tuxscaling_config::MotionQuality::High => MotionQuality::High,
        tuxscaling_config::MotionQuality::Balanced => MotionQuality::Balanced,
        tuxscaling_config::MotionQuality::Performance => MotionQuality::Performance,
    }
}

fn quality_budget(quality: tuxscaling_config::MotionQuality) -> f32 {
    match quality {
        tuxscaling_config::MotionQuality::Ultra => 12.0,
        tuxscaling_config::MotionQuality::High => 8.0,
        tuxscaling_config::MotionQuality::Balanced => 4.0,
        tuxscaling_config::MotionQuality::Performance => 2.5,
    }
}

pub fn map_damage_rect(
    game_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    rect: vk::Rect2D,
) -> vk::Rect2D {
    if game_extent == output_extent || game_extent.width == 0 || game_extent.height == 0 {
        return rect;
    }
    let viewport = content_viewport(game_extent, output_extent);
    let left = (viewport.offset[0] * output_extent.width as f32).round() as i32;
    let top = (viewport.offset[1] * output_extent.height as f32).round() as i32;
    let width = (viewport.size[0] * output_extent.width as f32).round() as i32;
    let height = (viewport.size[1] * output_extent.height as f32).round() as i32;
    let right = left.saturating_add(width.max(0));
    let bottom = top.saturating_add(height.max(0));
    let source_right = rect.offset.x.saturating_add(rect.extent.width as i32);
    let source_bottom = rect.offset.y.saturating_add(rect.extent.height as i32);
    let x0 = left
        .saturating_add(
            (rect.offset.x as f64 * width as f64 / game_extent.width as f64).floor() as i32,
        )
        .clamp(left, right);
    let y0 = top
        .saturating_add(
            (rect.offset.y as f64 * height as f64 / game_extent.height as f64).floor() as i32,
        )
        .clamp(top, bottom);
    let x1 = left
        .saturating_add(
            (source_right as f64 * width as f64 / game_extent.width as f64).ceil() as i32,
        )
        .clamp(left, right);
    let y1 = top
        .saturating_add(
            (source_bottom as f64 * height as f64 / game_extent.height as f64).ceil() as i32,
        )
        .clamp(top, bottom);
    vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D {
            width: x1.saturating_sub(x0) as u32,
            height: y1.saturating_sub(y0) as u32,
        },
    }
}

unsafe fn record_spatial_fallback(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source: &Capture,
    output: vk::Image,
    output_layout: vk::ImageLayout,
    output_extent: vk::Extent2D,
) {
    unsafe {
        record_spatial_blit(
            device,
            command,
            source.color.handle,
            source.color.extent,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            output,
            output_layout,
            output_extent,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_spatial_blit(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source: vk::Image,
    source_extent: vk::Extent2D,
    source_layout: vk::ImageLayout,
    output: vk::Image,
    output_layout: vk::ImageLayout,
    output_extent: vk::Extent2D,
) {
    let viewport = content_viewport(source_extent, output_extent);
    let left = (viewport.offset[0] * output_extent.width as f32).round() as i32;
    let top = (viewport.offset[1] * output_extent.height as f32).round() as i32;
    let right =
        ((viewport.offset[0] + viewport.size[0]) * output_extent.width as f32).round() as i32;
    let bottom =
        ((viewport.offset[1] + viewport.size[1]) * output_extent.height as f32).round() as i32;
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    unsafe {
        image_barrier(
            device,
            command,
            source,
            source_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        image_barrier(
            device,
            command,
            output,
            output_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        device.cmd_clear_color_image(
            command,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            &[tuxscaling_vulkan::color_range()],
        );
        memory_barrier(device, command);
        device.cmd_blit_image(
            command,
            source,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers)
                .dst_subresource(layers)
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: source_extent.width as i32,
                        y: source_extent.height as i32,
                        z: 1,
                    },
                ])
                .dst_offsets([
                    vk::Offset3D {
                        x: left,
                        y: top,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: right,
                        y: bottom,
                        z: 1,
                    },
                ])],
            vk::Filter::LINEAR,
        );
        image_barrier(
            device,
            command,
            source,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            source_layout,
        );
        image_barrier(
            device,
            command,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }
}

pub struct SwapchainRuntime {
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    info: SwapchainInfo,
    game_images: Vec<vk::Image>,
    output_images: Vec<vk::Image>,
    overlay: Option<OverlayRenderer>,
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    queue: Option<vk::Queue>,
    enabled: bool,
    set_loader_data: Option<SetLoaderData>,
    temporal: TemporalPipeline,
    mode: u32,
    output_presented: Vec<bool>,
    pending_output: Option<usize>,
    diagnostics: FrameDiagnostics,
}
impl SwapchainRuntime {
    pub unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        create: SwapchainRuntimeCreateInfo,
        set_loader_data: Option<SetLoaderData>,
    ) -> Result<Self, vk::Result> {
        let SwapchainRuntimeCreateInfo {
            info,
            images,
            capture_enabled,
            window,
            fullscreen,
        } = create;
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
        if !images.is_valid() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let resolution =
            ResolutionPlan::new(images.game_extent, info.extent, config.processing_scale);
        let overlay = unsafe {
            OverlayRenderer::new(
                instance,
                physical,
                device,
                info,
                &images.output_images,
                window,
            )
        }?;
        let temporal = unsafe {
            TemporalPipeline::new(TemporalPipelineDescriptor {
                instance,
                physical,
                device,
                info,
                resolution,
                config: &config,
                capture_enabled,
                image_count: images.output_images.len(),
            })
        }?;
        let image_count = images.output_images.len();
        let mode = match std::env::var("TUXSCALING_VIEW").as_deref() {
            Ok(value) => debug_mode_id(value).ok_or_else(|| {
                eprintln!("TuxScaling: unsupported TUXSCALING_VIEW={value}");
                vk::Result::ERROR_INITIALIZATION_FAILED
            })?,
            Err(_) => config.debug_view as u32,
        };
        let diagnostic_resolution = temporal.resolution;
        Ok(Self {
            instance: instance.clone(),
            physical,
            device: device.clone(),
            info,
            game_images: images.game_images,
            output_images: images.output_images,
            overlay: Some(overlay),
            pool: vk::CommandPool::null(),
            slots: Vec::new(),
            queue: None,
            enabled: true,
            set_loader_data,
            temporal,
            mode,
            output_presented: vec![false; image_count],
            pending_output: None,
            diagnostics: FrameDiagnostics {
                state: if capture_enabled {
                    "Capture ready"
                } else {
                    "Bypass: capture unsupported"
                }
                .into(),
                mode: mode_name(mode).into(),
                quality: config.motion_quality,
                processing_scale: config.processing_scale,
                presentation_mode: if !capture_enabled {
                    "Fallback: unsupported capture"
                } else if diagnostic_resolution.game_extent != diagnostic_resolution.output_extent {
                    "Virtual upscale"
                } else if fullscreen {
                    "Native AA"
                } else {
                    "Windowed 1:1"
                }
                .into(),
                game_extent: [
                    diagnostic_resolution.game_extent.width,
                    diagnostic_resolution.game_extent.height,
                ],
                processing_extent: [
                    diagnostic_resolution.processing_extent.width,
                    diagnostic_resolution.processing_extent.height,
                ],
                output_extent: [
                    diagnostic_resolution.output_extent.width,
                    diagnostic_resolution.output_extent.height,
                ],
                ..Default::default()
            },
        })
    }
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    pub fn map_damage_rect(&self, rect: vk::Rect2D) -> vk::Rect2D {
        map_damage_rect(
            self.temporal.resolution.game_extent,
            self.temporal.resolution.output_extent,
            rect,
        )
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
        if self.temporal.timestamp_period > 0.0 {
            self.temporal.queries = unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(self.output_images.len() as u32 * GPU_TIMESTAMPS as u32),
                    None,
                )
            }?;
        }
        let commands = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.output_images.len() as u32),
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

    unsafe fn apply_pending_processing_scale(&mut self) -> Result<(), vk::Result> {
        let Some(scale) = self.temporal.pending_processing_scale.take() else {
            return Ok(());
        };
        if (scale - self.diagnostics.processing_scale).abs() < f32::EPSILON {
            return Ok(());
        }
        for slot in &self.slots {
            unsafe { self.device.wait_for_fences(&[slot.fence], true, u64::MAX) }?;
        }
        let mut config = self.temporal.config.clone();
        config.processing_scale = scale;
        let resolution = ResolutionPlan::new(
            self.temporal.resolution.game_extent,
            self.temporal.resolution.output_extent,
            scale,
        );
        let capture_enabled = self.temporal.capture.is_some();
        unsafe {
            self.temporal.rebuild(
                TemporalPipelineDescriptor {
                    instance: &self.instance,
                    physical: self.physical,
                    device: &self.device,
                    info: self.info,
                    resolution,
                    config: &config,
                    capture_enabled,
                    image_count: self.output_images.len(),
                },
                &self.device,
            )?;
        }
        self.diagnostics.processing_scale = scale;
        self.temporal.reset_reason = GuidanceReset::PresetChanged;
        if self.temporal.timestamp_period > 0.0 {
            self.temporal.queries = unsafe {
                self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(self.output_images.len() as u32 * GPU_TIMESTAMPS as u32),
                    None,
                )
            }?;
        }
        self.diagnostics.processing_extent = [
            resolution.processing_extent.width,
            resolution.processing_extent.height,
        ];
        self.diagnostics.state = "Processing scale changed; history reset".into();
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
        if cfg!(debug_assertions)
            && std::env::var("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE")
                .ok()
                .as_deref()
                == Some("1")
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        unsafe { self.initialize(queue, family) }?;
        unsafe { self.apply_pending_processing_scale() }?;
        let index = index as usize;
        let slot = self
            .slots
            .get(index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        let output_layout = if self.output_presented[index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        unsafe { self.device.wait_for_fences(&[slot.fence], true, u64::MAX) }?;
        if let Some(quality) = self.temporal.pending_quality.take() {
            if let Some(motion) = &mut self.temporal.motion {
                motion.set_quality(quality);
            }
            self.temporal.history.reset();
            self.temporal.reset_reason = GuidanceReset::PresetChanged;
            if let Some(upscaler) = &mut self.temporal.upscaler {
                upscaler.reset();
            }
            self.diagnostics.state = "Preset changed; history reset".into();
        }
        if self.temporal.query_ready[index] && self.temporal.queries != vk::QueryPool::null() {
            let mut times = [0u64; GPU_TIMESTAMPS];
            if unsafe {
                self.device.get_query_pool_results(
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32,
                    &mut times,
                    vk::QueryResultFlags::TYPE_64,
                )
            }
            .is_ok()
            {
                let elapsed = |start: usize, end: usize| {
                    times[end].wrapping_sub(times[start]) as f32 * self.temporal.timestamp_period
                        / 1_000_000.0
                };
                let ms = [
                    elapsed(0, 1),
                    elapsed(2, 3),
                    elapsed(3, 4),
                    elapsed(4, 5),
                    elapsed(5, 6),
                    elapsed(6, 7),
                    elapsed(7, 8),
                    elapsed(8, 9),
                    elapsed(9, 10),
                    elapsed(11, 12),
                    elapsed(12, 13),
                    elapsed(13, 14),
                    elapsed(14, 15),
                ];
                self.diagnostics.capture_ms = ms[0];
                self.diagnostics.luma_ms = ms[1];
                self.diagnostics.pyramid_ms = ms[2];
                self.diagnostics.forward_flow_ms = ms[3];
                self.diagnostics.backward_flow_ms = ms[4];
                self.diagnostics.confidence_ms = ms[5];
                self.diagnostics.scene_ms = ms[7];
                self.diagnostics.invalidate_ms = ms[8];
                self.diagnostics.motion_ms = ms[1..9].iter().sum();
                self.diagnostics.reactive_ms = ms[9];
                self.diagnostics.exposure_ms = ms[10];
                self.diagnostics.guidance_ms = ms[9] + ms[10];
                self.diagnostics.reconstruction_ms = ms[11];
                self.diagnostics.overlay_ms = ms[12];
                let temporal_ms = self.diagnostics.motion_ms
                    + self.diagnostics.guidance_ms
                    + self.diagnostics.reconstruction_ms;
                self.diagnostics.budget_warning =
                    temporal_ms > quality_budget(self.diagnostics.quality);
                if self.temporal.history.frame_id >= 180 {
                    if self.temporal.timings.len() == 18000 {
                        self.temporal.timings.remove(0);
                    }
                    self.temporal.timings.push(ms);
                    let mut totals = self
                        .temporal
                        .timings
                        .iter()
                        .map(|timing| timing[1..12].iter().sum::<f32>())
                        .collect::<Vec<_>>();
                    totals.sort_by(f32::total_cmp);
                    self.diagnostics.p95_ms = totals[(totals.len() - 1) * 95 / 100];
                    let median = totals[totals.len() / 2];
                    self.diagnostics.budget_warning |= self.diagnostics.p95_ms > median * 1.35;
                }
            }
        }
        self.temporal.pending_time = self.temporal.start.elapsed();
        let (timing, timing_reset) = self.temporal.timing.sample(self.temporal.pending_time);
        self.temporal.pending_timing = timing;
        self.diagnostics.frame_delta_ms = timing.raw.as_secs_f32() * 1_000.0;
        if timing_reset.is_some() {
            self.temporal.reset_reason = GuidanceReset::LongPause;
            self.temporal.history.reset();
        }
        let valid = self.temporal.history.valid(self.temporal.pending_time);
        if self.temporal.motion.is_some() {
            self.diagnostics.state = if valid {
                "Estimated motion"
            } else {
                "History reset"
            }
            .into();
        }
        let guidance_view = match (&self.temporal.guidance, &self.temporal.motion) {
            (Some(guidance), Some(motion)) => Some(guidance.view(
                motion,
                self.temporal.history.frame_id + 1,
                self.temporal.resolution.processing_extent,
                valid,
                self.temporal.pending_timing,
                if valid {
                    GuidanceReset::None
                } else {
                    self.temporal.reset_reason
                },
            )),
            _ => None,
        };
        let guidance_view = guidance_view.and_then(|view| {
            let frame_extent = FrameExtent {
                width: self.temporal.resolution.processing_extent.width,
                height: self.temporal.resolution.processing_extent.height,
            };
            if view.is_valid_for(self.temporal.history.frame_id + 1, frame_extent) {
                Some(view)
            } else {
                eprintln!("TuxScaling: invalid guidance metadata; reconstruction bypassed");
                None
            }
        });
        let frame = self.overlay.as_mut().unwrap().prepare(
            queue,
            self.pool,
            index,
            &mut self.diagnostics,
        )?;
        if let Some(quality) = frame.requested_quality {
            self.temporal.pending_quality = Some(motion_quality(quality));
        }
        if let Some(scale) = frame.requested_processing_scale {
            self.temporal.pending_processing_scale = Some(scale);
        }
        unsafe {
            self.device
                .reset_command_buffer(slot.command, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                slot.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if self.temporal.queries != vk::QueryPool::null() {
                self.device.cmd_reset_query_pool(
                    slot.command,
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32,
                    GPU_TIMESTAMPS as u32,
                );
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32,
                );
            }
            if let Some(capture) = &mut self.temporal.capture {
                capture.record_scaled_from(
                    &self.device,
                    slot.command,
                    self.game_images[index],
                    self.temporal.resolution.game_extent,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    if self.game_images[index] == self.output_images[index] {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::PRESENT_SRC_KHR
                    },
                );
            } else {
                image_barrier(
                    &self.device,
                    slot.command,
                    self.output_images[index],
                    output_layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            if self.temporal.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32 + 1,
                );
            }
            if let Some(motion) = &mut self.temporal.motion {
                if self.temporal.queries != vk::QueryPool::null() {
                    motion.record_timed(
                        slot.command,
                        self.temporal.history.write_index(),
                        valid,
                        self.mode,
                        self.temporal.queries,
                        index as u32 * GPU_TIMESTAMPS as u32 + 2,
                    );
                } else {
                    motion.record(
                        slot.command,
                        self.temporal.history.write_index(),
                        valid,
                        self.mode,
                    );
                }
                memory_barrier(&self.device, slot.command);
            } else if self.temporal.queries != vk::QueryPool::null() {
                for offset in 2..=10 {
                    self.device.cmd_write_timestamp(
                        slot.command,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        self.temporal.queries,
                        index as u32 * GPU_TIMESTAMPS as u32 + offset,
                    );
                }
            }
            if let Some(guidance) = &mut self.temporal.guidance {
                if self.temporal.queries != vk::QueryPool::null() {
                    self.device.cmd_reset_query_pool(
                        slot.command,
                        self.temporal.queries,
                        index as u32 * GPU_TIMESTAMPS as u32 + 11,
                        3,
                    );
                    guidance.record_timed(
                        slot.command,
                        valid,
                        self.temporal.queries,
                        index as u32 * GPU_TIMESTAMPS as u32 + 11,
                    );
                } else {
                    guidance.record(slot.command, valid);
                }
                memory_barrier(&self.device, slot.command);
            } else if self.temporal.queries != vk::QueryPool::null() {
                for offset in 11..=13 {
                    self.device.cmd_write_timestamp(
                        slot.command,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        self.temporal.queries,
                        index as u32 * GPU_TIMESTAMPS as u32 + offset,
                    );
                }
            }
            if self.game_images[index] != self.output_images[index] {
                image_barrier(
                    &self.device,
                    slot.command,
                    self.output_images[index],
                    output_layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            if let (Some(upscaler), Some(guidance)) = (&mut self.temporal.upscaler, guidance_view) {
                upscaler.record(
                    slot.command,
                    self.output_images[index],
                    guidance,
                    valid,
                    self.temporal.history.write_index(),
                    index,
                    match self.mode {
                        6 => 1,
                        7 => 2,
                        8 => 3,
                        9 => 4,
                        10 => 5,
                        _ => 0,
                    },
                );
            } else if self.game_images[index] != self.output_images[index]
                && let Some(capture) = &self.temporal.capture
            {
                record_spatial_fallback(
                    &self.device,
                    slot.command,
                    capture,
                    self.output_images[index],
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    self.temporal.resolution.output_extent,
                );
            }
            if self.mode == 5
                && let Some(upscaler) = &self.temporal.upscaler
            {
                upscaler.record_debug(
                    slot.command,
                    self.output_images[index],
                    self.temporal.history.write_index(),
                );
            }
            if self.mode != 0
                && self.mode <= 3
                && let Some(motion) = &self.temporal.motion
            {
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
                    self.output_images[index],
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let layers = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);
                let region = vk::ImageBlit::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: motion.visualization.extent.width as i32,
                            y: motion.visualization.extent.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: self.info.extent.width as i32,
                            y: self.info.extent.height as i32,
                            z: 1,
                        },
                    ]);
                self.device.cmd_blit_image(
                    slot.command,
                    motion.visualization.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.output_images[index],
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
                    self.output_images[index],
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            if self.temporal.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32 + 14,
                );
            }
            self.overlay
                .as_mut()
                .unwrap()
                .record(slot.command, index, &frame)?;
            image_barrier(
                &self.device,
                slot.command,
                self.output_images[index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            if self.temporal.queries != vk::QueryPool::null() {
                self.device.cmd_write_timestamp(
                    slot.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.temporal.queries,
                    index as u32 * GPU_TIMESTAMPS as u32 + 15,
                );
            }
            self.device.end_command_buffer(slot.command)?;
        }
        if self.game_images[index] != self.output_images[index] {
            self.pending_output = Some(index);
        }
        self.temporal.query_ready[index] = true;
        Ok(FrameSubmission {
            command_buffer: slot.command,
            fence: slot.fence,
            render_complete: slot.semaphore,
        })
    }

    pub unsafe fn prepare_spatial_fallback(
        &mut self,
        queue: vk::Queue,
        family: u32,
        index: u32,
        reason: vk::Result,
    ) -> Result<FrameSubmission, vk::Result> {
        if !self.enabled {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        unsafe { self.initialize(queue, family) }?;
        unsafe { self.apply_pending_processing_scale() }?;
        let index = index as usize;
        let output_layout = if self.output_presented[index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let slot = self
            .slots
            .get(index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        unsafe { self.device.wait_for_fences(&[slot.fence], true, u64::MAX) }?;
        eprintln!("TuxScaling: temporal processing failed ({reason:?}); using spatial fallback");
        self.temporal.history.reset();
        self.temporal.reset_reason = GuidanceReset::ProviderFailure;
        if let Some(upscaler) = &mut self.temporal.upscaler {
            upscaler.reset();
        }
        self.diagnostics.state = "Temporal failure; spatial fallback".into();
        let frame = self
            .overlay
            .as_mut()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
            .prepare(queue, self.pool, index, &mut self.diagnostics)?;
        if let Some(quality) = frame.requested_quality {
            self.temporal.pending_quality = Some(motion_quality(quality));
        }
        if let Some(scale) = frame.requested_processing_scale {
            self.temporal.pending_processing_scale = Some(scale);
        }
        unsafe {
            self.device
                .reset_command_buffer(slot.command, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                slot.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if self.game_images[index] != self.output_images[index] {
                record_spatial_blit(
                    &self.device,
                    slot.command,
                    self.game_images[index],
                    self.temporal.resolution.game_extent,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    self.output_images[index],
                    output_layout,
                    self.temporal.resolution.output_extent,
                );
            } else {
                image_barrier(
                    &self.device,
                    slot.command,
                    self.output_images[index],
                    output_layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            self.overlay
                .as_mut()
                .unwrap()
                .record(slot.command, index, &frame)?;
            image_barrier(
                &self.device,
                slot.command,
                self.output_images[index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
            self.device.end_command_buffer(slot.command)?;
        }
        if self.game_images[index] != self.output_images[index] {
            self.pending_output = Some(index);
        }
        self.temporal.query_ready[index] = false;
        Ok(FrameSubmission {
            command_buffer: slot.command,
            fence: slot.fence,
            render_complete: slot.semaphore,
        })
    }

    pub fn submitted(&mut self) {
        self.temporal.history.commit(self.temporal.pending_time);
        if let Some(index) = self.pending_output.take() {
            self.output_presented[index] = true;
        }
        self.diagnostics.frame_id = self.temporal.history.frame_id;
    }
    pub fn presentation_failed(&mut self) {
        self.temporal.history.reset();
        self.output_presented.fill(false);
    }
    pub unsafe fn destroy(self, _device: &ash::Device) {
        drop(self);
    }
}
impl Drop for SwapchainRuntime {
    fn drop(&mut self) {
        if !self.temporal.timings.is_empty() {
            let mut summary = String::new();
            for (axis, name) in [
                "capture",
                "luma",
                "pyramid",
                "forward",
                "backward",
                "confidence",
                "stats_partial",
                "scene",
                "invalidate",
                "reactive",
                "exposure",
                "reconstruction",
                "overlay",
            ]
            .iter()
            .enumerate()
            {
                let mut values = self
                    .temporal
                    .timings
                    .iter()
                    .map(|v| v[axis])
                    .collect::<Vec<_>>();
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
                self.temporal.timings.len(),
                summary
            );
        }
        drop(self.temporal.motion.take());
        drop(self.temporal.capture.take());
        drop(self.overlay.take());
        unsafe {
            for slot in &self.slots {
                self.device.destroy_semaphore(slot.semaphore, None);
                self.device.destroy_fence(slot.fence, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_query_pool(self.temporal.queries, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::debug_mode_id;

    #[test]
    fn assigns_stable_debug_mode_ids_for_guidance_views() {
        assert_eq!(debug_mode_id("original"), Some(0));
        assert_eq!(debug_mode_id("depth"), Some(8));
        assert_eq!(debug_mode_id("composition"), Some(9));
        assert_eq!(debug_mode_id("exposure"), Some(10));
        assert_eq!(debug_mode_id("unsupported"), None);
    }
}
