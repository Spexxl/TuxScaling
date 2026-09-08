#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{
    Buffer, Image, compute_memory_barrier, image_barrier, transfer_memory_barrier,
};

use crate::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResource, GuidanceScalar, GuidanceView, JitterSample, MotionDirection, MotionUnits,
    SignalState, ValidRegion,
};

const DEPTH_RECORD_WORDS: u64 = 80;
const DEPTH_MODEL_WORDS: u64 = 32;

pub struct GuidanceEstimator {
    device: ash::Device,
    pub reactive: Image,
    pub disocclusion: Image,
    pub exposure: Image,
    pub depth: Image,
    pub transparency: Image,
    transparency_history: Image,
    transparency_residual: Image,
    depth_partials: Buffer,
    depth_models: Vec<Buffer>,
    sampler: vk::Sampler,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    depth_descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline: vk::Pipeline,
    depth_pipeline: vk::Pipeline,
    initialized: bool,
    history_initialized: bool,
    provider_failure: bool,
    extent: vk::Extent2D,
}

fn depth_group_stride(extent: vk::Extent2D) -> u32 {
    const MAX_DEPTH_GROUPS: u32 = 256;
    let groups = extent.width.div_ceil(8) * extent.height.div_ceil(8);
    groups.div_ceil(MAX_DEPTH_GROUPS).max(1)
}

impl GuidanceEstimator {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        current_view: vk::ImageView,
        previous_view: vk::ImageView,
        confidence_view: vk::ImageView,
        motion_view: vk::ImageView,
        statistics_buffer: vk::Buffer,
        compact_statistics_buffer: vk::Buffer,
    ) -> Result<Self, vk::Result> {
        unsafe {
            Self::new_with_slots(
                device,
                memory,
                extent,
                current_view,
                previous_view,
                confidence_view,
                motion_view,
                statistics_buffer,
                compact_statistics_buffer,
                1,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_with_slots(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        current_view: vk::ImageView,
        previous_view: vk::ImageView,
        confidence_view: vk::ImageView,
        motion_view: vk::ImageView,
        statistics_buffer: vk::Buffer,
        compact_statistics_buffer: vk::Buffer,
        slot_count: usize,
    ) -> Result<Self, vk::Result> {
        let slot_count = slot_count.max(1);
        let descriptor_count =
            u32::try_from(slot_count).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        let scaled_descriptor_count = |count: u32| {
            count
                .checked_mul(descriptor_count)
                .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        };
        let storage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let depth_groups =
            u64::from(extent.width.div_ceil(8)) * u64::from(extent.height.div_ceil(8));
        let depth_buffer_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let mut result = Self {
            device: device.clone(),
            reactive: unsafe { Image::new(device, memory, extent, vk::Format::R8_UNORM, storage) }?,
            disocclusion: unsafe {
                Image::new(device, memory, extent, vk::Format::R8_UNORM, storage)
            }?,
            exposure: unsafe {
                Image::new(
                    device,
                    memory,
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    vk::Format::R32_SFLOAT,
                    storage,
                )
            }?,
            depth: unsafe { Image::new(device, memory, extent, vk::Format::R32_SFLOAT, storage) }?,
            transparency: unsafe {
                Image::new(device, memory, extent, vk::Format::R8_UNORM, storage)
            }?,
            transparency_history: unsafe {
                Image::new(device, memory, extent, vk::Format::R8_UNORM, storage)
            }?,
            transparency_residual: unsafe {
                Image::new(device, memory, extent, vk::Format::R8_UNORM, storage)
            }?,
            depth_partials: unsafe {
                Buffer::new(
                    device,
                    memory,
                    depth_groups * DEPTH_RECORD_WORDS * 4,
                    depth_buffer_usage,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
            }?,
            depth_models: (0..slot_count)
                .map(|_| unsafe {
                    Buffer::new(
                        device,
                        memory,
                        DEPTH_MODEL_WORDS * 4,
                        depth_buffer_usage,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            sampler: vk::Sampler::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            depth_descriptor_sets: Vec::new(),
            pipeline: vk::Pipeline::null(),
            depth_pipeline: vk::Pipeline::null(),
            initialized: false,
            history_initialized: false,
            provider_failure: false,
            extent,
        };
        result.sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }?;
        let bindings = (0..=14)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(if binding < 2 || binding == 11 {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    } else if binding == 7 || binding == 10 || binding == 13 || binding == 14 {
                        vk::DescriptorType::STORAGE_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_IMAGE
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        result.descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        result.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(descriptor_count)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: scaled_descriptor_count(3)?,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: scaled_descriptor_count(8)?,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: scaled_descriptor_count(4)?,
                        },
                    ]),
                None,
            )
        }?;
        result.depth_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(result.descriptor_pool)
                    .set_layouts(&vec![result.descriptor_layout; slot_count]),
            )
        }?;
        let sampled = [
            vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(current_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(previous_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(result.transparency_history.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        for &descriptor_set in &result.depth_descriptor_sets {
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&sampled[0..1]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&sampled[1..2]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(11)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&sampled[2..3]),
                    ],
                    &[],
                );
            }
            for (binding, view) in [
                (2, confidence_view),
                (3, result.reactive.view),
                (4, result.disocclusion.view),
                (5, result.exposure.view),
                (6, result.depth.view),
                (8, motion_view),
                (9, result.transparency.view),
                (12, result.transparency_residual.view),
            ] {
                let image = [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&image)],
                        &[],
                    );
                }
            }
        }
        let stats = [vk::DescriptorBufferInfo::default()
            .buffer(statistics_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let compact_stats = [vk::DescriptorBufferInfo::default()
            .buffer(compact_statistics_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        for (descriptor_set, depth_model) in result
            .depth_descriptor_sets
            .iter()
            .zip(result.depth_models.iter())
        {
            let partials = [vk::DescriptorBufferInfo::default()
                .buffer(result.depth_partials.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let model = [vk::DescriptorBufferInfo::default()
                .buffer(depth_model.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*descriptor_set)
                            .dst_binding(7)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&stats),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*descriptor_set)
                            .dst_binding(10)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&compact_stats),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*descriptor_set)
                            .dst_binding(13)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&partials),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*descriptor_set)
                            .dst_binding(14)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&model),
                    ],
                    &[],
                );
            }
        }
        result.layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[result.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 32,
                    }]),
                None,
            )
        }?;
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/guidance.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(result.layout)],
                None,
            )
        };
        unsafe { device.destroy_shader_module(module, None) };
        result.pipeline = match pipeline {
            Ok(pipelines) => pipelines[0],
            Err((_, error)) => return Err(error),
        };
        let depth_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/depth_reduce.spv"));
        let depth_words = ash::util::read_spv(&mut std::io::Cursor::new(depth_bytes))
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let depth_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&depth_words),
                None,
            )
        }?;
        let depth_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(depth_module)
            .name(c"main");
        let depth_pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(depth_stage)
                    .layout(result.layout)],
                None,
            )
        };
        unsafe { device.destroy_shader_module(depth_module, None) };
        result.depth_pipeline = match depth_pipeline {
            Ok(pipelines) => pipelines[0],
            Err((_, error)) => return Err(error),
        };
        Ok(result)
    }

    pub unsafe fn record(&mut self, command: vk::CommandBuffer, valid: bool) {
        unsafe { self.record_with_timing_for_slot(command, valid, FrameTiming::default(), 0) };
    }

    pub unsafe fn record_with_timing(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
    ) {
        unsafe { self.record_with_timing_for_slot(command, valid, timing, 0) };
    }

    pub unsafe fn record_with_timing_for_slot(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
        slot: usize,
    ) {
        self.provider_failure = false;
        unsafe { self.record_inner(command, valid, timing, None, slot) };
    }

    pub unsafe fn record_timed(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        query_pool: vk::QueryPool,
        query_base: u32,
    ) {
        unsafe {
            self.record_timed_with_timing(
                command,
                valid,
                FrameTiming::default(),
                query_pool,
                query_base,
            )
        };
    }

    pub unsafe fn record_timed_with_timing(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
        query_pool: vk::QueryPool,
        query_base: u32,
    ) {
        unsafe {
            self.record_timed_with_timing_for_slot(
                command, valid, timing, query_pool, query_base, 0,
            )
        };
    }

    pub unsafe fn record_timed_with_timing_for_slot(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
        query_pool: vk::QueryPool,
        query_base: u32,
        slot: usize,
    ) {
        self.provider_failure = false;
        unsafe { self.record_inner(command, valid, timing, Some((query_pool, query_base)), slot) };
    }

    unsafe fn record_inner(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
        timestamps: Option<(vk::QueryPool, u32)>,
        slot: usize,
    ) {
        let slot = slot % self.depth_descriptor_sets.len();
        let depth_model = &self.depth_models[slot];
        unsafe {
            // Every depth phase consumes a complete, deterministic set of
            // records.  Clearing the device-local scratch buffers also makes
            // invalid/provider-failure frames safe before the first dispatch.
            self.device.cmd_fill_buffer(
                command,
                self.depth_partials.handle,
                0,
                self.depth_partials.size,
                0,
            );
            self.device
                .cmd_fill_buffer(command, depth_model.handle, 0, depth_model.size, 0);
            transfer_memory_barrier(&self.device, command);
            if !self.initialized {
                for image in [
                    &self.reactive,
                    &self.disocclusion,
                    &self.exposure,
                    &self.depth,
                    &self.transparency,
                    &self.transparency_history,
                    &self.transparency_residual,
                ] {
                    image_barrier(
                        &self.device,
                        command,
                        image.handle,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    );
                }
                self.device.cmd_clear_color_image(
                    command,
                    self.transparency.handle,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)],
                );
                self.device.cmd_clear_color_image(
                    command,
                    self.exposure.handle,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [1.0, 0.0, 0.0, 0.0],
                    },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)],
                );
                self.device.cmd_clear_color_image(
                    command,
                    self.transparency_history.handle,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)],
                );
                self.device.cmd_clear_color_image(
                    command,
                    self.transparency_residual.handle,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1)],
                );
                image_barrier(
                    &self.device,
                    command,
                    self.transparency_history.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                transfer_memory_barrier(&self.device, command);
            }
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.depth_descriptor_sets[slot]],
                &[],
            );
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base,
                );
            }
            let mut params = [
                self.extent.width,
                self.extent.height,
                u32::from(valid),
                0,
                timing.smoothed.as_secs_f32().to_bits(),
                u32::from(self.history_initialized),
                u32::from(self.provider_failure),
                depth_group_stride(self.extent),
            ];
            self.device.cmd_push_constants(
                command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            self.device.cmd_dispatch(
                command,
                self.extent.width.div_ceil(8),
                self.extent.height.div_ceil(8),
                1,
            );
            compute_memory_barrier(&self.device, command);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 1,
                );
            }
            params[3] = 1;
            self.device.cmd_push_constants(
                command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            self.device.cmd_dispatch(command, 1, 1, 1);
            compute_memory_barrier(&self.device, command);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 2,
                );
            }

            // Relative depth is deliberately a separate reduction pipeline:
            // first solve the compact affine model, then accumulate robust
            // residual percentiles, and finally normalize the output image.
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.depth_pipeline,
            );
            for (mode, width, height) in [
                (2u32, 1u32, 1u32),                            // compact affine solve
                (3u32, self.extent.width, self.extent.height), // residual histogram
                (4u32, 1u32, 1u32),                            // inlier/percentile final solve
                (5u32, self.extent.width, self.extent.height), // normalized depth
            ] {
                params[3] = mode;
                self.device.cmd_push_constants(
                    command,
                    self.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::cast_slice(&params),
                );
                self.device
                    .cmd_dispatch(command, width.div_ceil(8), height.div_ceil(8), 1);
                compute_memory_barrier(&self.device, command);
            }
            image_barrier(
                &self.device,
                command,
                self.transparency_residual.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                self.transparency_history.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.device.cmd_copy_image(
                command,
                self.transparency_residual.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.transparency_history.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                &self.device,
                command,
                self.transparency_history.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                self.transparency_residual.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            compute_memory_barrier(&self.device, command);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 3,
                );
            }
        }
        self.initialized = true;
        self.history_initialized = valid && !self.provider_failure;
    }

    /// Record coherent fallback values after a provider failure.
    pub unsafe fn record_provider_failure(&mut self, command: vk::CommandBuffer) {
        unsafe { self.record_provider_failure_for_slot(command, 0) };
    }

    pub unsafe fn record_provider_failure_for_slot(
        &mut self,
        command: vk::CommandBuffer,
        slot: usize,
    ) {
        self.provider_failure = true;
        self.history_initialized = false;
        unsafe { self.record_inner(command, false, FrameTiming::default(), None, slot) };
    }

    /// Clear the host-side history marker when an external reset invalidates
    /// the resources without recording a guidance dispatch.
    pub fn reset_history(&mut self) {
        self.history_initialized = false;
    }

    /// Mark the beginning of a new provider attempt after a fallback frame.
    /// The history marker remains cleared so the next successful dispatch is
    /// treated as a fresh temporal sample.
    pub fn clear_provider_failure(&mut self) {
        self.provider_failure = false;
    }

    pub fn view(
        &self,
        motion: &MotionEstimator,
        frame_id: u64,
        extent: vk::Extent2D,
        valid: bool,
        timing: FrameTiming,
        reset: GuidanceReset,
    ) -> GuidanceView {
        let extent = FrameExtent {
            width: extent.width,
            height: extent.height,
        };
        let reset = if self.provider_failure {
            GuidanceReset::ProviderFailure
        } else {
            reset
        };
        let metadata = GuidanceMetadata {
            frame_id,
            extent,
            valid: true,
            valid_region: ValidRegion::full(extent),
            reset,
            is_zero: !valid || self.provider_failure,
            requires_history_reset: !valid
                || self.provider_failure
                || !self.history_initialized
                || !matches!(reset, GuidanceReset::None),
        };
        // Support selection remains GPU-resident.  The host only reports that
        // a valid depth dispatch was recorded; the shader may still produce a
        // conservative exact flat-one image when its inlier thresholds are
        // not met.
        let depth_estimated = valid && !self.provider_failure;
        let resource =
            |image: vk::Image, view: vk::ImageView, format: vk::Format, state: SignalState| {
                GuidanceResource {
                    image,
                    view,
                    format,
                    metadata,
                    state,
                }
            };
        GuidanceView {
            motion: resource(
                motion.vectors.handle,
                motion.vectors.view,
                vk::Format::R16G16_SFLOAT,
                if valid && !self.provider_failure {
                    SignalState::Estimated
                } else {
                    SignalState::ConstantFallback
                },
            ),
            confidence: resource(
                motion.confidence.handle,
                motion.confidence.view,
                vk::Format::R8_UNORM,
                if valid && !self.provider_failure {
                    SignalState::Estimated
                } else {
                    SignalState::ConstantFallback
                },
            ),
            disocclusion: resource(
                self.disocclusion.handle,
                self.disocclusion.view,
                vk::Format::R8_UNORM,
                if !valid || self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            reactive: resource(
                self.reactive.handle,
                self.reactive.view,
                vk::Format::R8_UNORM,
                if !valid || self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            exposure: resource(
                self.exposure.handle,
                self.exposure.view,
                vk::Format::R32_SFLOAT,
                if !valid || self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            depth: resource(
                self.depth.handle,
                self.depth.view,
                vk::Format::R32_SFLOAT,
                if depth_estimated {
                    SignalState::Estimated
                } else {
                    SignalState::ConstantFallback
                },
            ),
            transparency_composition: resource(
                self.transparency.handle,
                self.transparency.view,
                vk::Format::R8_UNORM,
                if !valid || self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing,
            jitter: JitterSample::default(),
            depth_semantics: if depth_estimated {
                DepthSemantics::RelativeNearIsOne
            } else {
                DepthSemantics::FlatFallback
            },
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(extent, extent),
            requires_history_reset: metadata.requires_history_reset,
        }
    }
}

impl Drop for GuidanceEstimator {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.depth_pipeline, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::depth_group_stride;
    use ash::vk;

    #[test]
    fn depth_reduction_bounds_sampled_workgroups() {
        assert_eq!(
            depth_group_stride(vk::Extent2D {
                width: 64,
                height: 48,
            }),
            1
        );
        assert_eq!(
            depth_group_stride(vk::Extent2D {
                width: 1920,
                height: 1080,
            }),
            127
        );
    }
}
