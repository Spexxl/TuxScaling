#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{Image, image_barrier, memory_barrier};

use crate::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResource,
    GuidanceView, JitterSample, MotionDirection, MotionUnits, SignalState, ValidRegion,
};

pub struct GuidanceEstimator {
    device: ash::Device,
    pub reactive: Image,
    pub disocclusion: Image,
    pub exposure: Image,
    pub depth: Image,
    pub transparency: Image,
    transparency_history: Image,
    sampler: vk::Sampler,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    initialized: bool,
    history_initialized: bool,
    provider_failure: bool,
    extent: vk::Extent2D,
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
        let storage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
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
            sampler: vk::Sampler::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline: vk::Pipeline::null(),
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
        let bindings = (0..12)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(if binding < 2 || binding == 11 {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    } else if binding == 7 || binding == 10 {
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
                    .max_sets(1)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 3,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 7,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 2,
                        },
                    ]),
                None,
            )
        }?;
        result.descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(result.descriptor_pool)
                    .set_layouts(&[result.descriptor_layout]),
            )
        }?[0];
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
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&sampled[0..1]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&sampled[1..2]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
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
        ] {
            let image = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image)],
                    &[],
                );
            }
        }
        let stats = [vk::DescriptorBufferInfo::default()
            .buffer(statistics_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(result.descriptor_set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&stats)],
                &[],
            );
        }
        let compact_stats = [vk::DescriptorBufferInfo::default()
            .buffer(compact_statistics_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(result.descriptor_set)
                    .dst_binding(10)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&compact_stats)],
                &[],
            );
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
        Ok(result)
    }

    pub unsafe fn record(&mut self, command: vk::CommandBuffer, valid: bool) {
        unsafe { self.record_with_timing(command, valid, FrameTiming::default()) };
    }

    pub unsafe fn record_with_timing(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
    ) {
        self.provider_failure = false;
        unsafe { self.record_inner(command, valid, timing, None) };
    }

    pub unsafe fn record_timed(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        query_pool: vk::QueryPool,
        query_base: u32,
    ) {
        self.provider_failure = false;
        unsafe {
            self.record_inner(
                command,
                valid,
                FrameTiming::default(),
                Some((query_pool, query_base)),
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
        self.provider_failure = false;
        unsafe { self.record_inner(command, valid, timing, Some((query_pool, query_base))) };
    }

    unsafe fn record_inner(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timing: FrameTiming,
        timestamps: Option<(vk::QueryPool, u32)>,
    ) {
        unsafe {
            if !self.initialized {
                for image in [
                    &self.reactive,
                    &self.disocclusion,
                    &self.exposure,
                    &self.depth,
                    &self.transparency,
                    &self.transparency_history,
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
                image_barrier(
                    &self.device,
                    command,
                    self.transparency_history.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                memory_barrier(&self.device, command);
            }
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_set],
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
                0,
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
            memory_barrier(&self.device, command);
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
            memory_barrier(&self.device, command);
            image_barrier(
                &self.device,
                command,
                self.transparency.handle,
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
                self.transparency.handle,
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
                self.transparency.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            memory_barrier(&self.device, command);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 2,
                );
            }
        }
        self.initialized = true;
        if valid && !self.provider_failure {
            self.history_initialized = true;
        }
    }

    /// Record coherent fallback values after a provider failure.
    pub unsafe fn record_provider_failure(&mut self, command: vk::CommandBuffer) {
        self.provider_failure = true;
        unsafe { self.record_inner(command, false, FrameTiming::default(), None) };
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
                || !matches!(reset, GuidanceReset::None),
        };
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
                if self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            reactive: resource(
                self.reactive.handle,
                self.reactive.view,
                vk::Format::R8_UNORM,
                if self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            exposure: resource(
                self.exposure.handle,
                self.exposure.view,
                vk::Format::R32_SFLOAT,
                if self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            depth: resource(
                self.depth.handle,
                self.depth.view,
                vk::Format::R32_SFLOAT,
                SignalState::ConstantFallback,
            ),
            transparency_composition: resource(
                self.transparency.handle,
                self.transparency.view,
                vk::Format::R8_UNORM,
                if self.provider_failure {
                    SignalState::ConstantFallback
                } else {
                    SignalState::Estimated
                },
            ),
            pre_exposure: 1.0,
            timing,
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::FlatFallback,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            requires_history_reset: metadata.requires_history_reset,
        }
    }
}

impl Drop for GuidanceEstimator {
    fn drop(&mut self) {
        unsafe {
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
