#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{Image, image_barrier, memory_barrier};

use crate::{
    FrameExtent, GuidanceMetadata, GuidanceReset, GuidanceResource, GuidanceView, MotionDirection,
    MotionUnits, ValidRegion,
};

pub struct GuidanceEstimator {
    device: ash::Device,
    pub reactive: Image,
    pub disocclusion: Image,
    pub exposure: Image,
    pub depth: Image,
    sampler: vk::Sampler,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    initialized: bool,
    extent: vk::Extent2D,
}

impl GuidanceEstimator {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        current_view: vk::ImageView,
        previous_view: vk::ImageView,
        confidence_view: vk::ImageView,
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
            sampler: vk::Sampler::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline: vk::Pipeline::null(),
            initialized: false,
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
        let bindings = (0..7)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(if binding < 2 {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
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
                            descriptor_count: 2,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 5,
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
        result.layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[result.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 16,
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
        unsafe { self.record_inner(command, valid, None) };
    }

    pub unsafe fn record_timed(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        query_pool: vk::QueryPool,
        query_base: u32,
    ) {
        unsafe { self.record_inner(command, valid, Some((query_pool, query_base))) };
    }

    unsafe fn record_inner(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        timestamps: Option<(vk::QueryPool, u32)>,
    ) {
        unsafe {
            if !self.initialized {
                for image in [
                    &self.reactive,
                    &self.disocclusion,
                    &self.exposure,
                    &self.depth,
                ] {
                    image_barrier(
                        &self.device,
                        command,
                        image.handle,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    );
                }
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
            let mut params = [self.extent.width, self.extent.height, u32::from(valid), 0];
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
            params[2] = u32::from(valid);
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
    }

    pub fn view(
        &self,
        motion: &MotionEstimator,
        frame_id: u64,
        extent: vk::Extent2D,
        valid: bool,
        reset: GuidanceReset,
    ) -> GuidanceView {
        let extent = FrameExtent {
            width: extent.width,
            height: extent.height,
        };
        let metadata = GuidanceMetadata {
            frame_id,
            extent,
            valid: true,
            valid_region: ValidRegion::full(extent),
            reset,
            is_zero: !valid,
            requires_history_reset: !valid || !matches!(reset, GuidanceReset::None),
        };
        let resource =
            |image: vk::Image, view: vk::ImageView, format: vk::Format| GuidanceResource {
                image,
                view,
                format,
                metadata,
            };
        GuidanceView {
            motion: resource(
                motion.vectors.handle,
                motion.vectors.view,
                vk::Format::R16G16_SFLOAT,
            ),
            confidence: resource(
                motion.confidence.handle,
                motion.confidence.view,
                vk::Format::R8_UNORM,
            ),
            disocclusion: resource(
                self.disocclusion.handle,
                self.disocclusion.view,
                vk::Format::R8_UNORM,
            ),
            reactive: resource(
                self.reactive.handle,
                self.reactive.view,
                vk::Format::R8_UNORM,
            ),
            exposure: resource(
                self.exposure.handle,
                self.exposure.view,
                vk::Format::R32_SFLOAT,
            ),
            depth: resource(self.depth.handle, self.depth.view, vk::Format::R32_SFLOAT),
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
