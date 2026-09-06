#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_temporal::GuidanceView;
use tuxscaling_vulkan::{Image, image_barrier, memory_barrier};

pub fn scaled_extent(output: vk::Extent2D, scale: f32) -> vk::Extent2D {
    let scale = scale.clamp(0.5, 1.0);
    vk::Extent2D {
        width: ((output.width as f32 * scale).round() as u32).max(1),
        height: ((output.height as f32 * scale).round() as u32).max(1),
    }
}

pub struct ReferenceUpscaler {
    device: ash::Device,
    pub output: Image,
    pub history: [Image; 2],
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pub input_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
    initialized: bool,
}

impl ReferenceUpscaler {
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        input_view: vk::ImageView,
        input_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        output_format: vk::Format,
        guidance: GuidanceView,
        image_count: usize,
    ) -> Result<Self, vk::Result> {
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE;
        let mut result = Self {
            device: device.clone(),
            output: unsafe { Image::new(device, memory, output_extent, output_format, usage) }?,
            history: [
                unsafe { Image::new(device, memory, output_extent, output_format, usage) }?,
                unsafe { Image::new(device, memory, output_extent, output_format, usage) }?,
            ],
            sampler: vk::Sampler::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            input_extent,
            output_extent,
            initialized: false,
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
        let bindings = (0..10)
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
                    .max_sets(image_count.max(1) as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: (2 * image_count.max(1)) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: (8 * image_count.max(1)) as u32,
                        },
                    ]),
                None,
            )
        }?;
        result.descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(result.descriptor_pool)
                    .set_layouts(&vec![result.descriptor_layout; image_count.max(1)]),
            )
        }?;

        let sampled = [
            vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(input_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(result.history[0].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let storage = [
            (2, guidance.motion.view),
            (3, guidance.confidence.view),
            (4, guidance.reactive.view),
            (5, guidance.disocclusion.view),
            (6, guidance.exposure.view),
            (7, guidance.depth.view),
            (8, guidance.transparency_composition.view),
            (9, result.output.view),
        ];
        let mut writes = Vec::with_capacity(result.descriptor_sets.len() * (2 + storage.len()));
        for descriptor_set in &result.descriptor_sets {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&sampled[0..1]),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&sampled[1..2]),
            );
        }
        let storage_infos = storage
            .iter()
            .map(|(_, view)| {
                vk::DescriptorImageInfo::default()
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();
        for ((binding, _), image) in storage.iter().zip(storage_infos.iter()) {
            for descriptor_set in &result.descriptor_sets {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(*descriptor_set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(image)),
                );
            }
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        result.pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[result.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 36,
                    }]),
                None,
            )
        }?;
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/reconstruct.spv"));
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
                    .layout(result.pipeline_layout)],
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        swapchain: vk::Image,
        guidance: GuidanceView,
        valid: bool,
        history_write: usize,
        slot: usize,
        debug_view: u32,
    ) {
        let history_read = 1 - history_write;
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        unsafe {
            if !self.initialized || !valid {
                for history in &self.history {
                    image_barrier(
                        &self.device,
                        command,
                        history.handle,
                        if self.initialized {
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        } else {
                            vk::ImageLayout::UNDEFINED
                        },
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );
                    self.device.cmd_clear_color_image(
                        command,
                        history.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                        &[tuxscaling_vulkan::color_range()],
                    );
                    image_barrier(
                        &self.device,
                        command,
                        history.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                }
            }
            let sampled_history = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(self.history[history_read].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[slot % self.descriptor_sets.len()])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&sampled_history)],
                &[],
            );
            image_barrier(
                &self.device,
                command,
                self.output.handle,
                if self.initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::GENERAL,
            );
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[slot % self.descriptor_sets.len()]],
                &[],
            );
            let params = [
                self.output_extent.width,
                self.output_extent.height,
                self.input_extent.width,
                self.input_extent.height,
                guidance.motion.metadata.extent.width.div_ceil(4),
                guidance.motion.metadata.extent.height.div_ceil(4),
                u32::from(valid && !guidance.requires_history_reset),
                u32::from(guidance.requires_history_reset),
                debug_view,
            ];
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            self.device.cmd_dispatch(
                command,
                self.output_extent.width.div_ceil(8),
                self.output_extent.height.div_ceil(8),
                1,
            );
            memory_barrier(&self.device, command);
            image_barrier(
                &self.device,
                command,
                self.output.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                self.history[history_write].handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                swapchain,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let copy = vk::ImageCopy::default()
                .src_subresource(layers)
                .dst_subresource(layers)
                .extent(vk::Extent3D {
                    width: self.output_extent.width,
                    height: self.output_extent.height,
                    depth: 1,
                });
            self.device.cmd_copy_image(
                command,
                self.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.history[history_write].handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
            self.device.cmd_copy_image(
                command,
                self.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
            image_barrier(
                &self.device,
                command,
                self.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                self.history[history_write].handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                swapchain,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
        self.initialized = true;
    }

    pub unsafe fn record_debug(
        &self,
        command: vk::CommandBuffer,
        swapchain: vk::Image,
        history_write: usize,
    ) {
        let source = &self.history[1 - history_write];
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        unsafe {
            image_barrier(
                &self.device,
                command,
                source.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                swapchain,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.device.cmd_copy_image(
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: self.output_extent.width,
                        height: self.output_extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                &self.device,
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                swapchain,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

impl Drop for ReferenceUpscaler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
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
    use super::scaled_extent;
    use ash::vk;

    #[test]
    fn scaled_extent_clamps_and_rounds() {
        let output = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        assert_eq!(
            scaled_extent(output, 0.67),
            vk::Extent2D {
                width: 1286,
                height: 724
            }
        );
        assert_eq!(
            scaled_extent(output, 0.1),
            vk::Extent2D {
                width: 960,
                height: 540
            }
        );
        assert_eq!(scaled_extent(output, 1.2), output);
    }
}
