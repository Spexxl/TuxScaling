#![allow(dead_code)]

use crate::{BackendEnvironment, BackendError, BackendImage};
use ash::vk;
use std::io::Cursor;
use tuxscaling_temporal::{DepthSemantics, FrameExtent, GuidanceView};
use tuxscaling_vulkan::{Image, compute_memory_barrier, image_barrier};

pub(crate) struct FsrInputs {
    pub motion: BackendImage,
    pub depth: BackendImage,
    pub reactive: BackendImage,
    pub composition: BackendImage,
    pub exposure: BackendImage,
}

struct InputSlot {
    motion: Image,
    depth: Image,
    reactive: Image,
    composition: Image,
}

impl InputSlot {
    fn outputs(&self, extent: vk::Extent2D, exposure: BackendImage) -> FsrInputs {
        FsrInputs {
            motion: backend_image(
                self.motion.handle,
                self.motion.view,
                vk::Format::R16G16_SFLOAT,
                extent,
            ),
            depth: backend_image(
                self.depth.handle,
                self.depth.view,
                vk::Format::R32_SFLOAT,
                extent,
            ),
            reactive: backend_image(
                self.reactive.handle,
                self.reactive.view,
                vk::Format::R8_UNORM,
                extent,
            ),
            composition: backend_image(
                self.composition.handle,
                self.composition.view,
                vk::Format::R8_UNORM,
                extent,
            ),
            exposure,
        }
    }
}

pub(crate) struct FsrInputAdapter {
    device: ash::Device,
    extent: vk::Extent2D,
    slots: Vec<InputSlot>,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    initialized: Vec<bool>,
}

impl FsrInputAdapter {
    pub(crate) unsafe fn new(
        environment: &BackendEnvironment,
        extent: vk::Extent2D,
        image_count: usize,
    ) -> Result<Self, BackendError> {
        if extent.width == 0 || extent.height == 0 || image_count == 0 {
            return Err(BackendError::InvalidMetadata(
                "FidelityFX input adapter extent/slots",
            ));
        }

        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE;
        let mut slots = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            let motion = unsafe {
                Image::new(
                    &environment.device,
                    &environment.memory,
                    extent,
                    vk::Format::R16G16_SFLOAT,
                    usage,
                )
            }
            .map_err(|error| BackendError::Internal(format!("motion input image: {error:?}")))?;
            let depth = unsafe {
                Image::new(
                    &environment.device,
                    &environment.memory,
                    extent,
                    vk::Format::R32_SFLOAT,
                    usage,
                )
            }
            .map_err(|error| BackendError::Internal(format!("depth input image: {error:?}")))?;
            let reactive = unsafe {
                Image::new(
                    &environment.device,
                    &environment.memory,
                    extent,
                    vk::Format::R8_UNORM,
                    usage,
                )
            }
            .map_err(|error| BackendError::Internal(format!("reactive input image: {error:?}")))?;
            let composition = unsafe {
                Image::new(
                    &environment.device,
                    &environment.memory,
                    extent,
                    vk::Format::R8_UNORM,
                    usage,
                )
            }
            .map_err(|error| {
                BackendError::Internal(format!("composition input image: {error:?}"))
            })?;
            slots.push(InputSlot {
                motion,
                depth,
                reactive,
                composition,
            });
        }

        let device = environment.device.clone();
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("input sampler: {error:?}")))?;

        let bindings = (0..10)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(if binding < 6 {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    } else {
                        vk::DescriptorType::STORAGE_IMAGE
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("input descriptor layout: {error:?}")))?;
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(image_count as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 6 * image_count as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 4 * image_count as u32,
                        },
                    ]),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("input descriptor pool: {error:?}")))?;
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![descriptor_layout; image_count]),
            )
        }
        .map_err(|error| BackendError::Internal(format!("input descriptor sets: {error:?}")))?;

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 12,
                    }]),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("input pipeline layout: {error:?}")))?;
        let shader = include_bytes!(concat!(env!("OUT_DIR"), "/fidelityfx_input.spv"));
        let words = ash::util::read_spv(&mut Cursor::new(shader))
            .map_err(|_| BackendError::Internal("invalid FidelityFX input shader".into()))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|error| BackendError::Internal(format!("input shader module: {error:?}")))?;
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(pipeline_layout)],
                None,
            )
        };
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = pipeline
            .map_err(|(_, error)| BackendError::Internal(format!("input pipeline: {error:?}")))?
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Internal("input pipeline was not created".into()))?;

        Ok(Self {
            device,
            extent,
            slots,
            sampler,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            initialized: vec![false; image_count],
        })
    }

    pub(crate) fn outputs(&self, slot: usize, exposure: BackendImage) -> FsrInputs {
        self.slots[slot % self.slots.len()].outputs(self.extent, exposure)
    }

    pub(crate) unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        slot: usize,
        frame_id: u64,
        guidance: GuidanceView,
    ) -> Result<(), BackendError> {
        let extent = FrameExtent {
            width: self.extent.width,
            height: self.extent.height,
        };
        if !guidance.is_valid_for(frame_id, extent) {
            return Err(BackendError::InvalidMetadata("FidelityFX guidance view"));
        }
        let slot = slot % self.slots.len();
        let outputs = &self.slots[slot];
        let input_images = [
            guidance.motion,
            guidance.confidence,
            guidance.disocclusion,
            guidance.reactive,
            guidance.depth,
            guidance.transparency_composition,
        ];
        let output_views = [
            outputs.motion.view,
            outputs.depth.view,
            outputs.reactive.view,
            outputs.composition.view,
        ];
        let sampled = input_images.map(|image| {
            vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(image.view)
                .image_layout(vk::ImageLayout::GENERAL)
        });
        let storage = output_views.map(|view| {
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
        });
        let set = self.descriptor_sets[slot];
        let mut writes = Vec::with_capacity(10);
        for (binding, image) in sampled.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(image)),
            );
        }
        for (binding, image) in storage.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32 + 6)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(image)),
            );
        }
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        for image in [
            outputs.motion.handle,
            outputs.depth.handle,
            outputs.reactive.handle,
            outputs.composition.handle,
        ] {
            unsafe {
                image_barrier(
                    &self.device,
                    command,
                    image,
                    if self.initialized[slot] {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    vk::ImageLayout::GENERAL,
                );
            }
        }
        unsafe {
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let params = [
                self.extent.width,
                self.extent.height,
                u32::from(matches!(
                    guidance.depth_semantics,
                    DepthSemantics::RelativeNearIsOne
                )),
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
                self.extent.width.div_ceil(8),
                self.extent.height.div_ceil(8),
                1,
            );
            compute_memory_barrier(&self.device, command);
        }
        Ok(())
    }

    pub(crate) fn mark_initialized(&mut self, slot: usize) {
        let slot = slot % self.initialized.len();
        self.initialized[slot] = true;
    }

    pub(crate) fn reset(&mut self) {
        // The adapter is stateless. Native FSR history is reset separately.
    }
}

impl Drop for FsrInputAdapter {
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

fn backend_image(
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
) -> BackendImage {
    BackendImage {
        image,
        view,
        format,
        extent,
        layout: vk::ImageLayout::GENERAL,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_uses_the_canonical_guidance_sanitization_rules() {
        let shader = include_str!("../../../../shaders/upscaler/fidelityfx_input.comp");
        for expression in [
            "confidence > 0.05",
            "disocclusion * (1.0 - confidence)",
            "max(composition, disocclusion)",
            "clamp(texelFetch(depth_image, pixel, 0).r, 0.0, 1.0)",
            "fsr_depth_value = params.relative_depth != 0u",
        ] {
            assert!(
                shader.contains(expression),
                "missing adapter rule: {expression}"
            );
        }
    }
}
