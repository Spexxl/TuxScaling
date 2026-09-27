use crate::{BackendColorEncoding, BackendEnvironment, BackendError, BackendImage};
use ash::vk;
use std::io::Cursor;
use tuxscaling_vulkan::{Image, image_barrier};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsrOutputEncoding {
    SrgbNonlinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsrColorPlan {
    pub decode_in_shader: bool,
    pub sampled_view_decodes_srgb: bool,
    pub intermediate_format: vk::Format,
    pub output_encoding: FsrOutputEncoding,
}

impl FsrColorPlan {
    pub fn for_source(
        format: vk::Format,
        encoding: BackendColorEncoding,
    ) -> Result<Self, BackendError> {
        if encoding != BackendColorEncoding::SrgbNonlinear {
            return Err(BackendError::InvalidMetadata(
                "FidelityFX requires captured SDR sRGB color",
            ));
        }
        if !matches!(
            format,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::R16G16B16A16_SFLOAT
        ) {
            return Err(BackendError::UnsupportedFormat {
                role: "FidelityFX color source",
                format,
            });
        }
        let sampled_view_decodes_srgb = matches!(
            format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        );
        Ok(Self {
            decode_in_shader: !sampled_view_decodes_srgb,
            sampled_view_decodes_srgb,
            intermediate_format: vk::Format::R16G16B16A16_SFLOAT,
            output_encoding: FsrOutputEncoding::SrgbNonlinear,
        })
    }
}

pub(crate) struct FsrColorAdapter {
    device: ash::Device,
    plan: FsrColorPlan,
    outputs: Vec<Image>,
    initialized: Vec<bool>,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl FsrColorAdapter {
    pub(crate) unsafe fn new(
        environment: &BackendEnvironment,
        extent: vk::Extent2D,
        image_count: usize,
        plan: FsrColorPlan,
    ) -> Result<Self, BackendError> {
        if extent.width == 0 || extent.height == 0 || image_count == 0 {
            return Err(BackendError::InvalidMetadata(
                "FidelityFX color adapter extent/slots",
            ));
        }
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut outputs = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            outputs.push(
                unsafe {
                    Image::new(
                        &environment.device,
                        &environment.memory,
                        extent,
                        plan.intermediate_format,
                        usage,
                    )
                }
                .map_err(|error| {
                    BackendError::Internal(format!("FidelityFX linear color image: {error:?}"))
                })?,
            );
        }

        let device = environment.device.clone();
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("FidelityFX color sampler: {error:?}")))?;
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX color descriptor layout: {error:?}"))
        })?;
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(image_count as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: image_count as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: image_count as u32,
                        },
                    ]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX color descriptor pool: {error:?}"))
        })?;
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![descriptor_layout; image_count]),
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX color descriptor sets: {error:?}"))
        })?;
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
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX color pipeline layout: {error:?}"))
        })?;
        let shader = include_bytes!(concat!(env!("OUT_DIR"), "/fidelityfx_color.spv"));
        let words = ash::util::read_spv(&mut Cursor::new(shader))
            .map_err(|_| BackendError::Internal("invalid FidelityFX color shader".into()))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX color shader module: {error:?}"))
        })?;
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
            .map_err(|(_, error)| {
                BackendError::Internal(format!("FidelityFX color pipeline: {error:?}"))
            })?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BackendError::Internal("FidelityFX color pipeline was not created".into())
            })?;

        Ok(Self {
            device,
            plan,
            outputs,
            initialized: vec![false; image_count],
            sampler,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
        })
    }

    pub(crate) fn plan(&self) -> FsrColorPlan {
        self.plan
    }

    pub(crate) unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        slot: usize,
        source: BackendImage,
    ) -> BackendImage {
        let slot = slot % self.outputs.len();
        let sampled = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(source.view)
            .image_layout(source.layout);
        let storage = vk::DescriptorImageInfo::default()
            .image_view(self.outputs[slot].view)
            .image_layout(vk::ImageLayout::GENERAL);
        let set = self.descriptor_sets[slot];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&sampled)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&storage)),
        ];
        let previous_layout = if self.initialized[slot] {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
            image_barrier(
                &self.device,
                command,
                self.outputs[slot].handle,
                previous_layout,
                vk::ImageLayout::GENERAL,
            );
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
            let push = [
                source.extent.width,
                source.extent.height,
                u32::from(self.plan.decode_in_shader),
            ];
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push),
            );
            self.device.cmd_dispatch(
                command,
                source.extent.width.div_ceil(8),
                source.extent.height.div_ceil(8),
                1,
            );
            image_barrier(
                &self.device,
                command,
                self.outputs[slot].handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }
        self.initialized[slot] = true;
        BackendImage {
            image: self.outputs[slot].handle,
            view: self.outputs[slot].view,
            format: self.outputs[slot].format,
            extent: self.outputs[slot].extent,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}

impl Drop for FsrColorAdapter {
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
    use super::{FsrColorPlan, FsrOutputEncoding};
    use crate::BackendColorEncoding;
    use ash::vk;

    #[test]
    fn srgb_unorm_and_srgb_view_agree() {
        let unorm = FsrColorPlan::for_source(
            vk::Format::R8G8B8A8_UNORM,
            BackendColorEncoding::SrgbNonlinear,
        )
        .unwrap();
        let srgb = FsrColorPlan::for_source(
            vk::Format::R8G8B8A8_SRGB,
            BackendColorEncoding::SrgbNonlinear,
        )
        .unwrap();
        assert!(unorm.decode_in_shader);
        assert!(!unorm.sampled_view_decodes_srgb);
        assert!(!srgb.decode_in_shader);
        assert!(srgb.sampled_view_decodes_srgb);
        assert_eq!(unorm.intermediate_format, vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(unorm.output_encoding, FsrOutputEncoding::SrgbNonlinear);
        assert_eq!(unorm.output_encoding, srgb.output_encoding);
    }

    #[test]
    fn linear_color_is_not_decoded_twice() {
        let plan = FsrColorPlan::for_source(
            vk::Format::B8G8R8A8_SRGB,
            BackendColorEncoding::SrgbNonlinear,
        )
        .unwrap();
        assert!(plan.sampled_view_decodes_srgb);
        assert!(!plan.decode_in_shader);
    }

    #[test]
    fn unsupported_color_space_is_rejected() {
        assert!(
            FsrColorPlan::for_source(
                vk::Format::R8G8B8A8_UNORM,
                BackendColorEncoding::Unsupported(vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT),
            )
            .is_err()
        );
    }
}
