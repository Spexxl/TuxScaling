use super::{ProtectionSettings, RISK_DECAY_SECONDS, record_spatial_blit};
use crate::{BackendError, BackendImage, ContentViewport, ProtectionInputs};
use ash::vk;
use std::{io::Cursor, time::Duration};
use tuxscaling_vulkan::{Image, color_range, compute_memory_barrier, image_barrier};

const RISK_PUSH_BYTES: u32 = 16;
const PROTECT_PUSH_WORDS: usize = 28;
const PROTECT_PUSH_BYTES: u32 = (PROTECT_PUSH_WORDS * size_of::<u32>()) as u32;

pub struct ProtectionFrame {
    pub command: vk::CommandBuffer,
    pub slot: usize,
    pub frame_id: u64,
    pub source: BackendImage,
    pub output: BackendImage,
    pub guidance: ProtectionInputs,
    pub viewport: ContentViewport,
    pub settings: ProtectionSettings,
    pub elapsed: Duration,
    pub reset: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ProtectionOutputs {
    pub fsr_raw: BackendImage,
    pub spatial_weight: BackendImage,
}

struct ComputePass {
    device: ash::Device,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ComputePass {
    unsafe fn new(
        device: &ash::Device,
        descriptors: &[vk::DescriptorType],
        push_bytes: u32,
        shader: &[u8],
        slots: usize,
    ) -> Result<Self, BackendError> {
        let mut pass = Self {
            device: device.clone(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        let bindings = descriptors
            .iter()
            .enumerate()
            .map(|(index, descriptor)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(index as u32)
                    .descriptor_type(*descriptor)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        pass.descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("protection descriptor layout: {error:?}"))
        })?;
        let sampled = descriptors
            .iter()
            .filter(|descriptor| **descriptor == vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .count() as u32;
        let storage = descriptors.len() as u32 - sampled;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: sampled * slots as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: storage * slots as u32,
            },
        ];
        pass.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(slots as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("protection descriptor pool: {error:?}"))
        })?;
        pass.sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pass.descriptor_pool)
                    .set_layouts(&vec![pass.descriptor_layout; slots]),
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("protection descriptor sets: {error:?}"))
        })?;
        pass.layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[pass.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: push_bytes,
                    }]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("protection pipeline layout: {error:?}"))
        })?;
        let words = ash::util::read_spv(&mut Cursor::new(shader))
            .map_err(|_| BackendError::Internal("invalid protection shader".into()))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|error| BackendError::Internal(format!("protection shader module: {error:?}")))?;
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
                    .layout(pass.layout)],
                None,
            )
        };
        unsafe { device.destroy_shader_module(module, None) };
        pass.pipeline = pipeline
            .map_err(|(_, error)| {
                BackendError::Internal(format!("protection pipeline: {error:?}"))
            })?
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Internal("protection pipeline missing".into()))?;
        Ok(pass)
    }

    unsafe fn bind_images(
        &self,
        slot: usize,
        sampler: vk::Sampler,
        images: &[(vk::ImageView, vk::ImageLayout)],
        sampled_count: usize,
    ) {
        let infos = images
            .iter()
            .enumerate()
            .map(|(index, (view, layout))| {
                vk::DescriptorImageInfo::default()
                    .sampler(if index < sampled_count {
                        sampler
                    } else {
                        vk::Sampler::null()
                    })
                    .image_view(*view)
                    .image_layout(*layout)
            })
            .collect::<Vec<_>>();
        let writes = infos
            .iter()
            .enumerate()
            .map(|(index, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[slot])
                    .dst_binding(index as u32)
                    .descriptor_type(if index < sampled_count {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    } else {
                        vk::DescriptorType::STORAGE_IMAGE
                    })
                    .image_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    unsafe fn dispatch(
        &self,
        command: vk::CommandBuffer,
        slot: usize,
        push: &[u32],
        extent: vk::Extent2D,
    ) {
        unsafe {
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[slot]],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(push),
            );
            self.device.cmd_dispatch(
                command,
                extent.width.div_ceil(8),
                extent.height.div_ceil(8),
                1,
            );
        }
    }
}

impl Drop for ComputePass {
    fn drop(&mut self) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            if self.layout != vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(self.layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
        }
    }
}

pub struct ProtectionRenderer {
    device: ash::Device,
    source_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    output_format: vk::Format,
    raw: Vec<Image>,
    spatial: Vec<Image>,
    weight: Vec<Image>,
    risk_history: Vec<Image>,
    initialized_slots: Vec<bool>,
    risk_initialized: bool,
    risk_write_index: usize,
    last_frame_id: Option<u64>,
    sampler: vk::Sampler,
    risk_pass: ComputePass,
    protect_pass: ComputePass,
}

impl ProtectionRenderer {
    /// # Safety
    ///
    /// `device` and `memory` must refer to the same live Vulkan device.
    /// The caller must destroy this renderer before destroying that device.
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        source_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        output_format: vk::Format,
        slots: usize,
    ) -> Result<Self, BackendError> {
        if source_extent.width == 0
            || source_extent.height == 0
            || output_extent.width == 0
            || output_extent.height == 0
            || slots == 0
        {
            return Err(BackendError::InvalidMetadata("protection extent or slots"));
        }
        let output_usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let mask_usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let mut raw = Vec::with_capacity(slots);
        let mut spatial = Vec::with_capacity(slots);
        let mut weight = Vec::with_capacity(slots);
        for _ in 0..slots {
            raw.push(
                unsafe { Image::new(device, memory, output_extent, output_format, output_usage) }
                    .map_err(|error| BackendError::Internal(format!("raw FSR image: {error:?}")))?,
            );
            spatial.push(unsafe { Image::new(device, memory, output_extent, output_format, output_usage) }
                .map_err(|error| BackendError::Internal(format!("spatial Off image: {error:?}")))?);
            weight.push(
                unsafe {
                    Image::new(
                        device,
                        memory,
                        output_extent,
                        vk::Format::R8_UNORM,
                        mask_usage,
                    )
                }
                .map_err(|error| {
                    BackendError::Internal(format!("protection weight image: {error:?}"))
                })?,
            );
        }
        let mut risk_history = Vec::with_capacity(2);
        for _ in 0..2 {
            risk_history.push(
                unsafe {
                    Image::new(
                        device,
                        memory,
                        source_extent,
                        vk::Format::R8_UNORM,
                        mask_usage,
                    )
                }
                .map_err(|error| {
                    BackendError::Internal(format!("protection risk image: {error:?}"))
                })?,
            );
        }
        let sampled = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let storage = vk::DescriptorType::STORAGE_IMAGE;
        let risk_pass = unsafe {
            ComputePass::new(
                device,
                &[sampled, sampled, sampled, storage],
                RISK_PUSH_BYTES,
                include_bytes!(concat!(env!("OUT_DIR"), "/risk_update.spv")),
                slots,
            )
        }?;
        let protect_pass = unsafe {
            ComputePass::new(
                device,
                &[sampled, sampled, sampled, storage, storage],
                PROTECT_PUSH_BYTES,
                include_bytes!(concat!(env!("OUT_DIR"), "/protect.spv")),
                slots,
            )
        }?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("protection sampler: {error:?}")))?;
        Ok(Self {
            device: device.clone(),
            source_extent,
            output_extent,
            output_format,
            raw,
            spatial,
            weight,
            risk_history,
            initialized_slots: vec![false; slots],
            risk_initialized: false,
            risk_write_index: 0,
            last_frame_id: None,
            sampler,
            risk_pass,
            protect_pass,
        })
    }

    pub fn reset(&mut self) {
        self.last_frame_id = None;
    }

    /// # Safety
    ///
    /// All frame images, views and the command buffer must belong to this
    /// renderer's device, and their declared layouts must be current.
    /// Reuse of a slot requires completion of its previous submission.
    pub unsafe fn record(
        &mut self,
        frame: ProtectionFrame,
    ) -> Result<ProtectionOutputs, BackendError> {
        if !frame.settings.is_enabled()
            || frame.command == vk::CommandBuffer::null()
            || frame.slot >= self.raw.len()
            || frame.source.extent != self.source_extent
            || frame.output.extent != self.output_extent
            || frame.output.format != self.output_format
            || frame.guidance.history_risk.extent != self.source_extent
            || frame.guidance.motion.extent != self.source_extent
            || frame.source.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            || frame.output.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            || frame.guidance.history_risk.layout != vk::ImageLayout::GENERAL
            || frame.guidance.motion.layout != vk::ImageLayout::GENERAL
        {
            return Err(BackendError::InvalidMetadata("protection frame"));
        }
        let command = frame.command;
        let slot = frame.slot;
        let initialized = self.initialized_slots[slot];
        let raw = &self.raw[slot];
        let spatial = &self.spatial[slot];
        let weight = &self.weight[slot];
        let risk_write = self.risk_write_index;
        let risk_read = 1 - risk_write;
        let reset = frame.reset
            || self.last_frame_id.and_then(|id| id.checked_add(1)) != Some(frame.frame_id);

        unsafe {
            if !self.risk_initialized {
                for image in &self.risk_history {
                    image_barrier(
                        &self.device,
                        command,
                        image.handle,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );
                    self.device.cmd_clear_color_image(
                        command,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                        &[color_range()],
                    );
                    image_barrier(
                        &self.device,
                        command,
                        image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                    );
                }
            } else {
                image_barrier(
                    &self.device,
                    command,
                    self.risk_history[risk_write].handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                );
                image_barrier(
                    &self.device,
                    command,
                    self.risk_history[risk_read].handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::GENERAL,
                );
            }
            self.risk_pass.bind_images(
                slot,
                self.sampler,
                &[
                    (frame.guidance.history_risk.view, vk::ImageLayout::GENERAL),
                    (self.risk_history[risk_read].view, vk::ImageLayout::GENERAL),
                    (frame.guidance.motion.view, vk::ImageLayout::GENERAL),
                    (self.risk_history[risk_write].view, vk::ImageLayout::GENERAL),
                ],
                3,
            );
            let decay = (-frame.elapsed.as_secs_f32() / RISK_DECAY_SECONDS).exp();
            self.risk_pass.dispatch(
                command,
                slot,
                &[
                    self.source_extent.width,
                    self.source_extent.height,
                    decay.to_bits(),
                    u32::from(reset),
                ],
                self.source_extent,
            );
            compute_memory_barrier(&self.device, command);

            image_barrier(
                &self.device,
                command,
                frame.output.image,
                frame.output.layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                raw.handle,
                if initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.device.cmd_copy_image(
                command,
                frame.output.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                raw.handle,
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
                        width: self.output_extent.width,
                        height: self.output_extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                &self.device,
                command,
                raw.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                frame.output.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.output.layout,
            );

            record_spatial_blit(
                &self.device,
                command,
                frame.source.image,
                self.source_extent,
                frame.source.layout,
                spatial.handle,
                if initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                self.output_extent,
            );
            image_barrier(
                &self.device,
                command,
                spatial.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                weight.handle,
                if initialized {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::GENERAL,
            );
            image_barrier(
                &self.device,
                command,
                frame.output.image,
                frame.output.layout,
                vk::ImageLayout::GENERAL,
            );
            self.protect_pass.bind_images(
                slot,
                self.sampler,
                &[
                    (raw.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    (spatial.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    (self.risk_history[risk_write].view, vk::ImageLayout::GENERAL),
                    (frame.output.view, vk::ImageLayout::GENERAL),
                    (weight.view, vk::ImageLayout::GENERAL),
                ],
                3,
            );
            let mut push = [0_u32; PROTECT_PUSH_WORDS];
            push[..10].copy_from_slice(&[
                self.output_extent.width,
                self.output_extent.height,
                frame.viewport.offset[0].to_bits(),
                frame.viewport.offset[1].to_bits(),
                frame.viewport.size[0].to_bits(),
                frame.viewport.size[1].to_bits(),
                (self.source_extent.width as f32).to_bits(),
                (self.source_extent.height as f32).to_bits(),
                frame.settings.mode as u32,
                frame.settings.regions().count() as u32,
            ]);
            for (index, region) in frame.settings.regions().enumerate() {
                let min = region.min();
                let max = region.max();
                push[12 + index * 4..16 + index * 4].copy_from_slice(&[
                    min[0].to_bits(),
                    min[1].to_bits(),
                    max[0].to_bits(),
                    max[1].to_bits(),
                ]);
            }
            self.protect_pass
                .dispatch(command, slot, &push, self.output_extent);
            image_barrier(
                &self.device,
                command,
                frame.output.image,
                vk::ImageLayout::GENERAL,
                frame.output.layout,
            );
        }
        self.risk_initialized = true;
        self.risk_write_index = risk_read;
        self.last_frame_id = Some(frame.frame_id);
        self.initialized_slots[slot] = true;
        Ok(ProtectionOutputs {
            fsr_raw: BackendImage {
                image: raw.handle,
                view: raw.view,
                format: raw.format,
                extent: raw.extent,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            spatial_weight: BackendImage {
                image: weight.handle,
                view: weight.view,
                format: weight.format,
                extent: weight.extent,
                layout: vk::ImageLayout::GENERAL,
            },
        })
    }
}

impl Drop for ProtectionRenderer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_sampler(self.sampler, None) };
    }
}
