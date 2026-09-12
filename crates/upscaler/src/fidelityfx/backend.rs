use super::{
    FsrInputAdapter, NativeContext, TUX_FFX_CREATE_DEBUG_CHECKING, TUX_FFX_CREATE_DEPTH_INVERTED,
    TUX_FFX_CREATE_NON_LINEAR_COLORSPACE, TUX_FFX_IMAGE_STATE_COMPUTE_READ,
    TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS, TUX_FFX_IMAGE_USAGE_READ_ONLY, TUX_FFX_IMAGE_USAGE_UAV,
    TuxFfxCreateInfo, TuxFfxDispatchInfo,
};
use crate::{
    BackendCapabilities, BackendColorEncoding, BackendConfig, BackendEnvironment, BackendError,
    BackendFrame, BackendId, BackendImage, UpscalerBackend,
};
use ash::vk;
use ash::vk::Handle;
use std::io::Cursor;
use tuxscaling_temporal::{FrameExtent, GuidanceView};
use tuxscaling_vulkan::{Image, image_barrier};

const FSR_COLOR_FORMATS: &[vk::Format] = &[
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::R8G8B8A8_SRGB,
    vk::Format::B8G8R8A8_SRGB,
    vk::Format::R16G16B16A16_SFLOAT,
];

const INTERNAL_FORMATS: &[vk::Format] = &[
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R16G16_SFLOAT,
    vk::Format::R32_SFLOAT,
    vk::Format::R8_UNORM,
];

pub struct Fsr314Upscaler {
    device: ash::Device,
    native: NativeContext,
    input: FsrInputAdapter,
    outputs: Vec<Image>,
    output_initialized: Vec<bool>,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    config: BackendConfig,
}

impl Fsr314Upscaler {
    /// Creates an FSR 3.1.4 backend for the supplied Vulkan device.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `environment` refers to live Vulkan
    /// instance/device handles and that no other thread uses those handles
    /// while construction records or submits backend resources.
    pub unsafe fn new(
        environment: &BackendEnvironment,
        config: BackendConfig,
        guidance: GuidanceView,
        image_count: usize,
    ) -> Result<Self, BackendError> {
        environment.validate()?;
        if environment.vulkan_api_version < vk::API_VERSION_1_2 {
            return Err(BackendError::Unavailable);
        }
        let physical_properties = unsafe {
            environment
                .instance
                .get_physical_device_properties(environment.physical_device)
        };
        if physical_properties.api_version < vk::API_VERSION_1_2 {
            return Err(BackendError::Unavailable);
        }
        if image_count == 0 {
            return Err(BackendError::InvalidMetadata("FidelityFX frame slots"));
        }
        let capabilities = Self::capabilities_static();
        config.validate(capabilities)?;
        if config.color_encoding != BackendColorEncoding::SrgbNonlinear {
            return Err(BackendError::InvalidMetadata("FidelityFX color encoding"));
        }
        let game_extent = FrameExtent {
            width: config.game_extent.width,
            height: config.game_extent.height,
        };
        if !guidance.is_valid_for(guidance.motion.metadata.frame_id, game_extent) {
            return Err(BackendError::InvalidMetadata("FidelityFX initial guidance"));
        }

        let features = unsafe {
            environment
                .instance
                .get_physical_device_features(environment.physical_device)
        };
        if features.shader_storage_image_write_without_format == 0 {
            return Err(BackendError::Unavailable);
        }
        for &format in INTERNAL_FORMATS {
            let format_properties = unsafe {
                environment
                    .instance
                    .get_physical_device_format_properties(environment.physical_device, format)
            };
            let required =
                vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::STORAGE_IMAGE;
            if !format_properties.optimal_tiling_features.contains(required) {
                return Err(BackendError::UnsupportedFormat {
                    role: "FidelityFX internal",
                    format,
                });
            }
        }
        let output_properties = unsafe {
            environment.instance.get_physical_device_format_properties(
                environment.physical_device,
                config.output_format,
            )
        };
        if !output_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "FidelityFX output",
                format: config.output_format,
            });
        }
        let source_properties = unsafe {
            environment.instance.get_physical_device_format_properties(
                environment.physical_device,
                config.source_format,
            )
        };
        if !source_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "FidelityFX source",
                format: config.source_format,
            });
        }

        let content_extent = content_extent(config);
        let output_usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let mut outputs = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            outputs.push(
                unsafe {
                    Image::new(
                        &environment.device,
                        &environment.memory,
                        content_extent,
                        vk::Format::R16G16B16A16_SFLOAT,
                        output_usage,
                    )
                }
                .map_err(|error| {
                    BackendError::Internal(format!("FidelityFX output image: {error:?}"))
                })?,
            );
        }

        let input = unsafe { FsrInputAdapter::new(environment, config.game_extent, image_count) }?;
        let library = super::FidelityFxLibrary::load_bundled()?;
        let create_info = TuxFfxCreateInfo {
            physical_device: environment.physical_device.as_raw(),
            device: environment.device.handle().as_raw(),
            get_device_proc_addr: environment.get_device_proc_addr as usize as u64,
            enumerate_device_extension_properties: environment
                .instance
                .fp_v1_0()
                .enumerate_device_extension_properties
                as usize as u64,
            get_physical_device_features: environment
                .instance
                .fp_v1_0()
                .get_physical_device_features as usize
                as u64,
            get_physical_device_features2: environment
                .instance
                .fp_v1_1()
                .get_physical_device_features2 as usize
                as u64,
            get_physical_device_memory_properties: environment
                .instance
                .fp_v1_0()
                .get_physical_device_memory_properties
                as usize as u64,
            get_physical_device_properties: environment
                .instance
                .fp_v1_0()
                .get_physical_device_properties as usize
                as u64,
            get_physical_device_properties2: environment
                .instance
                .fp_v1_1()
                .get_physical_device_properties2
                as usize as u64,
            max_render_width: config.game_extent.width,
            max_render_height: config.game_extent.height,
            max_output_width: content_extent.width,
            max_output_height: content_extent.height,
            flags: TUX_FFX_CREATE_DEPTH_INVERTED
                | TUX_FFX_CREATE_NON_LINEAR_COLORSPACE
                | TUX_FFX_CREATE_DEBUG_CHECKING,
            vulkan_api_version: environment.vulkan_api_version,
        };
        let native = NativeContext::create(library, create_info)?;

        let device = environment.device.clone();
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|error| BackendError::Internal(format!("FidelityFX output sampler: {error:?}")))?;
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
            BackendError::Internal(format!("FidelityFX output descriptor layout: {error:?}"))
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
            BackendError::Internal(format!("FidelityFX output descriptor pool: {error:?}"))
        })?;
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![descriptor_layout; image_count]),
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX output descriptor sets: {error:?}"))
        })?;
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 24,
                    }]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("FidelityFX output pipeline layout: {error:?}"))
        })?;
        let shader = include_bytes!(concat!(env!("OUT_DIR"), "/fidelityfx_output.spv"));
        let words = ash::util::read_spv(&mut Cursor::new(shader))
            .map_err(|_| BackendError::Internal("invalid FidelityFX output shader".into()))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|error| BackendError::Internal(format!("FidelityFX output shader: {error:?}")))?;
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
                BackendError::Internal(format!("FidelityFX output pipeline: {error:?}"))
            })?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BackendError::Internal("FidelityFX output pipeline was not created".into())
            })?;

        Ok(Self {
            device,
            native,
            input,
            outputs,
            output_initialized: vec![false; image_count],
            sampler,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            config,
        })
    }

    fn capabilities_static() -> BackendCapabilities {
        BackendCapabilities {
            temporal: true,
            frame_generation: false,
            required_guidance: [true; 7],
            supported_source_formats: FSR_COLOR_FORMATS,
            supported_output_formats: FSR_COLOR_FORMATS,
        }
    }

    fn record_output(&mut self, frame: BackendFrame, slot: usize, fsr_output_view: vk::ImageView) {
        let sampled = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(fsr_output_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let storage = vk::DescriptorImageInfo::default()
            .image_view(frame.output.view)
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
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
            image_barrier(
                &self.device,
                frame.command_buffer,
                frame.output.image,
                frame.output.layout,
                vk::ImageLayout::GENERAL,
            );
            self.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let params = [
                frame.output.extent.width,
                frame.output.extent.height,
                self.config.viewport.offset[0].to_bits(),
                self.config.viewport.offset[1].to_bits(),
                self.config.viewport.size[0].to_bits(),
                self.config.viewport.size[1].to_bits(),
            ];
            self.device.cmd_push_constants(
                frame.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            self.device.cmd_dispatch(
                frame.command_buffer,
                frame.output.extent.width.div_ceil(8),
                frame.output.extent.height.div_ceil(8),
                1,
            );
            image_barrier(
                &self.device,
                frame.command_buffer,
                frame.output.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
    }
}

impl UpscalerBackend for Fsr314Upscaler {
    fn id(&self) -> BackendId {
        BackendId::Fsr314
    }

    fn capabilities(&self) -> BackendCapabilities {
        Self::capabilities_static()
    }

    fn configure(&mut self, config: BackendConfig) -> Result<(), BackendError> {
        config.validate(self.capabilities())?;
        if config.game_extent != self.config.game_extent
            || content_extent(config) != content_extent(self.config)
        {
            return Err(BackendError::IncompatibleExtent {
                role: "FidelityFX configuration",
                expected: self.config.output_extent,
                actual: config.output_extent,
            });
        }
        self.config = config;
        Ok(())
    }

    unsafe fn record(&mut self, frame: BackendFrame) -> Result<(), BackendError> {
        frame.validate(self.config, self.capabilities())?;
        let slot = frame.slot % self.outputs.len();
        let guidance = frame.guidance;
        let inputs = self.input.outputs(
            slot,
            BackendImage {
                image: guidance.exposure.image,
                view: guidance.exposure.view,
                format: guidance.exposure.format,
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                layout: vk::ImageLayout::GENERAL,
            },
        );
        unsafe {
            self.input
                .record(frame.command_buffer, slot, frame.frame_id, guidance)?;
        }

        let output_was_initialized = self.output_initialized[slot];
        if !output_was_initialized {
            unsafe {
                image_barrier(
                    &self.device,
                    frame.command_buffer,
                    self.outputs[slot].handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
        }
        let dispatch = TuxFfxDispatchInfo {
            command_buffer: frame.command_buffer.as_raw(),
            color: image_info(
                frame.source,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_COMPUTE_READ,
            ),
            depth: image_info(
                inputs.depth,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS,
            ),
            motion: image_info(
                inputs.motion,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS,
            ),
            exposure: image_info(
                inputs.exposure,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS,
            ),
            reactive: image_info(
                inputs.reactive,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS,
            ),
            composition: image_info(
                inputs.composition,
                TUX_FFX_IMAGE_USAGE_READ_ONLY,
                TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS,
            ),
            output: image_info(
                backend_image_from_owned(&self.outputs[slot], content_extent(self.config)),
                TUX_FFX_IMAGE_USAGE_UAV,
                if output_was_initialized {
                    TUX_FFX_IMAGE_STATE_COMPUTE_READ
                } else {
                    TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS
                },
            ),
            jitter_x: frame.guidance.jitter.current[0],
            jitter_y: frame.guidance.jitter.current[1],
            motion_scale_x: 1.0,
            motion_scale_y: 1.0,
            frame_time_ms: frame.guidance.timing.validated.as_secs_f32() * 1_000.0,
            pre_exposure: 1.0,
            camera_near: 0.1,
            camera_far: 1_000.0,
            camera_fov_y: 60.0_f32.to_radians(),
            view_space_to_meters: 1.0,
            render_width: self.config.game_extent.width,
            render_height: self.config.game_extent.height,
            output_width: content_extent(self.config).width,
            output_height: content_extent(self.config).height,
            reset: u32::from(frame.reset_history || frame.guidance.requires_history_reset),
        };
        unsafe { self.native.dispatch(&dispatch)? };
        if !output_was_initialized {
            unsafe {
                image_barrier(
                    &self.device,
                    frame.command_buffer,
                    self.outputs[slot].handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        }
        let fsr_output_view = self.outputs[slot].view;
        self.record_output(frame, slot, fsr_output_view);
        self.output_initialized[slot] = true;
        self.input.mark_initialized(slot);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), BackendError> {
        self.native.reset()?;
        self.input.reset();
        Ok(())
    }
}

impl Drop for Fsr314Upscaler {
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

fn image_info(image: BackendImage, usage: u32, state: u32) -> super::TuxFfxImage {
    super::TuxFfxImage {
        image: image.image.as_raw(),
        format: image.format.as_raw() as u32,
        width: image.extent.width,
        height: image.extent.height,
        usage,
        state,
    }
}

fn backend_image_from_owned(image: &Image, extent: vk::Extent2D) -> BackendImage {
    BackendImage {
        image: image.handle,
        view: image.view,
        format: image.format,
        extent,
        layout: vk::ImageLayout::GENERAL,
    }
}

fn content_extent(config: BackendConfig) -> vk::Extent2D {
    vk::Extent2D {
        width: ((config.output_extent.width as f32 * config.viewport.size[0]).round() as u32)
            .max(1),
        height: ((config.output_extent.height as f32 * config.viewport.size[1]).round() as u32)
            .max(1),
    }
}
