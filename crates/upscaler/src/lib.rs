use ash::vk;
use thiserror::Error;
use tuxscaling_temporal::{GuidanceCapabilities, GuidanceSignal, GuidanceView, SignalState};
use tuxscaling_vulkan::{Image, image_barrier};

mod reference;
pub use reference::{ReferenceUpscaler, scaled_extent};

#[cfg(feature = "fidelityfx")]
pub mod fidelityfx;

pub const CRATE_NAME: &str = "tuxscaling-upscaler";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendId {
    Reference,
    Fsr314,
}

impl BackendId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Fsr314 => "fsr_3_1_4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendColorEncoding {
    SrgbNonlinear,
    Unsupported(vk::ColorSpaceKHR),
}

impl From<vk::ColorSpaceKHR> for BackendColorEncoding {
    fn from(value: vk::ColorSpaceKHR) -> Self {
        if value == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            Self::SrgbNonlinear
        } else {
            Self::Unsupported(value)
        }
    }
}

#[derive(Clone)]
pub struct BackendEnvironment {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    pub vulkan_api_version: u32,
}

impl BackendEnvironment {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Self {
        Self::new_with_api_version(instance, physical_device, device, vk::API_VERSION_1_2)
    }

    pub fn new_with_api_version(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        vulkan_api_version: u32,
    ) -> Self {
        Self {
            instance: instance.clone(),
            physical_device,
            device: device.clone(),
            memory: unsafe { instance.get_physical_device_memory_properties(physical_device) },
            get_device_proc_addr: instance.fp_v1_0().get_device_proc_addr,
            vulkan_api_version,
        }
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if self.instance.handle() == vk::Instance::null()
            || self.physical_device == vk::PhysicalDevice::null()
            || self.device.handle() == vk::Device::null()
        {
            return Err(BackendError::InvalidMetadata("Vulkan backend environment"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub temporal: bool,
    pub frame_generation: bool,
    pub required_guidance: [bool; 7],
    pub supported_source_formats: &'static [vk::Format],
    pub supported_output_formats: &'static [vk::Format],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputResolution {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("upscaler backend is unavailable")]
    Unavailable,
    #[error("unsupported {role} format: {format:?}")]
    UnsupportedFormat {
        role: &'static str,
        format: vk::Format,
    },
    #[error("incompatible {role} extent: expected {expected:?}, got {actual:?}")]
    IncompatibleExtent {
        role: &'static str,
        expected: vk::Extent2D,
        actual: vk::Extent2D,
    },
    #[error("required guidance signal is unavailable: {signal:?}")]
    MissingSignal { signal: GuidanceSignal },
    #[error("invalid {role} image layout: {layout:?}")]
    InvalidLayout {
        role: &'static str,
        layout: vk::ImageLayout,
    },
    #[error("invalid backend metadata: {0}")]
    InvalidMetadata(&'static str),
    #[error("upscaler configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("upscaler backend failed: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy)]
pub struct ImageResource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub resolution: InputResolution,
}

#[derive(Debug, Clone, Copy)]
pub struct BackendImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct BackendConfig {
    pub game_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
    pub source_format: vk::Format,
    pub output_format: vk::Format,
    pub color_encoding: BackendColorEncoding,
    pub viewport: ContentViewport,
    pub guidance: GuidanceCapabilities,
}

impl BackendConfig {
    pub fn validate(self, capabilities: BackendCapabilities) -> Result<(), BackendError> {
        if !is_valid_extent(self.game_extent) {
            return Err(BackendError::IncompatibleExtent {
                role: "game",
                expected: self.game_extent,
                actual: self.game_extent,
            });
        }
        if !is_valid_extent(self.output_extent) {
            return Err(BackendError::IncompatibleExtent {
                role: "output",
                expected: self.output_extent,
                actual: self.output_extent,
            });
        }
        if !capabilities
            .supported_source_formats
            .contains(&self.source_format)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "source",
                format: self.source_format,
            });
        }
        if !capabilities
            .supported_output_formats
            .contains(&self.output_format)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "output",
                format: self.output_format,
            });
        }
        if !matches!(self.color_encoding, BackendColorEncoding::SrgbNonlinear) {
            return Err(BackendError::InvalidMetadata("color encoding"));
        }
        if !self.viewport.is_valid() {
            return Err(BackendError::InvalidMetadata("viewport"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BackendFrame {
    pub command_buffer: vk::CommandBuffer,
    pub slot: usize,
    pub source: BackendImage,
    pub output: BackendImage,
    pub guidance: GuidanceView,
    pub viewport: ContentViewport,
    pub frame_id: u64,
    pub reset_history: bool,
    pub debug_view: u32,
}

impl BackendFrame {
    pub fn validate(
        self,
        config: BackendConfig,
        capabilities: BackendCapabilities,
    ) -> Result<(), BackendError> {
        config.validate(capabilities)?;
        if self.command_buffer == vk::CommandBuffer::null()
            || self.source.image == vk::Image::null()
            || self.source.view == vk::ImageView::null()
            || self.output.image == vk::Image::null()
            || self.output.view == vk::ImageView::null()
        {
            return Err(BackendError::InvalidMetadata(
                "null command or image handle",
            ));
        }
        validate_extent("source", config.game_extent, self.source.extent)?;
        validate_extent("output", config.output_extent, self.output.extent)?;
        if self.source.format != config.source_format {
            return Err(BackendError::UnsupportedFormat {
                role: "source",
                format: self.source.format,
            });
        }
        if self.output.format != config.output_format {
            return Err(BackendError::UnsupportedFormat {
                role: "output",
                format: self.output.format,
            });
        }
        if self.source.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return Err(BackendError::InvalidLayout {
                role: "source",
                layout: self.source.layout,
            });
        }
        if self.output.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            return Err(BackendError::InvalidLayout {
                role: "output",
                layout: self.output.layout,
            });
        }
        if self.viewport != config.viewport {
            return Err(BackendError::InvalidMetadata(
                "viewport does not match configuration",
            ));
        }
        let extent = tuxscaling_temporal::FrameExtent {
            width: config.game_extent.width,
            height: config.game_extent.height,
        };
        if !self.guidance.is_valid_for(self.frame_id, extent) {
            return Err(BackendError::InvalidMetadata("guidance view"));
        }
        for signal in all_guidance_signals() {
            if capabilities.required_guidance[signal as usize]
                && self.guidance.resource(signal).state == SignalState::Unavailable
            {
                return Err(BackendError::MissingSignal { signal });
            }
        }
        Ok(())
    }
}

/// Computes an Off-versus-active wipe into the same output image before the
/// overlay is composited. The active image is copied to a per-slot scratch
/// image first so the sampled and storage images never alias.
pub struct ComparisonRenderer {
    device: ash::Device,
    input_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    scratch: Vec<Image>,
    initialized: Vec<bool>,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ComparisonRenderer {
    /// Creates the comparison pipeline and one scratch image per frame slot.
    ///
    /// # Safety
    ///
    /// The caller must provide a live Vulkan device and memory properties
    /// belonging to that device.
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        input_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        format: vk::Format,
        image_count: usize,
    ) -> Result<Self, BackendError> {
        if !is_valid_extent(input_extent) || !is_valid_extent(output_extent) || image_count == 0 {
            return Err(BackendError::InvalidMetadata("comparison extents or slots"));
        }
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE;
        let mut scratch = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            scratch.push(
                unsafe { Image::new(device, memory, output_extent, format, usage) }.map_err(
                    |error| BackendError::Internal(format!("comparison scratch image: {error:?}")),
                )?,
            );
        }
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
        .map_err(|error| BackendError::Internal(format!("comparison sampler: {error:?}")))?;
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
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("comparison descriptor layout: {error:?}"))
        })?;
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(image_count as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: (2 * image_count) as u32,
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
            BackendError::Internal(format!("comparison descriptor pool: {error:?}"))
        })?;
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![descriptor_layout; image_count]),
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("comparison descriptor sets: {error:?}"))
        })?;
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 20,
                    }]),
                None,
            )
        }
        .map_err(|error| {
            BackendError::Internal(format!("comparison pipeline layout: {error:?}"))
        })?;
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/compare.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
            .map_err(|_| BackendError::Internal("invalid comparison shader".into()))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|error| BackendError::Internal(format!("comparison shader: {error:?}")))?;
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
                BackendError::Internal(format!("comparison pipeline: {error:?}"))
            })?
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::Internal("comparison pipeline was not created".into()))?;

        Ok(Self {
            device: device.clone(),
            input_extent,
            output_extent,
            scratch,
            initialized: vec![false; image_count],
            sampler,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
        })
    }

    /// Records the wipe before the overlay is rendered.
    ///
    /// # Safety
    ///
    /// The command buffer must be recording, and the source/output images
    /// must remain alive until the submission completes.
    pub unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        slot: usize,
        source: BackendImage,
        output: BackendImage,
        split: f32,
    ) -> Result<(), BackendError> {
        if !split.is_finite()
            || !(0.0..=1.0).contains(&split)
            || source.extent != self.input_extent
            || output.extent != self.output_extent
            || source.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            || output.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            || slot >= self.scratch.len()
        {
            return Err(BackendError::InvalidMetadata("comparison frame"));
        }
        let scratch = &self.scratch[slot];
        let descriptor_set = self.descriptor_sets[slot];
        let sampled_source = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(source.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let sampled_active = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(scratch.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let output_info = vk::DescriptorImageInfo::default()
            .image_view(output.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&sampled_source)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&sampled_active)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
        ];
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
            image_barrier(
                &self.device,
                command,
                output.image,
                output.layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                scratch.handle,
                if self.initialized[slot] {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            self.device.cmd_copy_image(
                command,
                output.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                scratch.handle,
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
                scratch.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &self.device,
                command,
                output.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let params = [
                self.output_extent.width,
                self.output_extent.height,
                self.input_extent.width,
                self.input_extent.height,
                split.to_bits(),
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
            image_barrier(
                &self.device,
                command,
                output.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
        self.initialized[slot] = true;
        Ok(())
    }
}

impl Drop for ComparisonRenderer {
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

fn all_guidance_signals() -> [GuidanceSignal; 7] {
    [
        GuidanceSignal::Motion,
        GuidanceSignal::Confidence,
        GuidanceSignal::Disocclusion,
        GuidanceSignal::Reactive,
        GuidanceSignal::Exposure,
        GuidanceSignal::RelativeDepth,
        GuidanceSignal::TransparencyComposition,
    ]
}

fn is_valid_extent(extent: vk::Extent2D) -> bool {
    extent.width != 0 && extent.height != 0
}

fn validate_extent(
    role: &'static str,
    expected: vk::Extent2D,
    actual: vk::Extent2D,
) -> Result<(), BackendError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackendError::IncompatibleExtent {
            role,
            expected,
            actual,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionPlan {
    pub game_extent: vk::Extent2D,
    pub guidance_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
    pub presentation: PresentationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationMode {
    Direct,
    Virtual,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentViewport {
    pub offset: [f32; 2],
    pub size: [f32; 2],
}

impl ContentViewport {
    fn is_valid(self) -> bool {
        self.offset
            .iter()
            .chain(self.size.iter())
            .all(|value| value.is_finite())
            && self.offset.iter().all(|value| *value >= 0.0)
            && self.size.iter().all(|value| (0.0..=1.0).contains(value))
            && self.size.iter().all(|value| *value > 0.0)
    }
}

pub fn content_viewport(input: vk::Extent2D, output: vk::Extent2D) -> ContentViewport {
    let input_aspect = input.width as f32 / input.height as f32;
    let output_aspect = output.width as f32 / output.height as f32;
    let size = if output_aspect > input_aspect {
        [input_aspect / output_aspect, 1.0]
    } else {
        [1.0, output_aspect / input_aspect]
    };
    ContentViewport {
        offset: [(1.0 - size[0]) * 0.5, (1.0 - size[1]) * 0.5],
        size,
    }
}

impl ResolutionPlan {
    pub fn new(
        game_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        guidance_scale: f32,
    ) -> Self {
        Self {
            game_extent,
            guidance_extent: scaled_extent(game_extent, guidance_scale),
            output_extent,
            presentation: if game_extent == output_extent {
                PresentationMode::Direct
            } else {
                PresentationMode::Virtual
            },
        }
    }
}

pub trait UpscalerBackend: Send {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn configure(&mut self, config: BackendConfig) -> Result<(), BackendError>;
    /// Records backend commands into the supplied command buffer only.
    ///
    /// # Safety
    ///
    /// The caller must provide a recording command buffer and image layouts
    /// matching the validated `BackendFrame` contract. The command buffer and
    /// all referenced resources must remain valid until its submission fence
    /// signals.
    unsafe fn record(&mut self, frame: BackendFrame) -> Result<(), BackendError>;
    fn reset(&mut self) -> Result<(), BackendError>;
}

#[cfg(test)]
mod tests {
    use super::content_viewport;
    use super::{
        BackendCapabilities, BackendColorEncoding, BackendConfig, BackendError, BackendFrame,
        BackendId, BackendImage, ContentViewport, PresentationMode, ResolutionPlan,
        UpscalerBackend,
    };
    use ash::vk;
    use ash::vk::Handle;
    use tuxscaling_temporal::{
        DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset,
        GuidanceResolution, GuidanceResource, GuidanceScalar, GuidanceSignal, GuidanceView,
        JitterSample, MotionDirection, MotionUnits, SignalState,
    };

    struct Dummy;

    impl UpscalerBackend for Dummy {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
                required_guidance: [true; 7],
                supported_source_formats: &[vk::Format::R8G8B8A8_UNORM],
                supported_output_formats: &[vk::Format::R8G8B8A8_UNORM],
            }
        }
        fn configure(&mut self, _: BackendConfig) -> Result<(), BackendError> {
            Ok(())
        }
        unsafe fn record(&mut self, _: BackendFrame) -> Result<(), BackendError> {
            Ok(())
        }
        fn reset(&mut self) -> Result<(), BackendError> {
            Ok(())
        }
    }

    #[test]
    fn dummy_backend_satisfies_contract() {
        let mut backend = Dummy;
        assert_eq!(backend.id(), BackendId::Reference);
        assert_eq!(BackendId::Fsr314.as_str(), "fsr_3_1_4");
        assert_eq!(
            BackendColorEncoding::from(vk::ColorSpaceKHR::SRGB_NONLINEAR),
            BackendColorEncoding::SrgbNonlinear
        );
        assert!(backend.capabilities().temporal);
        backend
            .configure(BackendConfig {
                game_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                output_extent: vk::Extent2D {
                    width: 2,
                    height: 2,
                },
                source_format: vk::Format::R8G8B8A8_UNORM,
                output_format: vk::Format::R8G8B8A8_UNORM,
                color_encoding: BackendColorEncoding::SrgbNonlinear,
                viewport: ContentViewport {
                    offset: [0.0, 0.0],
                    size: [1.0, 1.0],
                },
                guidance: guidance().capabilities(),
            })
            .unwrap();
        backend.reset().unwrap();
    }

    const GAME: vk::Extent2D = vk::Extent2D {
        width: 1280,
        height: 720,
    };
    const OUTPUT: vk::Extent2D = vk::Extent2D {
        width: 1920,
        height: 1080,
    };

    fn guidance() -> GuidanceView {
        let extent = FrameExtent {
            width: GAME.width,
            height: GAME.height,
        };
        let metadata = GuidanceMetadata::zero(7, extent, GuidanceReset::None);
        let resource = |format| GuidanceResource {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            format,
            metadata,
            state: SignalState::Estimated,
        };
        GuidanceView {
            motion: resource(vk::Format::R16G16_SFLOAT),
            confidence: resource(vk::Format::R8_UNORM),
            disocclusion: resource(vk::Format::R8_UNORM),
            reactive: resource(vk::Format::R8_UNORM),
            exposure: resource(vk::Format::R32_SFLOAT),
            depth: resource(vk::Format::R32_SFLOAT),
            transparency_composition: resource(vk::Format::R8_UNORM),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing: FrameTiming::default(),
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::FlatFallback,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(extent, extent),
            requires_history_reset: false,
        }
    }

    fn config() -> BackendConfig {
        BackendConfig {
            game_extent: GAME,
            output_extent: OUTPUT,
            source_format: vk::Format::R8G8B8A8_UNORM,
            output_format: vk::Format::R8G8B8A8_UNORM,
            color_encoding: BackendColorEncoding::SrgbNonlinear,
            viewport: content_viewport(GAME, OUTPUT),
            guidance: guidance().capabilities(),
        }
    }

    fn frame() -> BackendFrame {
        BackendFrame {
            command_buffer: vk::CommandBuffer::from_raw(3),
            slot: 0,
            source: BackendImage {
                image: vk::Image::from_raw(4),
                view: vk::ImageView::from_raw(5),
                format: vk::Format::R8G8B8A8_UNORM,
                extent: GAME,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            output: BackendImage {
                image: vk::Image::from_raw(6),
                view: vk::ImageView::from_raw(7),
                format: vk::Format::R8G8B8A8_UNORM,
                extent: OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            guidance: guidance(),
            viewport: content_viewport(GAME, OUTPUT),
            frame_id: 7,
            reset_history: false,
            debug_view: 0,
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        config: Option<BackendConfig>,
        records: u32,
        resets: u32,
    }

    impl UpscalerBackend for RecordingBackend {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
                required_guidance: [true; 7],
                supported_source_formats: &[vk::Format::R8G8B8A8_UNORM],
                supported_output_formats: &[vk::Format::R8G8B8A8_UNORM],
            }
        }

        fn configure(&mut self, config: BackendConfig) -> Result<(), BackendError> {
            config.validate(self.capabilities())?;
            self.config = Some(config);
            Ok(())
        }

        unsafe fn record(&mut self, frame: BackendFrame) -> Result<(), BackendError> {
            let config = self.config.ok_or(BackendError::Unavailable)?;
            frame.validate(config, self.capabilities())?;
            self.records += 1;
            Ok(())
        }

        fn reset(&mut self) -> Result<(), BackendError> {
            self.resets += 1;
            Ok(())
        }
    }

    #[test]
    fn backend_contract_is_object_safe_and_dispatches_valid_frame() {
        let mut backend: Box<dyn UpscalerBackend> = Box::new(RecordingBackend::default());
        backend.configure(config()).unwrap();
        unsafe { backend.record(frame()) }.unwrap();
        backend.reset().unwrap();
    }

    #[test]
    fn backend_config_rejects_unsupported_format_and_extent() {
        let capabilities = RecordingBackend::default().capabilities();
        let mut invalid = config();
        invalid.source_format = vk::Format::R16G16_SFLOAT;
        assert!(matches!(
            invalid.validate(capabilities),
            Err(BackendError::UnsupportedFormat { .. })
        ));
        invalid = config();
        invalid.game_extent = vk::Extent2D {
            width: 0,
            height: GAME.height,
        };
        assert!(matches!(
            invalid.validate(capabilities),
            Err(BackendError::IncompatibleExtent { .. })
        ));
    }

    #[test]
    fn backend_frame_rejects_incoherent_resources_and_missing_signal() {
        let capabilities = RecordingBackend::default().capabilities();
        let mut invalid = frame();
        invalid.source.extent.width -= 1;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::IncompatibleExtent { .. })
        ));
        invalid = frame();
        invalid.guidance.motion.state = SignalState::Unavailable;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::MissingSignal {
                signal: GuidanceSignal::Motion
            })
        ));
        invalid = frame();
        invalid.output.layout = vk::ImageLayout::GENERAL;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::InvalidLayout { .. })
        ));
    }

    #[test]
    fn resolution_plan_keeps_game_input_at_full_guidance_scale() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            1.0,
        );

        assert_eq!(plan.game_extent, plan.guidance_extent);
        assert_eq!(plan.output_extent.width, 1920);
        assert_eq!(plan.presentation, PresentationMode::Virtual);
    }

    #[test]
    fn resolution_plan_scales_only_guidance_input() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            0.75,
        );

        assert_eq!(
            plan.guidance_extent,
            vk::Extent2D {
                width: 960,
                height: 540,
            }
        );
        assert_eq!(plan.presentation, PresentationMode::Virtual);
    }

    #[test]
    fn equal_game_and_output_extents_use_native_aa_mode() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            1.0,
        );

        assert_eq!(plan.presentation, PresentationMode::Direct);
    }

    #[test]
    fn content_viewport_preserves_the_input_aspect_ratio() {
        let viewport = content_viewport(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1200,
            },
        );

        assert!((viewport.offset[0] - 0.0).abs() < 0.0001);
        assert!((viewport.offset[1] - 0.05).abs() < 0.0001);
        assert!((viewport.size[0] - 1.0).abs() < 0.0001);
        assert!((viewport.size[1] - 0.9).abs() < 0.0001);
    }

    #[test]
    fn resolution_plan_keeps_game_guidance_and_output_extents_independent() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1279,
                height: 719,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            0.75,
        );

        assert_eq!(plan.game_extent.width, 1279);
        assert_eq!(plan.guidance_extent.width, 959);
        assert_eq!(plan.guidance_extent.height, 539);
        assert_eq!(plan.output_extent.width, 1920);
        assert!(plan.guidance_extent.width > 0 && plan.guidance_extent.height > 0);
    }

    #[test]
    fn changing_guidance_scale_does_not_change_game_or_output_extents() {
        let game = vk::Extent2D {
            width: 1281,
            height: 721,
        };
        let output = vk::Extent2D {
            width: 2560,
            height: 1440,
        };
        let full = ResolutionPlan::new(game, output, 1.0);
        let reduced = ResolutionPlan::new(game, output, 0.5);

        assert_eq!(full.game_extent, reduced.game_extent);
        assert_eq!(full.output_extent, reduced.output_extent);
        assert_ne!(full.guidance_extent, reduced.guidance_extent);
    }

    #[test]
    fn comparison_shader_contract_keeps_both_inputs_and_a_normalized_wipe() {
        let shader = include_str!("../../../shaders/upscaler/compare.comp");

        for token in [
            "off_image",
            "active_image",
            "output_image",
            "split",
            "content_min",
        ] {
            assert!(shader.contains(token), "comparison shader lacks {token}");
        }
    }
}
