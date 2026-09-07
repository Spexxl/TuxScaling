#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_config::JitterMode;
use tuxscaling_temporal::{JitterSample, SignalState};
use tuxscaling_vulkan::{Image, compute_memory_barrier, image_barrier};

const JITTER_PHASES: u32 = 8;

#[derive(Debug, Clone, Copy)]
pub struct JitterState {
    mode: JitterMode,
    phase: u32,
    current: [f32; 2],
    phase_restart: bool,
}

impl JitterState {
    pub fn new(mode: JitterMode) -> Self {
        Self {
            mode,
            phase: 0,
            current: [0.0, 0.0],
            phase_restart: false,
        }
    }

    pub fn mode(self) -> JitterMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: JitterMode) -> bool {
        if self.mode == mode {
            return false;
        }
        self.mode = mode;
        self.reset();
        true
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.current = [0.0, 0.0];
        self.phase_restart = false;
    }

    pub fn take_phase_restart(&mut self) -> bool {
        std::mem::take(&mut self.phase_restart)
    }

    pub fn sample(&mut self) -> JitterSample {
        if self.mode == JitterMode::Off {
            self.current = [0.0, 0.0];
            self.phase = 0;
            self.phase_restart = false;
            return JitterSample::default();
        }

        let phase = self.phase;
        let previous = self.current;
        let current = halton8(phase);
        self.phase_restart = phase == 0 && previous != [0.0, 0.0];
        self.current = current;
        self.phase = (phase + 1) % JITTER_PHASES;
        let sample = JitterSample {
            current,
            previous,
            phase,
        };
        debug_assert_eq!(sample.signal_state(), SignalState::Estimated);
        sample
    }
}

impl Default for JitterState {
    fn default() -> Self {
        Self::new(JitterMode::Off)
    }
}

pub fn halton8(phase: u32) -> [f32; 2] {
    let index = phase % 4 + 1;
    let sign = if phase % JITTER_PHASES < 4 { 1.0 } else { -1.0 };
    [
        sign * (halton(index, 2) - 0.5),
        sign * (halton(index, 3) - 0.5),
    ]
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f32;
    while index != 0 {
        result += fraction * (index % base) as f32;
        index /= base;
        fraction /= base as f32;
    }
    result
}

pub struct CapturedFrame {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub frame_id: u64,
    pub timestamp: std::time::Duration,
    pub generation: u64,
}
pub fn supported_format(format: vk::Format, space: vk::ColorSpaceKHR) -> bool {
    space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        && matches!(
            format,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::R16G16B16A16_SFLOAT
        )
}

pub fn requires_scaling(source: vk::Extent2D, destination: vk::Extent2D) -> bool {
    source != destination
}

pub struct Capture {
    device: ash::Device,
    pub color: Image,
    pub previous: Image,
    resample_source: Image,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    initialized: bool,
    resample_source_initialized: bool,
}
impl Capture {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self, vk::Result> {
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE;
        let mut result = Self {
            device: device.clone(),
            color: unsafe { Image::new(device, memory, extent, format, usage) }?,
            previous: unsafe { Image::new(device, memory, extent, format, usage) }?,
            resample_source: unsafe { Image::new(device, memory, extent, format, usage) }?,
            sampler: vk::Sampler::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            initialized: false,
            resample_source_initialized: false,
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
        result.descriptor_layout = unsafe {
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
        }?;
        result.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
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
        let sampled = [vk::DescriptorImageInfo::default()
            .sampler(result.sampler)
            .image_view(result.resample_source.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let storage = [vk::DescriptorImageInfo::default()
            .image_view(result.color.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        unsafe {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&sampled),
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&storage),
                ],
                &[],
            );
        }
        result.pipeline_layout = unsafe {
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
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/resample.spv"));
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
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
    ) {
        unsafe {
            self.record_from(device, command, source, vk::ImageLayout::PRESENT_SRC_KHR);
        }
    }

    pub unsafe fn record_from(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
        layout: vk::ImageLayout,
    ) {
        let extent = self.color.extent;
        unsafe {
            self.record_scaled_from(
                device,
                command,
                source,
                extent,
                layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
    }

    pub unsafe fn record_scaled_from(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_extent: vk::Extent2D,
        layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) {
        unsafe {
            self.record_scaled_from_with_jitter(
                device,
                command,
                source,
                source_extent,
                layout,
                final_layout,
                JitterSample::default(),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record_scaled_from_with_jitter(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_extent: vk::Extent2D,
        layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
        jitter: JitterSample,
    ) {
        let experimental = jitter.signal_state() == SignalState::Estimated;
        unsafe {
            if self.initialized {
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let layers = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);
                device.cmd_copy_image(
                    command,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageCopy::default()
                        .src_subresource(layers)
                        .dst_subresource(layers)
                        .extent(vk::Extent3D {
                            width: self.color.extent.width,
                            height: self.color.extent.height,
                            depth: 1,
                        })],
                );
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            } else {
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                device.cmd_clear_color_image(
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[tuxscaling_vulkan::color_range()],
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            if experimental {
                image_barrier(
                    device,
                    command,
                    source,
                    layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.resample_source.handle,
                    if self.resample_source_initialized {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                if requires_scaling(source_extent, self.resample_source.extent) {
                    device.cmd_blit_image(
                        command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.resample_source.handle,
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
                                vk::Offset3D::default(),
                                vk::Offset3D {
                                    x: self.resample_source.extent.width as i32,
                                    y: self.resample_source.extent.height as i32,
                                    z: 1,
                                },
                            ])],
                        vk::Filter::LINEAR,
                    );
                } else {
                    device.cmd_copy_image(
                        command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.resample_source.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageCopy::default()
                            .src_subresource(layers)
                            .dst_subresource(layers)
                            .extent(vk::Extent3D {
                                width: self.resample_source.extent.width,
                                height: self.resample_source.extent.height,
                                depth: 1,
                            })],
                    );
                }
                image_barrier(
                    device,
                    command,
                    self.resample_source.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    final_layout,
                );
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    if self.initialized {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    vk::ImageLayout::GENERAL,
                );
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                let params = [
                    self.color.extent.width,
                    self.color.extent.height,
                    jitter.current[0].to_bits(),
                    jitter.current[1].to_bits(),
                ];
                device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::cast_slice(&params),
                );
                device.cmd_dispatch(
                    command,
                    self.color.extent.width.div_ceil(8),
                    self.color.extent.height.div_ceil(8),
                    1,
                );
                compute_memory_barrier(device, command);
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                self.resample_source_initialized = true;
            } else {
                image_barrier(
                    device,
                    command,
                    source,
                    layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    if self.initialized {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                if requires_scaling(source_extent, self.color.extent) {
                    device.cmd_blit_image(
                        command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.color.handle,
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
                                vk::Offset3D::default(),
                                vk::Offset3D {
                                    x: self.color.extent.width as i32,
                                    y: self.color.extent.height as i32,
                                    z: 1,
                                },
                            ])],
                        vk::Filter::LINEAR,
                    );
                } else {
                    device.cmd_copy_image(
                        command,
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.color.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[vk::ImageCopy::default()
                            .src_subresource(layers)
                            .dst_subresource(layers)
                            .extent(vk::Extent3D {
                                width: self.color.extent.width,
                                height: self.color.extent.height,
                                depth: 1,
                            })],
                    );
                }
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    final_layout,
                );
            }
        }
        self.initialized = true;
    }
}

impl Drop for Capture {
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
    use super::*;
    use tuxscaling_config::JitterMode;

    #[test]
    fn identifies_when_capture_requires_scaling() {
        let source = vk::Extent2D {
            width: 1280,
            height: 720,
        };

        assert!(!requires_scaling(source, source));
        assert!(requires_scaling(
            source,
            vk::Extent2D {
                width: 960,
                height: 540,
            }
        ));
    }
    #[test]
    fn rejects_hdr_and_accepts_sdr() {
        assert!(supported_format(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        ));
        assert!(supported_format(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        ));
        assert!(!supported_format(
            vk::Format::R8G8B8A8_UNORM,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT
        ));
    }

    #[test]
    fn experimental_halton8_uses_the_exact_centered_zero_mean_sequence() {
        let mut state = JitterState::new(JitterMode::ExperimentalHalton8);
        let samples = (0..8).map(|_| state.sample()).collect::<Vec<_>>();
        let expected_sequence = [
            [0.0, -1.0 / 6.0],
            [-1.0 / 4.0, 1.0 / 6.0],
            [1.0 / 4.0, -7.0 / 18.0],
            [-3.0 / 8.0, -1.0 / 18.0],
            [0.0, 1.0 / 6.0],
            [1.0 / 4.0, -1.0 / 6.0],
            [-1.0 / 4.0, 7.0 / 18.0],
            [3.0 / 8.0, 1.0 / 18.0],
        ];
        for (phase, (sample, expected)) in samples.iter().zip(expected_sequence).enumerate() {
            assert_eq!(sample.phase, phase as u32);
            assert!((sample.current[0] - expected[0]).abs() < 1e-6);
            assert!((sample.current[1] - expected[1]).abs() < 1e-6);
            let previous = if phase == 0 {
                [0.0, 0.0]
            } else {
                expected_sequence[phase - 1]
            };
            assert!((sample.previous[0] - previous[0]).abs() < 1e-6);
            assert!((sample.previous[1] - previous[1]).abs() < 1e-6);
        }
        let mean = samples.iter().fold([0.0, 0.0], |sum, sample| {
            [sum[0] + sample.current[0], sum[1] + sample.current[1]]
        });
        assert!(mean[0].abs() < 1e-6);
        assert!(mean[1].abs() < 1e-6);
    }

    #[test]
    fn jitter_wraps_and_reports_a_phase_restart() {
        let mut state = JitterState::new(JitterMode::ExperimentalHalton8);
        for _ in 0..7 {
            let _ = state.sample();
            assert!(!state.take_phase_restart());
        }
        let wrapped = state.sample();
        assert_eq!(wrapped.phase, 7);
        assert!(!state.take_phase_restart());
        let first_after_wrap = state.sample();
        assert_eq!(first_after_wrap.phase, 0);
        assert!(state.take_phase_restart());
        assert!(!state.take_phase_restart());
    }

    #[test]
    fn jitter_mode_transition_and_reset_clear_previous_sample() {
        let mut state = JitterState::new(JitterMode::ExperimentalHalton8);
        let first = state.sample();
        assert_eq!(first.previous, [0.0, 0.0]);
        assert!(state.set_mode(JitterMode::Off));
        let off = state.sample();
        assert_eq!(off.current, [0.0, 0.0]);
        assert_eq!(off.previous, [0.0, 0.0]);
        assert_eq!(off.phase, 0);
        assert!(!state.set_mode(JitterMode::Off));
        assert!(state.set_mode(JitterMode::ExperimentalHalton8));
        let restarted = state.sample();
        assert_eq!(restarted.previous, [0.0, 0.0]);
        state.sample();
        state.reset();
        let reset = state.sample();
        assert_eq!(reset.previous, [0.0, 0.0]);
        assert_eq!(reset.phase, 0);
    }
}
