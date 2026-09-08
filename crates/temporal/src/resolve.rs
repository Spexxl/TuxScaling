#![allow(clippy::missing_safety_doc)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidanceResolvePolicy {
    Bypass,
    Resolve,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuGuidance {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<f32>,
    pub motion: Vec<[f32; 2]>,
    pub confidence: Vec<f32>,
    pub disocclusion: Vec<f32>,
    pub reactive: Vec<f32>,
    pub composition: Vec<f32>,
    pub depth: Vec<f32>,
    pub exposure: f32,
    pub policy: GuidanceResolvePolicy,
}

impl CpuGuidance {
    pub fn constant(
        width: u32,
        height: u32,
        motion: [f32; 2],
        confidence: f32,
        depth: f32,
        exposure: f32,
    ) -> Self {
        let len = (width * height) as usize;
        Self {
            width,
            height,
            luma: vec![0.5; len],
            motion: vec![motion; len],
            confidence: vec![confidence; len],
            disocclusion: vec![0.0; len],
            reactive: vec![0.0; len],
            composition: vec![0.0; len],
            depth: vec![depth; len],
            exposure,
            policy: GuidanceResolvePolicy::Resolve,
        }
    }
}

pub fn resolve_cpu_guidance(
    guidance: &CpuGuidance,
    game_extent: (u32, u32),
    game_luma: &[f32],
) -> CpuGuidance {
    let (game_width, game_height) = game_extent;
    assert_eq!(game_luma.len(), (game_width * game_height) as usize);
    assert_eq!(
        guidance.luma.len(),
        (guidance.width * guidance.height) as usize
    );
    if guidance.width == game_width && guidance.height == game_height {
        let mut result = guidance.clone();
        result.policy = GuidanceResolvePolicy::Bypass;
        return result;
    }

    let mut result = CpuGuidance::constant(
        game_width,
        game_height,
        [0.0, 0.0],
        0.0,
        1.0,
        guidance.exposure,
    );
    result.policy = GuidanceResolvePolicy::Resolve;
    for y in 0..game_height {
        for x in 0..game_width {
            let target = (y * game_width + x) as usize;
            let candidates = candidates(guidance, x, y, game_width, game_height);
            let weights = edge_weights(guidance, &candidates, game_luma[target]);
            result.motion[target] = weighted_motion(guidance, &candidates, &weights, game_extent);
            result.confidence[target] = candidates
                .iter()
                .map(|&(index, _)| guidance.confidence[index])
                .fold(1.0, f32::min);
            result.disocclusion[target] = candidates
                .iter()
                .map(|&(index, _)| guidance.disocclusion[index])
                .fold(0.0, f32::max);
            result.reactive[target] = candidates
                .iter()
                .map(|&(index, _)| guidance.reactive[index])
                .fold(0.0, f32::max);
            result.composition[target] = candidates
                .iter()
                .map(|&(index, _)| guidance.composition[index])
                .fold(0.0, f32::max);
            result.depth[target] = weighted_scalar(&guidance.depth, &candidates, &weights);
        }
    }
    let depth_min = guidance.depth.iter().copied().fold(f32::INFINITY, f32::min);
    let depth_max = guidance
        .depth
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if depth_max - depth_min < 0.001 {
        result.depth.fill(1.0);
    }
    result
}

fn candidates(
    guidance: &CpuGuidance,
    x: u32,
    y: u32,
    game_width: u32,
    game_height: u32,
) -> Vec<(usize, f32)> {
    let gx = ((x as f32 + 0.5) * guidance.width as f32 / game_width as f32 - 0.5)
        .clamp(0.0, guidance.width.saturating_sub(1) as f32);
    let gy = ((y as f32 + 0.5) * guidance.height as f32 / game_height as f32 - 0.5)
        .clamp(0.0, guidance.height.saturating_sub(1) as f32);
    let x0 = gx.floor() as u32;
    let y0 = gy.floor() as u32;
    let x1 = (x0 + 1).min(guidance.width.saturating_sub(1));
    let y1 = (y0 + 1).min(guidance.height.saturating_sub(1));
    let tx = gx - x0 as f32;
    let ty = gy - y0 as f32;
    [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    ]
    .into_iter()
    .map(|(x, y, weight)| ((y * guidance.width + x) as usize, weight))
    .collect()
}

fn edge_weights(guidance: &CpuGuidance, candidates: &[(usize, f32)], target_luma: f32) -> Vec<f32> {
    let min_difference = candidates
        .iter()
        .map(|&(index, _)| (guidance.luma[index] - target_luma).abs())
        .fold(f32::INFINITY, f32::min);
    candidates
        .iter()
        .map(|&(index, spatial)| {
            if (guidance.luma[index] - target_luma).abs() <= min_difference + 0.05 {
                spatial.max(0.0001)
            } else {
                0.0
            }
        })
        .collect()
}

fn weighted_motion(
    guidance: &CpuGuidance,
    candidates: &[(usize, f32)],
    weights: &[f32],
    game_extent: (u32, u32),
) -> [f32; 2] {
    let (game_width, game_height) = game_extent;
    let x_scale = game_width as f32 / guidance.width as f32;
    let y_scale = game_height as f32 / guidance.height as f32;
    let total = weights.iter().sum::<f32>().max(f32::EPSILON);
    candidates
        .iter()
        .zip(weights)
        .fold([0.0, 0.0], |mut value, (&(index, _), weight)| {
            value[0] += guidance.motion[index][0] * weight / total * x_scale;
            value[1] += guidance.motion[index][1] * weight / total * y_scale;
            value
        })
}

fn weighted_scalar(values: &[f32], candidates: &[(usize, f32)], weights: &[f32]) -> f32 {
    let total = weights.iter().sum::<f32>().max(f32::EPSILON);
    candidates
        .iter()
        .zip(weights)
        .map(|(&(index, _), weight)| values[index] * weight / total)
        .sum()
}

use ash::vk;
use tuxscaling_vulkan::{Image, compute_memory_barrier, image_barrier};

use crate::{
    DepthSemantics, FrameExtent, GuidanceMetadata, GuidanceResolution, GuidanceResource,
    GuidanceView,
};

const RESOLVED_SIGNAL_COUNT: usize = 6;

pub struct GuidanceResolver {
    device: ash::Device,
    pub game_extent: vk::Extent2D,
    pub estimator_extent: vk::Extent2D,
    motion: Image,
    confidence: Image,
    disocclusion: Image,
    reactive: Image,
    depth: Image,
    composition: Image,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipelines: [vk::Pipeline; 4],
    initialized: bool,
}

impl GuidanceResolver {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        game_extent: vk::Extent2D,
        estimator_extent: vk::Extent2D,
        source_view: vk::ImageView,
        guidance: GuidanceView,
    ) -> Result<Self, vk::Result> {
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let motion = unsafe {
            Image::new(
                device,
                memory,
                game_extent,
                vk::Format::R16G16_SFLOAT,
                usage,
            )
        }?;
        let confidence =
            unsafe { Image::new(device, memory, game_extent, vk::Format::R8_UNORM, usage) }?;
        let disocclusion =
            unsafe { Image::new(device, memory, game_extent, vk::Format::R8_UNORM, usage) }?;
        let reactive =
            unsafe { Image::new(device, memory, game_extent, vk::Format::R8_UNORM, usage) }?;
        let depth =
            unsafe { Image::new(device, memory, game_extent, vk::Format::R32_SFLOAT, usage) }?;
        let composition =
            unsafe { Image::new(device, memory, game_extent, vk::Format::R8_UNORM, usage) }?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }?;
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
        }?;
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(RESOLVED_SIGNAL_COUNT as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: (RESOLVED_SIGNAL_COUNT * 2) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: RESOLVED_SIGNAL_COUNT as u32,
                        },
                    ]),
                None,
            )
        }?;
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_layout; RESOLVED_SIGNAL_COUNT]),
            )
        }?;
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
        }?;
        let pipelines = [
            unsafe {
                pipeline_from_bytes(
                    device,
                    pipeline_layout,
                    include_bytes!(concat!(env!("OUT_DIR"), "/resolve_motion.spv")),
                )
            }?,
            unsafe {
                pipeline_from_bytes(
                    device,
                    pipeline_layout,
                    include_bytes!(concat!(env!("OUT_DIR"), "/resolve_confidence.spv")),
                )
            }?,
            unsafe {
                pipeline_from_bytes(
                    device,
                    pipeline_layout,
                    include_bytes!(concat!(env!("OUT_DIR"), "/resolve_mask.spv")),
                )
            }?,
            unsafe {
                pipeline_from_bytes(
                    device,
                    pipeline_layout,
                    include_bytes!(concat!(env!("OUT_DIR"), "/resolve_depth.spv")),
                )
            }?,
        ];
        let input = [
            guidance.motion,
            guidance.confidence,
            guidance.disocclusion,
            guidance.reactive,
            guidance.depth,
            guidance.transparency_composition,
        ];
        let output = [
            &motion,
            &confidence,
            &disocclusion,
            &reactive,
            &depth,
            &composition,
        ];
        let output_formats = [
            vk::Format::R16G16_SFLOAT,
            vk::Format::R8_UNORM,
            vk::Format::R8_UNORM,
            vk::Format::R8_UNORM,
            vk::Format::R32_SFLOAT,
            vk::Format::R8_UNORM,
        ];
        for index in 0..RESOLVED_SIGNAL_COUNT {
            let source_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(source_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let input_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(input[index].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let output_info = [vk::DescriptorImageInfo::default()
                .image_view(output[index].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_sets[index])
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&source_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_sets[index])
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&input_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_sets[index])
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&output_info),
                    ],
                    &[],
                );
            }
            debug_assert_eq!(output[index].format, output_formats[index]);
        }
        Ok(Self {
            device: device.clone(),
            game_extent,
            estimator_extent,
            motion,
            confidence,
            disocclusion,
            reactive,
            depth,
            composition,
            sampler,
            descriptor_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipelines,
            initialized: false,
        })
    }

    pub unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        valid: bool,
        depth_semantics: DepthSemantics,
    ) {
        let outputs = [
            self.motion.handle,
            self.confidence.handle,
            self.disocclusion.handle,
            self.reactive.handle,
            self.depth.handle,
            self.composition.handle,
        ];
        unsafe {
            for image in outputs {
                image_barrier(
                    &self.device,
                    command,
                    image,
                    if self.initialized {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    vk::ImageLayout::GENERAL,
                );
            }
            let pipeline_indices = [0usize, 1, 2, 2, 3, 2];
            for (index, pipeline_index) in pipeline_indices.into_iter().enumerate() {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipelines[pipeline_index],
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[index]],
                    &[],
                );
                let params = [
                    self.game_extent.width,
                    self.game_extent.height,
                    self.estimator_extent.width,
                    self.estimator_extent.height,
                    u32::from(valid),
                    u32::from(matches!(depth_semantics, DepthSemantics::FlatFallback)),
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
                    self.game_extent.width.div_ceil(8),
                    self.game_extent.height.div_ceil(8),
                    1,
                );
                compute_memory_barrier(&self.device, command);
            }
        }
        self.initialized = true;
    }

    pub fn view(&self, raw: GuidanceView) -> GuidanceView {
        let resolution = GuidanceResolution::new(
            FrameExtent {
                width: self.game_extent.width,
                height: self.game_extent.height,
            },
            FrameExtent {
                width: self.estimator_extent.width,
                height: self.estimator_extent.height,
            },
        );
        let metadata = |resource: GuidanceResource| GuidanceMetadata {
            extent: resolution.signal_extent,
            valid_region: crate::ValidRegion::full(resolution.signal_extent),
            ..resource.metadata
        };
        let resource = |image: vk::Image,
                        view: vk::ImageView,
                        format: vk::Format,
                        original: GuidanceResource| GuidanceResource {
            image,
            view,
            format,
            metadata: metadata(original),
            state: original.state,
        };
        GuidanceView {
            motion: resource(
                self.motion.handle,
                self.motion.view,
                vk::Format::R16G16_SFLOAT,
                raw.motion,
            ),
            confidence: resource(
                self.confidence.handle,
                self.confidence.view,
                vk::Format::R8_UNORM,
                raw.confidence,
            ),
            disocclusion: resource(
                self.disocclusion.handle,
                self.disocclusion.view,
                vk::Format::R8_UNORM,
                raw.disocclusion,
            ),
            reactive: resource(
                self.reactive.handle,
                self.reactive.view,
                vk::Format::R8_UNORM,
                raw.reactive,
            ),
            exposure: {
                let mut exposure = raw.exposure;
                exposure.metadata = metadata(raw.exposure);
                exposure
            },
            depth: resource(
                self.depth.handle,
                self.depth.view,
                vk::Format::R32_SFLOAT,
                raw.depth,
            ),
            transparency_composition: resource(
                self.composition.handle,
                self.composition.view,
                vk::Format::R8_UNORM,
                raw.transparency_composition,
            ),
            pre_exposure: raw.pre_exposure,
            timing: raw.timing,
            jitter: raw.jitter,
            depth_semantics: raw.depth_semantics,
            direction: raw.direction,
            units: raw.units,
            resolution,
            requires_history_reset: raw.requires_history_reset,
        }
    }
}

unsafe fn pipeline_from_bytes(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    bytes: &[u8],
) -> Result<vk::Pipeline, vk::Result> {
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
                .layout(layout)],
            None,
        )
    };
    unsafe { device.destroy_shader_module(module, None) };
    match pipeline {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, error)) => Err(error),
    }
}

impl Drop for GuidanceResolver {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.pipelines {
                self.device.destroy_pipeline(pipeline, None);
            }
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
    use super::{CpuGuidance, GuidanceResolvePolicy, resolve_cpu_guidance};

    fn extent(width: u32, height: u32) -> (u32, u32) {
        (width, height)
    }

    #[test]
    fn scales_motion_from_guidance_pixels_to_game_pixels() {
        let guidance = CpuGuidance::constant(2, 2, [1.0, -2.0], 0.5, 1.0, 0.0);
        let resolved = resolve_cpu_guidance(&guidance, extent(4, 4), &[0.5; 16]);

        assert!(
            resolved.motion.iter().all(|value| {
                (value[0] - 2.0).abs() < 0.0001 && (value[1] + 4.0).abs() < 0.0001
            })
        );
    }

    #[test]
    fn preserves_translation_and_scale_one_bypasses_the_resolver() {
        let guidance = CpuGuidance::constant(3, 5, [2.0, 1.0], 0.8, 1.0, 0.0);
        let resolved = resolve_cpu_guidance(&guidance, extent(3, 5), &[0.5; 15]);

        assert_eq!(resolved.motion, guidance.motion);
        assert_eq!(resolved.policy, GuidanceResolvePolicy::Bypass);
    }

    #[test]
    fn edge_aware_motion_does_not_bleed_between_luma_regions() {
        let guidance = CpuGuidance {
            width: 2,
            height: 1,
            luma: vec![0.0, 1.0],
            motion: vec![[0.0, 0.0], [8.0, 0.0]],
            confidence: vec![1.0, 1.0],
            disocclusion: vec![0.0, 0.0],
            reactive: vec![0.0, 0.0],
            composition: vec![0.0, 0.0],
            depth: vec![0.25, 0.75],
            exposure: 1.0,
            policy: GuidanceResolvePolicy::Resolve,
        };
        let resolved = resolve_cpu_guidance(&guidance, extent(4, 1), &[0.0, 0.0, 1.0, 1.0]);

        assert_eq!(resolved.motion[0], [0.0, 0.0]);
        assert!((resolved.motion[3][0] - 16.0).abs() < 0.0001);
        assert!(resolved.depth[0] < resolved.depth[3]);
    }

    #[test]
    fn conservative_masks_keep_thin_features_and_confidence_uses_a_minimum() {
        let guidance = CpuGuidance {
            width: 3,
            height: 1,
            luma: vec![0.0, 0.5, 1.0],
            motion: vec![[0.0, 0.0]; 3],
            confidence: vec![1.0, 0.2, 1.0],
            disocclusion: vec![0.0, 1.0, 0.0],
            reactive: vec![0.0, 1.0, 0.0],
            composition: vec![0.0, 1.0, 0.0],
            depth: vec![1.0, 1.0, 1.0],
            exposure: 1.25,
            policy: GuidanceResolvePolicy::Resolve,
        };
        let resolved =
            resolve_cpu_guidance(&guidance, extent(6, 1), &[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);

        assert!(resolved.reactive[2] > 0.0);
        assert!(resolved.disocclusion[2] > 0.0);
        assert!(resolved.composition[2] > 0.0);
        assert!(resolved.confidence[2] <= 0.2);
        assert_eq!(resolved.exposure, 1.25);
    }

    #[test]
    fn flat_depth_stays_flat_and_relative_depth_order_remains_monotonic() {
        let mut guidance = CpuGuidance::constant(2, 1, [0.0, 0.0], 1.0, 1.0, 0.0);
        guidance.depth = vec![0.1, 0.9];
        let resolved = resolve_cpu_guidance(&guidance, extent(8, 1), &[0.0; 8]);

        assert!(resolved.depth.windows(2).all(|pair| pair[0] <= pair[1]));

        guidance.depth.fill(0.4);
        let flat = resolve_cpu_guidance(&guidance, extent(8, 1), &[0.0; 8]);
        assert!(flat.depth.iter().all(|value| *value == 1.0));
    }
}
