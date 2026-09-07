#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_vulkan::{
    Buffer, Image, compute_memory_barrier, image_barrier, transfer_memory_barrier,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MotionQuality {
    #[default]
    Ultra,
    High,
    Balanced,
    Performance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionProfile {
    pub levels: usize,
    pub patch_radius: i32,
    pub coarse_radius: i32,
    pub fine_radius: i32,
}

impl MotionQuality {
    pub const fn mean_epe_limit(self) -> f32 {
        match self {
            Self::Ultra => 1.0,
            Self::High => 1.25,
            Self::Balanced => 1.75,
            Self::Performance => 2.5,
        }
    }

    pub const fn p95_epe_limit(self) -> f32 {
        self.mean_epe_limit() * 2.0
    }

    pub fn profile(self, available_levels: usize) -> MotionProfile {
        let profile = match self {
            Self::Ultra => MotionProfile {
                levels: 4,
                patch_radius: 3,
                coarse_radius: 4,
                fine_radius: 2,
            },
            Self::High => MotionProfile {
                levels: 4,
                patch_radius: 2,
                coarse_radius: 4,
                fine_radius: 2,
            },
            Self::Balanced => MotionProfile {
                levels: 3,
                patch_radius: 2,
                coarse_radius: 3,
                fine_radius: 1,
            },
            Self::Performance => MotionProfile {
                levels: 3,
                patch_radius: 1,
                coarse_radius: 2,
                fine_radius: 1,
            },
        };
        MotionProfile {
            levels: profile.levels.min(available_levels),
            ..profile
        }
    }
}

fn patch_sample_step(quality: MotionQuality) -> u32 {
    match quality {
        MotionQuality::Ultra | MotionQuality::High => 1,
        MotionQuality::Balanced | MotionQuality::Performance => 2,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub offset: u32,
    pub grid_offset: u32,
}
pub fn pyramid(width: u32, height: u32) -> Vec<Level> {
    let (mut w, mut h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
    let (mut offset, mut grid_offset) = (0, 0);
    let mut levels = Vec::new();
    for _ in 0..4 {
        levels.push(Level {
            width: w,
            height: h,
            offset,
            grid_offset,
        });
        offset += w * h;
        grid_offset += w.div_ceil(2) * h.div_ceil(2);
        if w <= 8 || h <= 8 {
            break;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    levels
}

fn visualization_needed(mode: u32) -> bool {
    (1..=3).contains(&mode)
}

pub struct MotionField {
    pub vectors: vk::Image,
    pub confidence: vk::Image,
    pub extent: vk::Extent2D,
    pub original: vk::Extent2D,
    pub previous_id: u64,
    pub current_id: u64,
}

const STATS_WORDS: u64 = 68;

pub struct MotionEstimator {
    device: ash::Device,
    levels: Vec<Level>,
    luma: Vec<Buffer>,
    forward: Buffer,
    backward: Buffer,
    pub metadata: Buffer,
    pub stats: Buffer,
    pub visualization: Image,
    pub vectors: Image,
    pub confidence: Image,
    dense_input: Image,
    sampler: vk::Sampler,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    initialized: bool,
    decode_srgb: bool,
    pub cut_thresholds: [f32; 2],
    quality: MotionQuality,
    stats_grid: vk::Extent2D,
}
impl MotionEstimator {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        color_view: vk::ImageView,
        decode_srgb: bool,
    ) -> Result<Self, vk::Result> {
        let levels = pyramid(extent.width, extent.height);
        let last = levels.last().unwrap();
        let floats = last.offset + last.width * last.height;
        let vectors = last.grid_offset + last.width.div_ceil(2) * last.height.div_ceil(2);
        let stats_grid = vk::Extent2D {
            width: levels[0].width.div_ceil(8),
            height: levels[0].height.div_ceil(8),
        };
        let stats_size = stats_grid.width as u64 * stats_grid.height as u64 * STATS_WORDS * 4;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let buffer = |size| unsafe { Buffer::new(device, memory, size, usage, flags) };
        let image = |extent, format| unsafe {
            Image::new(
                device,
                memory,
                extent,
                format,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
        };
        let mut result = Self {
            device: device.clone(),
            levels,
            luma: vec![buffer(floats as u64 * 4)?, buffer(floats as u64 * 4)?],
            forward: buffer(vectors as u64 * 16)?,
            backward: buffer(vectors as u64 * 16)?,
            metadata: buffer(32)?,
            stats: buffer(stats_size)?,
            visualization: image(extent, vk::Format::R8G8B8A8_UNORM)?,
            // The pyramid and its coarse flow buffers are implementation
            // details.  Consumers receive a dense processing-extent signal.
            vectors: image(extent, vk::Format::R16G16_SFLOAT)?,
            confidence: image(extent, vk::Format::R8_UNORM)?,
            dense_input: image(extent, vk::Format::R16G16_SFLOAT)?,
            sampler: vk::Sampler::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            initialized: false,
            decode_srgb,
            cut_thresholds: [0.5, 0.2],
            quality: MotionQuality::Ultra,
            stats_grid,
        };
        result.sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }?;
        let bindings = (0..=10)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(match binding {
                        0 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        1..=5 | 10 => vk::DescriptorType::STORAGE_BUFFER,
                        6..=9 => vk::DescriptorType::STORAGE_IMAGE,
                        _ => unreachable!(),
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
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 12,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 8,
            },
        ];
        result.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        result.sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(result.descriptor_pool)
                    .set_layouts(&[result.descriptor_layout; 2]),
            )
        }?;
        for (i, set) in result.sets.iter().enumerate() {
            let color = [vk::DescriptorImageInfo::default()
                .sampler(result.sampler)
                .image_view(color_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&color)],
                    &[],
                );
            }
            for (binding, buffer) in [
                (1, &result.luma[i]),
                (2, &result.luma[1 - i]),
                (3, &result.forward),
                (4, &result.backward),
                (5, &result.metadata),
            ] {
                let data = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .range(buffer.size)];
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&data)],
                        &[],
                    );
                }
            }
            for (binding, image) in [
                (6, &result.visualization),
                (7, &result.vectors),
                (8, &result.confidence),
                (9, &result.dense_input),
            ] {
                let data = [vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&data)],
                        &[],
                    );
                }
            }
            let data = [vk::DescriptorBufferInfo::default()
                .buffer(result.stats.handle)
                .range(result.stats.size)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(10)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&data)],
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
                        size: 64,
                    }]),
                None,
            )
        }?;
        for bytes in [
            include_bytes!(concat!(env!("OUT_DIR"), "/luma.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/downsample.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/flow.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/confidence.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/scene_reduce.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/invalidate.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/visualize.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"), "/dense_resolve.spv")).as_slice(),
        ] {
            let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let module = unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
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
            unsafe {
                device.destroy_shader_module(module, None);
            }
            match pipeline {
                Ok(p) => result.pipelines.extend(p),
                Err((partial, error)) => {
                    for p in partial {
                        unsafe {
                            device.destroy_pipeline(p, None);
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(result)
    }
    pub fn set_quality(&mut self, quality: MotionQuality) {
        self.quality = quality;
    }
    fn params(
        &self,
        index: usize,
        parent: usize,
        valid: bool,
        mode: u32,
        active_levels: usize,
    ) -> [u32; 16] {
        let l = self.levels[index];
        let p = self.levels[parent];
        [
            l.width,
            l.height,
            l.offset,
            l.grid_offset,
            p.width,
            p.height,
            p.offset,
            p.grid_offset,
            index as u32,
            u32::from(index + 1 == active_levels),
            u32::from(valid),
            u32::from(self.decode_srgb),
            mode,
            self.cut_thresholds[0].to_bits(),
            self.cut_thresholds[1].to_bits(),
            (self.quality as u32) << 8,
        ]
    }
    unsafe fn dispatch(
        &self,
        command: vk::CommandBuffer,
        pipeline: usize,
        params: [u32; 16],
        width: u32,
        height: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines[pipeline],
            );
            self.device.cmd_push_constants(
                command,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            self.device
                .cmd_dispatch(command, width.div_ceil(8), height.div_ceil(8), 1);
            compute_memory_barrier(&self.device, command);
        }
    }
    pub unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        write: usize,
        valid: bool,
        mode: u32,
    ) {
        unsafe { self.record_inner(command, write, valid, mode, None) };
    }

    pub unsafe fn record_timed(
        &mut self,
        command: vk::CommandBuffer,
        write: usize,
        valid: bool,
        mode: u32,
        query_pool: vk::QueryPool,
        query_base: u32,
    ) {
        unsafe { self.record_inner(command, write, valid, mode, Some((query_pool, query_base))) };
    }

    unsafe fn record_inner(
        &mut self,
        command: vk::CommandBuffer,
        write: usize,
        valid: bool,
        mode: u32,
        timestamps: Option<(vk::QueryPool, u32)>,
    ) {
        unsafe {
            compute_memory_barrier(&self.device, command);
            if !self.initialized {
                for buffer in &self.luma {
                    self.device
                        .cmd_fill_buffer(command, buffer.handle, 0, buffer.size, 0);
                }
                for image in [
                    &self.visualization,
                    &self.vectors,
                    &self.confidence,
                    &self.dense_input,
                ] {
                    image_barrier(
                        &self.device,
                        command,
                        image.handle,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    );
                }
                transfer_memory_barrier(&self.device, command);
            }
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[write]],
                &[],
            );
            let profile = self.quality.profile(self.levels.len());
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base,
                );
            }
            for (i, l) in self.levels.iter().take(profile.levels).enumerate() {
                self.dispatch(
                    command,
                    usize::from(i != 0),
                    self.params(i, i.saturating_sub(1), valid, mode, profile.levels),
                    l.width,
                    l.height,
                );
                if i == 0
                    && let Some((query_pool, query_base)) = timestamps
                {
                    self.device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        query_pool,
                        query_base + 1,
                    );
                }
            }
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 2,
                );
            }
            for direction in 0..2 {
                for (i, l) in self.levels.iter().take(profile.levels).enumerate().rev() {
                    let mut p = self.params(
                        i,
                        (i + 1).min(profile.levels - 1),
                        valid,
                        mode,
                        profile.levels,
                    );
                    p[15] = direction
                        | ((self.quality as u32) << 8)
                        | ((profile.patch_radius as u32) << 16)
                        | ((profile.coarse_radius as u32) << 20)
                        | ((profile.fine_radius as u32) << 24)
                        | (patch_sample_step(self.quality) << 28);
                    self.dispatch(command, 2, p, l.width.div_ceil(2), l.height.div_ceil(2));
                }
                if let Some((query_pool, query_base)) = timestamps {
                    self.device.cmd_write_timestamp(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        query_pool,
                        query_base + 3 + direction,
                    );
                }
            }
            let grid = self.vectors.extent;
            // Resolve the private level-0 coarse grid into processing-extent
            // motion before confidence and all external consumers sample it.
            let mut dense_params = self.params(0, 0, valid, mode, profile.levels);
            dense_params[0] = grid.width;
            dense_params[1] = grid.height;
            dense_params[4] = self.levels[0].width;
            dense_params[5] = self.levels[0].height;
            self.dispatch(command, 8, dense_params, grid.width, grid.height);
            self.dispatch(
                command,
                3,
                self.params(0, 0, valid, mode, profile.levels),
                grid.width,
                grid.height,
            );
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 5,
                );
            }
            // Statistics are collected from the level-0 luminance pyramid in
            // one workgroup per tile, then reduced over the compact partial
            // buffer.  This replaces the old invocation-0 full-frame loops.
            let level = self.levels[0];
            let mut stats_params = self.params(0, 0, valid, mode, profile.levels);
            stats_params[4] = self.stats_grid.width;
            stats_params[5] = self.stats_grid.height;
            self.dispatch(command, 4, stats_params, level.width, level.height);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 6,
                );
            }
            let mut reduce_params = self.params(0, 0, valid, mode, profile.levels);
            reduce_params[0] = self.stats_grid.width;
            reduce_params[1] = self.stats_grid.height;
            self.dispatch(command, 5, reduce_params, 8, 8);
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 7,
                );
            }
            self.dispatch(
                command,
                6,
                self.params(0, 0, valid, mode, profile.levels),
                grid.width,
                grid.height,
            );
            if visualization_needed(mode) {
                self.dispatch(
                    command,
                    7,
                    self.params(0, 0, valid, mode, profile.levels),
                    self.visualization.extent.width,
                    self.visualization.extent.height,
                );
            }
            if let Some((query_pool, query_base)) = timestamps {
                self.device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    query_base + 8,
                );
            }
        }
        self.initialized = true;
    }
}
impl Drop for MotionEstimator {
    fn drop(&mut self) {
        unsafe {
            for pipeline in &self.pipelines {
                self.device.destroy_pipeline(*pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
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

    #[test]
    fn balanced_and_performance_sample_sparse_patch_lattices() {
        assert_eq!(patch_sample_step(MotionQuality::Ultra), 1);
        assert_eq!(patch_sample_step(MotionQuality::High), 1);
        assert_eq!(patch_sample_step(MotionQuality::Balanced), 2);
        assert_eq!(patch_sample_step(MotionQuality::Performance), 2);
    }

    #[test]
    fn quality_profiles_trade_precision_for_work() {
        let ultra = MotionQuality::Ultra.profile(4);
        let balanced = MotionQuality::Balanced.profile(4);
        let performance = MotionQuality::Performance.profile(4);
        assert_eq!(ultra.levels, 4);
        assert!(balanced.levels < ultra.levels);
        assert!(performance.patch_radius < balanced.patch_radius);
        assert!(performance.coarse_radius < ultra.coarse_radius);
        assert_eq!(balanced.fine_radius, 1);
        assert_eq!(performance.fine_radius, 1);
    }

    #[test]
    fn skips_visualization_for_the_original_view() {
        assert!(!visualization_needed(0));
        assert!(visualization_needed(1));
        assert!(visualization_needed(2));
        assert!(visualization_needed(3));
        assert!(!visualization_needed(4));
    }

    #[test]
    fn odd_dimensions_keep_all_pixels_and_disjoint_levels() {
        let p = pyramid(127, 65);
        assert_eq!((p[0].width, p[0].height), (64, 33));
        for pair in p.windows(2) {
            assert_eq!(
                pair[1].offset,
                pair[0].offset + pair[0].width * pair[0].height
            );
            assert_eq!(
                pair[1].grid_offset,
                pair[0].grid_offset + pair[0].width.div_ceil(2) * pair[0].height.div_ceil(2)
            );
        }
    }
}
