use super::{Capture, SwapchainInfo};
use ash::vk;
use std::time::{Duration, Instant};
use tuxscaling_config::Config;
use tuxscaling_motion::{MotionEstimator, MotionQuality};
use tuxscaling_temporal::{GuidanceEstimator, GuidanceReset, History};
use tuxscaling_upscaler::{ReferenceUpscaler, ResolutionPlan};

pub struct TemporalPipelineDescriptor<'a> {
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    pub info: SwapchainInfo,
    pub resolution: ResolutionPlan,
    pub config: &'a Config,
    pub capture_enabled: bool,
    pub image_count: usize,
}

pub struct TemporalPipeline {
    pub(crate) resolution: ResolutionPlan,
    pub(crate) capture: Option<Capture>,
    pub(crate) motion: Option<MotionEstimator>,
    pub(crate) guidance: Option<GuidanceEstimator>,
    pub(crate) upscaler: Option<ReferenceUpscaler>,
    pub(crate) history: History,
    pub(crate) start: Instant,
    pub(crate) last_time: Option<Duration>,
    pub(crate) pending_time: Duration,
    pub(crate) reset_reason: GuidanceReset,
    pub(crate) pending_quality: Option<MotionQuality>,
    pub(crate) pending_processing_scale: Option<f32>,
    pub(crate) queries: vk::QueryPool,
    pub(crate) query_ready: Vec<bool>,
    pub(crate) timestamp_period: f32,
    pub(crate) timings: Vec<[f32; super::GPU_PHASES]>,
    pub(crate) config: Config,
}

impl TemporalPipeline {
    pub unsafe fn new(descriptor: TemporalPipelineDescriptor<'_>) -> Result<Self, vk::Result> {
        let TemporalPipelineDescriptor {
            instance,
            physical,
            device,
            info,
            resolution,
            config,
            capture_enabled,
            image_count,
        } = descriptor;
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let capture = if capture_enabled {
            Some(unsafe {
                Capture::new(device, &memory, resolution.processing_extent, info.format)
            }?)
        } else {
            None
        };
        let mut motion = if let Some(capture) = &capture {
            Some(unsafe {
                MotionEstimator::new(
                    device,
                    &memory,
                    resolution.processing_extent,
                    capture.color.view,
                    matches!(
                        info.format,
                        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
                    ),
                )
            }?)
        } else {
            None
        };
        if let Some(motion) = &mut motion {
            motion.set_quality(motion_quality(config.motion_quality));
            motion.cut_thresholds = [
                config.scene_distance_threshold,
                config.scene_consistency_threshold,
            ];
        }
        let guidance = if let (Some(capture), Some(motion)) = (&capture, &motion) {
            match unsafe {
                GuidanceEstimator::new(
                    device,
                    &memory,
                    resolution.processing_extent,
                    capture.color.view,
                    capture.previous.view,
                    motion.confidence.view,
                )
            } {
                Ok(value) => Some(value),
                Err(error) => {
                    eprintln!("TuxScaling: guidance estimation disabled: {error:?}");
                    None
                }
            }
        } else {
            None
        };
        let mut upscaler = None;
        if let (Some(guidance), Some(motion), Some(capture)) = (&guidance, &motion, &capture) {
            let features =
                unsafe { instance.get_physical_device_format_properties(physical, info.format) }
                    .optimal_tiling_features;
            let device_features = unsafe { instance.get_physical_device_features(physical) };
            let required = vk::FormatFeatureFlags::STORAGE_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::BLIT_SRC
                | vk::FormatFeatureFlags::BLIT_DST;
            if features.contains(required)
                && device_features.shader_storage_image_write_without_format != 0
            {
                let view = guidance.view(
                    motion,
                    0,
                    resolution.processing_extent,
                    false,
                    GuidanceReset::Initialize,
                );
                match unsafe {
                    ReferenceUpscaler::new(
                        device,
                        &memory,
                        capture.color.view,
                        resolution.processing_extent,
                        resolution.output_extent,
                        info.format,
                        view,
                    )
                } {
                    Ok(value) => upscaler = Some(value),
                    Err(error) => {
                        eprintln!("TuxScaling: reference reconstruction disabled: {error:?}")
                    }
                }
            } else {
                eprintln!(
                    "TuxScaling: swapchain format {:?} lacks storage-image support; reconstruction bypassed",
                    info.format
                );
            }
        }
        let timestamp_period = if properties.limits.timestamp_compute_and_graphics != 0 {
            properties.limits.timestamp_period
        } else {
            0.0
        };
        Ok(Self {
            resolution,
            capture,
            motion,
            guidance,
            upscaler,
            history: History::default(),
            start: Instant::now(),
            last_time: None,
            pending_time: Duration::ZERO,
            reset_reason: GuidanceReset::Initialize,
            pending_quality: None,
            pending_processing_scale: None,
            queries: vk::QueryPool::null(),
            query_ready: vec![false; image_count],
            timestamp_period,
            timings: Vec::new(),
            config: config.clone(),
        })
    }

    pub(crate) unsafe fn rebuild(
        &mut self,
        descriptor: TemporalPipelineDescriptor<'_>,
        device: &ash::Device,
    ) -> Result<(), vk::Result> {
        let replacement = unsafe { Self::new(descriptor) }?;
        let query_pool = std::mem::replace(&mut self.queries, vk::QueryPool::null());
        if query_pool != vk::QueryPool::null() {
            unsafe { device.destroy_query_pool(query_pool, None) };
        }
        *self = replacement;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn for_test(resolution: ResolutionPlan) -> Self {
        Self {
            resolution,
            capture: None,
            motion: None,
            guidance: None,
            upscaler: None,
            history: History::default(),
            start: Instant::now(),
            last_time: None,
            pending_time: Duration::ZERO,
            reset_reason: GuidanceReset::Initialize,
            pending_quality: None,
            pending_processing_scale: None,
            queries: vk::QueryPool::null(),
            query_ready: Vec::new(),
            timestamp_period: 0.0,
            timings: Vec::new(),
            config: Config::default(),
        }
    }
}

fn motion_quality(quality: tuxscaling_config::MotionQuality) -> MotionQuality {
    match quality {
        tuxscaling_config::MotionQuality::Ultra => MotionQuality::Ultra,
        tuxscaling_config::MotionQuality::High => MotionQuality::High,
        tuxscaling_config::MotionQuality::Balanced => MotionQuality::Balanced,
        tuxscaling_config::MotionQuality::Performance => MotionQuality::Performance,
    }
}
