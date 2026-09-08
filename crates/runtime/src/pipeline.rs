use super::{Capture, GpuTimingWindow, SwapchainInfo};
use ash::vk;
use std::time::{Duration, Instant};
use tuxscaling_capture::JitterState;
use tuxscaling_config::Config;
use tuxscaling_motion::{MotionEstimator, MotionQuality};
use tuxscaling_temporal::{
    FrameTiming, GuidanceEstimator, GuidanceReset, GuidanceResolver, History,
};
use tuxscaling_upscaler::{
    BackendConfig, ReferenceUpscaler, ResolutionPlan, UpscalerBackend, content_viewport,
};

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

/// Host-side frame clock validation shared by presentation and guidance.
///
/// The input is an elapsed monotonic timestamp, rather than wall-clock time,
/// so the state is deterministic and straightforward to exercise without a
/// Vulkan device.  Invalid deltas keep the last validated sample while a long
/// pause additionally requests a history reset.
#[derive(Debug, Clone, Copy)]
pub struct FrameTimingState {
    last_timestamp: Option<Duration>,
    last_valid: Duration,
    smoothed: Duration,
}

impl Default for FrameTimingState {
    fn default() -> Self {
        let nominal = Duration::from_micros(16_667);
        Self {
            last_timestamp: None,
            last_valid: nominal,
            smoothed: nominal,
        }
    }
}

impl FrameTimingState {
    pub const NOMINAL: Duration = Duration::from_micros(16_667);
    pub const MIN_VALID: Duration = Duration::from_micros(250);
    pub const MAX_VALID: Duration = Duration::from_millis(250);

    /// Consume a monotonic timestamp and return validated timing plus a reset,
    /// if the timestamp indicates a pause beyond the supported history gap.
    pub fn sample(
        &mut self,
        timestamp: Duration,
    ) -> (tuxscaling_temporal::FrameTiming, Option<GuidanceReset>) {
        let raw = self
            .last_timestamp
            .map_or(Self::NOMINAL, |last| timestamp.saturating_sub(last));
        self.last_timestamp = Some(timestamp);
        let reset = (raw > Self::MAX_VALID).then_some(GuidanceReset::LongPause);
        let validated = if (Self::MIN_VALID..=Self::MAX_VALID).contains(&raw) {
            self.last_valid = raw;
            raw
        } else {
            self.last_valid
        };
        self.smoothed = ema(self.smoothed, validated);
        (
            tuxscaling_temporal::FrameTiming {
                raw,
                validated,
                smoothed: self.smoothed,
            },
            reset,
        )
    }
}

/// Stable alias for callers that prefer the shorter state name.
pub type TimingState = FrameTimingState;

fn ema(previous: Duration, sample: Duration) -> Duration {
    let previous = previous.as_nanos();
    let sample = sample.as_nanos();
    Duration::from_nanos((((previous * 9) + sample) / 10) as u64)
}

pub struct TemporalPipeline {
    pub(crate) resolution: ResolutionPlan,
    pub(crate) capture: Option<Capture>,
    pub(crate) motion: Option<MotionEstimator>,
    pub(crate) guidance: Option<GuidanceEstimator>,
    pub(crate) resolver: Option<GuidanceResolver>,
    pub(crate) upscaler: Option<Box<dyn UpscalerBackend>>,
    pub(crate) history: History,
    pub(crate) start: Instant,
    pub(crate) pending_time: Duration,
    pub(crate) timing: FrameTimingState,
    pub(crate) pending_timing: FrameTiming,
    pub(crate) reset_reason: GuidanceReset,
    pub(crate) pending_quality: Option<MotionQuality>,
    pub(crate) pending_guidance_scale: Option<f32>,
    pub(crate) queries: vk::QueryPool,
    pub(crate) query_ready: Vec<bool>,
    pub(crate) timestamp_period: f32,
    pub(crate) timings: GpuTimingWindow,
    pub(crate) config: Config,
    pub(crate) jitter: JitterState,
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
                Capture::new(
                    device,
                    &memory,
                    resolution.game_extent,
                    resolution.guidance_extent,
                    info.format,
                )
            }?)
        } else {
            None
        };
        let mut motion = if let Some(capture) = &capture {
            Some(unsafe {
                MotionEstimator::new(
                    device,
                    &memory,
                    resolution.guidance_extent,
                    capture.guidance.current.view,
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
                GuidanceEstimator::new_with_slots(
                    device,
                    &memory,
                    resolution.guidance_extent,
                    capture.guidance.current.view,
                    capture.guidance.previous.view,
                    motion.confidence.view,
                    motion.vectors.view,
                    motion.metadata.handle,
                    motion.stats.handle,
                    image_count,
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
        let mut resolver = None;
        let mut upscaler: Option<Box<dyn UpscalerBackend>> = None;
        if let (Some(guidance), Some(motion), Some(capture)) = (&guidance, &motion, &capture) {
            let requires_guidance_resolve = resolution.game_extent != resolution.guidance_extent;
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
                let raw_view = guidance.view(
                    motion,
                    0,
                    resolution.guidance_extent,
                    false,
                    FrameTiming::default(),
                    GuidanceReset::Initialize,
                );
                if requires_guidance_resolve {
                    match unsafe {
                        GuidanceResolver::new(
                            device,
                            &memory,
                            resolution.game_extent,
                            resolution.guidance_extent,
                            capture.source.color.view,
                            raw_view,
                        )
                    } {
                        Ok(value) => resolver = Some(value),
                        Err(error) => {
                            eprintln!("TuxScaling: guidance resolve disabled: {error:?}")
                        }
                    }
                }
                let backend_view = resolver
                    .as_ref()
                    .map_or(raw_view, |value| value.view(raw_view));
                if requires_guidance_resolve && resolver.is_none() {
                    eprintln!(
                        "TuxScaling: reference reconstruction disabled because guidance resolve is unavailable"
                    );
                } else {
                    match unsafe {
                        ReferenceUpscaler::new(
                            device,
                            &memory,
                            capture.source.color.view,
                            resolution.game_extent,
                            resolution.output_extent,
                            info.format,
                            info.format,
                            backend_view,
                            image_count,
                        )
                    } {
                        Ok(mut value) => {
                            let backend_config = BackendConfig {
                                game_extent: resolution.game_extent,
                                output_extent: resolution.output_extent,
                                source_format: info.format,
                                output_format: info.format,
                                viewport: content_viewport(
                                    resolution.game_extent,
                                    resolution.output_extent,
                                ),
                                guidance: backend_view.capabilities(),
                            };
                            match value.configure(backend_config) {
                                Ok(()) => upscaler = Some(Box::new(value)),
                                Err(error) => {
                                    eprintln!(
                                        "TuxScaling: reference backend configuration failed: {error}"
                                    )
                                }
                            }
                        }
                        Err(error) => {
                            eprintln!("TuxScaling: reference reconstruction disabled: {error:?}")
                        }
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
            resolver,
            upscaler,
            history: History::default(),
            start: Instant::now(),
            pending_time: Duration::ZERO,
            timing: FrameTimingState::default(),
            pending_timing: FrameTiming::default(),
            reset_reason: GuidanceReset::Initialize,
            pending_quality: None,
            pending_guidance_scale: None,
            queries: vk::QueryPool::null(),
            query_ready: vec![false; image_count],
            timestamp_period,
            timings: GpuTimingWindow::default(),
            config: config.clone(),
            jitter: JitterState::new(config.jitter_mode),
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
            resolver: None,
            upscaler: None,
            history: History::default(),
            start: Instant::now(),
            pending_time: Duration::ZERO,
            timing: FrameTimingState::default(),
            pending_timing: FrameTiming::default(),
            reset_reason: GuidanceReset::Initialize,
            pending_quality: None,
            pending_guidance_scale: None,
            queries: vk::QueryPool::null(),
            query_ready: Vec::new(),
            timestamp_period: 0.0,
            timings: GpuTimingWindow::default(),
            config: Config::default(),
            jitter: JitterState::default(),
        }
    }

    pub(crate) fn reset_history(&mut self, reason: GuidanceReset) {
        self.reset_history_preserving_jitter(reason);
        self.jitter.reset();
    }

    pub(crate) fn reset_history_preserving_jitter(&mut self, reason: GuidanceReset) {
        self.history.reset();
        self.reset_reason = reason;
        if let Some(guidance) = &mut self.guidance {
            guidance.reset_history();
        }
        if let Some(upscaler) = &mut self.upscaler
            && let Err(error) = upscaler.reset()
        {
            eprintln!("TuxScaling: backend reset failed: {error}");
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

#[cfg(test)]
mod tests {
    use super::{FrameTimingState, TemporalPipeline};
    use ash::vk;
    use std::time::Duration;
    use tuxscaling_temporal::{FrameTiming, GuidanceReset};
    use tuxscaling_upscaler::{
        BackendCapabilities, BackendConfig, BackendError, BackendFrame, BackendId, UpscalerBackend,
    };

    struct ResetSpy {
        resets: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }

    impl UpscalerBackend for ResetSpy {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
                required_guidance: [false; 7],
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
            self.resets
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn timing_uses_nominal_first_sample() {
        let mut state = FrameTimingState::default();

        let (timing, reset) = state.sample(Duration::from_millis(10));

        assert_eq!(timing, FrameTiming::default());
        assert_eq!(reset, None);
    }

    #[test]
    fn timing_accepts_normal_sample_and_applies_ema() {
        let mut state = FrameTimingState::default();
        state.sample(Duration::from_millis(10));

        let (timing, reset) = state.sample(Duration::from_micros(30_000));

        assert_eq!(timing.raw, Duration::from_micros(20_000));
        assert_eq!(timing.validated, Duration::from_micros(20_000));
        assert_eq!(timing.smoothed, Duration::from_nanos(17_000_300));
        assert_eq!(reset, None);
        assert!(timing.smoothed > Duration::from_micros(16_667));
        assert!(timing.smoothed < Duration::from_micros(20_000));
    }

    #[test]
    fn timing_reuses_last_valid_delta_for_invalid_sample() {
        let mut state = FrameTimingState::default();
        state.sample(Duration::from_millis(10));
        state.sample(Duration::from_micros(30_000));

        let (timing, reset) = state.sample(Duration::from_micros(30_100));

        assert_eq!(timing.raw, Duration::from_micros(100));
        assert_eq!(timing.validated, Duration::from_micros(20_000));
        assert_eq!(reset, None);
    }

    #[test]
    fn timing_reports_long_pause_and_reuses_last_valid_delta() {
        let mut state = FrameTimingState::default();
        state.sample(Duration::from_millis(10));
        state.sample(Duration::from_micros(30_000));

        let (timing, reset) = state.sample(Duration::from_micros(330_000));

        assert_eq!(timing.raw, Duration::from_micros(300_000));
        assert_eq!(timing.validated, Duration::from_micros(20_000));
        assert_eq!(reset, Some(GuidanceReset::LongPause));
    }

    #[test]
    fn timing_accepts_both_validity_boundaries() {
        let mut state = FrameTimingState::default();
        state.sample(Duration::ZERO);

        let (minimum, minimum_reset) = state.sample(Duration::from_micros(250));
        assert_eq!(minimum.raw, Duration::from_micros(250));
        assert_eq!(minimum.validated, Duration::from_micros(250));
        assert_eq!(minimum_reset, None);

        let (maximum, maximum_reset) = state.sample(Duration::from_millis(250) + minimum.raw);
        assert_eq!(maximum.raw, Duration::from_millis(250));
        assert_eq!(maximum.validated, Duration::from_millis(250));
        assert_eq!(maximum_reset, None);
    }

    #[test]
    fn history_reset_is_forwarded_to_the_generic_backend() {
        let mut pipeline = TemporalPipeline::for_test(tuxscaling_upscaler::ResolutionPlan::new(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            1.0,
        ));
        let resets = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        pipeline.upscaler = Some(Box::new(ResetSpy {
            resets: resets.clone(),
        }));

        pipeline.reset_history(GuidanceReset::ProviderFailure);

        assert_eq!(resets.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(pipeline.reset_reason, GuidanceReset::ProviderFailure);
    }
}
