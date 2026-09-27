#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReactivePolicy {
    Neutral,
    Constant(f32),
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsrCapturePolicy {
    reactive: ReactivePolicy,
    use_composition: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyError {
    ReactiveOutOfRange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptedGuidance {
    pub motion: [f32; 2],
    pub reactive: f32,
    pub composition: f32,
    pub history_risk: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidanceSample {
    pub motion: [f32; 2],
    pub confidence: f32,
    pub disocclusion: f32,
    pub reactive: f32,
    pub composition: f32,
    pub valid: bool,
    pub in_bounds: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct FsrCapturePolicyState {
    active: FsrCapturePolicy,
    reset_pending: bool,
}

impl FsrCapturePolicy {
    pub const fn neutral() -> Self {
        Self {
            reactive: ReactivePolicy::Neutral,
            use_composition: false,
        }
    }

    pub fn new(reactive: ReactivePolicy, use_composition: bool) -> Result<Self, PolicyError> {
        if let ReactivePolicy::Constant(value) = reactive
            && (!value.is_finite() || !(0.0..=0.9).contains(&value))
        {
            return Err(PolicyError::ReactiveOutOfRange);
        }
        Ok(Self {
            reactive,
            use_composition,
        })
    }

    pub const fn reactive(self) -> ReactivePolicy {
        self.reactive
    }

    pub const fn use_composition(self) -> bool {
        self.use_composition
    }
}

impl Default for FsrCapturePolicy {
    fn default() -> Self {
        Self::neutral()
    }
}

impl FsrCapturePolicyState {
    pub const fn new(active: FsrCapturePolicy) -> Self {
        Self {
            active,
            reset_pending: false,
        }
    }

    pub const fn active(self) -> FsrCapturePolicy {
        self.active
    }

    pub const fn reset_pending(self) -> bool {
        self.reset_pending
    }

    pub fn request(&mut self, policy: FsrCapturePolicy) -> bool {
        if policy == self.active {
            return false;
        }
        self.active = policy;
        self.reset_pending = true;
        true
    }

    pub fn finish_dispatch(&mut self, succeeded: bool) {
        if succeeded {
            self.reset_pending = false;
        }
    }
}

pub fn adapt_guidance(sample: GuidanceSample, policy: FsrCapturePolicy) -> AdaptedGuidance {
    if !sample.valid {
        return AdaptedGuidance {
            motion: [0.0, 0.0],
            reactive: 0.0,
            composition: 0.0,
            history_risk: 1.0,
        };
    }

    let finite_motion = sample.motion.iter().all(|value| value.is_finite());
    let motion = if finite_motion {
        sample.motion
    } else {
        [0.0, 0.0]
    };
    let confidence = clamp01(sample.confidence);
    let disocclusion = clamp01(sample.disocclusion);
    let producer_reactive = clamp01(sample.reactive);
    let producer_composition = clamp01(sample.composition);
    let confidence_risk = smoothstep(0.2, 0.8, 1.0 - confidence);
    let bounds_risk = if sample.in_bounds { 0.0 } else { 1.0 };
    let finite_risk = if finite_motion { 0.0 } else { 1.0 };
    let history_risk = disocclusion
        .max(confidence_risk)
        .max(bounds_risk)
        .max(finite_risk);
    let reactive = match policy.reactive {
        ReactivePolicy::Neutral => 0.0,
        ReactivePolicy::Constant(value) => value,
        ReactivePolicy::Adaptive => {
            producer_reactive.max((0.9 * smoothstep(0.6, 0.95, history_risk)).clamp(0.0, 0.9))
        }
    };
    let composition = if policy.use_composition {
        producer_composition.max(0.5 * history_risk).clamp(0.0, 1.0)
    } else {
        0.0
    };

    AdaptedGuidance {
        motion,
        reactive: reactive.clamp(0.0, 0.9),
        composition,
        history_risk: history_risk.clamp(0.0, 1.0),
    }
}

pub fn motion_for_jitter_convention(
    measured_motion: [f32; 2],
    jitter: tuxscaling_temporal::JitterSample,
    cancellation_enabled: bool,
) -> [f32; 2] {
    if cancellation_enabled || jitter.signal_state() != tuxscaling_temporal::SignalState::Estimated
    {
        return measured_motion;
    }
    [
        measured_motion[0] + jitter.current[0] - jitter.previous[0],
        measured_motion[1] + jitter.current[1] - jitter.previous[1],
    ]
}

pub fn fsr_jitter_offset(jitter: tuxscaling_temporal::JitterSample) -> [f32; 2] {
    if jitter.signal_state() == tuxscaling_temporal::SignalState::Estimated {
        [-jitter.current[0], -jitter.current[1]]
    } else {
        [0.0, 0.0]
    }
}

fn clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::{
        FsrCapturePolicy, FsrCapturePolicyState, GuidanceSample, ReactivePolicy, adapt_guidance,
        motion_for_jitter_convention,
    };
    use tuxscaling_temporal::JitterSample;

    #[test]
    fn policy_rejects_non_finite_or_out_of_range_reactive_constants() {
        assert!(FsrCapturePolicy::new(ReactivePolicy::Constant(f32::NAN), false).is_err());
        assert!(FsrCapturePolicy::new(ReactivePolicy::Constant(-0.01), false).is_err());
        assert!(FsrCapturePolicy::new(ReactivePolicy::Constant(0.91), false).is_err());
        assert!(FsrCapturePolicy::new(ReactivePolicy::Constant(0.9), false).is_ok());
    }

    #[test]
    fn unreliable_motion_is_not_trusted_static_history() {
        let result = adapt_guidance(
            GuidanceSample {
                motion: [4.0, -2.0],
                confidence: 0.05,
                disocclusion: 0.0,
                reactive: 0.0,
                composition: 0.0,
                valid: true,
                in_bounds: true,
            },
            FsrCapturePolicy::new(ReactivePolicy::Adaptive, true).unwrap(),
        );

        assert_eq!(result.motion, [4.0, -2.0]);
        assert!(result.history_risk > 0.9);
        assert!(result.reactive > 0.8);
        assert!(result.composition > 0.0);
    }

    #[test]
    fn source_pixel_translations_keep_direction_and_scale() {
        let policy = FsrCapturePolicy::new(ReactivePolicy::Neutral, false).unwrap();
        for (extent, guidance_scale) in [((7, 5), 1.0), ((13, 9), 0.5)] {
            for motion in [
                [0.5, 0.0],
                [-0.5, 0.0],
                [4.0, 0.0],
                [0.0, 0.5],
                [0.0, -0.5],
                [0.0, 4.0],
            ] {
                let result = adapt_guidance(
                    GuidanceSample {
                        motion,
                        confidence: 1.0,
                        disocclusion: 0.0,
                        reactive: 0.0,
                        composition: 0.0,
                        valid: true,
                        in_bounds: true,
                    },
                    policy,
                );
                assert_eq!(
                    result.motion, motion,
                    "extent={extent:?} scale={guidance_scale}"
                );
            }
        }
    }

    #[test]
    fn disocclusion_increases_history_risk() {
        let policy = FsrCapturePolicy::new(ReactivePolicy::Adaptive, true).unwrap();
        let clear = adapt_guidance(
            GuidanceSample {
                motion: [1.0, 0.0],
                confidence: 1.0,
                disocclusion: 0.0,
                reactive: 0.0,
                composition: 0.0,
                valid: true,
                in_bounds: true,
            },
            policy,
        );
        let revealed = adapt_guidance(
            GuidanceSample {
                motion: [1.0, 0.0],
                confidence: 1.0,
                disocclusion: 1.0,
                reactive: 0.0,
                composition: 0.0,
                valid: true,
                in_bounds: true,
            },
            policy,
        );

        assert!(revealed.history_risk > clear.history_risk);
        assert_ne!(revealed.reactive, revealed.composition);
    }

    #[test]
    fn zero_guidance_is_neutral() {
        let result = adapt_guidance(
            GuidanceSample {
                motion: [3.0, -1.0],
                confidence: 0.0,
                disocclusion: 1.0,
                reactive: 1.0,
                composition: 1.0,
                valid: false,
                in_bounds: false,
            },
            FsrCapturePolicy::default(),
        );

        assert_eq!(result.motion, [0.0, 0.0]);
        assert_eq!(result.reactive, 0.0);
        assert_eq!(result.composition, 0.0);
        assert_eq!(result.history_risk, 1.0);
    }

    #[test]
    fn policy_change_resets_once_after_successful_dispatch() {
        let mut state = FsrCapturePolicyState::new(FsrCapturePolicy::default());
        let changed = FsrCapturePolicy::new(ReactivePolicy::Constant(0.2), true).unwrap();

        assert!(state.request(changed));
        assert!(state.reset_pending());
        assert!(!state.request(changed));
        state.finish_dispatch(true);
        assert!(!state.reset_pending());
        assert!(!state.request(changed));
    }

    #[test]
    fn failed_frame_does_not_advance_history() {
        let mut state = FsrCapturePolicyState::new(FsrCapturePolicy::default());
        let changed = FsrCapturePolicy::new(ReactivePolicy::Adaptive, false).unwrap();

        state.request(changed);
        state.finish_dispatch(false);
        assert!(state.reset_pending());
        state.finish_dispatch(true);
        assert!(!state.reset_pending());
    }

    #[test]
    fn jitter_adjustment_matches_resampled_color_conventions() {
        let jitter = JitterSample {
            current: [0.25, -0.125],
            previous: [-0.125, 0.166_666_67],
            phase: 2,
        };
        let world_motion = [3.0, -4.0];
        let measured_from_resampled_sequence = [
            world_motion[0] + jitter.previous[0] - jitter.current[0],
            world_motion[1] + jitter.previous[1] - jitter.current[1],
        ];

        let cancellation_enabled =
            motion_for_jitter_convention(measured_from_resampled_sequence, jitter, true);
        let cancellation_disabled =
            motion_for_jitter_convention(measured_from_resampled_sequence, jitter, false);

        assert_eq!(cancellation_enabled, measured_from_resampled_sequence);
        assert!(
            cancellation_disabled
                .iter()
                .zip(world_motion)
                .all(|(actual, expected)| (*actual - expected).abs() < 1e-6)
        );

        let sdk_cancelled_jitter = [
            cancellation_enabled[0] + jitter.current[0] - jitter.previous[0],
            cancellation_enabled[1] + jitter.current[1] - jitter.previous[1],
        ];
        for (enabled, disabled) in sdk_cancelled_jitter.iter().zip(cancellation_disabled) {
            assert!((enabled - disabled).abs() < 1e-6);
        }

        assert_eq!(
            motion_for_jitter_convention(world_motion, JitterSample::default(), true),
            world_motion
        );
        assert_eq!(
            motion_for_jitter_convention(world_motion, JitterSample::default(), false),
            world_motion
        );
    }

    #[test]
    fn capture_policy_fixture_freezes_the_training_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/fsr-capture-policy-v1.json"
        ))
        .unwrap();
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["selected_default"]["reactive"], "neutral");
        assert_eq!(fixture["selected_default"]["composition"], false);
        assert_eq!(fixture["selected_default"]["jitter_cancellation"], true);
        assert_eq!(FsrCapturePolicy::default(), FsrCapturePolicy::neutral());

        let candidates = fixture["training_candidates"]["constant_reactive"]
            .as_array()
            .unwrap();
        for value in candidates {
            let value = value.as_f64().unwrap() as f32;
            assert!(FsrCapturePolicy::new(ReactivePolicy::Constant(value), false).is_ok());
        }

        let adaptive = &fixture["training_candidates"]["adaptive"];
        let policy = FsrCapturePolicy::new(ReactivePolicy::Adaptive, true).unwrap();
        let risk_one = adapt_guidance(
            GuidanceSample {
                motion: [1.0, 0.0],
                confidence: 1.0,
                disocclusion: 1.0,
                reactive: 0.0,
                composition: 0.0,
                valid: true,
                in_bounds: true,
            },
            policy,
        );
        assert_eq!(risk_one.history_risk, 1.0);
        assert_eq!(
            risk_one.reactive,
            adaptive["reactive_cap"].as_f64().unwrap() as f32
        );
        assert_eq!(
            risk_one.composition,
            adaptive["composition_risk_gain"].as_f64().unwrap() as f32
        );

        let risk_start = adaptive["reactive_risk_smoothstep"][0].as_f64().unwrap() as f32;
        let at_risk_start = adapt_guidance(
            GuidanceSample {
                motion: [1.0, 0.0],
                confidence: 1.0,
                disocclusion: risk_start,
                reactive: 0.0,
                composition: 0.0,
                valid: true,
                in_bounds: true,
            },
            policy,
        );
        assert_eq!(at_risk_start.reactive, 0.0);
    }
}
