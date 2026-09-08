#![allow(dead_code)]
//! Shared deterministic acceptance gates for motion and temporal fixtures.

pub const GUIDANCE_SCALES: &[f32] = &[1.0, 0.75, 0.5];

pub const MAX_MEAN_EPE: f32 = 1.0;
pub const MAX_P95_EPE: f32 = 2.0;
pub const MIN_CONFIDENCE_AUROC: f32 = 0.90;
pub const MIN_DISOCCLUSION_F1: f32 = 0.75;
pub const MIN_REACTIVE_F1: f32 = 0.70;
pub const MIN_COMPOSITION_F1: f32 = 0.65;
pub const MAX_EXPOSURE_ERROR_EV: f32 = 0.15;
pub const MIN_DEPTH_ORDERING: f32 = 0.85;

pub const MAX_MOTION_REGRESSION_EPE: f32 = 0.01;
pub const MAX_CONFIDENCE_REGRESSION_AUROC: f32 = 0.002;
pub const MAX_MASK_REGRESSION_F1: f32 = 0.005;
pub const MAX_DEPTH_REGRESSION_ORDERING: f32 = 0.005;
pub const MAX_EXPOSURE_REGRESSION_EV: f32 = 0.005;

pub fn passes_lower_gate(value: f32, minimum: f32) -> bool {
    value.is_finite() && value >= minimum
}

pub fn passes_upper_gate(value: f32, maximum: f32) -> bool {
    value.is_finite() && value <= maximum
}
