pub const CRATE_NAME: &str = "tuxscaling-temporal";
mod gpu;
mod history;
mod types;
pub use gpu::*;
pub use history::*;
pub use types::*;

/// Deterministic, CPU-only sequences and metrics used by the temporal quality
/// laboratory.  The fixtures intentionally model only signals that can be
/// estimated from final captured color; they are not GPU producers.
pub mod quality {
    use std::cmp::Ordering;

    #[derive(Debug, Clone, PartialEq)]
    pub struct SequenceFixture {
        pub width: u32,
        pub height: u32,
        pub previous: Vec<[f32; 4]>,
        pub current: Vec<[f32; 4]>,
        pub motion: Vec<[f32; 2]>,
        pub valid: Vec<bool>,
        pub occlusion: Vec<bool>,
        pub transparency: Vec<bool>,
        pub hud: Vec<bool>,
        pub depth: Vec<f32>,
        pub exposure_ev: f32,
        pub scene_cut: bool,
    }

    pub type Sequence = SequenceFixture;
    pub type Fixture = SequenceFixture;

    impl SequenceFixture {
        pub fn len(&self) -> usize {
            self.previous.len()
        }

        pub fn is_empty(&self) -> bool {
            self.previous.is_empty()
        }
    }

    fn pixel(_width: u32, x: u32, y: u32, seed: u32) -> [f32; 4] {
        let mut value = x.wrapping_mul(0x9e37_79b9).rotate_left(7)
            ^ y.wrapping_mul(0x85eb_ca6b).rotate_left(13)
            ^ seed.wrapping_mul(0xc2b2_ae35);
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        let r = (value & 255) as f32 / 255.0;
        let g = ((value >> 8) & 255) as f32 / 255.0;
        let b = ((value >> 16) & 255) as f32 / 255.0;
        [r, g, b, 1.0]
    }

    fn scene(width: u32, height: u32, seed: u32) -> Vec<[f32; 4]> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| pixel(width, x, y, seed)))
            .collect()
    }

    fn shifted(
        previous: &[[f32; 4]],
        width: u32,
        height: u32,
        dx: f32,
        dy: f32,
    ) -> (Vec<[f32; 4]>, Vec<[f32; 2]>) {
        let mut current = Vec::with_capacity(previous.len());
        let mut motion = Vec::with_capacity(previous.len());
        for y in 0..height {
            for x in 0..width {
                let source_x = (x as f32 - dx)
                    .round()
                    .clamp(0.0, width.saturating_sub(1) as f32);
                let source_y = (y as f32 - dy)
                    .round()
                    .clamp(0.0, height.saturating_sub(1) as f32);
                let index = source_y as usize * width as usize + source_x as usize;
                current.push(previous[index]);
                // Motion is explicitly current-to-previous, matching the
                // public guidance contract.
                motion.push([-dx, -dy]);
            }
        }
        (current, motion)
    }

    fn empty_labels(len: usize) -> (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>) {
        (
            vec![true; len],
            vec![false; len],
            vec![false; len],
            vec![false; len],
        )
    }

    fn base_translation(width: u32, height: u32, dx: f32, dy: f32, seed: u32) -> SequenceFixture {
        let previous = scene(width, height, seed);
        let (current, motion) = shifted(&previous, width, height, dx, dy);
        let len = previous.len();
        let (valid, occlusion, transparency, hud) = empty_labels(len);
        SequenceFixture {
            width,
            height,
            previous,
            current,
            motion,
            valid,
            occlusion,
            transparency,
            hud,
            depth: vec![1.0; len],
            exposure_ev: 0.0,
            scene_cut: false,
        }
    }

    /// A textured translation with a known current-to-previous displacement.
    pub fn translation(width: u32, height: u32) -> SequenceFixture {
        base_translation(width, height, 5.0, -3.0, 1)
    }

    /// A small affine (scale plus translation) sequence.  Its motion labels
    /// are the exact displacement generated at each pixel.
    pub fn affine_motion(width: u32, height: u32) -> SequenceFixture {
        let previous = scene(width, height, 2);
        let cx = (width.saturating_sub(1) as f32) * 0.5;
        let cy = (height.saturating_sub(1) as f32) * 0.5;
        let scale = 1.035;
        let dx = 2.0;
        let dy = -1.5;
        let mut current = Vec::with_capacity(previous.len());
        let mut motion = Vec::with_capacity(previous.len());
        for y in 0..height {
            for x in 0..width {
                let source_x = (cx + (x as f32 - cx) / scale - dx)
                    .round()
                    .clamp(0.0, width.saturating_sub(1) as f32);
                let source_y = (cy + (y as f32 - cy) / scale - dy)
                    .round()
                    .clamp(0.0, height.saturating_sub(1) as f32);
                current.push(previous[source_y as usize * width as usize + source_x as usize]);
                motion.push([source_x - x as f32, source_y - y as f32]);
            }
        }
        let len = previous.len();
        let (valid, occlusion, transparency, hud) = empty_labels(len);
        SequenceFixture {
            width,
            height,
            previous,
            current,
            motion,
            valid,
            occlusion,
            transparency,
            hud,
            depth: vec![1.0; len],
            exposure_ev: 0.0,
            scene_cut: false,
        }
    }

    /// Two planes with independent translations and deterministic depth.
    pub fn layered_parallax(width: u32, height: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 2.0, 1.0, 3);
        result.depth.fill(0.35);
        let foreground_x = width / 4..(width * 3 / 4).max(width / 4 + 1);
        let foreground_y = height / 4..(height * 3 / 4).max(height / 4 + 1);
        for y in foreground_y {
            for x in foreground_x.clone() {
                let index = y as usize * width as usize + x as usize;
                result.motion[index] = [-7.0, -3.0];
                result.depth[index] = 1.0;
                result.current[index] = result.previous[index];
            }
        }
        result
    }

    /// A newly revealed rectangle marks disocclusion in the current frame.
    pub fn occlusion(width: u32, height: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 0.0, 0.0, 4);
        let x_range = width / 3..(width * 2 / 3).max(width / 3 + 1);
        let y_range = height / 3..(height * 2 / 3).max(height / 3 + 1);
        for y in y_range {
            for x in x_range.clone() {
                let index = y as usize * width as usize + x as usize;
                result.current[index] = [1.0, 1.0, 1.0, 1.0];
                result.occlusion[index] = true;
            }
        }
        result
    }

    /// A semi-transparent rectangle has a known alpha-like composition label.
    pub fn transparency(width: u32, height: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 1.0, 0.0, 5);
        let x_range = width / 4..(width / 2).max(width / 4 + 1);
        let y_range = height / 4..(height * 3 / 4).max(height / 4 + 1);
        for y in y_range {
            for x in x_range.clone() {
                let index = y as usize * width as usize + x as usize;
                let source = result.current[index];
                result.current[index] = [
                    source[0] * 0.5 + 0.5,
                    source[1] * 0.5 + 0.5,
                    source[2] * 0.5 + 0.5,
                    1.0,
                ];
                result.transparency[index] = true;
            }
        }
        result
    }

    /// A static HUD bar is intentionally independent of scene motion.
    pub fn hud(width: u32, height: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 4.0, 0.0, 6);
        for y in 0..height.min(4) {
            for x in 0..width {
                let index = y as usize * width as usize + x as usize;
                result.current[index] = [0.1, 0.9, 0.2, 1.0];
                result.hud[index] = true;
            }
        }
        result
    }

    fn exposure_fixture(width: u32, height: u32, multiplier: f32, seed: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 0.0, 0.0, seed);
        for pixel in &mut result.current {
            pixel[0] = (pixel[0] * multiplier).clamp(0.0, 1.0);
            pixel[1] = (pixel[1] * multiplier).clamp(0.0, 1.0);
            pixel[2] = (pixel[2] * multiplier).clamp(0.0, 1.0);
        }
        result.exposure_ev = multiplier.log2();
        result
    }

    /// A global luminance flash is not a scene cut.
    pub fn flash(width: u32, height: u32) -> SequenceFixture {
        exposure_fixture(width, height, 1.5, 7)
    }

    /// A global fade is not a scene cut.
    pub fn fade(width: u32, height: u32) -> SequenceFixture {
        exposure_fixture(width, height, 0.65, 8)
    }

    /// A new deterministic scene is a labelled scene cut.
    pub fn scene_cut(width: u32, height: u32) -> SequenceFixture {
        let mut result = base_translation(width, height, 0.0, 0.0, 9);
        result.current = scene(width, height, 10);
        result.scene_cut = true;
        result
    }

    pub fn endpoint_error(estimated: &[[f32; 2]], reference: &[[f32; 2]]) -> f32 {
        let count = estimated.len().min(reference.len());
        if count == 0 {
            return 0.0;
        }
        estimated
            .iter()
            .zip(reference.iter())
            .take(count)
            .map(|(estimate, expected)| {
                ((estimate[0] - expected[0]).powi(2) + (estimate[1] - expected[1]).powi(2)).sqrt()
            })
            .sum::<f32>()
            / count as f32
    }

    pub fn epe(estimated: &[[f32; 2]], reference: &[[f32; 2]]) -> f32 {
        endpoint_error(estimated, reference)
    }

    pub fn mean_epe(estimated: &[[f32; 2]], reference: &[[f32; 2]]) -> f32 {
        endpoint_error(estimated, reference)
    }

    /// Percentile accepts either a fraction (`0.95`) or a percentage (`95`).
    pub fn percentile(values: &[f32], requested: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if sorted.is_empty() {
            return 0.0;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let quantile = if requested <= 1.0 {
            requested.clamp(0.0, 1.0)
        } else {
            (requested / 100.0).clamp(0.0, 1.0)
        };
        let position = quantile * (sorted.len() - 1) as f32;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f32)
        }
    }

    pub fn f1(predicted: &[bool], expected: &[bool]) -> f32 {
        let mut true_positive = 0.0;
        let mut false_positive = 0.0;
        let mut false_negative = 0.0;
        for (prediction, label) in predicted.iter().zip(expected.iter()) {
            match (*prediction, *label) {
                (true, true) => true_positive += 1.0,
                (true, false) => false_positive += 1.0,
                (false, true) => false_negative += 1.0,
                (false, false) => {}
            }
        }
        let denominator = 2.0 * true_positive + false_positive + false_negative;
        if denominator == 0.0 {
            0.0
        } else {
            2.0 * true_positive / denominator
        }
    }

    pub fn f1_score(predicted: &[bool], expected: &[bool]) -> f32 {
        f1(predicted, expected)
    }

    /// Pairwise AUROC avoids threshold and ordering ambiguities for ties.
    pub fn auroc(scores: &[f32], labels: &[bool]) -> f32 {
        let positives: Vec<f32> = scores
            .iter()
            .zip(labels.iter())
            .filter_map(|(score, label)| (*label && score.is_finite()).then_some(*score))
            .collect();
        let negatives: Vec<f32> = scores
            .iter()
            .zip(labels.iter())
            .filter_map(|(score, label)| (!*label && score.is_finite()).then_some(*score))
            .collect();
        if positives.is_empty() || negatives.is_empty() {
            return 0.0;
        }
        let mut wins = 0.0;
        for positive in &positives {
            for negative in &negatives {
                wins += match positive.partial_cmp(negative).unwrap_or(Ordering::Equal) {
                    Ordering::Greater => 1.0,
                    Ordering::Equal => 0.5,
                    Ordering::Less => 0.0,
                };
            }
        }
        wins / (positives.len() * negatives.len()) as f32
    }

    pub fn ev_error(estimated: f32, expected: f32) -> f32 {
        (estimated - expected).abs()
    }

    pub fn exposure_error(estimated: f32, expected: f32) -> f32 {
        ev_error(estimated, expected)
    }

    /// Fraction of pairs whose relative near/far ordering is preserved.
    pub fn depth_order(estimated: &[f32], expected: &[f32]) -> f32 {
        let count = estimated.len().min(expected.len());
        if count < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut correct = 0.0;
        for first in 0..count {
            for second in first + 1..count {
                let expected_delta = expected[first] - expected[second];
                if expected_delta.abs() <= f32::EPSILON {
                    continue;
                }
                total += 1.0;
                let estimated_delta = estimated[first] - estimated[second];
                if estimated_delta.abs() <= f32::EPSILON
                    || expected_delta.signum() == estimated_delta.signum()
                {
                    correct += if estimated_delta.abs() <= f32::EPSILON {
                        0.5
                    } else {
                        1.0
                    };
                }
            }
        }
        if total == 0.0 { 0.0 } else { correct / total }
    }

    pub fn depth_order_accuracy(estimated: &[f32], expected: &[f32]) -> f32 {
        depth_order(estimated, expected)
    }

    /// Mean squared error over all available RGBA channels.
    pub fn image_error(estimated: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
        let count = estimated.len().min(expected.len());
        if count == 0 {
            return 0.0;
        }
        estimated
            .iter()
            .zip(expected.iter())
            .take(count)
            .flat_map(|(estimate, truth)| estimate.iter().zip(truth.iter()))
            .map(|(estimate, truth)| (estimate - truth).powi(2))
            .sum::<f32>()
            / (count * 4) as f32
    }

    pub fn mse(estimated: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
        image_error(estimated, expected)
    }
}
