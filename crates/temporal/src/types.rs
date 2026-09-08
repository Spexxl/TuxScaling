use std::time::Duration;

use ash::vk;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SignalState {
    Estimated,
    ConstantFallback,
    #[default]
    Unavailable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DepthSemantics {
    RelativeNearIsOne,
    #[default]
    FlatFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTiming {
    pub raw: Duration,
    pub validated: Duration,
    pub smoothed: Duration,
}

impl Default for FrameTiming {
    fn default() -> Self {
        let nominal = Duration::from_micros(16_667);
        Self {
            raw: nominal,
            validated: nominal,
            smoothed: nominal,
        }
    }
}

impl FrameTiming {
    pub const fn is_finite(self) -> bool {
        // Duration stores an integer number of nanoseconds, so every value is
        // finite.  Keeping this predicate on the contract makes validation
        // explicit and leaves room for a future representation change.
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JitterSample {
    pub current: [f32; 2],
    pub previous: [f32; 2],
    pub phase: u32,
}

impl Default for JitterSample {
    fn default() -> Self {
        Self {
            current: [0.0, 0.0],
            previous: [0.0, 0.0],
            phase: 0,
        }
    }
}

impl JitterSample {
    pub const fn is_finite(self) -> bool {
        self.current[0].is_finite()
            && self.current[1].is_finite()
            && self.previous[0].is_finite()
            && self.previous[1].is_finite()
    }

    pub const fn signal_state(self) -> SignalState {
        if self.is_finite()
            && !(self.current[0] == 0.0
                && self.current[1] == 0.0
                && self.previous[0] == 0.0
                && self.previous[1] == 0.0
                && self.phase == 0)
        {
            SignalState::Estimated
        } else {
            SignalState::Unavailable
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameExtent {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GuidanceResolution {
    pub signal_extent: FrameExtent,
    pub estimator_extent: FrameExtent,
}

impl GuidanceResolution {
    pub const fn new(signal_extent: FrameExtent, estimator_extent: FrameExtent) -> Self {
        Self {
            signal_extent,
            estimator_extent,
        }
    }

    pub const fn requires_resolve(self) -> bool {
        self.signal_extent.width != self.estimator_extent.width
            || self.signal_extent.height != self.estimator_extent.height
    }

    pub fn is_valid_for(self, signal_extent: FrameExtent) -> bool {
        self.signal_extent == signal_extent
            && self.signal_extent.is_valid()
            && self.estimator_extent.is_valid()
    }
}

impl FrameExtent {
    pub const fn is_valid(self) -> bool {
        self.width != 0 && self.height != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ValidRegion {
    pub const fn full(extent: FrameExtent) -> Self {
        Self {
            x: 0,
            y: 0,
            width: extent.width,
            height: extent.height,
        }
    }

    pub const fn is_inside(self, extent: FrameExtent) -> bool {
        self.width != 0
            && self.height != 0
            && self.x <= extent.width
            && self.y <= extent.height
            && self.width <= extent.width - self.x
            && self.height <= extent.height - self.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionDirection {
    CurrentToPrevious,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionUnits {
    SourcePixels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GuidanceSignal {
    Motion = 0,
    Confidence = 1,
    Disocclusion = 2,
    Reactive = 3,
    Exposure = 4,
    RelativeDepth = 5,
    TransparencyComposition = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidanceCapabilities {
    pub estimated: [bool; 7],
    pub direction: MotionDirection,
    pub units: MotionUnits,
}

impl GuidanceCapabilities {
    pub const fn is_estimated(self, signal: GuidanceSignal) -> bool {
        self.estimated[signal as usize]
    }
}

impl GuidanceSignal {
    pub const fn fallback_value(self) -> f32 {
        match self {
            Self::Motion | Self::Confidence => 0.0,
            Self::Disocclusion | Self::Reactive | Self::Exposure | Self::RelativeDepth => 1.0,
            Self::TransparencyComposition => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidanceScalar {
    pub value: f32,
    pub state: SignalState,
}

impl GuidanceScalar {
    pub const fn constant_fallback(value: f32) -> Self {
        Self {
            value,
            state: SignalState::ConstantFallback,
        }
    }

    pub fn is_valid(self) -> bool {
        self.value.is_finite() && self.value > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidanceReset {
    None,
    Initialize,
    Resize,
    SceneChange,
    CaptureInterrupted,
    Presentation,
    LongPause,
    PresetChanged,
    ProviderFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidanceMetadata {
    pub frame_id: u64,
    pub extent: FrameExtent,
    pub valid_region: ValidRegion,
    pub reset: GuidanceReset,
    pub valid: bool,
    pub is_zero: bool,
    pub requires_history_reset: bool,
}

impl GuidanceMetadata {
    pub const fn zero(frame_id: u64, extent: FrameExtent, reset: GuidanceReset) -> Self {
        Self {
            frame_id,
            extent,
            valid_region: ValidRegion::full(extent),
            reset,
            valid: true,
            is_zero: true,
            requires_history_reset: !matches!(reset, GuidanceReset::None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidanceResource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub metadata: GuidanceMetadata,
    pub state: SignalState,
}

impl GuidanceResource {
    pub fn is_valid_for(self, frame_id: u64, extent: FrameExtent) -> bool {
        extent.is_valid()
            && self.image != vk::Image::null()
            && self.view != vk::ImageView::null()
            && self.metadata.valid
            && self.metadata.frame_id == frame_id
            && self.metadata.extent == extent
            && self.metadata.valid_region.is_inside(extent)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GuidanceView {
    pub motion: GuidanceResource,
    pub confidence: GuidanceResource,
    pub disocclusion: GuidanceResource,
    pub reactive: GuidanceResource,
    pub exposure: GuidanceResource,
    pub depth: GuidanceResource,
    pub transparency_composition: GuidanceResource,
    pub pre_exposure: GuidanceScalar,
    pub timing: FrameTiming,
    pub jitter: JitterSample,
    pub depth_semantics: DepthSemantics,
    pub direction: MotionDirection,
    pub units: MotionUnits,
    pub resolution: GuidanceResolution,
    pub requires_history_reset: bool,
}

impl GuidanceView {
    pub fn capabilities(self) -> GuidanceCapabilities {
        GuidanceCapabilities {
            estimated: [
                self.motion.state,
                self.confidence.state,
                self.disocclusion.state,
                self.reactive.state,
                self.exposure.state,
                self.depth.state,
                self.transparency_composition.state,
            ]
            .map(|state| state == SignalState::Estimated),
            direction: self.direction,
            units: self.units,
        }
    }

    pub fn resource(self, signal: GuidanceSignal) -> GuidanceResource {
        match signal {
            GuidanceSignal::Motion => self.motion,
            GuidanceSignal::Confidence => self.confidence,
            GuidanceSignal::Disocclusion => self.disocclusion,
            GuidanceSignal::Reactive => self.reactive,
            GuidanceSignal::Exposure => self.exposure,
            GuidanceSignal::RelativeDepth => self.depth,
            GuidanceSignal::TransparencyComposition => self.transparency_composition,
        }
    }

    pub fn has_expected_formats(self) -> bool {
        self.motion.format == vk::Format::R16G16_SFLOAT
            && self.confidence.format == vk::Format::R8_UNORM
            && self.disocclusion.format == vk::Format::R8_UNORM
            && self.reactive.format == vk::Format::R8_UNORM
            && self.exposure.format == vk::Format::R32_SFLOAT
            && self.depth.format == vk::Format::R32_SFLOAT
            && self.transparency_composition.format == vk::Format::R8_UNORM
    }

    pub fn is_valid_for(self, frame_id: u64, extent: FrameExtent) -> bool {
        self.has_expected_formats()
            && self.motion.is_valid_for(frame_id, extent)
            && self.confidence.is_valid_for(frame_id, extent)
            && self.disocclusion.is_valid_for(frame_id, extent)
            && self.reactive.is_valid_for(frame_id, extent)
            && self.exposure.is_valid_for(frame_id, extent)
            && self.depth.is_valid_for(frame_id, extent)
            && self.transparency_composition.is_valid_for(frame_id, extent)
            && matches!(self.direction, MotionDirection::CurrentToPrevious)
            && matches!(self.units, MotionUnits::SourcePixels)
            && self.timing.is_finite()
            && self.pre_exposure.is_valid()
            && self.jitter.is_finite()
            && self.resolution.is_valid_for(extent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn resource(metadata: GuidanceMetadata, format: vk::Format) -> GuidanceResource {
        GuidanceResource {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            format,
            metadata,
            state: SignalState::ConstantFallback,
        }
    }

    #[test]
    fn zero_metadata_resets_history_only_for_nonzero_reasons() {
        let extent = FrameExtent {
            width: 128,
            height: 96,
        };
        assert!(!GuidanceMetadata::zero(1, extent, GuidanceReset::None).requires_history_reset);
        assert!(GuidanceMetadata::zero(1, extent, GuidanceReset::Resize).requires_history_reset);
        assert!(
            GuidanceMetadata::zero(1, extent, GuidanceReset::PresetChanged).requires_history_reset
        );
    }

    #[test]
    fn guidance_view_requires_coherent_frame_resources() {
        let extent = FrameExtent {
            width: 128,
            height: 96,
        };
        let metadata = GuidanceMetadata::zero(7, extent, GuidanceReset::None);
        let view = GuidanceView {
            motion: resource(metadata, vk::Format::R16G16_SFLOAT),
            confidence: resource(metadata, vk::Format::R8_UNORM),
            disocclusion: resource(metadata, vk::Format::R8_UNORM),
            reactive: resource(metadata, vk::Format::R8_UNORM),
            exposure: resource(metadata, vk::Format::R32_SFLOAT),
            depth: resource(metadata, vk::Format::R32_SFLOAT),
            transparency_composition: resource(metadata, vk::Format::R8_UNORM),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing: FrameTiming::default(),
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::FlatFallback,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(extent, extent),
            requires_history_reset: false,
        };
        assert!(view.is_valid_for(7, extent));
        assert!(view.has_expected_formats());
        assert!(!view.is_valid_for(8, extent));

        let mut wrong = view;
        wrong.motion.format = vk::Format::R32G32_SFLOAT;
        assert!(!wrong.is_valid_for(7, extent));
    }

    #[test]
    fn guidance_resolution_bypasses_resolve_at_matching_extents() {
        let extent = FrameExtent {
            width: 1280,
            height: 720,
        };
        assert!(!GuidanceResolution::new(extent, extent).requires_resolve());
        assert!(
            GuidanceResolution::new(
                extent,
                FrameExtent {
                    width: 960,
                    height: 540,
                },
            )
            .requires_resolve()
        );
    }
}
