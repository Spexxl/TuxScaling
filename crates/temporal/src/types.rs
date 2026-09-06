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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameExtent {
    pub width: u32,
    pub height: u32,
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
        self.image != vk::Image::null()
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
    pub pre_exposure: f32,
    pub timing: FrameTiming,
    pub jitter: JitterSample,
    pub depth_semantics: DepthSemantics,
    pub direction: MotionDirection,
    pub units: MotionUnits,
    pub requires_history_reset: bool,
}

impl GuidanceView {
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
            && self.pre_exposure.is_finite()
            && self.pre_exposure > 0.0
            && self.jitter.is_finite()
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
            pre_exposure: 1.0,
            timing: FrameTiming::default(),
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::FlatFallback,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            requires_history_reset: false,
        };
        assert!(view.is_valid_for(7, extent));
        assert!(view.has_expected_formats());
        assert!(!view.is_valid_for(8, extent));

        let mut wrong = view;
        wrong.motion.format = vk::Format::R32G32_SFLOAT;
        assert!(!wrong.is_valid_for(7, extent));
    }
}
