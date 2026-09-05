use ash::vk;

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
    pub direction: MotionDirection,
    pub units: MotionUnits,
    pub requires_history_reset: bool,
}

impl GuidanceView {
    pub fn is_valid_for(self, frame_id: u64, extent: FrameExtent) -> bool {
        self.motion.is_valid_for(frame_id, extent)
            && self.confidence.is_valid_for(frame_id, extent)
            && self.disocclusion.is_valid_for(frame_id, extent)
            && self.reactive.is_valid_for(frame_id, extent)
            && self.exposure.is_valid_for(frame_id, extent)
            && self.depth.is_valid_for(frame_id, extent)
            && matches!(self.direction, MotionDirection::CurrentToPrevious)
            && matches!(self.units, MotionUnits::SourcePixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn resource(metadata: GuidanceMetadata) -> GuidanceResource {
        GuidanceResource {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            format: vk::Format::R8_UNORM,
            metadata,
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
        assert!(GuidanceMetadata::zero(1, extent, GuidanceReset::PresetChanged).requires_history_reset);
    }

    #[test]
    fn guidance_view_requires_coherent_frame_resources() {
        let extent = FrameExtent {
            width: 128,
            height: 96,
        };
        let metadata = GuidanceMetadata::zero(7, extent, GuidanceReset::None);
        let view = GuidanceView {
            motion: resource(metadata),
            confidence: resource(metadata),
            disocclusion: resource(metadata),
            reactive: resource(metadata),
            exposure: resource(metadata),
            depth: resource(metadata),
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            requires_history_reset: false,
        };
        assert!(view.is_valid_for(7, extent));
        assert!(!view.is_valid_for(8, extent));
    }
}
