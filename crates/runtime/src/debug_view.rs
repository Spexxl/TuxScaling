use ash::vk;
use tuxscaling_upscaler::protection::record_spatial_blit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DebugImage {
    pub image: vk::Image,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

#[derive(Default)]
pub(crate) struct DebugImages {
    pub original: Option<DebugImage>,
    pub visualization: Option<DebugImage>,
    pub reactive_estimated: Option<DebugImage>,
    pub reactive_applied: Option<DebugImage>,
    pub disocclusion: Option<DebugImage>,
    pub depth: Option<DebugImage>,
    pub composition_estimated: Option<DebugImage>,
    pub composition_applied: Option<DebugImage>,
    pub exposure: Option<DebugImage>,
    pub fsr_raw: Option<DebugImage>,
    pub protection: Option<DebugImage>,
}

pub(crate) fn selected_debug_image(mode: u32, images: &DebugImages) -> Option<DebugImage> {
    match mode {
        0 => images.original,
        1..=3 => images.visualization,
        6 => images.reactive_estimated,
        7 => images.disocclusion,
        8 => images.depth,
        9 => images.composition_estimated,
        10 => images.exposure,
        11 => images.fsr_raw,
        12 => images.protection,
        13 => images.reactive_applied,
        14 => images.composition_applied,
        _ => None,
    }
}

/// # Safety
///
/// The source and output images must belong to `device`, and the declared
/// source/output layouts must match their current Vulkan layouts.
pub(crate) unsafe fn record_debug_image(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source: DebugImage,
    output: vk::Image,
    output_extent: vk::Extent2D,
) {
    unsafe {
        record_spatial_blit(
            device,
            command,
            source.image,
            source.extent,
            source.layout,
            output,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            output_extent,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{DebugImage, DebugImages, selected_debug_image};
    use ash::vk::{self, Handle};

    fn image(id: u64) -> DebugImage {
        DebugImage {
            image: vk::Image::from_raw(id),
            extent: vk::Extent2D {
                width: 16,
                height: 16,
            },
            layout: vk::ImageLayout::GENERAL,
        }
    }

    #[test]
    fn every_supported_view_selects_its_own_frame_resource() {
        let images = DebugImages {
            original: Some(image(1)),
            visualization: Some(image(2)),
            reactive_estimated: Some(image(3)),
            reactive_applied: Some(image(4)),
            disocclusion: Some(image(5)),
            depth: Some(image(6)),
            composition_estimated: Some(image(7)),
            composition_applied: Some(image(8)),
            exposure: Some(image(9)),
            fsr_raw: Some(image(10)),
            protection: Some(image(11)),
        };
        for (mode, expected) in [
            (0, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (6, 3),
            (7, 5),
            (8, 6),
            (9, 7),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 4),
            (14, 8),
        ] {
            assert_eq!(selected_debug_image(mode, &images), Some(image(expected)));
        }
        assert_eq!(selected_debug_image(4, &images), None);
        assert_eq!(selected_debug_image(5, &images), None);
    }

    #[test]
    fn unavailable_source_never_reuses_a_previous_signal() {
        let images = DebugImages {
            original: Some(image(1)),
            ..Default::default()
        };
        assert_eq!(selected_debug_image(0, &images), Some(image(1)));
        for mode in 1..=14 {
            assert_eq!(selected_debug_image(mode, &images), None);
        }
    }
}
