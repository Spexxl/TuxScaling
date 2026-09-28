use crate::content_viewport;
use ash::vk;
use std::time::Duration;
use thiserror::Error;
use tuxscaling_config::{ProtectionMode, ProtectionRegion};
use tuxscaling_vulkan::{image_barrier, transfer_memory_barrier};

mod gpu;
pub use gpu::{ProtectionFrame, ProtectionOutputs, ProtectionRenderer};

pub const MAX_PROTECTION_REGIONS: usize = 4;
pub const RISK_DECAY_SECONDS: f32 = 0.08;
pub const REGION_FEATHER_SOURCE_PIXELS: f32 = 2.0;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionError {
    #[error("at most four protection regions are supported")]
    TooManyRegions,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProtectionSettings {
    pub mode: ProtectionMode,
    regions: [Option<ProtectionRegion>; MAX_PROTECTION_REGIONS],
    region_count: usize,
}

impl ProtectionSettings {
    pub fn new(
        mode: ProtectionMode,
        regions: &[ProtectionRegion],
    ) -> Result<Self, ProtectionError> {
        if regions.len() > MAX_PROTECTION_REGIONS {
            return Err(ProtectionError::TooManyRegions);
        }
        let mut result = Self {
            mode,
            regions: [None; MAX_PROTECTION_REGIONS],
            region_count: regions.len(),
        };
        for (index, region) in regions.iter().enumerate() {
            result.regions[index] = Some(*region);
        }
        Ok(result)
    }

    pub fn regions(self) -> impl Iterator<Item = ProtectionRegion> {
        self.regions.into_iter().flatten().take(self.region_count)
    }

    pub fn is_enabled(self) -> bool {
        self.mode != ProtectionMode::Disabled
    }
}

impl Default for ProtectionSettings {
    fn default() -> Self {
        Self::new(ProtectionMode::Disabled, &[]).expect("empty regions are valid")
    }
}

pub fn decay_risk(
    current_risk: f32,
    previous_risk: f32,
    elapsed: Duration,
    reprojection_valid: bool,
) -> f32 {
    let current = if current_risk.is_finite() {
        current_risk.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let previous = if reprojection_valid && previous_risk.is_finite() {
        previous_risk.clamp(0.0, 1.0)
    } else {
        0.0
    };
    current.max(previous * (-elapsed.as_secs_f32() / RISK_DECAY_SECONDS).exp())
}

pub fn region_weight(region: ProtectionRegion, pixel: [f32; 2], extent: [f32; 2]) -> f32 {
    if !pixel.iter().all(|value| value.is_finite())
        || !extent.iter().all(|value| value.is_finite() && *value > 0.0)
    {
        return 0.0;
    }
    let min = region.min();
    let max = region.max();
    let distance = (pixel[0] - min[0] * extent[0])
        .min(max[0] * extent[0] - pixel[0])
        .min(pixel[1] - min[1] * extent[1])
        .min(max[1] * extent[1] - pixel[1]);
    (0.5 + distance / REGION_FEATHER_SOURCE_PIXELS).clamp(0.0, 1.0)
}

pub fn spatial_weight(
    settings: ProtectionSettings,
    risk: f32,
    source_position: [f32; 2],
    source_extent: [f32; 2],
) -> f32 {
    if !source_position
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return 0.0;
    }
    let adaptive = match settings.mode {
        ProtectionMode::Adaptive | ProtectionMode::AdaptiveAndRegions => {
            if risk.is_finite() {
                risk.clamp(0.0, 1.0)
            } else {
                1.0
            }
        }
        ProtectionMode::Disabled | ProtectionMode::Regions => 0.0,
    };
    let regions = match settings.mode {
        ProtectionMode::Regions | ProtectionMode::AdaptiveAndRegions => settings
            .regions()
            .map(|region| {
                region_weight(
                    region,
                    [
                        source_position[0] * source_extent[0],
                        source_position[1] * source_extent[1],
                    ],
                    source_extent,
                )
            })
            .fold(0.0_f32, f32::max),
        ProtectionMode::Disabled | ProtectionMode::Adaptive => 0.0,
    };
    adaptive.max(regions)
}

#[allow(clippy::too_many_arguments)]
/// # Safety
///
/// The images and command buffer must belong to `device`, and their supplied
/// layouts must match their actual layouts when this command is recorded.
pub unsafe fn record_spatial_blit(
    device: &ash::Device,
    command: vk::CommandBuffer,
    source: vk::Image,
    source_extent: vk::Extent2D,
    source_layout: vk::ImageLayout,
    output: vk::Image,
    output_layout: vk::ImageLayout,
    output_extent: vk::Extent2D,
) {
    let viewport = content_viewport(source_extent, output_extent);
    let left = (viewport.offset[0] * output_extent.width as f32).round() as i32;
    let top = (viewport.offset[1] * output_extent.height as f32).round() as i32;
    let right =
        ((viewport.offset[0] + viewport.size[0]) * output_extent.width as f32).round() as i32;
    let bottom =
        ((viewport.offset[1] + viewport.size[1]) * output_extent.height as f32).round() as i32;
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    unsafe {
        image_barrier(
            device,
            command,
            source,
            source_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        image_barrier(
            device,
            command,
            output,
            output_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        device.cmd_clear_color_image(
            command,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            &[tuxscaling_vulkan::color_range()],
        );
        transfer_memory_barrier(device, command);
        device.cmd_blit_image(
            command,
            source,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers)
                .dst_subresource(layers)
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: source_extent.width as i32,
                        y: source_extent.height as i32,
                        z: 1,
                    },
                ])
                .dst_offsets([
                    vk::Offset3D {
                        x: left,
                        y: top,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: right,
                        y: bottom,
                        z: 1,
                    },
                ])],
            vk::Filter::LINEAR,
        );
        image_barrier(
            device,
            command,
            source,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            source_layout,
        );
        image_barrier(
            device,
            command,
            output,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{ProtectionSettings, decay_risk, region_weight, spatial_weight};
    use std::time::Duration;
    use tuxscaling_config::{ProtectionMode, ProtectionRegion};

    #[test]
    fn region_feather_is_bounded_to_two_source_pixels() {
        let region = ProtectionRegion::new([0.25, 0.25], [0.75, 0.75]).unwrap();
        let extent = [100.0, 100.0];

        assert_eq!(region_weight(region, [50.0, 50.0], extent), 1.0);
        assert_eq!(region_weight(region, [23.0, 50.0], extent), 0.0);
        assert_eq!(region_weight(region, [25.0, 50.0], extent), 0.5);
        assert_eq!(region_weight(region, [27.0, 50.0], extent), 1.0);
    }

    #[test]
    fn static_geometry_is_not_automatically_spatial() {
        let region = ProtectionRegion::new([0.1, 0.1], [0.2, 0.2]).unwrap();
        let settings = ProtectionSettings::new(ProtectionMode::Adaptive, &[region]).unwrap();

        assert_eq!(
            spatial_weight(settings, 0.0, [0.5, 0.5], [100.0, 100.0]),
            0.0
        );
        assert_eq!(
            spatial_weight(settings, 1.0, [0.5, 0.5], [100.0, 100.0]),
            1.0
        );
    }

    #[test]
    fn explicit_regions_protect_screen_space_without_motion() {
        let region = ProtectionRegion::new([0.1, 0.1], [0.2, 0.2]).unwrap();
        let settings = ProtectionSettings::new(ProtectionMode::Regions, &[region]).unwrap();

        assert_eq!(
            spatial_weight(settings, 0.0, [0.15, 0.15], [100.0, 100.0]),
            1.0
        );
        assert_eq!(
            spatial_weight(settings, 1.0, [0.5, 0.5], [100.0, 100.0]),
            0.0
        );
    }

    #[test]
    fn risk_rises_immediately_and_decays_by_elapsed_time() {
        let half_time = Duration::from_secs_f32(0.04);
        let once = decay_risk(0.0, 1.0, half_time, true);
        let twice = decay_risk(0.0, once, half_time, true);
        let at_thirty_hz = decay_risk(0.0, 1.0, Duration::from_secs_f32(0.08), true);

        assert!((twice - at_thirty_hz).abs() < 1e-6);
        assert_eq!(decay_risk(1.0, 0.0, half_time, true), 1.0);
        assert_eq!(decay_risk(0.0, 1.0, half_time, false), 0.0);
    }

    #[test]
    fn region_count_is_bounded_for_gpu_push_constants() {
        let region = ProtectionRegion::new([0.0, 0.0], [1.0, 1.0]).unwrap();
        assert!(ProtectionSettings::new(ProtectionMode::Regions, &[region; 4]).is_ok());
        assert!(ProtectionSettings::new(ProtectionMode::Regions, &[region; 5]).is_err());
    }
}
