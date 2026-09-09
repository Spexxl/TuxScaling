pub const CRATE_NAME: &str = "tuxscaling-runtime";
mod generation;
pub use generation::*;
mod present;
pub use present::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    Disabled,
    Ready,
    Bypassed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentDecision {
    PassThrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runtime {
    state: RuntimeState,
}

impl Runtime {
    pub const fn new() -> Self {
        Self {
            state: RuntimeState::Bypassed,
        }
    }

    pub const fn state(self) -> RuntimeState {
        self.state
    }

    pub const fn process_present(&self) -> PresentDecision {
        PresentDecision::PassThrough
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PresentDecision, Runtime, RuntimeState, SwapchainImages, TemporalPipeline, map_damage_rect,
    };
    use ash::vk;
    use tuxscaling_upscaler::ResolutionPlan;

    #[test]
    fn starts_bypassed_and_passes_through() {
        let runtime = Runtime::new();
        assert_eq!(runtime.state(), RuntimeState::Bypassed);
        assert_eq!(runtime.process_present(), PresentDecision::PassThrough);
    }

    #[test]
    fn direct_swapchain_images_use_the_same_source_and_output_images() {
        let images = vec![vk::Image::null(), vk::Image::null()];
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };

        let swapchain = SwapchainImages::direct(images.clone(), extent);

        assert_eq!(swapchain.game_images, images);
        assert_eq!(swapchain.output_images, images);
        assert_eq!(swapchain.game_extent, extent);
    }

    #[test]
    fn temporal_pipeline_owns_the_explicit_resolution_plan() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            1.0,
        );

        let pipeline = TemporalPipeline::for_test(plan);

        assert_eq!(pipeline.resolution, plan);
    }

    #[test]
    fn maps_incremental_damage_into_the_aspect_fit_viewport() {
        let mapped = map_damage_rect(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1200,
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: 640, y: 360 },
                extent: vk::Extent2D {
                    width: 640,
                    height: 360,
                },
            },
        );

        assert_eq!(mapped.offset, vk::Offset2D { x: 960, y: 600 });
        assert_eq!(
            mapped.extent,
            vk::Extent2D {
                width: 960,
                height: 540
            }
        );
    }

    #[test]
    fn clips_damage_to_the_content_viewport() {
        let mapped = map_damage_rect(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1200,
            },
            vk::Rect2D {
                offset: vk::Offset2D { x: -50, y: -20 },
                extent: vk::Extent2D {
                    width: 100,
                    height: 50,
                },
            },
        );

        assert_eq!(mapped.offset, vk::Offset2D { x: 0, y: 60 });
        assert_eq!(
            mapped.extent,
            vk::Extent2D {
                width: 75,
                height: 45
            }
        );
    }
}
