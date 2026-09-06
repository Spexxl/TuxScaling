use ash::vk;
use thiserror::Error;

mod reference;
pub use reference::{ReferenceUpscaler, scaled_extent};

pub const CRATE_NAME: &str = "tuxscaling-upscaler";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendId {
    Reference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub temporal: bool,
    pub frame_generation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputResolution {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("upscaler backend is unavailable")]
    Unavailable,
    #[error("upscaler configuration is invalid: {0}")]
    InvalidConfiguration(String),
    #[error("upscaler backend failed: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy)]
pub struct ImageResource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub resolution: InputResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionPlan {
    pub game_extent: vk::Extent2D,
    pub processing_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentViewport {
    pub offset: [f32; 2],
    pub size: [f32; 2],
}

pub fn content_viewport(input: vk::Extent2D, output: vk::Extent2D) -> ContentViewport {
    let input_aspect = input.width as f32 / input.height as f32;
    let output_aspect = output.width as f32 / output.height as f32;
    let size = if output_aspect > input_aspect {
        [input_aspect / output_aspect, 1.0]
    } else {
        [1.0, output_aspect / input_aspect]
    };
    ContentViewport {
        offset: [(1.0 - size[0]) * 0.5, (1.0 - size[1]) * 0.5],
        size,
    }
}

impl ResolutionPlan {
    pub fn new(
        game_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        processing_scale: f32,
    ) -> Self {
        Self {
            game_extent,
            processing_extent: scaled_extent(game_extent, processing_scale),
            output_extent,
        }
    }
}

pub trait UpscalerBackend {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn resize(
        &mut self,
        input: InputResolution,
        output: InputResolution,
    ) -> Result<(), BackendError>;
    fn reset(&mut self) -> Result<(), BackendError>;
}

#[cfg(test)]
mod tests {
    use super::content_viewport;
    use super::{
        BackendCapabilities, BackendError, BackendId, InputResolution, ResolutionPlan,
        UpscalerBackend,
    };
    use ash::vk;

    struct Dummy;

    impl UpscalerBackend for Dummy {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
            }
        }
        fn resize(&mut self, _: InputResolution, _: InputResolution) -> Result<(), BackendError> {
            Ok(())
        }
        fn reset(&mut self) -> Result<(), BackendError> {
            Ok(())
        }
    }

    #[test]
    fn dummy_backend_satisfies_contract() {
        let mut backend = Dummy;
        assert_eq!(backend.id(), BackendId::Reference);
        assert!(backend.capabilities().temporal);
        backend
            .resize(
                InputResolution {
                    width: 1,
                    height: 1,
                },
                InputResolution {
                    width: 2,
                    height: 2,
                },
            )
            .unwrap();
        backend.reset().unwrap();
    }

    #[test]
    fn resolution_plan_keeps_game_input_at_full_processing_scale() {
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

        assert_eq!(plan.game_extent, plan.processing_extent);
        assert_eq!(plan.output_extent.width, 1920);
    }

    #[test]
    fn resolution_plan_scales_only_processing_input() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            0.75,
        );

        assert_eq!(
            plan.processing_extent,
            vk::Extent2D {
                width: 960,
                height: 540,
            }
        );
    }

    #[test]
    fn content_viewport_preserves_the_input_aspect_ratio() {
        let viewport = content_viewport(
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
            vk::Extent2D {
                width: 1920,
                height: 1200,
            },
        );

        assert!((viewport.offset[0] - 0.0).abs() < 0.0001);
        assert!((viewport.offset[1] - 0.05).abs() < 0.0001);
        assert!((viewport.size[0] - 1.0).abs() < 0.0001);
        assert!((viewport.size[1] - 0.9).abs() < 0.0001);
    }
}
