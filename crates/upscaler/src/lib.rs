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
    use super::{BackendCapabilities, BackendError, BackendId, InputResolution, UpscalerBackend};

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
}
