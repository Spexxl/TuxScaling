pub const CRATE_NAME: &str = "tuxscaling-runtime";
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
    use super::{PresentDecision, Runtime, RuntimeState, SwapchainImages};
    use ash::vk;

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
}
