//! egui state and widgets for the in-game TuxScaling overlay.

use egui::Context;

pub const CRATE_NAME: &str = "tuxscaling-overlay";

#[derive(Debug)]
pub struct OverlayState {
    pub visible: bool,
    pub context: Context,
}

impl OverlayState {
    pub fn new() -> Self {
        Self {
            visible: false,
            context: Context::default(),
        }
    }
}

impl Default for OverlayState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::OverlayState;

    #[test]
    fn starts_hidden() {
        assert!(!OverlayState::new().visible);
    }
}
