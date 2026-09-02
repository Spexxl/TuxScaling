//! Vulkan implicit-layer entry point.

pub const CRATE_NAME: &str = "tuxscaling-layer";

/// Reports the initial milestone's pass-through behavior.
pub const fn is_pass_through() -> bool {
    true
}
