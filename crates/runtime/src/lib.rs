//! Per-frame orchestration for the TuxScaling runtime.

pub const CRATE_NAME: &str = "tuxscaling-runtime";

/// The processing state for a swapchain runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    Disabled,
    Ready,
    Bypassed,
}

/// The result of a present-processing attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentDecision {
    PassThrough,
}

/// Minimal runtime shell used by the initial pass-through milestone.
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

    /// Processing is intentionally bypassed until capture and reconstruction are implemented.
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
    use super::{PresentDecision, Runtime, RuntimeState};

    #[test]
    fn starts_bypassed_and_passes_through() {
        let runtime = Runtime::new();
        assert_eq!(runtime.state(), RuntimeState::Bypassed);
        assert_eq!(runtime.process_present(), PresentDecision::PassThrough);
    }
}
