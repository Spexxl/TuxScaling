pub const CRATE_NAME: &str = "tuxscaling-runtime";

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
    use super::{PresentDecision, Runtime, RuntimeState};

    #[test]
    fn starts_bypassed_and_passes_through() {
        let runtime = Runtime::new();
        assert_eq!(runtime.state(), RuntimeState::Bypassed);
        assert_eq!(runtime.process_present(), PresentDecision::PassThrough);
    }
}
