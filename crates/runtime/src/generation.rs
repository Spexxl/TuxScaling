use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGeneration {
    id: u64,
    extent: vk::Extent2D,
    image_count: usize,
}

impl OutputGeneration {
    pub const fn new(id: u64, extent: vk::Extent2D, image_count: usize) -> Self {
        Self {
            id,
            extent,
            image_count,
        }
    }

    pub const fn id(self) -> u64 {
        self.id
    }

    pub const fn extent(self) -> vk::Extent2D {
        self.extent
    }

    pub const fn image_count(self) -> usize {
        self.image_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationError {
    Invalid,
    NonMonotonic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    SpatialBypass,
    TemporalBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalContractSnapshot {
    game_extent: vk::Extent2D,
    image_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeGeneration {
    logical: LogicalContractSnapshot,
    output: OutputGeneration,
    history_reset: bool,
}

impl RuntimeGeneration {
    pub const fn new(
        game_extent: vk::Extent2D,
        logical_image_count: usize,
        output: OutputGeneration,
    ) -> Self {
        Self {
            logical: LogicalContractSnapshot {
                game_extent,
                image_count: logical_image_count,
            },
            output,
            history_reset: false,
        }
    }

    pub const fn logical_snapshot(self) -> LogicalContractSnapshot {
        self.logical
    }

    pub const fn output(self) -> OutputGeneration {
        self.output
    }

    pub const fn history_reset_requested(self) -> bool {
        self.history_reset
    }

    pub fn publish(&mut self, output: OutputGeneration) -> Result<(), GenerationError> {
        if output.extent().width == 0 || output.extent().height == 0 || output.image_count() == 0 {
            return Err(GenerationError::Invalid);
        }
        if output.id() <= self.output.id() {
            return Err(GenerationError::NonMonotonic);
        }
        self.output = output;
        self.history_reset = true;
        Ok(())
    }

    pub const fn dispatch_mode(self, native_published: bool) -> DispatchMode {
        if native_published {
            DispatchMode::TemporalBackend
        } else {
            DispatchMode::SpatialBypass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{OutputGeneration, RuntimeGeneration};
    use ash::vk;

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn generation_replacement_keeps_logical_contract_and_requests_history_reset() {
        let mut runtime = RuntimeGeneration::new(
            extent(1280, 720),
            2,
            OutputGeneration::new(0, extent(1280, 720), 3),
        );
        let before = runtime.logical_snapshot();

        runtime
            .publish(OutputGeneration::new(1, extent(3440, 1440), 3))
            .unwrap();

        assert_eq!(runtime.logical_snapshot(), before);
        assert_eq!(
            runtime.output(),
            OutputGeneration::new(1, extent(3440, 1440), 3)
        );
        assert!(runtime.history_reset_requested());
    }

    #[test]
    fn pending_generation_uses_spatial_bypass_until_native_publication() {
        let runtime = RuntimeGeneration::new(
            extent(1280, 720),
            2,
            OutputGeneration::new(0, extent(1280, 720), 3),
        );

        assert_eq!(
            runtime.dispatch_mode(false),
            super::DispatchMode::SpatialBypass
        );
        assert_eq!(
            runtime.dispatch_mode(true),
            super::DispatchMode::TemporalBackend
        );
    }
}
