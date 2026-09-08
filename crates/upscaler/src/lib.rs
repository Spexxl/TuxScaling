use ash::vk;
use thiserror::Error;
use tuxscaling_temporal::{GuidanceCapabilities, GuidanceSignal, GuidanceView, SignalState};

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
    pub required_guidance: [bool; 7],
    pub supported_source_formats: &'static [vk::Format],
    pub supported_output_formats: &'static [vk::Format],
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
    #[error("unsupported {role} format: {format:?}")]
    UnsupportedFormat {
        role: &'static str,
        format: vk::Format,
    },
    #[error("incompatible {role} extent: expected {expected:?}, got {actual:?}")]
    IncompatibleExtent {
        role: &'static str,
        expected: vk::Extent2D,
        actual: vk::Extent2D,
    },
    #[error("required guidance signal is unavailable: {signal:?}")]
    MissingSignal { signal: GuidanceSignal },
    #[error("invalid {role} image layout: {layout:?}")]
    InvalidLayout {
        role: &'static str,
        layout: vk::ImageLayout,
    },
    #[error("invalid backend metadata: {0}")]
    InvalidMetadata(&'static str),
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

#[derive(Debug, Clone, Copy)]
pub struct BackendImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub layout: vk::ImageLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct BackendConfig {
    pub game_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
    pub source_format: vk::Format,
    pub output_format: vk::Format,
    pub viewport: ContentViewport,
    pub guidance: GuidanceCapabilities,
}

impl BackendConfig {
    pub fn validate(self, capabilities: BackendCapabilities) -> Result<(), BackendError> {
        if !is_valid_extent(self.game_extent) {
            return Err(BackendError::IncompatibleExtent {
                role: "game",
                expected: self.game_extent,
                actual: self.game_extent,
            });
        }
        if !is_valid_extent(self.output_extent) {
            return Err(BackendError::IncompatibleExtent {
                role: "output",
                expected: self.output_extent,
                actual: self.output_extent,
            });
        }
        if !capabilities
            .supported_source_formats
            .contains(&self.source_format)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "source",
                format: self.source_format,
            });
        }
        if !capabilities
            .supported_output_formats
            .contains(&self.output_format)
        {
            return Err(BackendError::UnsupportedFormat {
                role: "output",
                format: self.output_format,
            });
        }
        if !self.viewport.is_valid() {
            return Err(BackendError::InvalidMetadata("viewport"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BackendFrame {
    pub command_buffer: vk::CommandBuffer,
    pub slot: usize,
    pub source: BackendImage,
    pub output: BackendImage,
    pub guidance: GuidanceView,
    pub viewport: ContentViewport,
    pub frame_id: u64,
    pub reset_history: bool,
    pub debug_view: u32,
}

impl BackendFrame {
    pub fn validate(
        self,
        config: BackendConfig,
        capabilities: BackendCapabilities,
    ) -> Result<(), BackendError> {
        config.validate(capabilities)?;
        if self.command_buffer == vk::CommandBuffer::null()
            || self.source.image == vk::Image::null()
            || self.source.view == vk::ImageView::null()
            || self.output.image == vk::Image::null()
            || self.output.view == vk::ImageView::null()
        {
            return Err(BackendError::InvalidMetadata(
                "null command or image handle",
            ));
        }
        validate_extent("source", config.game_extent, self.source.extent)?;
        validate_extent("output", config.output_extent, self.output.extent)?;
        if self.source.format != config.source_format {
            return Err(BackendError::UnsupportedFormat {
                role: "source",
                format: self.source.format,
            });
        }
        if self.output.format != config.output_format {
            return Err(BackendError::UnsupportedFormat {
                role: "output",
                format: self.output.format,
            });
        }
        if self.source.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return Err(BackendError::InvalidLayout {
                role: "source",
                layout: self.source.layout,
            });
        }
        if self.output.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            return Err(BackendError::InvalidLayout {
                role: "output",
                layout: self.output.layout,
            });
        }
        if self.viewport != config.viewport {
            return Err(BackendError::InvalidMetadata(
                "viewport does not match configuration",
            ));
        }
        let extent = tuxscaling_temporal::FrameExtent {
            width: config.game_extent.width,
            height: config.game_extent.height,
        };
        if !self.guidance.is_valid_for(self.frame_id, extent) {
            return Err(BackendError::InvalidMetadata("guidance view"));
        }
        for signal in all_guidance_signals() {
            if capabilities.required_guidance[signal as usize]
                && self.guidance.resource(signal).state == SignalState::Unavailable
            {
                return Err(BackendError::MissingSignal { signal });
            }
        }
        Ok(())
    }
}

fn all_guidance_signals() -> [GuidanceSignal; 7] {
    [
        GuidanceSignal::Motion,
        GuidanceSignal::Confidence,
        GuidanceSignal::Disocclusion,
        GuidanceSignal::Reactive,
        GuidanceSignal::Exposure,
        GuidanceSignal::RelativeDepth,
        GuidanceSignal::TransparencyComposition,
    ]
}

fn is_valid_extent(extent: vk::Extent2D) -> bool {
    extent.width != 0 && extent.height != 0
}

fn validate_extent(
    role: &'static str,
    expected: vk::Extent2D,
    actual: vk::Extent2D,
) -> Result<(), BackendError> {
    if expected == actual {
        Ok(())
    } else {
        Err(BackendError::IncompatibleExtent {
            role,
            expected,
            actual,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionPlan {
    pub game_extent: vk::Extent2D,
    pub guidance_extent: vk::Extent2D,
    pub output_extent: vk::Extent2D,
    pub presentation: PresentationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationMode {
    Direct,
    Virtual,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentViewport {
    pub offset: [f32; 2],
    pub size: [f32; 2],
}

impl ContentViewport {
    fn is_valid(self) -> bool {
        self.offset
            .iter()
            .chain(self.size.iter())
            .all(|value| value.is_finite())
            && self.offset.iter().all(|value| *value >= 0.0)
            && self.size.iter().all(|value| (0.0..=1.0).contains(value))
            && self.size.iter().all(|value| *value > 0.0)
    }
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
        guidance_scale: f32,
    ) -> Self {
        Self {
            game_extent,
            guidance_extent: scaled_extent(game_extent, guidance_scale),
            output_extent,
            presentation: if game_extent == output_extent {
                PresentationMode::Direct
            } else {
                PresentationMode::Virtual
            },
        }
    }
}

pub trait UpscalerBackend: Send {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn configure(&mut self, config: BackendConfig) -> Result<(), BackendError>;
    /// Records backend commands into the supplied command buffer only.
    ///
    /// # Safety
    ///
    /// The caller must provide a recording command buffer and image layouts
    /// matching the validated `BackendFrame` contract. The command buffer and
    /// all referenced resources must remain valid until its submission fence
    /// signals.
    unsafe fn record(&mut self, frame: BackendFrame) -> Result<(), BackendError>;
    fn reset(&mut self) -> Result<(), BackendError>;
}

#[cfg(test)]
mod tests {
    use super::content_viewport;
    use super::{
        BackendCapabilities, BackendConfig, BackendError, BackendFrame, BackendId, BackendImage,
        ContentViewport, PresentationMode, ResolutionPlan, UpscalerBackend,
    };
    use ash::vk;
    use ash::vk::Handle;
    use tuxscaling_temporal::{
        DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset,
        GuidanceResolution, GuidanceResource, GuidanceScalar, GuidanceSignal, GuidanceView,
        JitterSample, MotionDirection, MotionUnits, SignalState,
    };

    struct Dummy;

    impl UpscalerBackend for Dummy {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
                required_guidance: [true; 7],
                supported_source_formats: &[vk::Format::R8G8B8A8_UNORM],
                supported_output_formats: &[vk::Format::R8G8B8A8_UNORM],
            }
        }
        fn configure(&mut self, _: BackendConfig) -> Result<(), BackendError> {
            Ok(())
        }
        unsafe fn record(&mut self, _: BackendFrame) -> Result<(), BackendError> {
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
            .configure(BackendConfig {
                game_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                output_extent: vk::Extent2D {
                    width: 2,
                    height: 2,
                },
                source_format: vk::Format::R8G8B8A8_UNORM,
                output_format: vk::Format::R8G8B8A8_UNORM,
                viewport: ContentViewport {
                    offset: [0.0, 0.0],
                    size: [1.0, 1.0],
                },
                guidance: guidance().capabilities(),
            })
            .unwrap();
        backend.reset().unwrap();
    }

    const GAME: vk::Extent2D = vk::Extent2D {
        width: 1280,
        height: 720,
    };
    const OUTPUT: vk::Extent2D = vk::Extent2D {
        width: 1920,
        height: 1080,
    };

    fn guidance() -> GuidanceView {
        let extent = FrameExtent {
            width: GAME.width,
            height: GAME.height,
        };
        let metadata = GuidanceMetadata::zero(7, extent, GuidanceReset::None);
        let resource = |format| GuidanceResource {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            format,
            metadata,
            state: SignalState::Estimated,
        };
        GuidanceView {
            motion: resource(vk::Format::R16G16_SFLOAT),
            confidence: resource(vk::Format::R8_UNORM),
            disocclusion: resource(vk::Format::R8_UNORM),
            reactive: resource(vk::Format::R8_UNORM),
            exposure: resource(vk::Format::R32_SFLOAT),
            depth: resource(vk::Format::R32_SFLOAT),
            transparency_composition: resource(vk::Format::R8_UNORM),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing: FrameTiming::default(),
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::FlatFallback,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(extent, extent),
            requires_history_reset: false,
        }
    }

    fn config() -> BackendConfig {
        BackendConfig {
            game_extent: GAME,
            output_extent: OUTPUT,
            source_format: vk::Format::R8G8B8A8_UNORM,
            output_format: vk::Format::R8G8B8A8_UNORM,
            viewport: content_viewport(GAME, OUTPUT),
            guidance: guidance().capabilities(),
        }
    }

    fn frame() -> BackendFrame {
        BackendFrame {
            command_buffer: vk::CommandBuffer::from_raw(3),
            slot: 0,
            source: BackendImage {
                image: vk::Image::from_raw(4),
                view: vk::ImageView::from_raw(5),
                format: vk::Format::R8G8B8A8_UNORM,
                extent: GAME,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            output: BackendImage {
                image: vk::Image::from_raw(6),
                view: vk::ImageView::from_raw(7),
                format: vk::Format::R8G8B8A8_UNORM,
                extent: OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            guidance: guidance(),
            viewport: content_viewport(GAME, OUTPUT),
            frame_id: 7,
            reset_history: false,
            debug_view: 0,
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        config: Option<BackendConfig>,
        records: u32,
        resets: u32,
    }

    impl UpscalerBackend for RecordingBackend {
        fn id(&self) -> BackendId {
            BackendId::Reference
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                temporal: true,
                frame_generation: false,
                required_guidance: [true; 7],
                supported_source_formats: &[vk::Format::R8G8B8A8_UNORM],
                supported_output_formats: &[vk::Format::R8G8B8A8_UNORM],
            }
        }

        fn configure(&mut self, config: BackendConfig) -> Result<(), BackendError> {
            config.validate(self.capabilities())?;
            self.config = Some(config);
            Ok(())
        }

        unsafe fn record(&mut self, frame: BackendFrame) -> Result<(), BackendError> {
            let config = self.config.ok_or(BackendError::Unavailable)?;
            frame.validate(config, self.capabilities())?;
            self.records += 1;
            Ok(())
        }

        fn reset(&mut self) -> Result<(), BackendError> {
            self.resets += 1;
            Ok(())
        }
    }

    #[test]
    fn backend_contract_is_object_safe_and_dispatches_valid_frame() {
        let mut backend: Box<dyn UpscalerBackend> = Box::new(RecordingBackend::default());
        backend.configure(config()).unwrap();
        unsafe { backend.record(frame()) }.unwrap();
        backend.reset().unwrap();
    }

    #[test]
    fn backend_config_rejects_unsupported_format_and_extent() {
        let capabilities = RecordingBackend::default().capabilities();
        let mut invalid = config();
        invalid.source_format = vk::Format::R16G16_SFLOAT;
        assert!(matches!(
            invalid.validate(capabilities),
            Err(BackendError::UnsupportedFormat { .. })
        ));
        invalid = config();
        invalid.game_extent = vk::Extent2D {
            width: 0,
            height: GAME.height,
        };
        assert!(matches!(
            invalid.validate(capabilities),
            Err(BackendError::IncompatibleExtent { .. })
        ));
    }

    #[test]
    fn backend_frame_rejects_incoherent_resources_and_missing_signal() {
        let capabilities = RecordingBackend::default().capabilities();
        let mut invalid = frame();
        invalid.source.extent.width -= 1;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::IncompatibleExtent { .. })
        ));
        invalid = frame();
        invalid.guidance.motion.state = SignalState::Unavailable;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::MissingSignal {
                signal: GuidanceSignal::Motion
            })
        ));
        invalid = frame();
        invalid.output.layout = vk::ImageLayout::GENERAL;
        assert!(matches!(
            invalid.validate(config(), capabilities),
            Err(BackendError::InvalidLayout { .. })
        ));
    }

    #[test]
    fn resolution_plan_keeps_game_input_at_full_guidance_scale() {
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

        assert_eq!(plan.game_extent, plan.guidance_extent);
        assert_eq!(plan.output_extent.width, 1920);
        assert_eq!(plan.presentation, PresentationMode::Virtual);
    }

    #[test]
    fn resolution_plan_scales_only_guidance_input() {
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
            plan.guidance_extent,
            vk::Extent2D {
                width: 960,
                height: 540,
            }
        );
        assert_eq!(plan.presentation, PresentationMode::Virtual);
    }

    #[test]
    fn equal_game_and_output_extents_use_native_aa_mode() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            1.0,
        );

        assert_eq!(plan.presentation, PresentationMode::Direct);
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

    #[test]
    fn resolution_plan_keeps_game_guidance_and_output_extents_independent() {
        let plan = ResolutionPlan::new(
            vk::Extent2D {
                width: 1279,
                height: 719,
            },
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            0.75,
        );

        assert_eq!(plan.game_extent.width, 1279);
        assert_eq!(plan.guidance_extent.width, 959);
        assert_eq!(plan.guidance_extent.height, 539);
        assert_eq!(plan.output_extent.width, 1920);
        assert!(plan.guidance_extent.width > 0 && plan.guidance_extent.height > 0);
    }

    #[test]
    fn changing_guidance_scale_does_not_change_game_or_output_extents() {
        let game = vk::Extent2D {
            width: 1281,
            height: 721,
        };
        let output = vk::Extent2D {
            width: 2560,
            height: 1440,
        };
        let full = ResolutionPlan::new(game, output, 1.0);
        let reduced = ResolutionPlan::new(game, output, 0.5);

        assert_eq!(full.game_extent, reduced.game_extent);
        assert_eq!(full.output_extent, reduced.output_extent);
        assert_ne!(full.guidance_extent, reduced.guidance_extent);
    }
}
