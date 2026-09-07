use serde::{Deserialize, Deserializer};
use thiserror::Error;

pub const CRATE_NAME: &str = "tuxscaling-config";

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub enabled: bool,
    pub quality: String,
    pub toggle_key: String,
    pub debug_view: DebugView,
    pub jitter_mode: JitterMode,
    pub motion_quality: MotionQuality,
    pub output_resolution: OutputResolution,
    #[serde(alias = "processing_scale", alias = "render_scale")]
    pub guidance_scale: f32,
    pub scene_distance_threshold: f32,
    pub scene_consistency_threshold: f32,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MotionQuality {
    High,
    #[default]
    Ultra,
    Balanced,
    Performance,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JitterMode {
    #[default]
    Off,
    ExperimentalHalton8,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DebugView {
    #[default]
    Original,
    Luminance,
    Motion,
    Confidence,
    Reconstructed,
    History,
    Reactive,
    Disocclusion,
    Depth,
    Composition,
    Exposure,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputResolution {
    #[default]
    Native,
    Swapchain,
    Fixed {
        width: u32,
        height: u32,
    },
}

impl<'de> Deserialize<'de> for OutputResolution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "native" => Ok(Self::Native),
            "swapchain" => Ok(Self::Swapchain),
            _ => {
                let Some((width, height)) = value.split_once('x') else {
                    return Err(serde::de::Error::custom(
                        "output resolution must be native, swapchain, or WIDTHxHEIGHT",
                    ));
                };
                let width = width.parse::<u32>().map_err(serde::de::Error::custom)?;
                let height = height.parse::<u32>().map_err(serde::de::Error::custom)?;
                if width == 0 || height == 0 {
                    return Err(serde::de::Error::custom(
                        "output resolution dimensions must be positive",
                    ));
                }
                Ok(Self::Fixed { width, height })
            }
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: "balanced".into(),
            toggle_key: "Insert".into(),
            debug_view: DebugView::Original,
            jitter_mode: JitterMode::Off,
            motion_quality: MotionQuality::Ultra,
            output_resolution: OutputResolution::Native,
            guidance_scale: 1.0,
            scene_distance_threshold: 0.5,
            scene_consistency_threshold: 0.2,
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration file is invalid: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("guidance scale must be finite in [0.5, 1.0] and scene thresholds must be valid")]
    InvalidThresholds,
}

impl Config {
    pub fn parse(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source)?;
        if !config.guidance_scale.is_finite()
            || !(0.5..=1.0).contains(&config.guidance_scale)
            || !config.scene_distance_threshold.is_finite()
            || config.scene_distance_threshold <= 0.0
            || config.scene_distance_threshold > 2.0
            || !config.scene_consistency_threshold.is_finite()
            || !(0.0..=1.0).contains(&config.scene_consistency_threshold)
        {
            return Err(ConfigError::InvalidThresholds);
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError, DebugView, JitterMode, MotionQuality, OutputResolution};

    #[test]
    fn applies_defaults() {
        assert_eq!(Config::parse("").unwrap(), Config::default());
    }

    #[test]
    fn rejects_invalid_toml() {
        assert!(matches!(
            Config::parse("enabled = ["),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn validates_diagnostic_options() {
        assert!(Config::parse("scene_distance_threshold = nan").is_err());
        assert!(Config::parse("scene_consistency_threshold = 1.1").is_err());
        assert!(Config::parse("debug_view = 'unknown'").is_err());
        assert_eq!(
            Config::parse("debug_view = 'motion'").unwrap().debug_view,
            super::DebugView::Motion
        );
    }

    #[test]
    fn parses_motion_quality_presets() {
        assert_eq!(
            Config::parse("motion_quality = 'performance'")
                .unwrap()
                .motion_quality,
            MotionQuality::Performance
        );
        assert_eq!(Config::default().motion_quality, MotionQuality::Ultra);
    }

    #[test]
    fn validates_legacy_scale_alias() {
        assert!(Config::parse("render_scale = 0.75").is_ok());
        assert!(Config::parse("render_scale = 0.25").is_err());
    }

    #[test]
    fn defaults_to_native_output_without_guidance_downscale() {
        let config = Config::parse("").unwrap();

        assert_eq!(config.output_resolution, OutputResolution::Native);
        assert_eq!(config.guidance_scale, 1.0);
    }

    #[test]
    fn defaults_jitter_off_and_parses_experimental_halton_mode() {
        assert_eq!(Config::default().jitter_mode, JitterMode::Off);
        assert_eq!(
            Config::parse("jitter_mode = 'experimental_halton8'")
                .unwrap()
                .jitter_mode,
            JitterMode::ExperimentalHalton8
        );
        assert_eq!(
            Config::parse("debug_view = 'depth'").unwrap().debug_view,
            DebugView::Depth
        );
        assert_eq!(
            Config::parse("debug_view = 'composition'")
                .unwrap()
                .debug_view,
            DebugView::Composition
        );
        assert_eq!(
            Config::parse("debug_view = 'exposure'").unwrap().debug_view,
            DebugView::Exposure
        );
        assert_eq!(DebugView::Depth as u32, 8);
        assert_eq!(DebugView::Composition as u32, 9);
        assert_eq!(DebugView::Exposure as u32, 10);
    }

    #[test]
    fn parses_fixed_output_and_legacy_guidance_scale() {
        let config = Config::parse("output_resolution = '1920x1080'\nrender_scale = 0.75").unwrap();

        assert_eq!(
            config.output_resolution,
            OutputResolution::Fixed {
                width: 1920,
                height: 1080,
            }
        );
        assert_eq!(config.guidance_scale, 0.75);
    }

    #[test]
    fn rejects_invalid_output_resolution_and_guidance_scale() {
        assert!(Config::parse("output_resolution = '0x1080'").is_err());
        assert!(Config::parse("processing_scale = 1.1").is_err());
    }

    #[test]
    fn defaults_to_full_resolution_guidance_and_ultra_motion() {
        let config = Config::parse("").unwrap();

        assert_eq!(config.guidance_scale, 1.0);
        assert_eq!(config.motion_quality, MotionQuality::Ultra);
    }

    #[test]
    fn validates_guidance_scale_boundaries() {
        assert!(Config::parse("guidance_scale = 0.5").is_ok());
        assert!(Config::parse("guidance_scale = 1.0").is_ok());
        assert!(Config::parse("guidance_scale = 0.49").is_err());
        assert!(Config::parse("guidance_scale = 1.01").is_err());
    }

    #[test]
    fn accepts_legacy_scale_names_as_deserialization_aliases() {
        assert_eq!(
            Config::parse("processing_scale = 0.75")
                .unwrap()
                .guidance_scale,
            0.75
        );
        assert_eq!(
            Config::parse("render_scale = 0.75").unwrap().guidance_scale,
            0.75
        );
    }
}
