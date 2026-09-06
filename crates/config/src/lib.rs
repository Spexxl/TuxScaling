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
    pub motion_quality: MotionQuality,
    pub output_resolution: OutputResolution,
    #[serde(alias = "render_scale")]
    pub processing_scale: f32,
    pub scene_distance_threshold: f32,
    pub scene_consistency_threshold: f32,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MotionQuality {
    Ultra,
    High,
    #[default]
    Balanced,
    Performance,
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
            motion_quality: MotionQuality::Balanced,
            output_resolution: OutputResolution::Native,
            processing_scale: 1.0,
            scene_distance_threshold: 0.5,
            scene_consistency_threshold: 0.2,
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration file is invalid: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("processing scale must be finite in [0.5, 1.0] and scene thresholds must be valid")]
    InvalidThresholds,
}

impl Config {
    pub fn parse(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source)?;
        if !config.processing_scale.is_finite()
            || !(0.5..=1.0).contains(&config.processing_scale)
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
    use super::{Config, ConfigError, MotionQuality, OutputResolution};

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
        assert_eq!(Config::default().motion_quality, MotionQuality::Balanced);
    }

    #[test]
    fn validates_render_scale() {
        assert!(Config::parse("render_scale = 0.75").is_ok());
        assert!(Config::parse("render_scale = 0.25").is_err());
    }

    #[test]
    fn defaults_to_native_output_without_processing_downscale() {
        let config = Config::parse("").unwrap();

        assert_eq!(config.output_resolution, OutputResolution::Native);
        assert_eq!(config.processing_scale, 1.0);
    }

    #[test]
    fn parses_fixed_output_and_legacy_processing_scale() {
        let config = Config::parse("output_resolution = '1920x1080'\nrender_scale = 0.75").unwrap();

        assert_eq!(
            config.output_resolution,
            OutputResolution::Fixed {
                width: 1920,
                height: 1080,
            }
        );
        assert_eq!(config.processing_scale, 0.75);
    }

    #[test]
    fn rejects_invalid_output_resolution_and_processing_scale() {
        assert!(Config::parse("output_resolution = '0x1080'").is_err());
        assert!(Config::parse("processing_scale = 1.1").is_err());
    }
}
