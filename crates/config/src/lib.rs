use serde::Deserialize;
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: "balanced".into(),
            toggle_key: "Insert".into(),
            debug_view: DebugView::Original,
            motion_quality: MotionQuality::Balanced,
            scene_distance_threshold: 0.5,
            scene_consistency_threshold: 0.2,
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration file is invalid: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("scene thresholds must be finite: distance in (0, 2], consistency in [0, 1]")]
    InvalidThresholds,
}

impl Config {
    pub fn parse(source: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(source)?;
        if !config.scene_distance_threshold.is_finite()
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
    use super::{Config, ConfigError, MotionQuality};

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
}
