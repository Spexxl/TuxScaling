//! Versioned TOML configuration and validation.

use serde::Deserialize;
use thiserror::Error;

pub const CRATE_NAME: &str = "tuxscaling-config";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Config {
    pub enabled: bool,
    pub quality: String,
    pub toggle_key: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: "balanced".into(),
            toggle_key: "Insert".into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration file is invalid: {0}")]
    Parse(#[from] toml::de::Error),
}

impl Config {
    pub fn parse(source: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(source)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError};

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
}
