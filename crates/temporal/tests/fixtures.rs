use tuxscaling_temporal::quality::{
    SequenceFixture, affine_motion, noise, particles, pause, rotation, scene_cut, thin_geometry,
    translation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualQualityCategory {
    Translation,
    RotationScaling,
    ThinGeometryVegetation,
    HudText,
    TransparencyParticles,
    EmissiveNoise,
    OcclusionDisocclusion,
    PauseResumeSceneCut,
}

pub const VISUAL_QUALITY_CATEGORIES: [VisualQualityCategory; 8] = [
    VisualQualityCategory::Translation,
    VisualQualityCategory::RotationScaling,
    VisualQualityCategory::ThinGeometryVegetation,
    VisualQualityCategory::HudText,
    VisualQualityCategory::TransparencyParticles,
    VisualQualityCategory::EmissiveNoise,
    VisualQualityCategory::OcclusionDisocclusion,
    VisualQualityCategory::PauseResumeSceneCut,
];

pub fn catalog(width: u32, height: u32) -> Vec<(VisualQualityCategory, SequenceFixture)> {
    vec![
        (
            VisualQualityCategory::Translation,
            translation(width, height),
        ),
        (
            VisualQualityCategory::RotationScaling,
            rotation(width, height),
        ),
        (
            VisualQualityCategory::ThinGeometryVegetation,
            thin_geometry(width, height),
        ),
        (VisualQualityCategory::HudText, affine_motion(width, height)),
        (
            VisualQualityCategory::TransparencyParticles,
            particles(width, height),
        ),
        (VisualQualityCategory::EmissiveNoise, noise(width, height)),
        (
            VisualQualityCategory::OcclusionDisocclusion,
            scene_cut(width, height),
        ),
        (
            VisualQualityCategory::PauseResumeSceneCut,
            pause(width, height),
        ),
    ]
}

#[test]
fn visual_quality_catalog_is_complete_and_deterministic() {
    assert_eq!(VISUAL_QUALITY_CATEGORIES.len(), 8);
    let first = catalog(32, 24);
    let second = catalog(32, 24);
    assert_eq!(first, second);
    assert!(first.iter().all(|(_, fixture)| {
        fixture.previous.len() == 32 * 24
            && fixture.current.len() == 32 * 24
            && fixture.motion.len() == 32 * 24
            && fixture.depth.iter().all(|value| value.is_finite())
    }));
}
