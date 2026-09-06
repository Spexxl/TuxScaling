#[path = "../../../tests/support/sequence.rs"]
mod sequence;

use ash::vk;
use sequence::{
    affine_motion, auroc, depth_order, endpoint_error_masked, ev_error, f1, fade, flash, hud,
    image_error, layered_parallax, occlusion, percentile, scene_cut, translation, transparency,
};
use std::time::Duration;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResource,
    GuidanceView, JitterSample, MotionDirection, MotionUnits, SignalState,
};

fn resource(
    metadata: GuidanceMetadata,
    format: vk::Format,
    state: SignalState,
) -> GuidanceResource {
    use ash::vk::Handle;
    GuidanceResource {
        image: vk::Image::from_raw(1),
        view: vk::ImageView::from_raw(2),
        format,
        metadata,
        state,
    }
}

fn view() -> GuidanceView {
    let extent = FrameExtent {
        width: 64,
        height: 48,
    };
    let metadata = GuidanceMetadata::zero(7, extent, GuidanceReset::None);
    GuidanceView {
        motion: resource(metadata, vk::Format::R16G16_SFLOAT, SignalState::Estimated),
        confidence: resource(metadata, vk::Format::R8_UNORM, SignalState::Estimated),
        disocclusion: resource(metadata, vk::Format::R8_UNORM, SignalState::Estimated),
        reactive: resource(metadata, vk::Format::R8_UNORM, SignalState::Estimated),
        exposure: resource(metadata, vk::Format::R32_SFLOAT, SignalState::Estimated),
        depth: resource(
            metadata,
            vk::Format::R32_SFLOAT,
            SignalState::ConstantFallback,
        ),
        transparency_composition: resource(
            metadata,
            vk::Format::R8_UNORM,
            SignalState::ConstantFallback,
        ),
        pre_exposure: 1.0,
        timing: FrameTiming {
            raw: Duration::from_micros(16_667),
            validated: Duration::from_micros(16_667),
            smoothed: Duration::from_micros(16_667),
        },
        jitter: JitterSample {
            current: [0.0, 0.0],
            previous: [0.0, 0.0],
            phase: 0,
        },
        depth_semantics: DepthSemantics::FlatFallback,
        direction: MotionDirection::CurrentToPrevious,
        units: MotionUnits::SourcePixels,
        requires_history_reset: false,
    }
}

#[test]
fn contract_validates_new_signals_and_rejects_non_finite_values() {
    let valid = view();
    assert!(valid.is_valid_for(
        7,
        FrameExtent {
            width: 64,
            height: 48
        }
    ));

    let mut wrong = valid;
    wrong.pre_exposure = f32::NAN;
    assert!(!wrong.is_valid_for(
        7,
        FrameExtent {
            width: 64,
            height: 48
        }
    ));

    let mut wrong = valid;
    wrong.jitter.current[0] = f32::INFINITY;
    assert!(!wrong.is_valid_for(
        7,
        FrameExtent {
            width: 64,
            height: 48
        }
    ));
}

#[test]
fn off_jitter_and_flat_depth_are_explicit_fallbacks() {
    let view = view();
    assert_eq!(view.jitter.current, [0.0, 0.0]);
    assert_eq!(view.jitter.previous, [0.0, 0.0]);
    assert_eq!(view.jitter.phase, 0);
    assert_eq!(view.jitter.signal_state(), SignalState::Unavailable);
    assert_eq!(view.depth_semantics, DepthSemantics::FlatFallback);
    assert_eq!(view.depth.state, SignalState::ConstantFallback);
    assert_eq!(view.disocclusion.state, SignalState::Estimated);
    assert_eq!(view.reactive.state, SignalState::Estimated);
    assert_eq!(view.exposure.state, SignalState::Estimated);
    assert_eq!(
        view.transparency_composition.state,
        SignalState::ConstantFallback
    );
}

#[test]
fn procedural_fixtures_have_deterministic_ground_truth_labels() {
    let translation_fixture = translation(64, 48);
    assert_eq!(translation_fixture.motion[64 * 20 + 20], [-5.0, 3.0]);
    assert!(!translation_fixture.scene_cut);

    let parallax_fixture = layered_parallax(64, 48);
    assert!(
        parallax_fixture
            .depth
            .iter()
            .any(|depth| (*depth - 1.0).abs() > f32::EPSILON)
    );
    assert!(
        parallax_fixture
            .motion
            .iter()
            .any(|motion| *motion == [-7.0, -3.0])
    );
    for y in 12..36 {
        for x in 16..48 {
            let index = y * 64 + x;
            let source_x = (x as i32 - 7).clamp(0, 63) as usize;
            let source_y = (y as i32 - 3).clamp(0, 47) as usize;
            assert_eq!(
                parallax_fixture.current[index],
                parallax_fixture.previous[source_y * 64 + source_x]
            );
            assert!(parallax_fixture.valid[index]);
        }
    }

    assert!(!translation_fixture.valid[0]);
    assert!(!translation_fixture.valid[4]);
    assert!(!translation_fixture.valid[47 * 64]);
    assert!(translation_fixture.valid[20 * 64 + 20]);

    assert!(occlusion(64, 48).occlusion.iter().any(|label| *label));
    assert!(transparency(64, 48).transparency.iter().any(|label| *label));
    assert!(hud(64, 48).hud.iter().any(|label| *label));
    assert!(flash(64, 48).exposure_ev > 0.0);
    assert!(fade(64, 48).exposure_ev < 0.0);
    assert!(scene_cut(64, 48).scene_cut);

    for fixture in [
        translation_fixture,
        affine_motion(64, 48),
        parallax_fixture,
        occlusion(64, 48),
        transparency(64, 48),
        hud(64, 48),
        flash(64, 48),
        fade(64, 48),
        scene_cut(64, 48),
    ] {
        assert_eq!(fixture.previous.len(), 64 * 48);
        assert_eq!(fixture.current.len(), 64 * 48);
        assert_eq!(fixture.motion.len(), 64 * 48);
        assert!(fixture.valid.iter().any(|label| *label));
    }
    assert_eq!(translation(64, 48), translation(64, 48));
}

#[test]
fn quality_metrics_report_expected_values() {
    let truth = vec![[1.0, 0.0], [0.0, -2.0], [3.0, 4.0]];
    let estimate = vec![[1.0, 0.0], [0.0, -1.0], [3.0, 4.0]];
    assert!(
        (endpoint_error_masked(&estimate, &truth, &[true, true, true]) - 1.0 / 3.0).abs() < 1e-6
    );
    assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
    assert_eq!(
        f1(&[true, true, false, false], &[true, false, true, false]),
        0.5
    );
    assert!((auroc(&[0.9, 0.8, 0.2, 0.1], &[true, true, false, false]) - 1.0).abs() < 1e-6);
    assert_eq!(ev_error(1.5, 1.0), 0.5);
    assert_eq!(depth_order(&[1.0, 0.8, 0.2], &[1.0, 0.6, 0.1]), 1.0);
    assert_eq!(
        image_error(&[[0.0; 4], [1.0; 4]], &[[0.0; 4], [0.5; 4]]),
        0.125
    );

    let fixture = translation(16, 12);
    let mut estimated = fixture.motion.clone();
    estimated[0] = [100.0, 100.0];
    assert_eq!(
        endpoint_error_masked(&estimated, &fixture.motion, &fixture.valid),
        0.0
    );
    let parallax = layered_parallax(16, 12);
    assert_eq!(depth_order(&parallax.depth, &parallax.depth), 1.0);
    let occluded = occlusion(16, 12);
    assert_eq!(f1(&occluded.occlusion, &occluded.occlusion), 1.0);
    assert_eq!(auroc(&[1.0, 0.0], &[true, false]), 1.0);
    assert!(ev_error(flash(16, 12).exposure_ev, 0.0) > 0.0);
}
