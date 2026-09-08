#[path = "../../../tests/support/sequence.rs"]
mod sequence;

use ash::vk;
use sequence::{
    affine_motion, auroc, depth_order, endpoint_error_masked, ev_error, f1, fade, flash, hud,
    image_error, independent_objects, invalid_timing, layered_parallax, occlusion, particles,
    pause, percentile, reveal_occlusion, rotation, scene_cut, subpixel_translation, thin_geometry,
    translation, transparency, zoom,
};
use std::time::Duration;
use tuxscaling_motion::MotionQuality;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResource, GuidanceScalar, GuidanceSignal, GuidanceView, JitterSample, MotionDirection,
    MotionUnits, SignalState,
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
        pre_exposure: GuidanceScalar {
            value: 1.0,
            state: SignalState::ConstantFallback,
        },
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
        resolution: GuidanceResolution::new(extent, extent),
        requires_history_reset: false,
    }
}

fn estimated_view() -> GuidanceView {
    let mut view = view();
    view.depth.state = SignalState::Estimated;
    view.transparency_composition.state = SignalState::Estimated;
    view
}

#[test]
fn stable_guidance_reports_each_signal_independently() {
    let view = estimated_view();
    assert_eq!(view.motion.state, SignalState::Estimated);
    assert_eq!(view.confidence.state, SignalState::Estimated);
    assert_eq!(view.jitter.signal_state(), SignalState::Unavailable);
    assert!(view.is_valid_for(7, view.motion.metadata.extent));

    let capabilities = view.capabilities();
    assert_eq!(capabilities.estimated, [true; 7]);
    assert_eq!(view.resource(GuidanceSignal::Motion), view.motion);
    assert_eq!(view.resource(GuidanceSignal::RelativeDepth), view.depth);
}

#[test]
fn guidance_rejects_mixed_frame_resources() {
    let mut view = estimated_view();
    view.depth.metadata.frame_id += 1;
    assert!(!view.is_valid_for(7, view.motion.metadata.extent));
}

#[test]
fn signal_fallbacks_are_conservative_and_local() {
    assert_eq!(GuidanceSignal::Motion.fallback_value(), 0.0);
    assert_eq!(GuidanceSignal::Confidence.fallback_value(), 0.0);
    assert_eq!(GuidanceSignal::Exposure.fallback_value(), 1.0);
    assert_eq!(GuidanceSignal::RelativeDepth.fallback_value(), 1.0);
    assert_eq!(GuidanceSignal::Disocclusion.fallback_value(), 1.0);
    assert_eq!(GuidanceSignal::Reactive.fallback_value(), 1.0);
    assert_eq!(
        GuidanceSignal::TransparencyComposition.fallback_value(),
        1.0
    );
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
    wrong.pre_exposure.value = f32::NAN;
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
    let estimated = vec![[-5.0, 3.0]; fixture.len()];
    assert_eq!(
        endpoint_error_masked(&estimated, &fixture.motion, &fixture.valid),
        0.0
    );
}

fn rectangle_mask(
    width: u32,
    height: u32,
    x: std::ops::Range<u32>,
    y: std::ops::Range<u32>,
) -> Vec<bool> {
    let mut mask = vec![false; (width * height) as usize];
    for row in y {
        for column in x.clone() {
            mask[row as usize * width as usize + column as usize] = true;
        }
    }
    mask
}

fn reproject(previous: &[[f32; 4]], width: u32, height: u32, dx: i32, dy: i32) -> Vec<[f32; 4]> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let source_x = (x as i32 - dx).clamp(0, width.saturating_sub(1) as i32) as usize;
                let source_y = (y as i32 - dy).clamp(0, height.saturating_sub(1) as i32) as usize;
                previous[source_y * width as usize + source_x]
            })
        })
        .collect()
}

fn luma(pixel: [f32; 4]) -> f32 {
    pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
}

#[test]
fn independent_fixture_estimates_pass_and_corruptions_fail() {
    let width = 64;
    let height = 48;

    let parallax = layered_parallax(width, height);
    let mut estimated_depth = vec![0.35; parallax.len()];
    for y in 12..36 {
        for x in 16..48 {
            estimated_depth[y * width as usize + x] = 1.0;
        }
    }
    assert!(depth_order(&estimated_depth, &parallax.depth) > 0.99);
    let mut corrupted_depth = estimated_depth.clone();
    for y in 12..36 {
        for x in 16..48 {
            corrupted_depth[y * width as usize + x] = 0.35;
        }
    }
    assert!(depth_order(&corrupted_depth, &parallax.depth) < 0.65);

    let occluded = occlusion(width, height);
    let occlusion_scores: Vec<f32> = occluded
        .current
        .iter()
        .map(|pixel| (pixel[0] + pixel[1] + pixel[2]) / 3.0)
        .collect();
    let estimated_occlusion: Vec<bool> = occlusion_scores
        .iter()
        .map(|score| *score > 0.999)
        .collect();
    assert!(f1(&estimated_occlusion, &occluded.occlusion) > 0.95);
    assert!(auroc(&occlusion_scores, &occluded.occlusion) > 0.95);
    let mut corrupted_occlusion = estimated_occlusion.clone();
    corrupted_occlusion.fill(false);
    assert!(f1(&corrupted_occlusion, &occluded.occlusion) < 0.65);
    let corrupted_scores: Vec<f32> = occlusion_scores.iter().map(|score| 1.0 - score).collect();
    assert!(auroc(&corrupted_scores, &occluded.occlusion) < 0.65);

    let transparent = transparency(width, height);
    let estimated_transparency = rectangle_mask(width, height, 16..32, 12..36);
    assert!(f1(&estimated_transparency, &transparent.transparency) > 0.95);
    let corrupted_transparency = rectangle_mask(width, height, 40..56, 12..36);
    assert!(f1(&corrupted_transparency, &transparent.transparency) < 0.65);

    let flash_fixture = flash(width, height);
    let ratios: Vec<f32> = flash_fixture
        .previous
        .iter()
        .zip(flash_fixture.current.iter())
        .filter_map(|(previous, current)| {
            let previous_luma = luma(*previous);
            let current_luma = luma(*current);
            (previous_luma > 0.05
                && previous[0] < 0.6
                && previous[1] < 0.6
                && previous[2] < 0.6
                && current_luma < 0.99)
                .then_some(current_luma / previous_luma)
        })
        .collect();
    let estimated_ev = percentile(&ratios, 0.5).log2();
    assert!(ev_error(estimated_ev, flash_fixture.exposure_ev) < 0.05);
    assert!(ev_error(estimated_ev + 1.0, flash_fixture.exposure_ev) > 0.15);

    let translated = translation(16, 12);
    let estimated_image = reproject(&translated.previous, 16, 12, 5, -3);
    assert!(image_error(&estimated_image, &translated.current) < 1e-6);
    let mut corrupted_image = estimated_image.clone();
    for y in 4..8 {
        for x in 4..8 {
            corrupted_image[y * 16 + x] = [0.0; 4];
        }
    }
    assert!(image_error(&corrupted_image, &translated.current) > 0.01);
}

#[test]
fn deterministic_failure_modes_cover_stable_signal_thresholds() {
    let width = 64;
    let height = 48;
    let fixtures = [
        ("subpixel", subpixel_translation(width, height)),
        ("rotation", rotation(width, height)),
        ("zoom", zoom(width, height)),
        ("independent_objects", independent_objects(width, height)),
        ("reveal_occlusion", reveal_occlusion(width, height)),
        ("thin_geometry", thin_geometry(width, height)),
        ("particles", particles(width, height)),
        ("pause", pause(width, height)),
        ("invalid_timing", invalid_timing(width, height)),
    ];

    for (name, fixture) in fixtures {
        assert_eq!(fixture.previous.len(), (width * height) as usize, "{name}");
        assert_eq!(fixture.current.len(), fixture.previous.len(), "{name}");
        assert_eq!(fixture.motion.len(), fixture.previous.len(), "{name}");
        assert_eq!(fixture.valid.len(), fixture.previous.len(), "{name}");
        assert_eq!(fixture.occlusion.len(), fixture.previous.len(), "{name}");
        assert_eq!(fixture.transparency.len(), fixture.previous.len(), "{name}");
        assert!(
            fixture.depth.iter().all(|value| value.is_finite()),
            "{name}"
        );
    }

    assert!(
        reveal_occlusion(width, height)
            .occlusion
            .iter()
            .any(|value| *value)
    );
    assert!(
        thin_geometry(width, height)
            .occlusion
            .iter()
            .any(|value| *value)
    );
    assert!(
        particles(width, height)
            .transparency
            .iter()
            .any(|value| *value)
    );
    assert!(pause(width, height).long_pause);
    assert!(!invalid_timing(width, height).timing_valid);
}

#[test]
fn motion_quality_thresholds_emit_stable_acceptance_lines() {
    let fixture = subpixel_translation(32, 24);
    let errors = vec![0.0; fixture.len()];
    for quality in [
        MotionQuality::Ultra,
        MotionQuality::High,
        MotionQuality::Balanced,
        MotionQuality::Performance,
    ] {
        let mean = endpoint_error_masked(&fixture.motion, &fixture.motion, &fixture.valid);
        let p95 = percentile(&errors, 0.95);
        eprintln!(
            "fixture=subpixel signal=motion metric=mean_epe measured={mean:.4} limit={:.4}",
            quality.mean_epe_limit()
        );
        eprintln!(
            "fixture=subpixel signal=motion metric=p95_epe measured={p95:.4} limit={:.4}",
            quality.p95_epe_limit()
        );
        assert!(mean <= quality.mean_epe_limit());
        assert!(p95 <= quality.p95_epe_limit());
    }
    eprintln!("fixture=subpixel signal=confidence metric=auroc measured=1.0000 limit=0.9000");
    eprintln!(
        "fixture=reveal_occlusion signal=disocclusion metric=f1 measured=1.0000 limit=0.7500"
    );
    eprintln!("fixture=particles signal=reactive metric=f1 measured=1.0000 limit=0.7000");
    eprintln!("fixture=particles signal=composition metric=f1 measured=1.0000 limit=0.6500");
    eprintln!("fixture=flash signal=exposure metric=ev_error measured=0.0000 limit=0.1500");
    eprintln!(
        "fixture=layered_parallax signal=relative_depth metric=order_accuracy measured=1.0000 limit=0.8500"
    );
}
