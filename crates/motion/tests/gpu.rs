#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_motion::{MotionEstimator, MotionQuality};
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};
#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

#[test]
fn confidence_reads_a_separate_dense_motion_source() {
    let shader = include_str!("../../../shaders/motion/confidence.comp");
    assert!(shader.contains("dense_motion_input"));
    assert!(shader.contains("imageLoad(dense_motion_input,n)"));
    assert!(!shader.contains("imageLoad(motion_image,n)"));
}

#[test]
fn flow_cost_uses_two_axis_luminance_gradients() {
    let shader = include_str!("../../../shaders/motion/flow.comp");
    assert!(shader.contains("vec2 source_gradient"));
    assert!(shader.contains("vec2 target_gradient"));
    assert!(shader.contains("length(source_gradient"));
}

fn noise(a: i32, b: i32) -> f32 {
    let mut v = (a as u32).wrapping_mul(1664525) ^ (b as u32).wrapping_mul(1013904223) ^ 0x91e10da5;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    (v & 255) as f32
}

fn sample_pattern(px: f32, py: f32) -> u8 {
    let ax = px.floor() as i32;
    let ay = py.floor() as i32;
    let fx = px.fract();
    let fy = py.fract();
    let a = noise(ax, ay) * (1.0 - fx) + noise(ax + 1, ay) * fx;
    let b = noise(ax, ay + 1) * (1.0 - fx) + noise(ax + 1, ay + 1) * fx;
    (a * (1.0 - fy) + b * fy).clamp(0.0, 255.0) as u8
}

fn pattern(width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
    let mut pixels = vec![0; (width * height * 4) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let px = x - dx;
            let py = y - dy;
            let v = sample_pattern(px as f32 / 4.0, py as f32 / 4.0);
            let i = (y as u32 * width + x as u32) as usize * 4;
            pixels[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    pixels
}

fn affine_pattern(width: u32, height: u32) -> Vec<u8> {
    let theta = 0.025_f32;
    let scale = 1.02_f32;
    let (s, c) = theta.sin_cos();
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let mut pixels = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let px = x as f32 - cx;
            let py = y as f32 - cy;
            let source_x = (scale * (c * px - s * py) + cx) / 4.0;
            let source_y = (scale * (s * px + c * py) + cy) / 4.0;
            let value = sample_pattern(source_x, source_y);
            let i = (y * width + x) as usize * 4;
            pixels[i..i + 4].copy_from_slice(&[value, value, value, 255]);
        }
    }
    pixels
}

fn percentile95(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) * 95) / 100]
}

fn auroc(positive: &[f32], negative: &[f32]) -> f32 {
    let mut wins = 0.0;
    let total = (positive.len() * negative.len()) as f32;
    for &left in positive {
        for &right in negative {
            wins += match left.total_cmp(&right) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        }
    }
    wins / total
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn dense_outputs_match_processing_extent() {
    let gpu = unsafe { Gpu::new() };
    let extent = vk::Extent2D {
        width: 129,
        height: 97,
    };
    let color = unsafe {
        Image::new(
            &gpu.device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
    }
    .unwrap();
    let estimator =
        unsafe { MotionEstimator::new(&gpu.device, &gpu.memory, extent, color.view, false) }
            .unwrap();
    assert_eq!(estimator.vectors.extent, extent);
    assert_eq!(estimator.confidence.extent, extent);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn dense_confidence_output_is_stable_across_repeated_dispatches() {
    let width = 129;
    let height = 97;
    let previous = pattern(width, height, 0, 0);
    let current = pattern(width, height, 5, -3);
    let first =
        unsafe { pair_quality(width, height, &previous, &current, MotionQuality::Balanced) };
    let second =
        unsafe { pair_quality(width, height, &previous, &current, MotionQuality::Balanced) };
    let max_motion_delta = first
        .0
        .iter()
        .zip(second.0.iter())
        .map(|(left, right)| (left[0] - right[0]).abs().max((left[1] - right[1]).abs()))
        .fold(0.0, f32::max);
    let max_confidence_delta = first
        .1
        .iter()
        .zip(second.1.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "repeated dense output deltas: motion={max_motion_delta:.6}, confidence={max_confidence_delta:.6}"
    );
    assert!(max_motion_delta <= 0.001);
    assert!(max_confidence_delta <= 0.001);
}
unsafe fn pair(
    width: u32,
    height: u32,
    previous: &[u8],
    current: &[u8],
) -> (Vec<[f32; 2]>, Vec<f32>, u32, f32) {
    unsafe { pair_quality(width, height, previous, current, MotionQuality::Ultra) }
}

unsafe fn pair_quality(
    width: u32,
    height: u32,
    previous: &[u8],
    current: &[u8],
    quality: MotionQuality,
) -> (Vec<[f32; 2]>, Vec<f32>, u32, f32) {
    unsafe { pair_quality_valid(width, height, previous, current, quality, true) }
}

unsafe fn pair_quality_invalid(
    width: u32,
    height: u32,
    previous: &[u8],
    current: &[u8],
    quality: MotionQuality,
) -> (Vec<[f32; 2]>, Vec<f32>, u32, f32) {
    unsafe { pair_quality_valid(width, height, previous, current, quality, false) }
}

unsafe fn pair_quality_valid(
    width: u32,
    height: u32,
    previous: &[u8],
    current: &[u8],
    quality: MotionQuality,
    history_valid: bool,
) -> (Vec<[f32; 2]>, Vec<f32>, u32, f32) {
    let gpu = unsafe { Gpu::new() };
    let d = &gpu.device;
    let extent = vk::Extent2D { width, height };
    let color = unsafe {
        Image::new(
            d,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
    }
    .unwrap();
    let mut estimator =
        unsafe { MotionEstimator::new(d, &gpu.memory, extent, color.view, false) }.unwrap();
    estimator.set_quality(quality);
    let upload = unsafe {
        Buffer::new(
            d,
            &gpu.memory,
            (width * height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let count = width as u64 * height as u64;
    let vectors_bytes = count * 4;
    let confidence_offset = vectors_bytes;
    let metadata_offset = (confidence_offset + count + 3) & !3;
    let download = unsafe {
        Buffer::new(
            d,
            &gpu.memory,
            metadata_offset + 32,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    for (i, data) in [previous, current].iter().enumerate() {
        unsafe { upload.write(data) }.unwrap();
        unsafe {
            gpu.submit(|command| {
                image_barrier(
                    d,
                    command,
                    color.handle,
                    if i == 0 {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    },
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let layers = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(layers)
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                d.cmd_copy_buffer_to_image(
                    command,
                    upload.handle,
                    color.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
                image_barrier(
                    d,
                    command,
                    color.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                estimator.record(command, i, i != 0 && history_valid, 2);
            });
        }
    }
    unsafe {
        gpu.submit(|command| {
            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            for (image, offset) in [
                (&estimator.vectors, 0),
                (&estimator.confidence, confidence_offset),
            ] {
                image_barrier(
                    d,
                    command,
                    image.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(layers)
                    .image_extent(vk::Extent3D {
                        width: image.extent.width,
                        height: image.extent.height,
                        depth: 1,
                    });
                d.cmd_copy_image_to_buffer(
                    command,
                    image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    download.handle,
                    &[copy],
                );
            }
            d.cmd_copy_buffer(
                command,
                estimator.metadata.handle,
                download.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: metadata_offset,
                    size: 32,
                }],
            );
            memory_barrier(d, command);
        });
    }
    let mut bytes = vec![0; (metadata_offset + 32) as usize];
    unsafe { download.read(&mut bytes) }.unwrap();
    let half = |offset: usize| {
        let bits = u16::from_ne_bytes(bytes[offset..offset + 2].try_into().unwrap());
        let sign = f32::from((bits >> 15) & 1);
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let fraction = f32::from(bits & 0x3ff);
        if exponent == 0 {
            (if sign == 0.0 { 1.0 } else { -1.0 }) * (fraction / 1024.0) * 2f32.powi(-14)
        } else if exponent == 31 {
            f32::NAN
        } else {
            (if sign == 0.0 { 1.0 } else { -1.0 })
                * (1.0 + fraction / 1024.0)
                * 2f32.powi(exponent - 15)
        }
    };
    let vectors = (0..count as usize)
        .map(|i| [half(i * 4), half(i * 4 + 2)])
        .collect();
    let confidence = (0..count as usize)
        .map(|i| f32::from(bytes[confidence_offset as usize + i]) / 255.0)
        .collect();
    let cut = u32::from_ne_bytes(
        bytes[metadata_offset as usize..metadata_offset as usize + 4]
            .try_into()
            .unwrap(),
    );
    let exposure = f32::from_ne_bytes(
        bytes[metadata_offset as usize + 16..metadata_offset as usize + 20]
            .try_into()
            .unwrap(),
    );
    (vectors, confidence, cut, exposure)
}
#[test]
#[ignore = "requires a Vulkan GPU"]
fn known_motion_and_scene_cut() {
    for (width, height, dx, dy) in [
        (128, 96, 0, 0),
        (128, 96, 8, 0),
        (128, 96, -8, 4),
        (129, 97, 16, -8),
    ] {
        let previous = pattern(width, height, 0, 0);
        let current = pattern(width, height, dx, dy);
        let (vectors, _, cut, _) = unsafe { pair(width, height, &previous, &current) };
        let mut sum = 0.0;
        let mut count = 0;
        for y in 16..height - 16 {
            for x in 16..width - 16 {
                let v = vectors[(y * width + x) as usize];
                sum += ((v[0] + dx as f32).powi(2) + (v[1] + dy as f32).powi(2)).sqrt();
                count += 1;
            }
        }
        let error = sum / count as f32;
        eprintln!("{width}x{height} displacement=({dx},{dy}) EPE={error:.4} cut={cut}");
        assert_eq!(cut, 0);
        assert!(
            error <= if dx == 0 && dy == 0 { 0.25 } else { 1.0 },
            "EPE={error}"
        );
    }
    let black = vec![0; 128 * 96 * 4];
    let white = vec![255; 128 * 96 * 4];
    let (_, confidence, cut, exposure) = unsafe { pair(128, 96, &black, &white) };
    assert_eq!(cut, 1);
    assert!(confidence.iter().all(|c| c.is_finite() && *c <= 0.01));
    assert!(exposure.is_finite() && exposure > 0.0);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn all_quality_presets_keep_dense_odd_extent_translation() {
    let width = 129;
    let height = 97;
    let previous = pattern(width, height, 0, 0);
    let current = pattern(width, height, 5, -3);
    for quality in [
        MotionQuality::Ultra,
        MotionQuality::High,
        MotionQuality::Balanced,
        MotionQuality::Performance,
    ] {
        let (vectors, confidence, cut, _) =
            unsafe { pair_quality(width, height, &previous, &current, quality) };
        let mut sum = 0.0;
        let mut errors = Vec::new();
        let mut samples = 0;
        for y in 16..height - 16 {
            for x in 16..width - 16 {
                let value = vectors[(y * width + x) as usize];
                assert!(value[0].is_finite() && value[1].is_finite());
                let error = ((value[0] + 5.0).powi(2) + (value[1] - 3.0).powi(2)).sqrt();
                sum += error;
                errors.push(error);
                samples += 1;
            }
        }
        let epe = sum / samples as f32;
        assert_eq!(cut, 0);
        assert!(confidence.iter().all(|value| value.is_finite()));
        let p95 = percentile95(errors);
        eprintln!("{quality:?} dense translation EPE={epe:.4} p95={p95:.4}");
        let limit = match quality {
            MotionQuality::Ultra => 1.0,
            MotionQuality::High => 1.25,
            MotionQuality::Balanced => 1.75,
            MotionQuality::Performance => 2.5,
        };
        assert!(epe <= limit, "{quality:?} EPE={epe} > {limit}");
        assert!(
            p95 <= limit * 2.0,
            "{quality:?} p95={p95} > {}",
            limit * 2.0
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn rotation_zoom_and_affine_fixture_remain_finite_and_calibrated() {
    let width = 129;
    let height = 97;
    let previous = pattern(width, height, 0, 0);
    let current = affine_pattern(width, height);
    let theta = 0.025_f32;
    let scale = 1.02_f32;
    let (s, c) = theta.sin_cos();
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    for quality in [
        MotionQuality::Ultra,
        MotionQuality::High,
        MotionQuality::Balanced,
        MotionQuality::Performance,
    ] {
        let (vectors, confidence, cut, _) =
            unsafe { pair_quality(width, height, &previous, &current, quality) };
        let mut sum = 0.0;
        let mut errors = Vec::new();
        let mut samples = 0;
        for y in 20..height - 20 {
            for x in 20..width - 20 {
                let px = x as f32 - cx;
                let py = y as f32 - cy;
                let expected = [
                    scale * (c * px - s * py) + cx - x as f32,
                    scale * (s * px + c * py) + cy - y as f32,
                ];
                let value = vectors[(y * width + x) as usize];
                assert!(value[0].is_finite() && value[1].is_finite());
                let error =
                    ((value[0] - expected[0]).powi(2) + (value[1] - expected[1]).powi(2)).sqrt();
                sum += error;
                errors.push(error);
                samples += 1;
            }
        }
        let epe = sum / samples as f32;
        assert_eq!(cut, 0);
        assert!(confidence.iter().all(|value| value.is_finite()));
        let p95 = percentile95(errors);
        eprintln!("{quality:?} affine EPE={epe:.4} p95={p95:.4}");
        let limit = match quality {
            MotionQuality::Ultra => 1.0,
            MotionQuality::High => 1.25,
            MotionQuality::Balanced => 1.75,
            MotionQuality::Performance => 2.5,
        };
        assert!(epe <= limit, "{quality:?} EPE={epe} > {limit}");
        assert!(
            p95 <= limit * 2.0,
            "{quality:?} p95={p95} > {}",
            limit * 2.0
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn independent_object_and_uniform_frames_keep_guidance_calibrated() {
    let width = 129;
    let height = 97;
    let previous = pattern(width, height, 0, 0);
    let mut current = pattern(width, height, 5, -3);
    for y in 32..64 {
        for x in 40..88 {
            let offset = (y * width + x) as usize * 4;
            current[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    for quality in [
        MotionQuality::Ultra,
        MotionQuality::High,
        MotionQuality::Balanced,
        MotionQuality::Performance,
    ] {
        let (_, confidence, cut, _) =
            unsafe { pair_quality(width, height, &previous, &current, quality) };
        assert_eq!(cut, 0);
        assert!(confidence.iter().all(|value| value.is_finite()));
        let mut object = 0.0;
        let mut object_scores = Vec::new();
        for y in 48..80 {
            for x in 48..80 {
                let value = confidence[y * width as usize + x];
                object += value;
                if y < 64 {
                    object_scores.push(1.0 - value);
                }
            }
        }
        object /= (32 * 32) as f32;
        let mut background = 0.0;
        let mut background_scores = Vec::new();
        for y in 16..28 {
            for x in 16..28 {
                let value = confidence[y * width as usize + x];
                background += value;
                background_scores.push(1.0 - value);
            }
        }
        background /= (12 * 12) as f32;
        eprintln!(
            "{quality:?} independent object confidence={object:.4} background={background:.4}"
        );
        assert!(
            object < background,
            "{quality:?} object confidence was not lower"
        );
        let score = auroc(&object_scores, &background_scores);
        eprintln!("{quality:?} occlusion AUROC={score:.4}");
        assert!(score >= 0.90, "{quality:?} AUROC={score} < 0.90");
    }

    let uniform = vec![128; (width * height * 4) as usize];
    for quality in [
        MotionQuality::Ultra,
        MotionQuality::High,
        MotionQuality::Balanced,
        MotionQuality::Performance,
    ] {
        let (vectors, confidence, cut, exposure) =
            unsafe { pair_quality(width, height, &uniform, &uniform, quality) };
        assert_eq!(cut, 0);
        assert!(exposure.is_finite() && exposure > 0.0);
        assert!(
            vectors
                .iter()
                .all(|value| value[0].is_finite() && value[1].is_finite())
        );
        assert!(confidence.iter().all(|value| value.is_finite()));
    }

    let (vectors, confidence, _, _) = unsafe {
        pair_quality_invalid(width, height, &previous, &current, MotionQuality::Balanced)
    };
    assert!(vectors.iter().all(|value| *value == [0.0, 0.0]));
    assert!(confidence.iter().all(|value| *value == 0.0));
}

fn expected_exposure(pixels: &[u8], width: u32, height: u32) -> f32 {
    let (pixels, _) = pixels.as_chunks::<4>();
    let sum = pixels
        .iter()
        .map(|pixel| {
            let luma = (f32::from(pixel[0]) * 0.2126
                + f32::from(pixel[1]) * 0.7152
                + f32::from(pixel[2]) * 0.0722)
                / 255.0;
            luma.max(1e-4).log2()
        })
        .sum::<f32>();
    2.0_f32.powf(-sum / (width * height) as f32)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn bounded_flash_and_fade_are_not_scene_cuts() {
    let width = 128;
    let height = 96;
    let previous = pattern(width, height, 0, 0);
    for multiplier in [1.5_f32, 0.65_f32] {
        let current = previous
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| {
                [
                    (f32::from(pixel[0]) * multiplier).min(255.0) as u8,
                    (f32::from(pixel[1]) * multiplier).min(255.0) as u8,
                    (f32::from(pixel[2]) * multiplier).min(255.0) as u8,
                    255,
                ]
            })
            .collect::<Vec<_>>();
        let (_, _, cut, exposure) = unsafe { pair(width, height, &previous, &current) };
        eprintln!("exposure multiplier={multiplier} estimated={exposure:.3} cut={cut}");
        assert_eq!(cut, 0);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn parallel_exposure_is_finite_and_accurate_for_uniform_and_gradient_frames() {
    let width = 128;
    let height = 96;
    let uniform = vec![128; (width * height * 4) as usize];
    let (_, _, cut, exposure) = unsafe { pair(width, height, &uniform, &uniform) };
    let expected = expected_exposure(&uniform, width, height);
    assert_eq!(cut, 0);
    assert!(exposure.is_finite() && exposure > 0.0);
    assert!((exposure.log2() - expected.log2()).abs() <= 0.15);

    let gradient = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let value = ((x + y) * 255 / (width + height - 2)) as u8;
                [value, value, value, 255]
            })
        })
        .collect::<Vec<_>>();
    let (_, _, cut, exposure) = unsafe { pair(width, height, &gradient, &gradient) };
    let expected = expected_exposure(&gradient, width, height);
    assert_eq!(cut, 0);
    assert!(exposure.is_finite() && exposure > 0.0);
    assert!((exposure.log2() - expected.log2()).abs() <= 0.15);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn occlusion_reduces_confidence() {
    let previous = pattern(128, 96, 0, 0);
    let mut current = previous.clone();
    for y in 32..64 {
        for x in 40..88 {
            let offset = (y * 128 + x) * 4;
            current[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let (_, confidence, _, _) = unsafe { pair(128, 96, &previous, &current) };
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for y in 24..72 {
        for x in 32..96 {
            let value = confidence[y * 128 + x];
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if (40..56).contains(&y) && (48..80).contains(&x) {
                inside.push(value);
            } else if !(36..60).contains(&y) || !(44..84).contains(&x) {
                outside.push(value);
            }
        }
    }
    let average = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;
    eprintln!(
        "occlusion confidence: inside={:.4}, outside={:.4}",
        average(&inside),
        average(&outside)
    );
    assert!(average(&inside) < average(&outside) * 0.5);
}
