#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};
#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

fn pattern(width: u32, height: u32, dx: i32, dy: i32) -> Vec<u8> {
    let mut pixels = vec![0; (width * height * 4) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let px = x - dx;
            let py = y - dy;
            let noise = |a: i32, b: i32| {
                let mut v = (a as u32).wrapping_mul(1664525)
                    ^ (b as u32).wrapping_mul(1013904223)
                    ^ 0x91e10da5;
                v ^= v >> 16;
                v = v.wrapping_mul(0x7feb352d);
                v ^= v >> 15;
                (v & 255) as f32
            };
            let ax = px.div_euclid(4);
            let ay = py.div_euclid(4);
            let fx = px.rem_euclid(4) as f32 / 4.0;
            let fy = py.rem_euclid(4) as f32 / 4.0;
            let a = noise(ax, ay) * (1.0 - fx) + noise(ax + 1, ay) * fx;
            let b = noise(ax, ay + 1) * (1.0 - fx) + noise(ax + 1, ay + 1) * fx;
            let v = (a * (1.0 - fy) + b * fy) as u8;
            let i = (y as u32 * width + x as u32) as usize * 4;
            pixels[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    pixels
}
unsafe fn pair(
    width: u32,
    height: u32,
    previous: &[u8],
    current: &[u8],
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
    let count = width.div_ceil(4) * height.div_ceil(4);
    let download = unsafe {
        Buffer::new(
            d,
            &gpu.memory,
            count as u64 * 12 + 32,
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
                estimator.record(command, i, i != 0, 2);
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
                (&estimator.confidence, count as u64 * 8),
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
                    dst_offset: count as u64 * 12,
                    size: 32,
                }],
            );
            memory_barrier(d, command);
        });
    }
    let mut bytes = vec![0; (count * 12 + 32) as usize];
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
        .map(|i| f32::from(bytes[count as usize * 8 + i]) / 255.0)
        .collect();
    let cut = u32::from_ne_bytes(
        bytes[count as usize * 12..count as usize * 12 + 4]
            .try_into()
            .unwrap(),
    );
    let exposure = f32::from_ne_bytes(
        bytes[count as usize * 12 + 16..count as usize * 12 + 20]
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
        let gw = width.div_ceil(4);
        let gh = height.div_ceil(4);
        let mut sum = 0.0;
        let mut count = 0;
        for y in 8..gh - 8 {
            for x in 8..gw - 8 {
                let v = vectors[(y * gw + x) as usize];
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
    for y in 4..20 {
        for x in 4..28 {
            let value = confidence[y * 32 + x];
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
            if (9..15).contains(&y) && (12..20).contains(&x) {
                inside.push(value);
            } else if !(6..18).contains(&y) || !(8..24).contains(&x) {
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
