#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

#[test]
fn guidance_shader_contains_reprojected_mask_and_exposure_producers() {
    let shader = include_str!("../../../shaders/temporal/guidance.comp");
    for term in [
        "dense_motion",
        "reproject",
        "occupancy",
        "divergence",
        "transparency",
        "percentile",
        "adaptation",
        "scene_consistency",
        "provider_failure",
    ] {
        assert!(shader.contains(term), "guidance shader is missing {term}");
    }
    assert!(!shader.contains("disocclusion = 1.0 - confidence"));
    let scene = include_str!("../../../shaders/motion/scene_reduce.comp");
    assert!(scene.contains("raw_shift"));
    assert!(scene.contains("motion_consistency"));
}

unsafe fn copy_upload(gpu: &Gpu, image: &Image, upload: &Buffer, extent: vk::Extent2D) {
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                device,
                command,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });
    }
}

unsafe fn copy_upload_reuse(gpu: &Gpu, image: &Image, upload: &Buffer, extent: vk::Extent2D) {
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                image.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                device,
                command,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });
    }
}

unsafe fn guidance_masks(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    unsafe {
        let gpu = Gpu::new();
        let device = &gpu.device;
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let current = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )
        .unwrap();
        let previous = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )
        .unwrap();
        let upload = Buffer::new(
            device,
            &gpu.memory,
            (extent.width * extent.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        upload.write(previous_pixels).unwrap();
        copy_upload(&gpu, &current, &upload, extent);
        let mut motion =
            MotionEstimator::new(device, &gpu.memory, extent, current.view, false).unwrap();
        gpu.submit(|command| motion.record(command, 0, false, 0));
        upload.write(current_pixels).unwrap();
        copy_upload_reuse(&gpu, &current, &upload, extent);
        gpu.submit(|command| motion.record(command, 1, true, 0));
        upload.write(previous_pixels).unwrap();
        copy_upload(&gpu, &previous, &upload, extent);
        let mut guidance = tuxscaling_temporal::GuidanceEstimator::new(
            device,
            &gpu.memory,
            extent,
            current.view,
            previous.view,
            motion.confidence.view,
            motion.vectors.view,
            motion.metadata.handle,
            motion.stats.handle,
        )
        .unwrap();
        let count = (extent.width * extent.height) as u64;
        let reactive_offset = 0;
        let disocclusion_offset = count;
        let transparency_offset = count * 2;
        let readback = Buffer::new(
            device,
            &gpu.memory,
            count * 3,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| {
            guidance.record(command, true);
            for (image, offset) in [
                (&guidance.reactive, reactive_offset),
                (&guidance.disocclusion, disocclusion_offset),
                (&guidance.transparency, transparency_offset),
            ] {
                image_barrier(
                    device,
                    command,
                    image.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                device.cmd_copy_image_to_buffer(
                    command,
                    image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.handle,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })],
                );
            }
            memory_barrier(device, command);
        });
        let mut bytes = vec![0u8; readback.size as usize];
        readback.read(&mut bytes).unwrap();
        (
            bytes[reactive_offset as usize..disocclusion_offset as usize].to_vec(),
            bytes[disocclusion_offset as usize..transparency_offset as usize].to_vec(),
            bytes[transparency_offset as usize..].to_vec(),
        )
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn normal_guidance_record_updates_the_exposure_image() {
    unsafe {
        let gpu = Gpu::new();
        let device = &gpu.device;
        let extent = vk::Extent2D {
            width: 64,
            height: 48,
        };
        let image_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let current = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            image_usage,
        )
        .unwrap();
        let previous = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            image_usage,
        )
        .unwrap();
        let pixels = vec![128u8; (extent.width * extent.height * 4) as usize];
        let upload = Buffer::new(
            device,
            &gpu.memory,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        upload.write(&pixels).unwrap();
        copy_upload(&gpu, &current, &upload, extent);
        copy_upload(&gpu, &previous, &upload, extent);

        let mut motion =
            MotionEstimator::new(device, &gpu.memory, extent, current.view, false).unwrap();
        let mut guidance = tuxscaling_temporal::GuidanceEstimator::new(
            device,
            &gpu.memory,
            extent,
            current.view,
            previous.view,
            motion.confidence.view,
            motion.vectors.view,
            motion.metadata.handle,
            motion.stats.handle,
        )
        .unwrap();
        let count = (extent.width * extent.height) as u64;
        let reactive_offset = 4;
        let disocclusion_offset = reactive_offset + count;
        let transparency_offset = disocclusion_offset + count;
        let exposure_readback = Buffer::new(
            device,
            &gpu.memory,
            transparency_offset + count,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| {
            motion.record(command, 0, true, 0);
            memory_barrier(device, command);
            guidance.record(command, true);
            image_barrier(
                device,
                command,
                guidance.exposure.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                guidance.exposure.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                exposure_readback.handle,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    })],
            );
            for (image, offset) in [
                (&guidance.reactive, reactive_offset),
                (&guidance.disocclusion, disocclusion_offset),
                (&guidance.transparency, transparency_offset),
            ] {
                image_barrier(
                    device,
                    command,
                    image.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                device.cmd_copy_image_to_buffer(
                    command,
                    image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    exposure_readback.handle,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })],
                );
            }
            memory_barrier(device, command);
        });
        let mut bytes = vec![0u8; exposure_readback.size as usize];
        exposure_readback.read(&mut bytes).unwrap();
        let exposure = f32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let expected: f32 = 1.0 / (128.0 / 255.0);
        assert!(exposure.is_finite() && exposure > 0.0);
        assert!((exposure.log2() - expected.log2()).abs() <= 0.15);
        let interior = ((extent.width / 4) as usize, (extent.height / 4) as usize);
        let interior_index = interior.1 * extent.width as usize + interior.0;
        eprintln!(
            "guidance interior reactive={} disocclusion={} transparency={}",
            bytes[reactive_offset as usize + interior_index],
            bytes[disocclusion_offset as usize + interior_index],
            bytes[transparency_offset as usize + interior_index]
        );
        assert!(bytes[reactive_offset as usize + interior_index] < 96);
        assert!(bytes[disocclusion_offset as usize + interior_index] < 250);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn guidance_masks_separate_transparency_from_a_static_background() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let previous = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                let value = ((x * 17 + y * 31) & 255) as u8;
                [
                    value,
                    value.saturating_add(17),
                    value.saturating_add(31),
                    255,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut current = previous.clone();
    for y in 12..36 {
        for x in 16..32 {
            let index = (y * extent.width + x) as usize * 4;
            for channel in &mut current[index..index + 3] {
                *channel = ((*channel as u16 + 255) / 2) as u8;
            }
        }
    }
    let (_, _, transparency) = unsafe { guidance_masks(extent, &previous, &current) };
    let labels = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| (12..36).contains(&y) && (16..32).contains(&x))
        })
        .collect::<Vec<_>>();
    let predicted = transparency
        .iter()
        .map(|value| *value as f32 / 255.0 > 0.20)
        .collect::<Vec<_>>();
    let score = tuxscaling_temporal::quality::f1(&predicted, &labels);
    eprintln!("transparency F1={score:.3}");
    assert!(score >= 0.65);
    let mut inside = 0.0;
    let mut outside = 0.0;
    let mut inside_count = 0;
    let mut outside_count = 0;
    for y in 8..40 {
        for x in 8..56 {
            let value = transparency[(y * extent.width + x) as usize] as f32 / 255.0;
            if (12..36).contains(&y) && (16..32).contains(&x) {
                inside += value;
                inside_count += 1;
            } else {
                outside += value;
                outside_count += 1;
            }
        }
    }
    let inside = inside / inside_count as f32;
    let outside = outside / outside_count as f32;
    eprintln!("transparency mask: inside={inside:.3} outside={outside:.3}");
    assert!(inside > outside + 0.08);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn guidance_disocclusion_uses_flow_holes_and_boundaries() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let previous = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                let value = ((x * 17 + y * 31) & 255) as u8;
                [value, value, value, 255]
            })
        })
        .collect::<Vec<_>>();
    let mut current = previous.clone();
    for y in 16..32 {
        for x in 24..40 {
            let index = (y * extent.width + x) as usize * 4;
            current[index..index + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let (reactive, disocclusion, _) = unsafe { guidance_masks(extent, &previous, &current) };
    let labels = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| (16..32).contains(&y) && (24..40).contains(&x))
        })
        .collect::<Vec<_>>();
    let predicted = disocclusion
        .iter()
        .map(|value| *value as f32 / 255.0 > 0.60)
        .collect::<Vec<_>>();
    let score = tuxscaling_temporal::quality::f1(&predicted, &labels);
    let tp = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| **p && **l)
        .count();
    let fp = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| **p && !**l)
        .count();
    let fn_ = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| !**p && **l)
        .count();
    eprintln!("disocclusion counts tp={tp} fp={fp} fn={fn_}");
    eprintln!("disocclusion F1={score:.3}");
    assert!(score >= 0.65);
    let reactive_predicted = reactive
        .iter()
        .map(|value| *value as f32 / 255.0 > 0.25)
        .collect::<Vec<_>>();
    let reactive_score = tuxscaling_temporal::quality::f1(&reactive_predicted, &labels);
    eprintln!("reactive F1={reactive_score:.3}");
    assert!(reactive_score >= 0.70);
    let average = |x: std::ops::Range<u32>, y: std::ops::Range<u32>| {
        let mut total = 0.0;
        let mut count = 0;
        for row in y {
            for column in x.clone() {
                total += disocclusion[(row * extent.width + column) as usize] as f32 / 255.0;
                count += 1;
            }
        }
        total / count as f32
    };
    let inside = average(24..40, 16..32);
    let outside = average(8..24, 16..32);
    eprintln!("disocclusion mask: inside={inside:.3} outside={outside:.3}");
    assert!(inside > outside + 0.10);
}
