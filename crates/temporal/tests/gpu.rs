#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_temporal::{GuidanceReset, SignalState};
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
        "transparency_residual_image",
        "imageStore(transparency_residual_image",
    ] {
        assert!(shader.contains(term), "guidance shader is missing {term}");
    }
    assert!(!shader.contains("disocclusion = 1.0 - confidence"));
    let scene = include_str!("../../../shaders/motion/scene_reduce.comp");
    assert!(scene.contains("raw_shift"));
    assert!(scene.contains("motion_consistency"));
    assert!(shader.contains("((p.width + 1u) / 2u) * ((p.height + 1u) / 2u)"));
    let runtime = include_str!("../../../crates/runtime/src/present.rs");
    assert!(runtime.contains("guidance.record_provider_failure"));
    assert!(runtime.contains("guidance.reset_history"));
    assert!(runtime.contains("guidance.clear_provider_failure"));
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

fn flow_noise(x: i32, y: i32) -> f32 {
    let mut value =
        (x as u32).wrapping_mul(1_664_525) ^ (y as u32).wrapping_mul(1_013_904_223) ^ 0x91e1_0da5;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    (value & 255) as f32
}

fn flow_sample(px: f32, py: f32) -> u8 {
    let ax = px.floor() as i32;
    let ay = py.floor() as i32;
    let fx = px.fract();
    let fy = py.fract();
    let a = flow_noise(ax, ay) * (1.0 - fx) + flow_noise(ax + 1, ay) * fx;
    let b = flow_noise(ax, ay + 1) * (1.0 - fx) + flow_noise(ax + 1, ay + 1) * fx;
    (a * (1.0 - fy) + b * fy).clamp(0.0, 255.0) as u8
}

fn translated_flow_fixture(
    extent: vk::Extent2D,
    dx: i32,
    dy: i32,
) -> (Vec<u8>, Vec<u8>, Vec<bool>) {
    let mut previous = vec![0u8; (extent.width * extent.height * 4) as usize];
    let mut current = previous.clone();
    let mut holes = vec![false; (extent.width * extent.height) as usize];
    for y in 0..extent.height as i32 {
        for x in 0..extent.width as i32 {
            let index = (y * extent.width as i32 + x) as usize;
            let value = flow_sample(x as f32 / 4.0, y as f32 / 4.0);
            previous[index * 4..index * 4 + 4].copy_from_slice(&[value, value, value, 255]);
            let source_x = x - dx;
            let source_y = y - dy;
            if source_x < 0
                || source_x >= extent.width as i32
                || source_y < 0
                || source_y >= extent.height as i32
            {
                current[index * 4..index * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
                holes[index] = true;
            } else {
                let source = (source_y * extent.width as i32 + source_x) as usize * 4;
                current[index * 4..index * 4 + 4].copy_from_slice(&previous[source..source + 4]);
            }
        }
    }
    (previous, current, holes)
}

unsafe fn clear_motion(gpu: &Gpu, motion: &Image, value: [f32; 4]) {
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                motion.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_clear_color_image(
                command,
                motion.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: value },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)],
            );
            image_barrier(
                device,
                command,
                motion.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        });
    }
}

unsafe fn guidance_masks(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (reactive, disocclusion, transparency, ..) = unsafe {
        guidance_masks_with_options(extent, previous_pixels, current_pixels, None, false)
    };
    (reactive, disocclusion, transparency)
}

unsafe fn guidance_masks_with_motion(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
    forced_motion: Option<[f32; 4]>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (reactive, disocclusion, transparency, ..) = unsafe {
        guidance_masks_with_options(
            extent,
            previous_pixels,
            current_pixels,
            forced_motion,
            false,
        )
    };
    (reactive, disocclusion, transparency)
}

unsafe fn guidance_provider_failure(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>, f32, [SignalState; 4], bool) {
    unsafe { guidance_masks_with_options(extent, previous_pixels, current_pixels, None, true) }
}

unsafe fn read_transparency(
    gpu: &Gpu,
    device: &ash::Device,
    guidance: &tuxscaling_temporal::GuidanceEstimator,
    readback: &Buffer,
    extent: vk::Extent2D,
) -> Vec<u8> {
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                guidance.transparency.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                guidance.transparency.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.handle,
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
                guidance.transparency.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            memory_barrier(device, command);
        });
        let mut bytes = vec![0u8; readback.size as usize];
        readback.read(&mut bytes).unwrap();
        bytes
    }
}

unsafe fn read_exposure(
    gpu: &Gpu,
    device: &ash::Device,
    guidance: &tuxscaling_temporal::GuidanceEstimator,
    readback: &Buffer,
) -> f32 {
    unsafe {
        gpu.submit(|command| {
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
                readback.handle,
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
            image_barrier(
                device,
                command,
                guidance.exposure.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            memory_barrier(device, command);
        });
        let mut bytes = [0u8; 4];
        readback.read(&mut bytes).unwrap();
        f32::from_ne_bytes(bytes)
    }
}

unsafe fn guidance_two_frame_translated_history(
    extent: vk::Extent2D,
) -> (Vec<u8>, Vec<u8>, f32, f32) {
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
        let count = (extent.width * extent.height) as usize;
        let upload = Buffer::new(
            device,
            &gpu.memory,
            (count * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let mut frame0 = vec![32u8; count * 4];
        let mut frame1 = frame0.clone();
        let mut frame2 = vec![64u8; count * 4];
        let paint = |frame: &mut [u8], x0: u32, x1: u32| {
            for y in 12..24 {
                for x in x0..x1 {
                    let index = (y * extent.width + x) as usize * 4;
                    frame[index..index + 4].copy_from_slice(&[144, 144, 144, 255]);
                }
            }
        };
        paint(&mut frame1, 16, 28);
        paint(&mut frame2, 20, 32);
        for frame in [&mut frame0, &mut frame1, &mut frame2] {
            for pixel in frame.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
        }
        upload.write(&frame0).unwrap();
        copy_upload(&gpu, &current, &upload, extent);
        let mut motion =
            MotionEstimator::new(device, &gpu.memory, extent, current.view, false).unwrap();
        gpu.submit(|command| motion.record(command, 0, false, 0));
        upload.write(&frame1).unwrap();
        copy_upload_reuse(&gpu, &current, &upload, extent);
        gpu.submit(|command| motion.record(command, 1, true, 0));
        clear_motion(&gpu, &motion.vectors, [-4.0, 0.0, 0.0, 0.0]);
        upload.write(&frame0).unwrap();
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
        let readback = Buffer::new(
            device,
            &gpu.memory,
            (count) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let exposure_readback = Buffer::new(
            device,
            &gpu.memory,
            4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| guidance.record(command, true));
        let first = read_transparency(&gpu, device, &guidance, &readback, extent);
        let first_exposure = read_exposure(&gpu, device, &guidance, &exposure_readback);

        upload.write(&frame2).unwrap();
        copy_upload_reuse(&gpu, &current, &upload, extent);
        gpu.submit(|command| motion.record(command, 0, true, 0));
        clear_motion(&gpu, &motion.vectors, [-4.0, 0.0, 0.0, 0.0]);
        upload.write(&frame1).unwrap();
        copy_upload_reuse(&gpu, &previous, &upload, extent);
        gpu.submit(|command| guidance.record(command, true));
        let second = read_transparency(&gpu, device, &guidance, &readback, extent);
        let second_exposure = read_exposure(&gpu, device, &guidance, &exposure_readback);
        (first, second, first_exposure, second_exposure)
    }
}

unsafe fn guidance_exposure_after_provider_failure(extent: vk::Extent2D) -> (f32, f32, f32) {
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
        gpu.submit(|command| motion.record(command, 0, true, 0));
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
        let readback = Buffer::new(
            device,
            &gpu.memory,
            4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| guidance.record(command, true));
        let first = read_exposure(&gpu, device, &guidance, &readback);
        gpu.submit(|command| guidance.record_provider_failure(command));
        let fallback = read_exposure(&gpu, device, &guidance, &readback);
        gpu.submit(|command| guidance.record(command, true));
        let fresh = read_exposure(&gpu, device, &guidance, &readback);
        (first, fallback, fresh)
    }
}

unsafe fn guidance_masks_with_options(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
    forced_motion: Option<[f32; 4]>,
    provider_failure: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, f32, [SignalState; 4], bool) {
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
        if let Some(forced_motion) = forced_motion {
            clear_motion(&gpu, &motion.vectors, forced_motion);
        }
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
        let exposure_offset = count * 3;
        let readback = Buffer::new(
            device,
            &gpu.memory,
            exposure_offset + 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| {
            if provider_failure {
                guidance.record_provider_failure(command);
            } else {
                guidance.record(command, true);
            }
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
                readback.handle,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(exposure_offset)
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
            memory_barrier(device, command);
        });
        let mut bytes = vec![0u8; readback.size as usize];
        readback.read(&mut bytes).unwrap();
        let exposure = f32::from_ne_bytes(
            bytes[exposure_offset as usize..exposure_offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let view = guidance.view(
            &motion,
            1,
            extent,
            !provider_failure,
            tuxscaling_temporal::FrameTiming::default(),
            if provider_failure {
                GuidanceReset::ProviderFailure
            } else {
                GuidanceReset::None
            },
        );
        (
            bytes[reactive_offset as usize..disocclusion_offset as usize].to_vec(),
            bytes[disocclusion_offset as usize..transparency_offset as usize].to_vec(),
            bytes[transparency_offset as usize..exposure_offset as usize].to_vec(),
            exposure,
            [
                view.reactive.state,
                view.disocclusion.state,
                view.exposure.state,
                view.transparency_composition.state,
            ],
            view.requires_history_reset,
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
    let (previous, current, labels) = translated_flow_fixture(extent, 5, 0);
    let (reactive, disocclusion, _) = unsafe {
        guidance_masks_with_motion(extent, &previous, &current, Some([-5.0, 0.0, 0.0, 0.0]))
    };
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
    assert!(score >= 0.75);
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
    let inside = average(0..2, 0..48);
    let outside = average(8..14, 0..48);
    eprintln!("disocclusion mask: inside={inside:.3} outside={outside:.3}");
    assert!(inside > outside + 0.10);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn guidance_coverage_does_not_turn_low_confidence_into_disocclusion() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (previous, mut current, _) = translated_flow_fixture(extent, 0, 0);
    for pixel in current.chunks_exact_mut(4) {
        pixel[0] = 255 - pixel[0];
        pixel[1] = 255 - pixel[1];
        pixel[2] = 255 - pixel[2];
    }
    let (_, disocclusion, _) = unsafe {
        guidance_masks_with_motion(extent, &previous, &current, Some([0.0, 0.0, 0.0, 0.0]))
    };
    let maximum = disocclusion.iter().copied().max().unwrap_or_default();
    let average =
        disocclusion.iter().map(|value| *value as f32).sum::<f32>() / disocclusion.len() as f32;
    eprintln!("low-confidence coverage disocclusion: max={maximum} average={average:.1}");
    assert!(maximum < 160);
    assert!(average < 96.0);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn provider_failure_writes_fallback_guidance_and_labels_the_view() {
    let extent = vk::Extent2D {
        width: 32,
        height: 24,
    };
    let (reactive, disocclusion, transparency, exposure, states, reset) = unsafe {
        let (previous, current, _) = translated_flow_fixture(extent, 0, 0);
        let (reactive, disocclusion, transparency, exposure, states, reset) =
            guidance_provider_failure(extent, &previous, &current);
        (
            reactive,
            disocclusion,
            transparency,
            exposure,
            states,
            reset,
        )
    };
    assert!(reactive.iter().all(|value| *value == 0));
    assert!(disocclusion.iter().all(|value| *value == 0));
    assert!(transparency.iter().all(|value| *value == 0));
    assert!((exposure - 1.0).abs() <= f32::EPSILON);
    assert_eq!(states, [SignalState::ConstantFallback; 4]);
    assert!(reset);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn transparency_history_reprojects_a_persistent_residual_across_frames() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (first, second, first_exposure, second_exposure) =
        unsafe { guidance_two_frame_translated_history(extent) };
    let average = |values: &[u8], x0: u32, x1: u32| {
        let mut total = 0.0;
        let mut count = 0;
        for y in 12..24 {
            for x in x0..x1 {
                total += values[(y * extent.width + x) as usize] as f32 / 255.0;
                count += 1;
            }
        }
        total / count as f32
    };
    let first_overlay = average(&first, 16, 28);
    let second_overlay = average(&second, 20, 32);
    let second_background = average(&second, 36, 48);
    eprintln!(
        "translated history: first={first_overlay:.3} second={second_overlay:.3} background={second_background:.3} exposure={first_exposure:.3}->{second_exposure:.3}"
    );
    assert!(first_overlay > 0.20);
    assert!(second_overlay > second_background + 0.08);
    assert!(first_exposure.is_finite() && second_exposure.is_finite());
    assert!(second_exposure < first_exposure);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn provider_failure_resets_history_before_the_next_valid_frame() {
    let (first, fallback, fresh) = unsafe {
        guidance_exposure_after_provider_failure(vk::Extent2D {
            width: 32,
            height: 24,
        })
    };
    eprintln!("provider reset exposure: first={first:.3} fallback={fallback:.3} fresh={fresh:.3}");
    assert!(first > 1.5);
    assert!((fallback - 1.0).abs() <= f32::EPSILON);
    assert!(fresh > 1.5);
}
