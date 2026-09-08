#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResolver, GuidanceResource, GuidanceScalar, GuidanceView, JitterSample,
    MotionDirection, MotionUnits, SignalState, ValidRegion,
};
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

type GuidanceOutputs = (
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<f32>,
    f32,
    [SignalState; 4],
    SignalState,
    DepthSemantics,
    bool,
);

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

#[test]
fn guidance_reuses_the_motion_statistics_exposure() {
    let shader = include_str!("../../../shaders/temporal/guidance.comp");
    assert!(shader.contains("return metadata.exposure"));
}

#[test]
fn resolve_shader_guides_weights_with_each_input_sample_luma() {
    let shader = include_str!("../../../shaders/temporal/resolve.comp");
    assert!(shader.contains("vec2(pixel) + 0.5"));
    assert!(shader.contains("signal_luma(sample_pixel)"));
    assert!(shader.contains("flat_depth"));
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn guidance_resolver_outputs_full_resolution_signals() {
    unsafe {
        let gpu = Gpu::new();
        let device = &gpu.device;
        let game_extent = vk::Extent2D {
            width: 4,
            height: 4,
        };
        let estimator_extent = vk::Extent2D {
            width: 2,
            height: 2,
        };
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE;
        let source = Image::new(
            device,
            &gpu.memory,
            game_extent,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )
        .unwrap();
        let motion = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R16G16_SFLOAT,
            usage,
        )
        .unwrap();
        let confidence = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R8_UNORM,
            usage,
        )
        .unwrap();
        let disocclusion = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R8_UNORM,
            usage,
        )
        .unwrap();
        let reactive = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R8_UNORM,
            usage,
        )
        .unwrap();
        let depth = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R32_SFLOAT,
            usage,
        )
        .unwrap();
        let composition = Image::new(
            device,
            &gpu.memory,
            estimator_extent,
            vk::Format::R8_UNORM,
            usage,
        )
        .unwrap();
        let exposure = Image::new(
            device,
            &gpu.memory,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R32_SFLOAT,
            usage,
        )
        .unwrap();

        let source_bytes = vec![
            0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0,
            0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255,
            255, 255,
        ];
        let motion_bytes = [
            f32_to_f16(1.0).to_ne_bytes(),
            f32_to_f16(-2.0).to_ne_bytes(),
            f32_to_f16(1.0).to_ne_bytes(),
            f32_to_f16(-2.0).to_ne_bytes(),
            f32_to_f16(1.0).to_ne_bytes(),
            f32_to_f16(-2.0).to_ne_bytes(),
            f32_to_f16(1.0).to_ne_bytes(),
            f32_to_f16(-2.0).to_ne_bytes(),
        ]
        .concat();
        let confidence_bytes = vec![128u8; 4];
        let disocclusion_bytes = vec![64u8; 4];
        let reactive_bytes = vec![96u8; 4];
        let composition_bytes = vec![192u8; 4];
        let depth_bytes = [0.25f32, 0.75, 0.25, 0.75]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let exposure_bytes = 1.0f32.to_ne_bytes();
        let mut uploads = Vec::new();
        let mut upload = |image: &Image, bytes: &[u8], layout, extent| {
            uploads.push(upload_image_layout(&gpu, image, bytes, layout, extent));
        };
        upload(
            &source,
            &source_bytes,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            game_extent,
        );
        upload(
            &motion,
            &motion_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &confidence,
            &confidence_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &disocclusion,
            &disocclusion_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &reactive,
            &reactive_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &composition,
            &composition_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &depth,
            &depth_bytes,
            vk::ImageLayout::GENERAL,
            estimator_extent,
        );
        upload(
            &exposure,
            &exposure_bytes,
            vk::ImageLayout::GENERAL,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
        );

        let input_extent = FrameExtent {
            width: estimator_extent.width,
            height: estimator_extent.height,
        };
        let metadata = GuidanceMetadata {
            frame_id: 1,
            extent: input_extent,
            valid_region: ValidRegion::full(input_extent),
            reset: GuidanceReset::None,
            valid: true,
            is_zero: false,
            requires_history_reset: false,
        };
        let resource = |image: &Image, format: vk::Format| GuidanceResource {
            image: image.handle,
            view: image.view,
            format,
            metadata,
            state: SignalState::Estimated,
        };
        let raw = GuidanceView {
            motion: resource(&motion, vk::Format::R16G16_SFLOAT),
            confidence: resource(&confidence, vk::Format::R8_UNORM),
            disocclusion: resource(&disocclusion, vk::Format::R8_UNORM),
            reactive: resource(&reactive, vk::Format::R8_UNORM),
            exposure: resource(&exposure, vk::Format::R32_SFLOAT),
            depth: resource(&depth, vk::Format::R32_SFLOAT),
            transparency_composition: resource(&composition, vk::Format::R8_UNORM),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing: FrameTiming::default(),
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::RelativeNearIsOne,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(input_extent, input_extent),
            requires_history_reset: false,
        };
        let mut resolver = GuidanceResolver::new(
            device,
            &gpu.memory,
            game_extent,
            estimator_extent,
            source.view,
            raw,
        )
        .unwrap();
        gpu.submit(|command| resolver.record(command, true, raw.depth_semantics));
        let resolved = resolver.view(raw);
        let game_frame_extent = FrameExtent {
            width: game_extent.width,
            height: game_extent.height,
        };
        assert!(resolved.is_valid_for(1, game_frame_extent));
        assert_eq!(resolved.resolution.signal_extent, game_frame_extent);
        assert_eq!(resolved.resolution.estimator_extent, input_extent);

        let motion_output = read_resolved(
            &gpu,
            resolved.motion.image,
            game_extent,
            std::mem::size_of::<u16>() * 2,
        );
        let confidence_output = read_resolved(&gpu, resolved.confidence.image, game_extent, 1);
        let disocclusion_output = read_resolved(&gpu, resolved.disocclusion.image, game_extent, 1);
        let reactive_output = read_resolved(&gpu, resolved.reactive.image, game_extent, 1);
        let composition_output = read_resolved(
            &gpu,
            resolved.transparency_composition.image,
            game_extent,
            1,
        );
        let depth_output = read_resolved(
            &gpu,
            resolved.depth.image,
            game_extent,
            std::mem::size_of::<f32>(),
        );
        for &[x0, x1, y0, y1] in motion_output.as_slice().as_chunks::<4>().0 {
            assert!((f16_to_f32(u16::from_ne_bytes([x0, x1])) - 2.0).abs() < 0.01);
            assert!((f16_to_f32(u16::from_ne_bytes([y0, y1])) + 4.0).abs() < 0.01);
        }
        assert!(
            confidence_output
                .iter()
                .all(|value| (*value as i32 - 128).abs() <= 1)
        );
        assert!(
            disocclusion_output
                .iter()
                .all(|value| (*value as i32 - 64).abs() <= 1)
        );
        assert!(
            reactive_output
                .iter()
                .all(|value| (*value as i32 - 96).abs() <= 1)
        );
        assert!(
            composition_output
                .iter()
                .all(|value| (*value as i32 - 192).abs() <= 1)
        );
        let depth_values = depth_output
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_ne_bytes(*value))
            .collect::<Vec<_>>();
        assert!(
            depth_values
                .as_slice()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|row| row
                    .as_slice()
                    .windows(2)
                    .all(|pair| pair[0] <= pair[1] + 0.001))
        );
        assert!(depth_values[0] < depth_values[3]);
    }
}

#[test]
fn guidance_shader_contains_relative_depth_and_gradient_rejection_producers() {
    let guidance = include_str!("../../../shaders/temporal/guidance.comp");
    for term in ["DepthPartials", "normal_equations"] {
        assert!(guidance.contains(term), "guidance shader is missing {term}");
    }
    let depth = include_str!("../../../shaders/temporal/depth_reduce.comp");
    for term in [
        "solve_affine",
        "affine_inliers",
        "global_motion",
        "0.25",
        "0.50",
        "percentile_5",
        "percentile_95",
        "relative_parallax",
        "flow_jacobian_trace",
        "FlatFallback",
        "RelativeNearIsOne",
    ] {
        assert!(depth.contains(term), "depth shader is missing {term}");
    }
    let reconstruct = include_str!("../../../shaders/upscaler/reconstruct.comp");
    assert!(reconstruct.contains("depth_discontinuity"));
    assert!(!reconstruct.contains("* clamp(depth, 0.0, 1.0)"));
    let runtime = include_str!("../../../crates/runtime/src/present.rs");
    assert!(runtime.contains("guidance.view("));
}

#[test]
fn production_depth_status_stays_gpu_resident() {
    let gpu = include_str!("../src/gpu.rs");
    let runtime = include_str!("../../runtime/src/present.rs");
    assert!(!gpu.contains("MemoryPropertyFlags::HOST_VISIBLE"));
    assert!(!gpu.contains("map_memory"));
    assert!(!gpu.contains("refresh_depth_status"));
    assert!(!runtime.contains("refresh_depth_status"));
}

#[test]
fn production_depth_status_is_slot_scoped() {
    let gpu = include_str!("../src/gpu.rs");
    let runtime = include_str!("../../runtime/src/present.rs");
    assert!(gpu.contains("depth_models: Vec<Buffer>"));
    assert!(gpu.contains("depth_descriptor_sets: Vec<vk::DescriptorSet>"));
    assert!(runtime.contains("record_with_timing_for_slot"));
    assert!(runtime.contains("record_provider_failure_for_slot"));
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

unsafe fn upload_image_layout(
    gpu: &Gpu,
    image: &Image,
    bytes: &[u8],
    layout: vk::ImageLayout,
    extent: vk::Extent2D,
) -> Buffer {
    let device = &gpu.device;
    let staging = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    unsafe { staging.write(bytes) }.unwrap();
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
                staging.handle,
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
                layout,
            );
        });
    }
    staging
}

unsafe fn read_resolved(
    gpu: &Gpu,
    image: vk::Image,
    extent: vk::Extent2D,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let device = &gpu.device;
    let readback = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            (extent.width * extent.height) as u64 * bytes_per_pixel as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                image,
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
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        });
    }
    let mut bytes = vec![0u8; readback.size as usize];
    unsafe { readback.read(&mut bytes) }.unwrap();
    bytes
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa = mantissa;
                let mut exponent = -14i32;
                while mantissa & 0x400 == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        exponent => sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
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

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ff_ff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let shifted = (mantissa | 0x80_00_00) >> (1 - exponent);
        return sign | ((shifted + 0x1000) >> 13) as u16;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exponent as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
}

unsafe fn clear_motion_field(gpu: &Gpu, motion: &Image, values: &[[f32; 2]], extent: vk::Extent2D) {
    unsafe {
        let device = &gpu.device;
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&f32_to_f16(value[0]).to_ne_bytes());
            bytes.extend_from_slice(&f32_to_f16(value[1]).to_ne_bytes());
        }
        let upload = Buffer::new(
            device,
            &gpu.memory,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        upload.write(&bytes).unwrap();
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                motion.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload.handle,
                motion.handle,
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
    let (reactive, disocclusion, transparency, _depth, exposure, states, _, _, reset) =
        unsafe { guidance_masks_with_options(extent, previous_pixels, current_pixels, None, true) };
    (
        reactive,
        disocclusion,
        transparency,
        exposure,
        states,
        reset,
    )
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
            for pixel in frame.as_chunks_mut::<4>().0 {
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
) -> GuidanceOutputs {
    unsafe {
        guidance_masks_with_field_options(
            extent,
            previous_pixels,
            current_pixels,
            forced_motion,
            None,
            provider_failure,
        )
    }
}

unsafe fn guidance_masks_with_field_options(
    extent: vk::Extent2D,
    previous_pixels: &[u8],
    current_pixels: &[u8],
    forced_motion: Option<[f32; 4]>,
    forced_motion_field: Option<&[[f32; 2]]>,
    provider_failure: bool,
) -> GuidanceOutputs {
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
        if let Some(forced_motion_field) = forced_motion_field {
            clear_motion_field(&gpu, &motion.vectors, forced_motion_field, extent);
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
        let depth_offset = exposure_offset + 4;
        let readback = Buffer::new(
            device,
            &gpu.memory,
            depth_offset + count * 4,
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
            image_barrier(
                device,
                command,
                guidance.depth.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                guidance.depth.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.handle,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(depth_offset)
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
            memory_barrier(device, command);
        });
        let mut bytes = vec![0u8; readback.size as usize];
        readback.read(&mut bytes).unwrap();
        let exposure = f32::from_ne_bytes(
            bytes[exposure_offset as usize..exposure_offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let depth = (0..count as usize)
            .map(|index| {
                let offset = depth_offset as usize + index * 4;
                f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
            })
            .collect::<Vec<_>>();
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
            depth,
            exposure,
            [
                view.reactive.state,
                view.disocclusion.state,
                view.exposure.state,
                view.transparency_composition.state,
            ],
            view.depth.state,
            view.depth_semantics,
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
    for pixel in current.as_chunks_mut::<4>().0 {
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
    assert!(reactive.iter().all(|value| *value == 255));
    assert!(disocclusion.iter().all(|value| *value == 255));
    assert!(transparency.iter().all(|value| *value == 255));
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

fn layered_parallax_flow_fixture(
    extent: vk::Extent2D,
) -> (Vec<u8>, Vec<u8>, Vec<[f32; 2]>, Vec<f32>) {
    let count = (extent.width * extent.height) as usize;
    let mut previous = vec![0u8; count * 4];
    for y in 0..extent.height {
        for x in 0..extent.width {
            let value = flow_sample(x as f32 / 3.0, y as f32 / 3.0);
            let index = (y * extent.width + x) as usize * 4;
            previous[index..index + 4].copy_from_slice(&[value, value, value, 255]);
        }
    }
    let mut current = previous.clone();
    let mut motion = vec![[0.0, 0.0]; count];
    let mut expected = vec![0.35; count];
    for y in 0..extent.height {
        for x in 0..extent.width {
            let foreground = (extent.width / 4..extent.width * 3 / 4).contains(&x)
                && (extent.height / 4..extent.height * 3 / 4).contains(&y);
            let displacement = if foreground {
                [-7.0, -3.0]
            } else {
                [-2.0, -1.0]
            };
            let index = (y * extent.width + x) as usize;
            motion[index] = displacement;
            expected[index] = if foreground { 1.0 } else { 0.35 };
            let source_x = (x as i32 + displacement[0] as i32)
                .clamp(0, extent.width.saturating_sub(1) as i32) as u32;
            let source_y = (y as i32 + displacement[1] as i32)
                .clamp(0, extent.height.saturating_sub(1) as i32) as u32;
            let source = (source_y * extent.width + source_x) as usize * 4;
            current[index * 4..index * 4 + 4].copy_from_slice(&previous[source..source + 4]);
        }
    }
    (previous, current, motion, expected)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn relative_depth_orders_independent_parallax_planes() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (previous, current, motion, expected) = layered_parallax_flow_fixture(extent);
    let (_, _, _, depth, _, _, depth_state, depth_semantics, _) = unsafe {
        guidance_masks_with_field_options(extent, &previous, &current, None, Some(&motion), false)
    };
    assert!(depth.iter().all(|value| value.is_finite()));
    // A recorded dispatch is an estimated resource even when the GPU chooses
    // the exact flat-one conservative output for this scene.
    assert_eq!(depth_state, SignalState::Estimated);
    assert_eq!(depth_semantics, DepthSemantics::RelativeNearIsOne);
    let score = tuxscaling_temporal::quality::depth_order(&depth, &expected);
    let foreground = (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| (x, y)))
        .filter(|(x, y)| {
            (extent.width / 4..extent.width * 3 / 4).contains(x)
                && (extent.height / 4..extent.height * 3 / 4).contains(y)
        })
        .map(|(x, y)| depth[(y * extent.width + x) as usize])
        .collect::<Vec<_>>();
    let background = depth
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let x = *index as u32 % extent.width;
            let y = *index as u32 / extent.width;
            !(extent.width / 4..extent.width * 3 / 4).contains(&x)
                || !(extent.height / 4..extent.height * 3 / 4).contains(&y)
        })
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let foreground_mean = foreground.iter().sum::<f32>() / foreground.len() as f32;
    let background_mean = background.iter().sum::<f32>() / background.len() as f32;
    eprintln!(
        "relative depth parallax: order={score:.3} foreground={foreground_mean:.3} background={background_mean:.3}"
    );
    assert!(score >= 0.85);
    assert!(foreground_mean > background_mean + 0.10);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn valid_depth_dispatch_advertises_estimated_relative_semantics() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (previous, current, motion, _) = layered_parallax_flow_fixture(extent);
    let (_, _, _, depth, _, _, depth_state, depth_semantics, _) = unsafe {
        guidance_masks_with_field_options(extent, &previous, &current, None, Some(&motion), false)
    };
    assert!(depth.iter().all(|value| value.is_finite()));
    assert_eq!(depth_state, SignalState::Estimated);
    assert_eq!(depth_semantics, DepthSemantics::RelativeNearIsOne);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn relative_depth_uses_flat_fallback_below_global_motion_threshold() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (previous, current, _) = translated_flow_fixture(extent, 5, 0);
    let (_, _, _, depth, _, states, depth_state, depth_semantics, _) = unsafe {
        guidance_masks_with_options(
            extent,
            &previous,
            &current,
            Some([0.0, 0.0, 0.0, 0.0]),
            false,
        )
    };
    assert!(depth.iter().all(|value| *value == 1.0));
    assert!(depth.iter().all(|value| value.is_finite()));
    assert_eq!(states[0], SignalState::Estimated);
    assert_eq!(depth_state, SignalState::Estimated);
    assert_eq!(depth_semantics, DepthSemantics::RelativeNearIsOne);
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn relative_depth_uses_flat_fallback_below_affine_inlier_threshold() {
    let extent = vk::Extent2D {
        width: 64,
        height: 48,
    };
    let (previous, current, _) = translated_flow_fixture(extent, 0, 0);
    let motion = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| {
                if (x + y) % 2 == 0 {
                    [-1.0, 0.0]
                } else {
                    [-8.0, 0.0]
                }
            })
        })
        .collect::<Vec<_>>();
    let (_, _, _, depth, _, _, depth_state, depth_semantics, _) = unsafe {
        guidance_masks_with_field_options(extent, &previous, &current, None, Some(&motion), false)
    };
    assert!(depth.iter().all(|value| *value == 1.0));
    assert!(depth.iter().all(|value| value.is_finite()));
    assert_eq!(depth_state, SignalState::Estimated);
    assert_eq!(depth_semantics, DepthSemantics::RelativeNearIsOne);
}
