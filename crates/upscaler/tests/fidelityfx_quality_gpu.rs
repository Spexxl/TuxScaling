#![cfg(feature = "fidelityfx")]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use ash::vk;
use std::time::Duration;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResource, GuidanceScalar, GuidanceView, JitterSample, MotionDirection, MotionUnits,
    SignalState,
};
use tuxscaling_upscaler::fidelityfx::Fsr314Upscaler;
use tuxscaling_upscaler::{
    BackendColorEncoding, BackendConfig, BackendEnvironment, BackendFrame, BackendImage,
    UpscalerBackend, content_viewport,
};
use tuxscaling_vulkan::{Buffer, Image, color_range, image_barrier, memory_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

const INPUT: vk::Extent2D = vk::Extent2D {
    width: 32,
    height: 32,
};
const OUTPUT: vk::Extent2D = vk::Extent2D {
    width: 64,
    height: 64,
};

fn resource(image: &Image, format: vk::Format, metadata: GuidanceMetadata) -> GuidanceResource {
    GuidanceResource {
        image: image.handle,
        view: image.view,
        format,
        metadata,
        state: SignalState::Estimated,
    }
}

fn guidance(
    motion: &Image,
    confidence: &Image,
    disocclusion: &Image,
    reactive: &Image,
    depth: &Image,
    transparency: &Image,
    exposure: &Image,
    frame_id: u64,
) -> GuidanceView {
    let extent = FrameExtent {
        width: INPUT.width,
        height: INPUT.height,
    };
    let metadata = GuidanceMetadata::zero(frame_id, extent, GuidanceReset::None);
    GuidanceView {
        motion: resource(motion, vk::Format::R16G16_SFLOAT, metadata),
        confidence: resource(confidence, vk::Format::R8_UNORM, metadata),
        disocclusion: resource(disocclusion, vk::Format::R8_UNORM, metadata),
        reactive: resource(reactive, vk::Format::R8_UNORM, metadata),
        exposure: resource(exposure, vk::Format::R32_SFLOAT, metadata),
        depth: resource(depth, vk::Format::R32_SFLOAT, metadata),
        transparency_composition: resource(transparency, vk::Format::R8_UNORM, metadata),
        pre_exposure: GuidanceScalar::constant_fallback(1.0),
        timing: FrameTiming {
            raw: Duration::from_micros(16_667),
            validated: Duration::from_micros(16_667),
            smoothed: Duration::from_micros(16_667),
        },
        jitter: JitterSample::default(),
        depth_semantics: DepthSemantics::FlatFallback,
        direction: MotionDirection::CurrentToPrevious,
        units: MotionUnits::SourcePixels,
        resolution: GuidanceResolution::new(extent, extent),
        requires_history_reset: false,
    }
}

fn scene_pixel(x: u32, y: u32, extent: vk::Extent2D) -> [f32; 4] {
    let tile_x = x * 8 / extent.width.max(1);
    let tile_y = y * 8 / extent.height.max(1);
    let checker = (tile_x + tile_y) % 2;
    let edge = if (x * 17 / extent.width.max(1) + y * 13 / extent.height.max(1)).is_multiple_of(5) {
        1.0
    } else {
        0.0
    };
    [
        if checker == 0 { 0.1 + edge * 0.8 } else { 0.9 },
        if checker == 0 { 0.8 } else { 0.15 + edge * 0.7 },
        if checker == 0 { 0.2 + edge * 0.7 } else { 0.95 },
        1.0,
    ]
}

fn rgba8(extent: vk::Extent2D) -> Vec<u8> {
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                scene_pixel(x, y, extent)
                    .map(|value| (value * 255.0).round() as u8)
                    .into_iter()
            })
        })
        .collect()
}

fn expected_high_resolution(extent: vk::Extent2D) -> Vec<[f32; 4]> {
    (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| scene_pixel(x, y, extent)))
        .collect()
}

fn bilinear(
    source: &[u8],
    source_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
) -> Vec<[f32; 4]> {
    let sample = |x: u32, y: u32| {
        let index = ((y * source_extent.width + x) * 4) as usize;
        [
            source[index] as f32 / 255.0,
            source[index + 1] as f32 / 255.0,
            source[index + 2] as f32 / 255.0,
            source[index + 3] as f32 / 255.0,
        ]
    };
    (0..output_extent.height)
        .flat_map(|y| {
            (0..output_extent.width).map(move |x| {
                let source_x =
                    x as f32 * (source_extent.width - 1) as f32 / (output_extent.width - 1) as f32;
                let source_y = y as f32 * (source_extent.height - 1) as f32
                    / (output_extent.height - 1) as f32;
                let x0 = source_x.floor() as u32;
                let y0 = source_y.floor() as u32;
                let x1 = (x0 + 1).min(source_extent.width - 1);
                let y1 = (y0 + 1).min(source_extent.height - 1);
                let tx = source_x - x0 as f32;
                let ty = source_y - y0 as f32;
                let top: [f32; 4] = std::array::from_fn(|channel| {
                    sample(x0, y0)[channel].mul_add(1.0 - tx, sample(x1, y0)[channel] * tx)
                });
                let bottom: [f32; 4] = std::array::from_fn(|channel| {
                    sample(x0, y1)[channel].mul_add(1.0 - tx, sample(x1, y1)[channel] * tx)
                });
                std::array::from_fn(|channel| top[channel].mul_add(1.0 - ty, bottom[channel] * ty))
            })
        })
        .collect()
}

fn image_mse(actual: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .flat_map(|(actual, expected)| actual.iter().zip(expected))
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>()
        / (actual.len() * 4) as f32
}

fn psnr(actual: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
    let mse = image_mse(actual, expected).max(f32::MIN_POSITIVE);
    10.0 * (1.0 / mse).log10()
}

fn ssim(actual: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
    let luminance = |pixel: &[f32; 4]| 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
    let actual = actual.iter().map(luminance).collect::<Vec<_>>();
    let expected = expected.iter().map(luminance).collect::<Vec<_>>();
    let mean_actual = actual.iter().sum::<f32>() / actual.len() as f32;
    let mean_expected = expected.iter().sum::<f32>() / expected.len() as f32;
    let variance_actual = actual
        .iter()
        .map(|value| (value - mean_actual).powi(2))
        .sum::<f32>()
        / actual.len() as f32;
    let variance_expected = expected
        .iter()
        .map(|value| (value - mean_expected).powi(2))
        .sum::<f32>()
        / expected.len() as f32;
    let covariance = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - mean_actual) * (expected - mean_expected))
        .sum::<f32>()
        / actual.len() as f32;
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);
    ((2.0 * mean_actual * mean_expected + c1) * (2.0 * covariance + c2))
        / ((mean_actual.powi(2) + mean_expected.powi(2) + c1)
            * (variance_actual + variance_expected + c2))
}

fn run_frame(
    gpu: &Gpu,
    output: &Image,
    readback_buffer: &Buffer,
    bytes: &mut [u8],
    backend: &mut Fsr314Upscaler,
    frame: BackendFrame,
    guidance_images: &[&Image],
    exposure: &Image,
    source: &Image,
    staging: &Buffer,
    source_bytes: &[u8],
    first: bool,
) {
    unsafe {
        gpu.submit(|command| {
            if first {
                staging.write(source_bytes).unwrap();
                image_barrier(
                    &gpu.device,
                    command,
                    source.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                gpu.device.cmd_copy_buffer_to_image(
                    command,
                    staging.handle,
                    source.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: INPUT.width,
                            height: INPUT.height,
                            depth: 1,
                        })],
                );
                image_barrier(
                    &gpu.device,
                    command,
                    source.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                image_barrier(
                    &gpu.device,
                    command,
                    output.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
                for image in guidance_images
                    .iter()
                    .copied()
                    .chain(std::iter::once(exposure))
                {
                    image_barrier(
                        &gpu.device,
                        command,
                        image.handle,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    );
                    let clear = if image.format == vk::Format::R32_SFLOAT {
                        vk::ClearColorValue {
                            float32: [1.0, 1.0, 1.0, 1.0],
                        }
                    } else {
                        vk::ClearColorValue { float32: [0.0; 4] }
                    };
                    gpu.device.cmd_clear_color_image(
                        command,
                        image.handle,
                        vk::ImageLayout::GENERAL,
                        &clear,
                        &[color_range()],
                    );
                }
                memory_barrier(&gpu.device, command);
            }
            let mut frame = frame;
            frame.command_buffer = command;
            backend.record(frame).unwrap();
            image_barrier(
                &gpu.device,
                command,
                output.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            gpu.device.cmd_copy_image_to_buffer(
                command,
                output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buffer.handle,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: OUTPUT.width,
                        height: OUTPUT.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                &gpu.device,
                command,
                output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        });
        readback_buffer.read(bytes).unwrap();
    }
}

fn rgba8_pixels(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            [
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                pixel[3] as f32 / 255.0,
            ]
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU with the FidelityFX storage-image formats"]
fn fsr314_upscale_beats_bilinear_on_deterministic_static_fixture() {
    let gpu = unsafe { Gpu::new() };
    let physical = unsafe { gpu.instance.enumerate_physical_devices() }.unwrap()[0];
    let environment = BackendEnvironment::new(&gpu.instance, physical, &gpu.device);
    let memory = gpu.memory;
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let guidance_usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::TRANSFER_DST;
    let source = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )
    }
    .unwrap();
    let output = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            OUTPUT,
            vk::Format::R8G8B8A8_UNORM,
            usage,
        )
    }
    .unwrap();
    let motion = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R16G16_SFLOAT,
            guidance_usage,
        )
    }
    .unwrap();
    let confidence = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R8_UNORM,
            guidance_usage,
        )
    }
    .unwrap();
    let disocclusion = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R8_UNORM,
            guidance_usage,
        )
    }
    .unwrap();
    let reactive = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R8_UNORM,
            guidance_usage,
        )
    }
    .unwrap();
    let depth = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R32_SFLOAT,
            guidance_usage,
        )
    }
    .unwrap();
    let transparency = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            INPUT,
            vk::Format::R8_UNORM,
            guidance_usage,
        )
    }
    .unwrap();
    let exposure = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R32_SFLOAT,
            guidance_usage,
        )
    }
    .unwrap();
    let guidance_images = [
        &motion,
        &confidence,
        &disocclusion,
        &reactive,
        &depth,
        &transparency,
    ];
    let guidance = guidance(
        &motion,
        &confidence,
        &disocclusion,
        &reactive,
        &depth,
        &transparency,
        &exposure,
        1,
    );
    let config = BackendConfig {
        game_extent: INPUT,
        output_extent: OUTPUT,
        source_format: vk::Format::R8G8B8A8_UNORM,
        output_format: vk::Format::R8G8B8A8_UNORM,
        color_encoding: BackendColorEncoding::SrgbNonlinear,
        viewport: content_viewport(INPUT, OUTPUT),
        guidance: guidance.capabilities(),
    };
    let mut backend = unsafe { Fsr314Upscaler::new(&environment, config, guidance, 1) }.unwrap();
    let source_bytes = rgba8(INPUT);
    let expected = expected_high_resolution(OUTPUT);
    let bilinear = bilinear(&source_bytes, INPUT, OUTPUT);
    let staging = unsafe {
        Buffer::new(
            &gpu.device,
            &memory,
            source_bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let readback_buffer = unsafe {
        Buffer::new(
            &gpu.device,
            &memory,
            (OUTPUT.width * OUTPUT.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let mut first_bytes = vec![0_u8; OUTPUT.width as usize * OUTPUT.height as usize * 4];
    let frame = BackendFrame {
        command_buffer: vk::CommandBuffer::null(),
        slot: 0,
        source: BackendImage {
            image: source.handle,
            view: source.view,
            format: source.format,
            extent: INPUT,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        output: BackendImage {
            image: output.handle,
            view: output.view,
            format: output.format,
            extent: OUTPUT,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        guidance,
        viewport: config.viewport,
        frame_id: 1,
        reset_history: true,
        debug_view: 0,
    };
    run_frame(
        &gpu,
        &output,
        &readback_buffer,
        &mut first_bytes,
        &mut backend,
        frame,
        &guidance_images,
        &exposure,
        &source,
        &staging,
        &source_bytes,
        true,
    );
    let mut second_bytes = vec![0_u8; first_bytes.len()];
    let mut second_frame = frame;
    second_frame.reset_history = false;
    run_frame(
        &gpu,
        &output,
        &readback_buffer,
        &mut second_bytes,
        &mut backend,
        second_frame,
        &guidance_images,
        &exposure,
        &source,
        &staging,
        &source_bytes,
        false,
    );
    let first = rgba8_pixels(&first_bytes);
    let second = rgba8_pixels(&second_bytes);
    let fsr_psnr = psnr(&first, &expected);
    let bilinear_psnr = psnr(&bilinear, &expected);
    let fsr_ssim = ssim(&first, &expected);
    let bilinear_ssim = ssim(&bilinear, &expected);
    let flicker = image_mse(&first, &second);
    eprintln!(
        "FSR quality fixture: psnr={fsr_psnr:.3}dB bilinear={bilinear_psnr:.3}dB ssim={fsr_ssim:.5} bilinear_ssim={bilinear_ssim:.5} flicker_mse={flicker:.6}"
    );
    assert!(
        first
            .iter()
            .any(|pixel| pixel[..3].iter().any(|value| *value > 0.01))
    );
    assert!(fsr_psnr > bilinear_psnr);
    assert!(fsr_ssim > bilinear_ssim);
    assert!(flicker <= 0.002);
}
