#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResource, GuidanceScalar, GuidanceView, JitterSample, MotionDirection, MotionUnits,
    SignalState, ValidRegion,
};
use tuxscaling_upscaler::ReferenceUpscaler;
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

fn scene_value(x: f32, y: f32, changed: bool) -> [f32; 3] {
    let mut value = [
        0.5 + 0.22 * (x * 17.0 + y * 3.0).sin(),
        0.5 + 0.19 * (x * 5.0 - y * 13.0).cos(),
        0.5 + 0.16 * (x * 11.0 + y * 7.0).sin(),
    ];
    if changed && (0.25..0.75).contains(&x) && (0.25..0.75).contains(&y) {
        value[0] = (value[0] + 0.32).min(1.0);
        value[1] = (value[1] + 0.18).min(1.0);
    }
    value
}

fn frame(extent: vk::Extent2D, changed: bool) -> Vec<u8> {
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                let value = scene_value(
                    (x as f32 + 0.5) / extent.width as f32,
                    (y as f32 + 0.5) / extent.height as f32,
                    changed,
                );
                [
                    (value[0] * 255.0).round() as u8,
                    (value[1] * 255.0).round() as u8,
                    (value[2] * 255.0).round() as u8,
                    255,
                ]
            })
        })
        .collect()
}

unsafe fn upload_image(
    gpu: &Gpu,
    image: &Image,
    staging: &Buffer,
    bytes: &[u8],
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    extent: vk::Extent2D,
) {
    unsafe { staging.write(bytes) }.unwrap();
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                image.handle,
                old_layout,
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
                new_layout,
            );
        });
    }
}

unsafe fn clear_image(gpu: &Gpu, image: &Image, old_layout: vk::ImageLayout, value: [f32; 4]) {
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                image.handle,
                old_layout,
                vk::ImageLayout::GENERAL,
            );
            device.cmd_clear_color_image(
                command,
                image.handle,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: value },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)],
            );
            memory_barrier(device, command);
        });
    }
}

fn guidance_resource(image: &Image, format: vk::Format, state: SignalState) -> GuidanceResource {
    GuidanceResource {
        image: image.handle,
        view: image.view,
        format,
        metadata: GuidanceMetadata {
            frame_id: 1,
            extent: FrameExtent {
                width: image.extent.width,
                height: image.extent.height,
            },
            valid: true,
            valid_region: ValidRegion::full(FrameExtent {
                width: image.extent.width,
                height: image.extent.height,
            }),
            reset: GuidanceReset::None,
            is_zero: false,
            requires_history_reset: false,
        },
        state,
    }
}

#[allow(clippy::too_many_arguments)]
fn guidance(
    motion: &Image,
    confidence: &Image,
    reactive: &Image,
    disocclusion: &Image,
    exposure: &Image,
    depth: &Image,
    composition: &Image,
    requires_history_reset: bool,
) -> GuidanceView {
    GuidanceView {
        motion: guidance_resource(motion, vk::Format::R16G16_SFLOAT, SignalState::Estimated),
        confidence: guidance_resource(confidence, vk::Format::R8_UNORM, SignalState::Estimated),
        reactive: guidance_resource(reactive, vk::Format::R8_UNORM, SignalState::Estimated),
        disocclusion: guidance_resource(disocclusion, vk::Format::R8_UNORM, SignalState::Estimated),
        exposure: guidance_resource(exposure, vk::Format::R32_SFLOAT, SignalState::Estimated),
        depth: guidance_resource(depth, vk::Format::R32_SFLOAT, SignalState::Estimated),
        transparency_composition: guidance_resource(
            composition,
            vk::Format::R8_UNORM,
            SignalState::ConstantFallback,
        ),
        pre_exposure: GuidanceScalar::constant_fallback(1.0),
        timing: FrameTiming::default(),
        jitter: JitterSample::default(),
        depth_semantics: DepthSemantics::RelativeNearIsOne,
        direction: MotionDirection::CurrentToPrevious,
        units: MotionUnits::SourcePixels,
        resolution: GuidanceResolution::new(
            FrameExtent {
                width: 64,
                height: 48,
            },
            FrameExtent {
                width: 64,
                height: 48,
            },
        ),
        requires_history_reset,
    }
}

unsafe fn read_output(gpu: &Gpu, upscaler: &ReferenceUpscaler, readback: &Buffer) -> Vec<u8> {
    let device = &gpu.device;
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                upscaler.output.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                upscaler.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.handle,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: upscaler.output_extent.width,
                        height: upscaler.output_extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                device,
                command,
                upscaler.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            memory_barrier(device, command);
        });
    }
    let mut bytes = vec![0u8; readback.size as usize];
    unsafe { readback.read(&mut bytes) }.unwrap();
    bytes
}

fn region_error(bytes: &[u8], extent: vk::Extent2D) -> f32 {
    let mut error = 0.0;
    let mut samples = 0;
    for y in 0..extent.height {
        for x in 0..extent.width {
            let uv = [
                (x as f32 + 0.5) / extent.width as f32,
                (y as f32 + 0.5) / extent.height as f32,
            ];
            if !(0.25..0.75).contains(&uv[0]) || !(0.25..0.75).contains(&uv[1]) {
                continue;
            }
            let expected = scene_value(uv[0], uv[1], true);
            let offset = (y * extent.width + x) as usize * 4;
            for channel in 0..3 {
                let actual = bytes[offset + channel] as f32 / 255.0;
                error += (actual - expected[channel]).powi(2);
            }
            samples += 3;
        }
    }
    error / samples as f32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn disoccluded_reconstruction_does_not_exceed_bilinear_baseline() {
    let input_extent = vk::Extent2D {
        width: 16,
        height: 16,
    };
    let output_extent = vk::Extent2D {
        width: 32,
        height: 32,
    };
    let gpu = unsafe { Gpu::new() };
    let device = &gpu.device;
    let image_usage = vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE;
    let swapchain_usage = image_usage | vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let input = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R8G8B8A8_UNORM,
            image_usage,
        )
    }
    .unwrap();
    let swapchain = unsafe {
        Image::new(
            device,
            &gpu.memory,
            output_extent,
            vk::Format::R8G8B8A8_UNORM,
            swapchain_usage,
        )
    }
    .unwrap();
    let motion = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R16G16_SFLOAT,
            image_usage,
        )
    }
    .unwrap();
    let confidence = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R8_UNORM,
            image_usage,
        )
    }
    .unwrap();
    let reactive = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R8_UNORM,
            image_usage,
        )
    }
    .unwrap();
    let disocclusion = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R8_UNORM,
            image_usage,
        )
    }
    .unwrap();
    let exposure = unsafe {
        Image::new(
            device,
            &gpu.memory,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R32_SFLOAT,
            image_usage,
        )
    }
    .unwrap();
    let depth = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R32_SFLOAT,
            image_usage,
        )
    }
    .unwrap();
    let composition = unsafe {
        Image::new(
            device,
            &gpu.memory,
            input_extent,
            vk::Format::R8_UNORM,
            image_usage,
        )
    }
    .unwrap();
    let staging = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            (output_extent.width * output_extent.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let readback = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            (output_extent.width * output_extent.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();

    unsafe {
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, false),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        clear_image(
            &gpu,
            &swapchain,
            vk::ImageLayout::UNDEFINED,
            [0.0, 0.0, 0.0, 1.0],
        );
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                swapchain.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        });
        for (image, value) in [
            (&motion, [0.0, 0.0, 0.0, 0.0]),
            (&confidence, [1.0, 0.0, 0.0, 0.0]),
            (&reactive, [0.0, 0.0, 0.0, 0.0]),
            (&disocclusion, [1.0, 0.0, 0.0, 0.0]),
            (&exposure, [1.0, 0.0, 0.0, 0.0]),
            (&depth, [1.0, 0.0, 0.0, 0.0]),
            (&composition, [0.0, 0.0, 0.0, 0.0]),
        ] {
            clear_image(&gpu, image, vk::ImageLayout::UNDEFINED, value);
        }
    }
    let mut upscaler = unsafe {
        ReferenceUpscaler::new(
            device,
            &gpu.memory,
            input.view,
            input_extent,
            output_extent,
            vk::Format::R8G8B8A8_UNORM,
            guidance(
                &motion,
                &confidence,
                &reactive,
                &disocclusion,
                &exposure,
                &depth,
                &composition,
                true,
            ),
            1,
        )
    }
    .unwrap();
    let mut first_guidance = guidance(
        &motion,
        &confidence,
        &reactive,
        &disocclusion,
        &exposure,
        &depth,
        &composition,
        true,
    );
    let mut second_guidance = first_guidance;
    second_guidance.requires_history_reset = false;

    unsafe {
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, first_guidance, false, 0, 0, 0)
        });
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, true),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        clear_image(
            &gpu,
            &disocclusion,
            vk::ImageLayout::GENERAL,
            [1.0, 0.0, 0.0, 0.0],
        );
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, second_guidance, true, 0, 0, 0)
        });
    }
    let bilinear = unsafe { read_output(&gpu, &upscaler, &readback) };
    let bilinear_error = region_error(&bilinear, output_extent);

    upscaler.reset();
    unsafe {
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, false),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        let depth_bands = (0..input_extent.height)
            .flat_map(|_| {
                (0..input_extent.width).flat_map(|x| {
                    let value = if x / 4 % 2 == 0 { 0.2f32 } else { 0.8f32 };
                    value.to_ne_bytes()
                })
            })
            .collect::<Vec<_>>();
        upload_image(
            &gpu,
            &depth,
            &staging,
            &depth_bands,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
            input_extent,
        );
        clear_image(
            &gpu,
            &disocclusion,
            vk::ImageLayout::GENERAL,
            [0.0, 0.0, 0.0, 0.0],
        );
        first_guidance.depth_semantics = DepthSemantics::RelativeNearIsOne;
        second_guidance.depth_semantics = DepthSemantics::RelativeNearIsOne;
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, first_guidance, false, 0, 0, 0)
        });
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, true),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, second_guidance, true, 0, 0, 0)
        });
    }
    let reconstructed = unsafe { read_output(&gpu, &upscaler, &readback) };
    let reconstructed_error = region_error(&reconstructed, output_extent);
    eprintln!(
        "disoccluded reconstruction error: reconstructed={reconstructed_error:.6} bilinear={bilinear_error:.6}"
    );
    assert!(reconstructed_error <= bilinear_error + 0.001);

    upscaler.reset();
    first_guidance.jitter = JitterSample::default();
    first_guidance.depth_semantics = DepthSemantics::FlatFallback;
    second_guidance.jitter = JitterSample {
        current: [0.25, 0.0],
        previous: [0.0, 0.0],
        phase: 1,
    };
    second_guidance.depth_semantics = DepthSemantics::FlatFallback;
    unsafe {
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, false),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, first_guidance, false, 0, 0, 0)
        });
        upload_image(
            &gpu,
            &input,
            &staging,
            &frame(input_extent, true),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            input_extent,
        );
        gpu.submit(|command| {
            upscaler.record(command, swapchain.handle, second_guidance, true, 0, 0, 0)
        });
    }
    let jittered = unsafe { read_output(&gpu, &upscaler, &readback) };
    let different_pixels = reconstructed
        .as_chunks::<4>()
        .0
        .iter()
        .zip(jittered.as_chunks::<4>().0.iter())
        .filter(|(baseline, jittered)| baseline != jittered)
        .count();
    assert!(different_pixels > 0);
}
