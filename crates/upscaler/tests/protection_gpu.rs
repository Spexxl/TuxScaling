#![allow(clippy::missing_safety_doc)]

use ash::vk;
use std::time::Duration;
use tuxscaling_config::{ProtectionMode, ProtectionRegion};
use tuxscaling_upscaler::protection::{
    ProtectionFrame, ProtectionRenderer, ProtectionSettings, record_spatial_blit,
};
use tuxscaling_upscaler::{BackendImage, ProtectionInputs, content_viewport};
use tuxscaling_vulkan::{Buffer, Image, color_range, image_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

const SOURCE: vk::Extent2D = vk::Extent2D {
    width: 16,
    height: 16,
};
const OUTPUT: vk::Extent2D = vk::Extent2D {
    width: 40,
    height: 32,
};

unsafe fn clear(gpu: &Gpu, image: &Image, value: [f32; 4], final_layout: vk::ImageLayout) {
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            gpu.device.cmd_clear_color_image(
                command,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: value },
                &[color_range()],
            );
            image_barrier(
                &gpu.device,
                command,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                final_layout,
            );
        });
    }
}

fn backend_image(image: &Image, layout: vk::ImageLayout) -> BackendImage {
    BackendImage {
        image: image.handle,
        view: image.view,
        format: image.format,
        extent: image.extent,
        layout,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn protected_pixels_match_off_and_unprotected_pixels_keep_raw_fsr() {
    unsafe {
        let gpu = Gpu::new();
        let source = Image::new(
            &gpu.device,
            &gpu.memory,
            SOURCE,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .unwrap();
        let output = Image::new(
            &gpu.device,
            &gpu.memory,
            OUTPUT,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .unwrap();
        let spatial_reference = Image::new(
            &gpu.device,
            &gpu.memory,
            OUTPUT,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .unwrap();
        let risk = Image::new(
            &gpu.device,
            &gpu.memory,
            SOURCE,
            vk::Format::R8_UNORM,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .unwrap();
        let motion = Image::new(
            &gpu.device,
            &gpu.memory,
            SOURCE,
            vk::Format::R16G16_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .unwrap();
        clear(
            &gpu,
            &source,
            [1.0, 0.0, 0.0, 1.0],
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let mut source_pixels = vec![0_u8; (SOURCE.width * SOURCE.height * 4) as usize];
        for y in 0..SOURCE.height as usize {
            for x in 0..SOURCE.width as usize {
                let offset = (y * SOURCE.width as usize + x) * 4;
                let pixel = if x == 8 || y == 8 {
                    [255, 255, 255, 255]
                } else {
                    [(x * 11) as u8, (y * 13) as u8, ((x ^ y) * 8) as u8, 255]
                };
                source_pixels[offset..offset + 4].copy_from_slice(&pixel);
            }
        }
        let source_staging = Buffer::new(
            &gpu.device,
            &gpu.memory,
            source_pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        source_staging.write(&source_pixels).unwrap();
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                source.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            gpu.device.cmd_copy_buffer_to_image(
                command,
                source_staging.handle,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: SOURCE.width,
                        height: SOURCE.height,
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
        });
        clear(
            &gpu,
            &output,
            [0.0, 0.0, 1.0, 1.0],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        clear(&gpu, &risk, [0.0; 4], vk::ImageLayout::GENERAL);
        clear(&gpu, &motion, [0.0; 4], vk::ImageLayout::GENERAL);

        let mut renderer =
            ProtectionRenderer::new(&gpu.device, &gpu.memory, SOURCE, OUTPUT, output.format, 1)
                .unwrap();
        let output_buffer = Buffer::new(
            &gpu.device,
            &gpu.memory,
            u64::from(OUTPUT.width) * u64::from(OUTPUT.height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let weight_buffer = Buffer::new(
            &gpu.device,
            &gpu.memory,
            u64::from(OUTPUT.width) * u64::from(OUTPUT.height),
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let reference_buffer = Buffer::new(
            &gpu.device,
            &gpu.memory,
            u64::from(OUTPUT.width) * u64::from(OUTPUT.height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| {
            let result = renderer
                .record(ProtectionFrame {
                    command,
                    slot: 0,
                    frame_id: 1,
                    source: backend_image(&source, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    output: backend_image(&output, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    guidance: ProtectionInputs {
                        motion: backend_image(&motion, vk::ImageLayout::GENERAL),
                        history_risk: backend_image(&risk, vk::ImageLayout::GENERAL),
                    },
                    viewport: content_viewport(SOURCE, OUTPUT),
                    settings: ProtectionSettings::new(
                        ProtectionMode::Regions,
                        &[ProtectionRegion::new([0.25, 0.25], [0.75, 0.75]).unwrap()],
                    )
                    .unwrap(),
                    elapsed: Duration::from_millis(16),
                    reset: true,
                })
                .unwrap();
            record_spatial_blit(
                &gpu.device,
                command,
                source.handle,
                SOURCE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                spatial_reference.handle,
                vk::ImageLayout::UNDEFINED,
                OUTPUT,
            );
            image_barrier(
                &gpu.device,
                command,
                output.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &gpu.device,
                command,
                spatial_reference.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &gpu.device,
                command,
                result.spatial_weight.image,
                result.spatial_weight.layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let copy = |image, buffer| {
                gpu.device.cmd_copy_image_to_buffer(
                    command,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
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
            };
            copy(output.handle, output_buffer.handle);
            copy(result.spatial_weight.image, weight_buffer.handle);
            copy(spatial_reference.handle, reference_buffer.handle);
        });
        let mut pixels = vec![0_u8; (OUTPUT.width * OUTPUT.height * 4) as usize];
        let mut weights = vec![0_u8; (OUTPUT.width * OUTPUT.height) as usize];
        let mut reference = vec![0_u8; pixels.len()];
        output_buffer.read(&mut pixels).unwrap();
        weight_buffer.read(&mut weights).unwrap();
        reference_buffer.read(&mut reference).unwrap();
        let pixel = |x: usize, y: usize| {
            let offset = (y * OUTPUT.width as usize + x) * 4;
            &pixels[offset..offset + 4]
        };
        assert_eq!(weights[16 * OUTPUT.width as usize + 20], 255);
        let fully_protected = weights
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, weight)| *weight == 255)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert!(fully_protected.len() > 100);
        for index in fully_protected {
            let offset = index * 4;
            for channel in 0..4 {
                assert!(
                    pixels[offset + channel].abs_diff(reference[offset + channel]) <= 1,
                    "protected pixel {index} channel {channel} differs from Off"
                );
            }
        }
        assert_eq!(pixel(2, 2), &[0, 0, 0, 255]);
        assert_eq!(weights[2 * OUTPUT.width as usize + 2], 0);
        assert_eq!(pixel(6, 2), &[0, 0, 255, 255]);
        assert_eq!(weights[2 * OUTPUT.width as usize + 6], 0);
    }
}
