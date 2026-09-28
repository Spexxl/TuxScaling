#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_vulkan::{Buffer, Image, color_range, image_barrier};

#[path = "../src/debug_view.rs"]
mod debug_view;
#[path = "../../../tests/support/gpu.rs"]
mod support;

use debug_view::{DebugImage, record_debug_image};
use support::Gpu;

const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 8,
    height: 8,
};

unsafe fn read_debug_pixel(format: vk::Format, color: [f32; 4]) -> [u8; 4] {
    let gpu = unsafe { Gpu::new() };
    let source = unsafe {
        Image::new(
            &gpu.device,
            &gpu.memory,
            EXTENT,
            format,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )
    }
    .unwrap();
    let output = unsafe {
        Image::new(
            &gpu.device,
            &gpu.memory,
            EXTENT,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
    }
    .unwrap();
    let readback = unsafe {
        Buffer::new(
            &gpu.device,
            &gpu.memory,
            u64::from(EXTENT.width) * u64::from(EXTENT.height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                source.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            gpu.device.cmd_clear_color_image(
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: color },
                &[color_range()],
            );
            image_barrier(
                &gpu.device,
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
            image_barrier(
                &gpu.device,
                command,
                output.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            record_debug_image(
                &gpu.device,
                command,
                DebugImage {
                    image: source.handle,
                    extent: EXTENT,
                    layout: vk::ImageLayout::GENERAL,
                },
                output.handle,
                EXTENT,
            );
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
                readback.handle,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: EXTENT.width,
                        height: EXTENT.height,
                        depth: 1,
                    })],
            );
        });
        let mut pixels = vec![0; (EXTENT.width * EXTENT.height * 4) as usize];
        readback.read(&mut pixels).unwrap();
        [pixels[0], pixels[1], pixels[2], pixels[3]]
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn debug_blit_renders_color_masks_depth_and_motion_with_distinct_pixels() {
    unsafe {
        assert_eq!(
            read_debug_pixel(vk::Format::R8G8B8A8_UNORM, [0.2, 0.4, 0.6, 1.0]),
            [51, 102, 153, 255]
        );
        assert_eq!(
            read_debug_pixel(vk::Format::R8_UNORM, [0.75, 0.0, 0.0, 1.0]),
            [191, 0, 0, 255]
        );
        assert_eq!(
            read_debug_pixel(vk::Format::R32_SFLOAT, [0.25, 0.0, 0.0, 1.0]),
            [64, 0, 0, 255]
        );
        assert_eq!(
            read_debug_pixel(vk::Format::R16G16_SFLOAT, [-1.0, 0.5, 0.0, 1.0]),
            [0, 128, 0, 255]
        );
    }
}
