use ash::vk;
use tuxscaling_capture::{Capture, JitterState};
use tuxscaling_config::JitterMode;
use tuxscaling_vulkan::{Buffer, Image, color_range, image_barrier, memory_barrier};
#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

#[test]
#[ignore = "requires a Vulkan GPU"]
fn capture_excludes_later_overlay_writes() {
    unsafe {
        let gpu = Gpu::new();
        let device = &gpu.device;
        let extent = vk::Extent2D {
            width: 32,
            height: 32,
        };
        let source = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .unwrap();
        let mut capture = Capture::new(
            device,
            &gpu.memory,
            extent,
            extent,
            vk::Format::R8G8B8A8_UNORM,
        )
        .unwrap();
        let download = Buffer::new(
            device,
            &gpu.memory,
            32 * 32 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                source.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_clear_color_image(
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 1.0, 1.0],
                },
                &[color_range()],
            );
            capture.record_from(
                device,
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            image_barrier(
                device,
                command,
                source.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_clear_color_image(
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
                &[color_range()],
            );
            image_barrier(
                device,
                command,
                capture.source.color.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: 32,
                    height: 32,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                command,
                capture.source.color.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                download.handle,
                &[copy],
            );
            memory_barrier(device, command);
        });
        let mut bytes = vec![0; 32 * 32 * 4];
        download.read(&mut bytes).unwrap();
        assert!(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| *pixel == [0, 0, 255, 255])
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn capture_resampling_applies_experimental_subpixel_jitter_without_changing_off() {
    unsafe {
        let gpu = Gpu::new();
        let device = &gpu.device;
        let extent = vk::Extent2D {
            width: 8,
            height: 8,
        };
        let source = Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .unwrap();
        let mut capture = Capture::new(
            device,
            &gpu.memory,
            extent,
            extent,
            vk::Format::R8G8B8A8_UNORM,
        )
        .unwrap();
        let staging = Buffer::new(
            device,
            &gpu.memory,
            (extent.width * extent.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let download = Buffer::new(
            device,
            &gpu.memory,
            (extent.width * extent.height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let pixels = (0..extent.height)
            .flat_map(|y| {
                (0..extent.width).flat_map(move |x| {
                    [x * 32, y * 32, 0, 255]
                        .into_iter()
                        .map(|value| value as u8)
                })
            })
            .collect::<Vec<_>>();
        staging.write(&pixels).unwrap();

        let mut jitter = JitterState::new(JitterMode::ExperimentalHalton8);
        let _ = jitter.sample();
        let experimental = jitter.sample();
        assert_ne!(experimental.current, [0.0, 0.0]);
        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        let read = |capture: &Capture, command: vk::CommandBuffer| {
            image_barrier(
                device,
                command,
                capture.guidance.current.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                capture.guidance.current.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                download.handle,
                &[copy],
            );
            image_barrier(
                device,
                command,
                capture.guidance.current.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            memory_barrier(device, command);
        };
        let capture_frame = |capture: &mut Capture,
                             source_layout: vk::ImageLayout,
                             jitter: tuxscaling_temporal::JitterSample,
                             command: vk::CommandBuffer| {
            image_barrier(
                device,
                command,
                source.handle,
                source_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                staging.handle,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            capture.record_scaled_from_with_jitter(
                device,
                command,
                source.handle,
                extent,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                jitter,
            );
        };

        gpu.submit(|command| {
            capture_frame(
                &mut capture,
                vk::ImageLayout::UNDEFINED,
                tuxscaling_temporal::JitterSample::default(),
                command,
            );
            read(&capture, command);
        });
        let mut off = vec![0u8; pixels.len()];
        download.read(&mut off).unwrap();

        gpu.submit(|command| {
            capture_frame(
                &mut capture,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                experimental,
                command,
            );
            read(&capture, command);
        });
        let mut on = vec![0u8; pixels.len()];
        download.read(&mut on).unwrap();

        assert_eq!(&off[0..4], &[0, 0, 0, 255]);
        assert_eq!(&off[4..8], &[32, 0, 0, 255]);
        assert_ne!(&on[4..8], &[32, 0, 0, 255]);
        assert!(on[4] > 32);
    }
}
