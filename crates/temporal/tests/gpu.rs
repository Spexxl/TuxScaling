#![allow(clippy::missing_safety_doc)]

use ash::vk;
use tuxscaling_motion::MotionEstimator;
use tuxscaling_vulkan::{Buffer, Image, image_barrier, memory_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

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
            motion.metadata.handle,
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
            memory_barrier(device, command);
        });
        let mut bytes = [0u8; 4];
        exposure_readback.read(&mut bytes).unwrap();
        let exposure = f32::from_ne_bytes(bytes);
        let expected: f32 = 1.0 / (128.0 / 255.0);
        assert!(exposure.is_finite() && exposure > 0.0);
        assert!((exposure.log2() - expected.log2()).abs() <= 0.15);
    }
}
