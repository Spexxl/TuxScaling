#![allow(clippy::missing_safety_doc)]

use ash::vk;
use std::io::Cursor;
use tuxscaling_upscaler::{BackendColorEncoding, fidelityfx::color::FsrColorPlan};
use tuxscaling_vulkan::{Buffer, Image, image_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

fn decode_srgb(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn encode_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let rounded = significand + (1 << (shift - 1)) - 1 + ((significand >> shift) & 1);
        sign | (rounded >> shift) as u16
    } else if exponent >= 31 {
        sign | 0x7c00
    } else {
        let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
        let rounded_exponent = exponent + i32::from((rounded & 0x80_0000) != 0);
        sign | ((rounded_exponent as u16) << 10) | ((rounded >> 13) as u16 & 0x03ff)
    }
}

unsafe fn resolve_linear_to_unorm(
    gpu: &Gpu,
    source_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    viewport: [f32; 4],
    pixels: &[[f32; 4]],
) -> Vec<u8> {
    let device = &gpu.device;
    let source = unsafe {
        Image::new(
            device,
            &gpu.memory,
            source_extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
    }
    .unwrap();
    let output = unsafe {
        Image::new(
            device,
            &gpu.memory,
            output_extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }
    .unwrap();
    let input_bytes = pixels
        .iter()
        .flat_map(|pixel| pixel.iter().copied().map(f32_to_f16_bits))
        .flat_map(u16::to_ne_bytes)
        .collect::<Vec<_>>();
    let upload = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            input_bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    unsafe { upload.write(&input_bytes) }.unwrap();
    let readback_size = u64::from(output_extent.width) * u64::from(output_extent.height) * 4;
    let readback = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            readback_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.0),
            None,
        )
    }
    .unwrap();
    let descriptor_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ]),
            None,
        )
    }
    .unwrap();
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    },
                ]),
            None,
        )
    }
    .unwrap();
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_layout]),
        )
    }
    .unwrap()[0];
    let shader_words = ash::util::read_spv(&mut Cursor::new(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/fidelityfx_output.spv"
    ))))
    .unwrap();
    let shader = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader_words),
            None,
        )
    }
    .unwrap();
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[descriptor_layout])
                .push_constant_ranges(&[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: 28,
                }]),
            None,
        )
    }
    .unwrap();
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(c"main"),
                )
                .layout(pipeline_layout)],
            None,
        )
    }
    .unwrap()[0];
    let sampled = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(source.view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let storage = vk::DescriptorImageInfo::default()
        .image_view(output.view)
        .image_layout(vk::ImageLayout::GENERAL);
    unsafe {
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sampled)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&storage)),
            ],
            &[],
        );
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                source.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload.handle,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: source_extent.width,
                        height: source_extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                device,
                command,
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                device,
                command,
                output.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let push = [
                output_extent.width,
                output_extent.height,
                viewport[0].to_bits(),
                viewport[1].to_bits(),
                viewport[2].to_bits(),
                viewport[3].to_bits(),
                1,
            ];
            device.cmd_push_constants(
                command,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push),
            );
            device.cmd_dispatch(
                command,
                output_extent.width.div_ceil(8),
                output_extent.height.div_ceil(8),
                1,
            );
            image_barrier(
                device,
                command,
                output.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
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
                        width: output_extent.width,
                        height: output_extent.height,
                        depth: 1,
                    })],
            );
        });
    }
    let mut output_bytes = vec![0; readback_size as usize];
    unsafe { readback.read(&mut output_bytes) }.unwrap();
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(descriptor_layout, None);
        device.destroy_sampler(sampler, None);
    }
    output_bytes
}

fn f16_bits_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = u32::from((value >> 10) & 0x1f);
    let mantissa = u32::from(value & 0x03ff);
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut exponent = 113_u32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

unsafe fn convert_encoded_source_to_linear(
    gpu: &Gpu,
    format: vk::Format,
    decode_in_shader: bool,
    extent: vk::Extent2D,
    encoded_bytes: &[u8],
) -> Vec<u8> {
    let device = &gpu.device;
    let source = unsafe {
        Image::new(
            device,
            &gpu.memory,
            extent,
            format,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )
    }
    .unwrap();
    let linear = unsafe {
        Image::new(
            device,
            &gpu.memory,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )
    }
    .unwrap();
    let upload = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            encoded_bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    unsafe { upload.write(encoded_bytes) }.unwrap();
    let readback_size = u64::from(extent.width) * u64::from(extent.height) * 8;
    let readback = unsafe {
        Buffer::new(
            device,
            &gpu.memory,
            readback_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }
    .unwrap();
    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.0),
            None,
        )
    }
    .unwrap();
    let descriptor_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ]),
            None,
        )
    }
    .unwrap();
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                    },
                ]),
            None,
        )
    }
    .unwrap();
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_layout]),
        )
    }
    .unwrap()[0];
    let shader_words = ash::util::read_spv(&mut Cursor::new(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/fidelityfx_color.spv"
    ))))
    .unwrap();
    let shader = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader_words),
            None,
        )
    }
    .unwrap();
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[descriptor_layout])
                .push_constant_ranges(&[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: 12,
                }]),
            None,
        )
    }
    .unwrap();
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(c"main"),
                )
                .layout(pipeline_layout)],
            None,
        )
    }
    .unwrap()[0];
    let sampled = vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(source.view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let storage = vk::DescriptorImageInfo::default()
        .image_view(linear.view)
        .image_layout(vk::ImageLayout::GENERAL);
    unsafe {
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sampled)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&storage)),
            ],
            &[],
        );
        gpu.submit(|command| {
            image_barrier(
                device,
                command,
                source.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload.handle,
                source.handle,
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
                source.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                device,
                command,
                linear.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            let push = [extent.width, extent.height, u32::from(decode_in_shader)];
            device.cmd_push_constants(
                command,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push),
            );
            device.cmd_dispatch(
                command,
                extent.width.div_ceil(8),
                extent.height.div_ceil(8),
                1,
            );
            image_barrier(
                device,
                command,
                linear.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                linear.handle,
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
        });
    }
    let mut bytes = vec![0; readback_size as usize];
    unsafe { readback.read(&mut bytes) }.unwrap();
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(descriptor_layout, None);
        device.destroy_sampler(sampler, None);
    }
    bytes
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn srgb_round_trip_preserves_alpha() {
    let gpu = unsafe { Gpu::new() };
    let encoded = [0_u8, 64, 128, 255];
    let pixels = encoded
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            [
                decode_srgb(f32::from(*channel) / 255.0),
                decode_srgb(f32::from(*channel) / 255.0),
                decode_srgb(f32::from(*channel) / 255.0),
                f32::from([0_u8, 64, 128, 255][index]) / 255.0,
            ]
        })
        .collect::<Vec<_>>();
    let result = unsafe {
        resolve_linear_to_unorm(
            &gpu,
            vk::Extent2D {
                width: 4,
                height: 1,
            },
            vk::Extent2D {
                width: 4,
                height: 1,
            },
            [0.0, 0.0, 1.0, 1.0],
            &pixels,
        )
    };

    for (index, expected) in encoded.into_iter().enumerate() {
        let offset = index * 4;
        assert!(
            result[offset].abs_diff(expected) <= 1,
            "red at pixel {index}"
        );
        assert!(
            result[offset + 1].abs_diff(expected) <= 1,
            "green at pixel {index}"
        );
        assert!(
            result[offset + 2].abs_diff(expected) <= 1,
            "blue at pixel {index}"
        );
        assert_eq!(result[offset + 3], expected, "alpha at pixel {index}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn srgb_unorm_and_srgb_view_agree() {
    let gpu = unsafe { Gpu::new() };
    let extent = vk::Extent2D {
        width: 5,
        height: 1,
    };
    let encoded_pixels = [
        [0_u8, 255, 32, 0],
        [16, 240, 64, 64],
        [64, 192, 128, 128],
        [128, 128, 192, 192],
        [255, 0, 255, 255],
    ];
    let encoded_bytes = encoded_pixels.into_iter().flatten().collect::<Vec<_>>();
    let unorm_plan = FsrColorPlan::for_source(
        vk::Format::R8G8B8A8_UNORM,
        BackendColorEncoding::SrgbNonlinear,
    )
    .unwrap();
    let srgb_plan = FsrColorPlan::for_source(
        vk::Format::R8G8B8A8_SRGB,
        BackendColorEncoding::SrgbNonlinear,
    )
    .unwrap();
    let unorm = unsafe {
        convert_encoded_source_to_linear(
            &gpu,
            vk::Format::R8G8B8A8_UNORM,
            unorm_plan.decode_in_shader,
            extent,
            &encoded_bytes,
        )
    };
    let srgb = unsafe {
        convert_encoded_source_to_linear(
            &gpu,
            vk::Format::R8G8B8A8_SRGB,
            srgb_plan.decode_in_shader,
            extent,
            &encoded_bytes,
        )
    };

    for (pixel_index, pixel) in encoded_pixels.iter().enumerate() {
        for (channel, value) in pixel.iter().enumerate() {
            let offset = (pixel_index * 4 + channel) * 2;
            let unorm_value =
                f16_bits_to_f32(u16::from_ne_bytes([unorm[offset], unorm[offset + 1]]));
            let srgb_value = f16_bits_to_f32(u16::from_ne_bytes([srgb[offset], srgb[offset + 1]]));
            let expected = if channel == 3 {
                f32::from(*value) / 255.0
            } else {
                decode_srgb(f32::from(*value) / 255.0)
            };
            assert!(
                (unorm_value - expected).abs() <= 0.002,
                "UNORM channel {channel} at pixel {pixel_index}: {unorm_value} != {expected}"
            );
            assert!(
                (srgb_value - expected).abs() <= 0.002,
                "sRGB view channel {channel} at pixel {pixel_index}: {srgb_value} != {expected}"
            );
            assert!(
                (unorm_value - srgb_value).abs() <= 0.002,
                "UNORM and sRGB views differ at pixel {pixel_index}, channel {channel}"
            );
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn identity_resolve_has_no_half_pixel_shift() {
    let gpu = unsafe { Gpu::new() };
    let extent = vk::Extent2D {
        width: 4,
        height: 4,
    };
    let pixels = (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| {
                let value = if x == 1 && y == 1 {
                    1.0
                } else if (x + y) % 2 == 0 {
                    0.2
                } else {
                    0.05
                };
                [value, value, value, 1.0]
            })
        })
        .collect::<Vec<_>>();
    let output =
        unsafe { resolve_linear_to_unorm(&gpu, extent, extent, [0.0, 0.0, 1.0, 1.0], &pixels) };

    for (index, pixel) in pixels.iter().enumerate() {
        let expected = (encode_srgb(pixel[0]) * 255.0).round() as u8;
        for channel in 0..3 {
            assert!(
                output[index * 4 + channel].abs_diff(expected) <= 1,
                "identity resolve shifted pixel {index}, channel {channel}"
            );
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn odd_extent_letterbox_does_not_stretch() {
    let gpu = unsafe { Gpu::new() };
    let source_extent = vk::Extent2D {
        width: 5,
        height: 5,
    };
    let output_extent = vk::Extent2D {
        width: 7,
        height: 5,
    };
    let pixels = (0..source_extent.height)
        .flat_map(|y| {
            (0..source_extent.width).map(move |x| {
                [
                    0.05 + x as f32 * 0.1,
                    0.1 + y as f32 * 0.1,
                    0.2 + (x + y) as f32 * 0.05,
                    1.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let output = unsafe {
        resolve_linear_to_unorm(
            &gpu,
            source_extent,
            output_extent,
            [1.0 / 7.0, 0.0, 5.0 / 7.0, 1.0],
            &pixels,
        )
    };

    for y in 0..output_extent.height {
        for x in 0..output_extent.width {
            let output_index = (y * output_extent.width + x) as usize * 4;
            if x == 0 || x == output_extent.width - 1 {
                assert_eq!(&output[output_index..output_index + 3], &[0, 0, 0]);
                continue;
            }
            let source_index = (y * source_extent.width + (x - 1)) as usize;
            for channel in 0..3 {
                let expected = (encode_srgb(pixels[source_index][channel]) * 255.0).round() as u8;
                assert!(
                    output[output_index + channel].abs_diff(expected) <= 1,
                    "letterbox changed source pixel ({}, {}) channel {channel}",
                    x - 1,
                    y
                );
            }
        }
    }
}
