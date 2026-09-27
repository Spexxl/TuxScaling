#![allow(clippy::missing_safety_doc)]

use ash::vk;
use std::io::Cursor;
use tuxscaling_vulkan::{Buffer, Image, compute_memory_barrier, image_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

struct Inputs {
    motion: Image,
    confidence: Image,
    disocclusion: Image,
    reactive: Image,
    depth: Image,
    composition: Image,
}

struct Outputs {
    motion: Image,
    depth: Image,
    reactive: Image,
    composition: Image,
    risk: Image,
}

struct Readback {
    motion: Buffer,
    reactive: Buffer,
    composition: Buffer,
    risk: Buffer,
}

fn bytes_for(extent: vk::Extent2D, bytes_per_pixel: u64) -> u64 {
    u64::from(extent.width) * u64::from(extent.height) * bytes_per_pixel
}

unsafe fn image(
    gpu: &Gpu,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Image {
    unsafe { Image::new(&gpu.device, &gpu.memory, extent, format, usage) }.unwrap()
}

unsafe fn clear_input(gpu: &Gpu, image: &Image, value: [f32; 4]) {
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
            device.cmd_clear_color_image(
                command,
                image.handle,
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
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        });
    }
}

unsafe fn execute_case(
    guidance_enabled: bool,
    motion_value: [f32; 2],
    extent: vk::Extent2D,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let gpu = unsafe { Gpu::new() };
    let device = &gpu.device;
    let sampled = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
    let inputs = Inputs {
        motion: unsafe { image(&gpu, extent, vk::Format::R16G16_SFLOAT, sampled) },
        confidence: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, sampled) },
        disocclusion: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, sampled) },
        reactive: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, sampled) },
        depth: unsafe { image(&gpu, extent, vk::Format::R32_SFLOAT, sampled) },
        composition: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, sampled) },
    };
    let outputs = Outputs {
        motion: unsafe { image(&gpu, extent, vk::Format::R16G16_SFLOAT, storage) },
        depth: unsafe { image(&gpu, extent, vk::Format::R32_SFLOAT, storage) },
        reactive: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, storage) },
        composition: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, storage) },
        risk: unsafe { image(&gpu, extent, vk::Format::R8_UNORM, storage) },
    };
    let readback = Readback {
        motion: unsafe {
            Buffer::new(
                device,
                &gpu.memory,
                bytes_for(extent, 4),
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        }
        .unwrap(),
        reactive: unsafe {
            Buffer::new(
                device,
                &gpu.memory,
                bytes_for(extent, 1),
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        }
        .unwrap(),
        composition: unsafe {
            Buffer::new(
                device,
                &gpu.memory,
                bytes_for(extent, 1),
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        }
        .unwrap(),
        risk: unsafe {
            Buffer::new(
                device,
                &gpu.memory,
                bytes_for(extent, 1),
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        }
        .unwrap(),
    };

    unsafe {
        clear_input(
            &gpu,
            &inputs.motion,
            [motion_value[0], motion_value[1], 0.0, 0.0],
        )
    };
    unsafe { clear_input(&gpu, &inputs.confidence, [0.05, 0.0, 0.0, 0.0]) };
    unsafe { clear_input(&gpu, &inputs.disocclusion, [0.0; 4]) };
    unsafe { clear_input(&gpu, &inputs.reactive, [1.0, 0.0, 0.0, 0.0]) };
    unsafe { clear_input(&gpu, &inputs.depth, [1.0, 0.0, 0.0, 0.0]) };
    unsafe { clear_input(&gpu, &inputs.composition, [0.0; 4]) };

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
    let bindings = (0..11)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(if binding < 6 {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                } else {
                    vk::DescriptorType::STORAGE_IMAGE
                })
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
    let descriptor_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
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
                        descriptor_count: 6,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 5,
                    },
                ]),
            None,
        )
    }
    .unwrap();
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_layout]),
        )
    }
    .unwrap()[0];
    let sampled_images = [
        &inputs.motion,
        &inputs.confidence,
        &inputs.disocclusion,
        &inputs.reactive,
        &inputs.depth,
        &inputs.composition,
    ];
    let output_images = [
        &outputs.motion,
        &outputs.depth,
        &outputs.reactive,
        &outputs.composition,
        &outputs.risk,
    ];
    let sampled_infos = sampled_images
        .iter()
        .map(|input| {
            [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(input.view)
                .image_layout(vk::ImageLayout::GENERAL)]
        })
        .collect::<Vec<_>>();
    let output_infos = output_images
        .iter()
        .map(|output| {
            [vk::DescriptorImageInfo::default()
                .image_view(output.view)
                .image_layout(vk::ImageLayout::GENERAL)]
        })
        .collect::<Vec<_>>();
    let mut writes = Vec::with_capacity(11);
    for (binding, image) in sampled_infos.iter().enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image),
        );
    }
    for (index, image) in output_infos.iter().enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(index as u32 + 6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(image),
        );
    }
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    let shader_words = ash::util::read_spv(&mut Cursor::new(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/fidelityfx_input.spv"
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
                    size: 24,
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

    let mut motion_bytes = vec![0; bytes_for(extent, 4) as usize];
    let mut reactive_bytes = vec![0; bytes_for(extent, 1) as usize];
    let mut composition_bytes = vec![0; bytes_for(extent, 1) as usize];
    let mut risk_bytes = vec![0; bytes_for(extent, 1) as usize];
    unsafe {
        gpu.submit(|command| {
            for output in output_images {
                image_barrier(
                    device,
                    command,
                    output.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[set],
                &[],
            );
            let params = [
                extent.width,
                extent.height,
                u32::from(guidance_enabled),
                2,
                0.0_f32.to_bits(),
                1,
            ];
            device.cmd_push_constants(
                command,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            device.cmd_dispatch(
                command,
                extent.width.div_ceil(8),
                extent.height.div_ceil(8),
                1,
            );
            compute_memory_barrier(device, command);
            for (image, buffer) in [
                (&outputs.motion, &readback.motion),
                (&outputs.reactive, &readback.reactive),
                (&outputs.composition, &readback.composition),
                (&outputs.risk, &readback.risk),
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
                    buffer.handle,
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
            }
        });
        readback.motion.read(&mut motion_bytes).unwrap();
        readback.reactive.read(&mut reactive_bytes).unwrap();
        readback.composition.read(&mut composition_bytes).unwrap();
        readback.risk.read(&mut risk_bytes).unwrap();
    }

    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_descriptor_set_layout(descriptor_layout, None);
        device.destroy_sampler(sampler, None);
    }

    (motion_bytes, reactive_bytes, composition_bytes, risk_bytes)
}

fn f16_to_f32(value: u16) -> f32 {
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

#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn adaptive_policy_preserves_motion_and_emits_independent_masks() {
    let extent = vk::Extent2D {
        width: 13,
        height: 9,
    };
    let (motion, reactive, composition, risk) = unsafe { execute_case(true, [4.0, -2.0], extent) };
    let center = (4 * 13 + 6) as usize;
    let motion_offset = center * 4;
    let actual_motion = [
        f16_to_f32(u16::from_ne_bytes([
            motion[motion_offset],
            motion[motion_offset + 1],
        ])),
        f16_to_f32(u16::from_ne_bytes([
            motion[motion_offset + 2],
            motion[motion_offset + 3],
        ])),
    ];
    assert!((actual_motion[0] - 4.0).abs() < 0.1);
    assert!((actual_motion[1] + 2.0).abs() < 0.1);
    assert!((220..=230).contains(&reactive[center]));
    assert!(risk[center] > 220);
    assert!((20..=150).contains(&composition[center]));
    assert_ne!(reactive[center], composition[center]);
}

#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn zero_guidance_keeps_masks_neutral_and_marks_history_unsafe() {
    let extent = vk::Extent2D {
        width: 13,
        height: 9,
    };
    let (motion, reactive, composition, risk) = unsafe { execute_case(false, [4.0, -2.0], extent) };
    let center = (4 * 13 + 6) as usize;
    assert_eq!(&motion[center * 4..center * 4 + 4], &[0, 0, 0, 0]);
    assert_eq!(reactive[center], 0);
    assert_eq!(composition[center], 0);
    assert_eq!(risk[center], 255);
}

#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn source_pixel_translation_magnitudes_survive_adapter_without_rescaling() {
    for extent in [
        vk::Extent2D {
            width: 13,
            height: 9,
        },
        vk::Extent2D {
            width: 7,
            height: 5,
        },
    ] {
        for expected in [
            [0.5, 0.0],
            [-0.5, 0.0],
            [4.0, 0.0],
            [0.0, 0.5],
            [0.0, -0.5],
            [0.0, 4.0],
        ] {
            let (motion, _, _, _) = unsafe { execute_case(true, expected, extent) };
            let center = ((extent.height / 2 * extent.width) + extent.width / 2) as usize;
            let offset = center * 4;
            let actual = [
                f16_to_f32(u16::from_ne_bytes([motion[offset], motion[offset + 1]])),
                f16_to_f32(u16::from_ne_bytes([motion[offset + 2], motion[offset + 3]])),
            ];
            assert!((actual[0] - expected[0]).abs() < 0.1);
            assert!((actual[1] - expected[1]).abs() < 0.1);
        }
    }
}
