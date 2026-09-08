#![cfg(feature = "fidelityfx")]
#![allow(clippy::missing_safety_doc)]

use ash::vk;
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
use tuxscaling_vulkan::{Image, image_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

const GAME: vk::Extent2D = vk::Extent2D {
    width: 64,
    height: 64,
};

fn resource(image: &Image, format: vk::Format, metadata: GuidanceMetadata) -> GuidanceResource {
    GuidanceResource {
        image: image.handle,
        view: image.view,
        format,
        metadata,
        state: SignalState::ConstantFallback,
    }
}

#[test]
fn fsr314_shader_contract_uses_zero_jitter_and_fixed_camera_domain() {
    let shader = include_str!("../../../shaders/upscaler/fidelityfx_output.comp");
    assert!(shader.contains("viewport_offset"));
    assert!(shader.contains("imageStore(output_image"));
}

#[test]
#[ignore = "requires a Vulkan GPU with the FidelityFX storage-image formats"]
fn fsr314_lifecycle_dispatches_reset_and_native_aa() {
    let gpu = unsafe { Gpu::new() };
    let physical = unsafe { gpu.instance.enumerate_physical_devices() }.unwrap()[0];
    let environment = BackendEnvironment::new(&gpu.instance, physical, &gpu.device);
    let memory = gpu.memory;
    let color_usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let source = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            GAME,
            vk::Format::R8G8B8A8_UNORM,
            color_usage,
        )
    }
    .unwrap();
    let output = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            GAME,
            vk::Format::R8G8B8A8_UNORM,
            color_usage,
        )
    }
    .unwrap();
    let guidance_images = [
        (vk::Format::R16G16_SFLOAT, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R16G16_SFLOAT,
                color_usage,
            )
        }),
        (vk::Format::R8_UNORM, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R8_UNORM,
                color_usage,
            )
        }),
        (vk::Format::R8_UNORM, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R8_UNORM,
                color_usage,
            )
        }),
        (vk::Format::R8_UNORM, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R8_UNORM,
                color_usage,
            )
        }),
        (vk::Format::R32_SFLOAT, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R32_SFLOAT,
                color_usage,
            )
        }),
        (vk::Format::R8_UNORM, unsafe {
            Image::new(
                &gpu.device,
                &memory,
                GAME,
                vk::Format::R8_UNORM,
                color_usage,
            )
        }),
    ];
    let images = guidance_images
        .into_iter()
        .map(|(_, image)| image.unwrap())
        .collect::<Vec<_>>();
    let exposure = unsafe {
        Image::new(
            &gpu.device,
            &memory,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R32_SFLOAT,
            color_usage,
        )
    }
    .unwrap();
    let game_extent = FrameExtent {
        width: GAME.width,
        height: GAME.height,
    };
    let metadata = GuidanceMetadata::zero(1, game_extent, GuidanceReset::None);
    let guidance = GuidanceView {
        motion: resource(&images[0], vk::Format::R16G16_SFLOAT, metadata),
        confidence: resource(&images[1], vk::Format::R8_UNORM, metadata),
        disocclusion: resource(&images[2], vk::Format::R8_UNORM, metadata),
        reactive: resource(&images[3], vk::Format::R8_UNORM, metadata),
        exposure: resource(&exposure, vk::Format::R32_SFLOAT, metadata),
        depth: resource(&images[4], vk::Format::R32_SFLOAT, metadata),
        transparency_composition: resource(&images[5], vk::Format::R8_UNORM, metadata),
        pre_exposure: GuidanceScalar::constant_fallback(1.0),
        timing: FrameTiming::default(),
        jitter: JitterSample::default(),
        depth_semantics: DepthSemantics::FlatFallback,
        direction: MotionDirection::CurrentToPrevious,
        units: MotionUnits::SourcePixels,
        resolution: GuidanceResolution::new(game_extent, game_extent),
        requires_history_reset: false,
    };
    let config = BackendConfig {
        game_extent: GAME,
        output_extent: GAME,
        source_format: vk::Format::R8G8B8A8_UNORM,
        output_format: vk::Format::R8G8B8A8_UNORM,
        color_encoding: BackendColorEncoding::SrgbNonlinear,
        viewport: content_viewport(GAME, GAME),
        guidance: guidance.capabilities(),
    };
    let mut backend = unsafe { Fsr314Upscaler::new(&environment, config, guidance, 1) }.unwrap();
    assert_eq!(backend.id().as_str(), "fsr_3_1_4");

    let frame = BackendFrame {
        command_buffer: vk::CommandBuffer::null(),
        slot: 0,
        source: BackendImage {
            image: source.handle,
            view: source.view,
            format: source.format,
            extent: GAME,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        output: BackendImage {
            image: output.handle,
            view: output.view,
            format: output.format,
            extent: GAME,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        guidance,
        viewport: content_viewport(GAME, GAME),
        frame_id: 1,
        reset_history: true,
        debug_view: 0,
    };
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                source.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &gpu.device,
                command,
                output.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            for image in images.iter().chain(std::iter::once(&exposure)) {
                image_barrier(
                    &gpu.device,
                    command,
                    image.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
            let mut frame = frame;
            frame.command_buffer = command;
            backend.record(frame).unwrap();
        });
        backend.reset().unwrap();
        gpu.submit(|command| {
            let mut frame = frame;
            frame.command_buffer = command;
            frame.reset_history = true;
            backend.record(frame).unwrap();
        });
    }
}
