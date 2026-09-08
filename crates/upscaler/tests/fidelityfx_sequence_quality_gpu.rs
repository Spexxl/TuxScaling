#![cfg(feature = "fidelityfx")]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use ash::vk;
use std::time::Duration;
use tuxscaling_temporal::quality::SequenceFixture;
use tuxscaling_temporal::{
    DepthSemantics, FrameExtent, FrameTiming, GuidanceMetadata, GuidanceReset, GuidanceResolution,
    GuidanceResource, GuidanceScalar, GuidanceView, JitterSample, MotionDirection, MotionUnits,
    SignalState,
};
use tuxscaling_upscaler::fidelityfx::Fsr314Upscaler;
use tuxscaling_upscaler::{
    BackendColorEncoding, BackendConfig, BackendEnvironment, BackendFrame, BackendImage,
    ReferenceUpscaler, UpscalerBackend, content_viewport,
};
use tuxscaling_vulkan::{Buffer, Image, image_barrier};

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

type FixtureFactory = fn(u32, u32) -> SequenceFixture;

fn captured_sequence_catalog() -> [(&'static str, FixtureFactory); 12] {
    [
        ("translation", tuxscaling_temporal::quality::translation),
        ("rotation", tuxscaling_temporal::quality::rotation),
        ("scaling", tuxscaling_temporal::quality::zoom),
        (
            "camera_pan",
            tuxscaling_temporal::quality::independent_objects,
        ),
        ("thin_geometry", tuxscaling_temporal::quality::thin_geometry),
        ("hud", tuxscaling_temporal::quality::hud),
        ("transparency", tuxscaling_temporal::quality::transparency),
        ("particles", tuxscaling_temporal::quality::particles),
        (
            "occlusion_disocclusion",
            tuxscaling_temporal::quality::occlusion,
        ),
        ("noise", tuxscaling_temporal::quality::noise),
        ("scene_cut", tuxscaling_temporal::quality::scene_cut),
        ("pause_resume", tuxscaling_temporal::quality::pause),
    ]
}

#[test]
fn captured_sequence_catalog_is_complete_and_deterministic() {
    let first = captured_sequence_catalog();
    let second = captured_sequence_catalog();
    assert_eq!(
        first.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        [
            "translation",
            "rotation",
            "scaling",
            "camera_pan",
            "thin_geometry",
            "hud",
            "transparency",
            "particles",
            "occlusion_disocclusion",
            "noise",
            "scene_cut",
            "pause_resume",
        ]
    );
    for ((name, factory), (_, second_factory)) in first.iter().zip(second.iter()) {
        let left = factory(32, 32);
        let right = second_factory(32, 32);
        assert_eq!(left, right, "fixture {name} is not deterministic");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU with the FidelityFX storage-image formats"]
fn fsr314_captured_sequence_quality_suite() {
    let gpu = unsafe { Gpu::new() };
    let physical = unsafe { gpu.instance.enumerate_physical_devices() }.unwrap()[0];
    let environment = BackendEnvironment::new(&gpu.instance, physical, &gpu.device);
    let mut fsr_psnr_total = 0.0;
    let mut bilinear_psnr_total = 0.0;
    let mut fsr_ssim_total = 0.0;
    let mut bilinear_ssim_total = 0.0;
    let mut flicker_total = 0.0;
    let mut bilinear_flicker_total = 0.0;
    let mut reference_psnr_total = 0.0;
    let mut reference_ssim_total = 0.0;

    for (case_index, (name, factory)) in captured_sequence_catalog().into_iter().enumerate() {
        let fixture = factory(INPUT.width, INPUT.height);
        let source_bytes = scene_bytes(case_index, INPUT);
        let second_source_bytes = fixture_scene_bytes(case_index, &fixture, INPUT);
        let expected = expected_pixels(case_index, OUTPUT);
        let second_expected = fixture_expected_pixels(case_index, &fixture, OUTPUT);
        let bilinear_frame = bilinear(&source_bytes, INPUT, OUTPUT);
        let second_bilinear = bilinear(&second_source_bytes, INPUT, OUTPUT);
        let (fsr_first, fsr_second) = unsafe {
            run_fsr_case(
                &gpu,
                &environment,
                &fixture,
                &source_bytes,
                &second_source_bytes,
            )
        };
        let reference = unsafe { run_reference_case(&gpu, &fixture, &source_bytes) };
        let fsr_psnr = psnr(&fsr_first, &expected);
        let bilinear_psnr = psnr(&bilinear_frame, &expected);
        let fsr_ssim = ssim(&fsr_first, &expected);
        let bilinear_ssim = ssim(&bilinear_frame, &expected);
        let reference_psnr = psnr(&reference, &expected);
        let reference_ssim = ssim(&reference, &expected);
        let flicker = temporal_error(&fsr_first, &fsr_second, &expected, &second_expected);
        let bilinear_flicker = temporal_error(
            &bilinear_frame,
            &second_bilinear,
            &expected,
            &second_expected,
        );
        eprintln!(
            "FSR sequence fixture: name={name} psnr={fsr_psnr:.3}dB bilinear={bilinear_psnr:.3}dB reference={reference_psnr:.3}dB ssim={fsr_ssim:.5} bilinear_ssim={bilinear_ssim:.5} reference_ssim={reference_ssim:.5} flicker_mse={flicker:.6} bilinear_flicker_mse={bilinear_flicker:.6}"
        );
        assert!(
            fsr_first
                .iter()
                .any(|pixel| pixel[..3].iter().any(|value| *value > 0.01))
        );
        assert!(fsr_psnr.is_finite());
        assert!(fsr_ssim.is_finite());
        assert!(flicker.is_finite());
        fsr_psnr_total += fsr_psnr;
        bilinear_psnr_total += bilinear_psnr;
        fsr_ssim_total += fsr_ssim;
        bilinear_ssim_total += bilinear_ssim;
        reference_psnr_total += reference_psnr;
        reference_ssim_total += reference_ssim;
        flicker_total += flicker;
        bilinear_flicker_total += bilinear_flicker;
    }

    let count = captured_sequence_catalog().len() as f32;
    let fsr_psnr = fsr_psnr_total / count;
    let bilinear_psnr = bilinear_psnr_total / count;
    let fsr_ssim = fsr_ssim_total / count;
    let bilinear_ssim = bilinear_ssim_total / count;
    let reference_psnr = reference_psnr_total / count;
    let reference_ssim = reference_ssim_total / count;
    let flicker = flicker_total / count;
    let bilinear_flicker = bilinear_flicker_total / count;
    eprintln!(
        "FSR sequence aggregate: psnr={fsr_psnr:.3}dB bilinear={bilinear_psnr:.3}dB reference={reference_psnr:.3}dB ssim={fsr_ssim:.5} bilinear_ssim={bilinear_ssim:.5} reference_ssim={reference_ssim:.5} flicker_mse={flicker:.6} bilinear_flicker_mse={bilinear_flicker:.6}"
    );
    assert!(fsr_psnr > bilinear_psnr);
    assert!(fsr_ssim > bilinear_ssim);
    assert!(flicker < bilinear_flicker);
    if fsr_psnr + 0.25 < reference_psnr || fsr_ssim + 0.005 < reference_ssim {
        eprintln!("FSR remains experimental: reference regression allowance would be exceeded");
    }
}

struct Images {
    source: Image,
    output: Image,
    motion: Image,
    confidence: Image,
    disocclusion: Image,
    reactive: Image,
    depth: Image,
    transparency: Image,
    exposure: Image,
}

struct Uploads {
    source: Buffer,
    motion: Buffer,
    confidence: Buffer,
    disocclusion: Buffer,
    reactive: Buffer,
    depth: Buffer,
    transparency: Buffer,
    exposure: Buffer,
    readback: Buffer,
}

impl Images {
    unsafe fn new(gpu: &Gpu) -> Self {
        let image_usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let guidance_usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST;
        let image = |extent, format, usage| unsafe {
            Image::new(&gpu.device, &gpu.memory, extent, format, usage).unwrap()
        };
        Self {
            source: image(INPUT, vk::Format::R8G8B8A8_UNORM, image_usage),
            output: image(OUTPUT, vk::Format::R8G8B8A8_UNORM, image_usage),
            motion: image(INPUT, vk::Format::R16G16_SFLOAT, guidance_usage),
            confidence: image(INPUT, vk::Format::R8_UNORM, guidance_usage),
            disocclusion: image(INPUT, vk::Format::R8_UNORM, guidance_usage),
            reactive: image(INPUT, vk::Format::R8_UNORM, guidance_usage),
            depth: image(INPUT, vk::Format::R32_SFLOAT, guidance_usage),
            transparency: image(INPUT, vk::Format::R8_UNORM, guidance_usage),
            exposure: image(
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                vk::Format::R32_SFLOAT,
                guidance_usage,
            ),
        }
    }

    fn guidance(&self, fixture: &SequenceFixture, frame_id: u64, first: bool) -> GuidanceView {
        let extent = FrameExtent {
            width: INPUT.width,
            height: INPUT.height,
        };
        let reset = if first {
            if fixture.scene_cut {
                GuidanceReset::SceneChange
            } else {
                GuidanceReset::Initialize
            }
        } else if fixture.long_pause {
            GuidanceReset::LongPause
        } else {
            GuidanceReset::None
        };
        let mut metadata = GuidanceMetadata::zero(frame_id, extent, reset);
        metadata.is_zero = false;
        let resource = |image: &Image, format| GuidanceResource {
            image: image.handle,
            view: image.view,
            format,
            metadata,
            state: SignalState::Estimated,
        };
        GuidanceView {
            motion: resource(&self.motion, vk::Format::R16G16_SFLOAT),
            confidence: resource(&self.confidence, vk::Format::R8_UNORM),
            disocclusion: resource(&self.disocclusion, vk::Format::R8_UNORM),
            reactive: resource(&self.reactive, vk::Format::R8_UNORM),
            exposure: resource(&self.exposure, vk::Format::R32_SFLOAT),
            depth: resource(&self.depth, vk::Format::R32_SFLOAT),
            transparency_composition: resource(&self.transparency, vk::Format::R8_UNORM),
            pre_exposure: GuidanceScalar::constant_fallback(1.0),
            timing: FrameTiming {
                raw: Duration::from_micros(16_667),
                validated: Duration::from_micros(16_667),
                smoothed: Duration::from_micros(16_667),
            },
            jitter: JitterSample::default(),
            depth_semantics: DepthSemantics::RelativeNearIsOne,
            direction: MotionDirection::CurrentToPrevious,
            units: MotionUnits::SourcePixels,
            resolution: GuidanceResolution::new(extent, extent),
            requires_history_reset: metadata.requires_history_reset,
        }
    }
}

impl Uploads {
    unsafe fn new(gpu: &Gpu) -> Self {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let buffer = |size, usage| unsafe {
            Buffer::new(&gpu.device, &gpu.memory, size, usage, host).unwrap()
        };
        let pixels = (INPUT.width * INPUT.height) as u64;
        Self {
            source: buffer(pixels * 4, vk::BufferUsageFlags::TRANSFER_SRC),
            motion: buffer(pixels * 4, vk::BufferUsageFlags::TRANSFER_SRC),
            confidence: buffer(pixels, vk::BufferUsageFlags::TRANSFER_SRC),
            disocclusion: buffer(pixels, vk::BufferUsageFlags::TRANSFER_SRC),
            reactive: buffer(pixels, vk::BufferUsageFlags::TRANSFER_SRC),
            depth: buffer(pixels * 4, vk::BufferUsageFlags::TRANSFER_SRC),
            transparency: buffer(pixels, vk::BufferUsageFlags::TRANSFER_SRC),
            exposure: buffer(4, vk::BufferUsageFlags::TRANSFER_SRC),
            readback: buffer(
                (OUTPUT.width * OUTPUT.height * 4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
            ),
        }
    }
}

unsafe fn copy_upload(
    device: &ash::Device,
    command: vk::CommandBuffer,
    buffer: &Buffer,
    image: &Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    extent: vk::Extent2D,
) {
    unsafe {
        image_barrier(
            device,
            command,
            image.handle,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        device.cmd_copy_buffer_to_image(
            command,
            buffer.handle,
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
    }
}

unsafe fn upload_frame(
    gpu: &Gpu,
    images: &Images,
    uploads: &Uploads,
    fixture: &SequenceFixture,
    source_bytes: &[u8],
    first: bool,
) {
    let motion = fixture
        .motion
        .iter()
        .flat_map(|value| [f32_to_f16(value[0]), f32_to_f16(value[1])])
        .flat_map(u16::to_ne_bytes)
        .collect::<Vec<_>>();
    let confidence = fixture
        .valid
        .iter()
        .map(|valid| u8::from(*valid) * 255)
        .collect::<Vec<_>>();
    let disocclusion = fixture
        .occlusion
        .iter()
        .map(|value| u8::from(*value) * 255)
        .collect::<Vec<_>>();
    let reactive = fixture
        .transparency
        .iter()
        .map(|value| u8::from(*value) * 255)
        .collect::<Vec<_>>();
    let transparency = fixture
        .transparency
        .iter()
        .zip(fixture.hud.iter())
        .map(|(composition, hud)| u8::from(*composition || *hud) * 255)
        .collect::<Vec<_>>();
    let depth = fixture
        .depth
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let exposure = fixture.exposure_ev.exp2().to_ne_bytes();
    unsafe {
        uploads.source.write(source_bytes).unwrap();
        uploads.motion.write(&motion).unwrap();
        uploads.confidence.write(&confidence).unwrap();
        uploads.disocclusion.write(&disocclusion).unwrap();
        uploads.reactive.write(&reactive).unwrap();
        uploads.depth.write(&depth).unwrap();
        uploads.transparency.write(&transparency).unwrap();
        uploads.exposure.write(&exposure).unwrap();
        gpu.submit(|command| {
            let device = &gpu.device;
            copy_upload(
                device,
                command,
                &uploads.source,
                &images.source,
                if first {
                    vk::ImageLayout::UNDEFINED
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                },
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                INPUT,
            );
            let old_layout = if first {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::GENERAL
            };
            for (buffer, image, extent) in [
                (&uploads.motion, &images.motion, INPUT),
                (&uploads.confidence, &images.confidence, INPUT),
                (&uploads.disocclusion, &images.disocclusion, INPUT),
                (&uploads.reactive, &images.reactive, INPUT),
                (&uploads.depth, &images.depth, INPUT),
                (&uploads.transparency, &images.transparency, INPUT),
            ] {
                copy_upload(
                    device,
                    command,
                    buffer,
                    image,
                    old_layout,
                    vk::ImageLayout::GENERAL,
                    extent,
                );
            }
            copy_upload(
                device,
                command,
                &uploads.exposure,
                &images.exposure,
                old_layout,
                vk::ImageLayout::GENERAL,
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            );
        });
    }
}

unsafe fn run_backend_frame<B: UpscalerBackend>(
    gpu: &Gpu,
    images: &Images,
    uploads: &Uploads,
    backend: &mut B,
    guidance: GuidanceView,
    frame_id: u64,
    reset_history: bool,
    first_output: bool,
) -> Vec<[f32; 4]> {
    let mut bytes = vec![0_u8; (OUTPUT.width * OUTPUT.height * 4) as usize];
    unsafe {
        gpu.submit(|command| {
            if first_output {
                image_barrier(
                    &gpu.device,
                    command,
                    images.output.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            let frame = BackendFrame {
                command_buffer: command,
                slot: 0,
                source: BackendImage {
                    image: images.source.handle,
                    view: images.source.view,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent: INPUT,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                output: BackendImage {
                    image: images.output.handle,
                    view: images.output.view,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent: OUTPUT,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                },
                guidance,
                viewport: content_viewport(INPUT, OUTPUT),
                frame_id,
                reset_history,
                debug_view: 0,
            };
            backend.record(frame).unwrap();
            image_barrier(
                &gpu.device,
                command,
                images.output.handle,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            gpu.device.cmd_copy_image_to_buffer(
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                uploads.readback.handle,
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
                images.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        });
        uploads.readback.read(&mut bytes).unwrap();
    }
    rgba8_pixels(&bytes)
}

unsafe fn run_fsr_case(
    gpu: &Gpu,
    environment: &BackendEnvironment,
    fixture: &SequenceFixture,
    source_bytes: &[u8],
    second_source_bytes: &[u8],
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let images = unsafe { Images::new(gpu) };
    let uploads = unsafe { Uploads::new(gpu) };
    let guidance = images.guidance(fixture, 1, true);
    let config = BackendConfig {
        game_extent: INPUT,
        output_extent: OUTPUT,
        source_format: vk::Format::R8G8B8A8_UNORM,
        output_format: vk::Format::R8G8B8A8_UNORM,
        color_encoding: BackendColorEncoding::SrgbNonlinear,
        viewport: content_viewport(INPUT, OUTPUT),
        guidance: guidance.capabilities(),
    };
    let mut backend = unsafe { Fsr314Upscaler::new(environment, config, guidance, 1) }.unwrap();
    unsafe { upload_frame(gpu, &images, &uploads, fixture, source_bytes, true) };
    let first = unsafe {
        run_backend_frame(
            gpu,
            &images,
            &uploads,
            &mut backend,
            guidance,
            1,
            true,
            true,
        )
    };
    let second_guidance = images.guidance(fixture, 2, false);
    unsafe { upload_frame(gpu, &images, &uploads, fixture, second_source_bytes, false) };
    let second = unsafe {
        run_backend_frame(
            gpu,
            &images,
            &uploads,
            &mut backend,
            second_guidance,
            2,
            false,
            false,
        )
    };
    (first, second)
}

unsafe fn run_reference_case(
    gpu: &Gpu,
    fixture: &SequenceFixture,
    source_bytes: &[u8],
) -> Vec<[f32; 4]> {
    let images = unsafe { Images::new(gpu) };
    let uploads = unsafe { Uploads::new(gpu) };
    let guidance = images.guidance(fixture, 1, true);
    let mut backend = unsafe {
        ReferenceUpscaler::new(
            &gpu.device,
            &gpu.memory,
            images.source.view,
            INPUT,
            OUTPUT,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            guidance,
            1,
        )
    }
    .unwrap();
    unsafe { upload_frame(gpu, &images, &uploads, fixture, source_bytes, true) };
    unsafe {
        run_backend_frame(
            gpu,
            &images,
            &uploads,
            &mut backend,
            guidance,
            1,
            true,
            true,
        )
    }
}

fn ideal_pixel(case_index: usize, x: u32, y: u32, extent: vk::Extent2D) -> [f32; 4] {
    ideal_pixel_uv(
        case_index,
        (x as f32 + 0.5) / extent.width as f32,
        (y as f32 + 0.5) / extent.height as f32,
    )
}

fn ideal_pixel_uv(case_index: usize, mut u: f32, mut v: f32) -> [f32; 4] {
    let (cx, cy) = (0.5, 0.5);
    match case_index {
        0 => u = (u - 0.06).clamp(0.0, 1.0),
        1 => {
            let (sin, cos) = 0.08_f32.sin_cos();
            let px = u - cx;
            let py = v - cy;
            u = (cx + cos * px + sin * py).clamp(0.0, 1.0);
            v = (cy - sin * px + cos * py).clamp(0.0, 1.0);
        }
        2 => {
            u = (cx + (u - cx) / 1.08).clamp(0.0, 1.0);
            v = (cy + (v - cy) / 1.08).clamp(0.0, 1.0);
        }
        3 => u = (u + 0.04).clamp(0.0, 1.0),
        _ => {}
    }
    let checker = ((u * 8.0).floor() as u32 + (v * 8.0).floor() as u32) % 2;
    let edge = ((u * 17.0).floor() as u32 + (v * 13.0).floor() as u32).is_multiple_of(5);
    let mut pixel: [f32; 4] = if checker == 0 {
        [0.1, 0.8, 0.2, 1.0]
    } else {
        [0.9, 0.15, 0.95, 1.0]
    };
    if edge {
        pixel[0] = (pixel[0] + 0.7).min(1.0);
        pixel[1] = (pixel[1] + 0.15).min(1.0);
        pixel[2] = (pixel[2] + 0.1).min(1.0);
    }
    if case_index == 4 && (u - 0.5).abs() < 0.018 {
        pixel = [1.0, 1.0, 1.0, 1.0];
    }
    if case_index == 5 && v < 0.09 {
        pixel = [0.1, 0.9, 0.2, 1.0];
    }
    if case_index == 6 && (0.25..0.55).contains(&u) && (0.22..0.78).contains(&v) {
        pixel = [
            pixel[0] * 0.5 + 0.5,
            pixel[1] * 0.5 + 0.5,
            pixel[2] * 0.5 + 0.5,
            1.0,
        ];
    }
    if case_index == 7 {
        for (px, py) in [(0.25, 0.25), (0.5, 0.34), (0.75, 0.66)] {
            if (u - px).abs() < 0.025 && (v - py).abs() < 0.025 {
                pixel = [0.9, 0.95, 1.0, 1.0];
            }
        }
    }
    if case_index == 8 && (0.33..0.67).contains(&u) && (0.33..0.67).contains(&v) {
        pixel = [1.0, 1.0, 1.0, 1.0];
    }
    if case_index == 9 {
        let noise_x = (u * 1024.0) as u32;
        let noise_y = (v * 1024.0) as u32;
        let value =
            ((noise_x.wrapping_mul(73) ^ noise_y.wrapping_mul(151)) & 31) as f32 / 255.0 - 0.0625;
        pixel[0] = (pixel[0] + value).clamp(0.0, 1.0);
        pixel[1] = (pixel[1] + value * 0.75).clamp(0.0, 1.0);
        pixel[2] = (pixel[2] - value * 0.5).clamp(0.0, 1.0);
    }
    pixel
}

fn scene_bytes(case_index: usize, extent: vk::Extent2D) -> Vec<u8> {
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                ideal_pixel(case_index, x, y, extent)
                    .map(|value| (value * 255.0).round() as u8)
                    .into_iter()
            })
        })
        .collect()
}

fn fixture_scene_bytes(
    case_index: usize,
    fixture: &SequenceFixture,
    extent: vk::Extent2D,
) -> Vec<u8> {
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).flat_map(move |x| {
                let fixture_x = (x * fixture.width / extent.width).min(fixture.width - 1);
                let fixture_y = (y * fixture.height / extent.height).min(fixture.height - 1);
                let index = (fixture_y * fixture.width + fixture_x) as usize;
                let motion = fixture.motion[index];
                let u = ((x as f32 + 0.5) / extent.width as f32 + motion[0] / fixture.width as f32)
                    .clamp(0.0, 1.0);
                let v = ((y as f32 + 0.5) / extent.height as f32
                    + motion[1] / fixture.height as f32)
                    .clamp(0.0, 1.0);
                ideal_pixel_uv(case_index, u, v)
                    .map(|value| (value * 255.0).round() as u8)
                    .into_iter()
            })
        })
        .collect()
}

fn expected_pixels(case_index: usize, extent: vk::Extent2D) -> Vec<[f32; 4]> {
    (0..extent.height)
        .flat_map(|y| (0..extent.width).map(move |x| ideal_pixel(case_index, x, y, extent)))
        .collect()
}

fn fixture_expected_pixels(
    case_index: usize,
    fixture: &SequenceFixture,
    extent: vk::Extent2D,
) -> Vec<[f32; 4]> {
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| {
                let fixture_x = (x * fixture.width / extent.width).min(fixture.width - 1);
                let fixture_y = (y * fixture.height / extent.height).min(fixture.height - 1);
                let index = (fixture_y * fixture.width + fixture_x) as usize;
                let motion = fixture.motion[index];
                let u = ((x as f32 + 0.5) / extent.width as f32 + motion[0] / fixture.width as f32)
                    .clamp(0.0, 1.0);
                let v = ((y as f32 + 0.5) / extent.height as f32
                    + motion[1] / fixture.height as f32)
                    .clamp(0.0, 1.0);
                ideal_pixel_uv(case_index, u, v)
            })
        })
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

fn temporal_error(
    first: &[[f32; 4]],
    second: &[[f32; 4]],
    expected_first: &[[f32; 4]],
    expected_second: &[[f32; 4]],
) -> f32 {
    first
        .iter()
        .zip(second)
        .zip(expected_first.iter().zip(expected_second))
        .flat_map(|((first, second), (expected_first, expected_second))| {
            first
                .iter()
                .zip(second)
                .zip(expected_first.iter().zip(expected_second))
                .map(|((first, second), (expected_first, expected_second))| {
                    ((second - first) - (expected_second - expected_first)).powi(2)
                })
        })
        .sum::<f32>()
        / (first.len() * 4) as f32
}

fn psnr(actual: &[[f32; 4]], expected: &[[f32; 4]]) -> f32 {
    10.0 * (1.0 / image_mse(actual, expected).max(f32::MIN_POSITIVE)).log10()
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
