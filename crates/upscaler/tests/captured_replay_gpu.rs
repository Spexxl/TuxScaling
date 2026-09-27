#![cfg(feature = "fidelityfx")]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use ash::vk;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, process::Command, time::Duration};
use tuxscaling_motion::{MotionEstimator, MotionQuality};
use tuxscaling_temporal::{FrameTiming, GuidanceAblations, GuidanceReset};
use tuxscaling_upscaler::fidelityfx::Fsr314Upscaler;
use tuxscaling_upscaler::{
    BackendColorEncoding, BackendConfig, BackendEnvironment, BackendFrame, BackendImage,
    OutputSharpening, UpscalerBackend, content_viewport,
};
use tuxscaling_vulkan::{Buffer, Image, image_barrier};

#[path = "../../../tests/support/gpu.rs"]
mod support;
use support::Gpu;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ReplayFrame {
    frame_id: u64,
    timestamp_ns: u64,
    slot_index: u32,
    motion_slot_index: u32,
    source_file: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct ReplayManifest {
    schema_version: u32,
    sequence: ReplaySequence,
    variants: Vec<ReplayVariant>,
}

#[derive(Debug, Deserialize)]
struct ReplaySequence {
    id: String,
    source_extent: [u32; 2],
    source_format: String,
    numeric_encoding: String,
    output_extent: [u32; 2],
    content_viewport: [u32; 4],
    slot_count: u32,
    reset_frame_ids: Vec<u64>,
    frames: Vec<ReplayFrame>,
}

#[derive(Debug, Deserialize)]
struct ReplayVariant {
    name: String,
    backend: String,
    guidance: String,
    settings: ReplaySettings,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplaySettings {
    jitter: bool,
    sharpening: f32,
    motion_quality: String,
}

impl Default for ReplaySettings {
    fn default() -> Self {
        Self {
            jitter: false,
            sharpening: 0.0,
            motion_quality: "balanced".into(),
        }
    }
}

#[derive(Serialize)]
struct ReplayReport {
    schema_version: u32,
    manifest_sha256: String,
    implementation_verified: bool,
    quality_accepted: Option<bool>,
    variants: Vec<ReplayVariantReport>,
}

#[derive(Serialize)]
struct ReplayVariantReport {
    name: String,
    backend: String,
    guidance: String,
    settings: ReplaySettings,
    consumed_frames: Vec<ReplayFrame>,
    metrics: BTreeMap<String, f64>,
}

struct ReplayImages {
    current: Image,
    previous: Image,
    output: Image,
    upload_current: Buffer,
    upload_previous: Buffer,
    readback: Buffer,
}

fn source_format(name: &str) -> Result<vk::Format, String> {
    match name {
        "R8G8B8A8_UNORM" => Ok(vk::Format::R8G8B8A8_UNORM),
        "B8G8R8A8_UNORM" => Ok(vk::Format::B8G8R8A8_UNORM),
        "R8G8B8A8_SRGB" => Ok(vk::Format::R8G8B8A8_SRGB),
        "B8G8R8A8_SRGB" => Ok(vk::Format::B8G8R8A8_SRGB),
        other => Err(format!(
            "captured replay does not support source format {other}"
        )),
    }
}

fn checked_rgba_bytes(extent: [u32; 2]) -> Result<usize, String> {
    if extent[0] < 32 || extent[1] < 32 {
        return Err("GPU replay input dimensions must be at least 32x32".into());
    }
    (extent[0] as usize)
        .checked_mul(extent[1] as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "GPU replay image size overflows addressable memory".into())
}

fn sha256(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|error| format!("run sha256sum: {error}"))?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase())
}

fn aspect_fit_pixels(source: [u32; 2], output: [u32; 2]) -> [u32; 4] {
    let input_aspect = source[0] as f32 / source[1] as f32;
    let output_aspect = output[0] as f32 / output[1] as f32;
    let size = if output_aspect > input_aspect {
        [input_aspect / output_aspect, 1.0]
    } else {
        [1.0, output_aspect / input_aspect]
    };
    let offset = [(1.0 - size[0]) * 0.5, (1.0 - size[1]) * 0.5];
    let left = (offset[0] * output[0] as f32).round() as u32;
    let top = (offset[1] * output[1] as f32).round() as u32;
    let right = ((offset[0] + size[0]) * output[0] as f32).round() as u32;
    let bottom = ((offset[1] + size[1]) * output[1] as f32).round() as u32;
    [left, top, right - left, bottom - top]
}

fn read_manifest(path: &Path) -> Result<(ReplayManifest, String), String> {
    let path = path
        .canonicalize()
        .map_err(|error| format!("open replay manifest {}: {error}", path.display()))?;
    let bytes = fs::read(&path).map_err(|error| format!("read replay manifest: {error}"))?;
    let manifest: ReplayManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse replay manifest: {error}"))?;
    if manifest.schema_version != 1 {
        return Err("unsupported replay manifest schema_version".into());
    }
    if manifest.sequence.numeric_encoding != "srgb_nonlinear" {
        return Err("captured replay worker currently requires srgb_nonlinear input".into());
    }
    if manifest.sequence.id.trim().is_empty() {
        return Err("replay sequence id must not be empty".into());
    }
    if manifest.sequence.slot_count == 0 {
        return Err("replay sequence slot_count must be nonzero".into());
    }
    source_format(&manifest.sequence.source_format)?;
    let expected_size = checked_rgba_bytes(manifest.sequence.source_extent)?;
    if manifest.sequence.output_extent.contains(&0)
        || manifest.sequence.content_viewport
            != aspect_fit_pixels(
                manifest.sequence.source_extent,
                manifest.sequence.output_extent,
            )
    {
        return Err("captured replay content_viewport does not match aspect-fit output".into());
    }
    if manifest.sequence.frames.len() < 2 || manifest.variants.is_empty() {
        return Err("GPU replay requires at least two frames and one variant".into());
    }
    let root = path
        .parent()
        .ok_or_else(|| "replay manifest has no parent".to_owned())?;
    let mut previous_id = None;
    let mut previous_time = None;
    let mut frame_ids = std::collections::BTreeSet::new();
    for frame in &manifest.sequence.frames {
        if frame.slot_index >= manifest.sequence.slot_count {
            return Err(format!(
                "replay frame {} has a slot_index outside slot_count",
                frame.frame_id
            ));
        }
        if frame.motion_slot_index >= 2 {
            return Err(format!(
                "replay frame {} has a motion_slot_index outside the two-slot history",
                frame.frame_id
            ));
        }
        if previous_id.is_some_and(|id| frame.frame_id <= id)
            || previous_time.is_some_and(|time| frame.timestamp_ns <= time)
        {
            return Err("replay frame IDs and timestamps must increase".into());
        }
        previous_id = Some(frame.frame_id);
        previous_time = Some(frame.timestamp_ns);
        frame_ids.insert(frame.frame_id);
        let source = Path::new(&frame.source_file);
        if source.is_absolute()
            || source
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err("replay source paths must stay inside the manifest directory".into());
        }
        let source = root
            .join(source)
            .canonicalize()
            .map_err(|error| format!("missing source frame {}: {error}", frame.frame_id))?;
        if !source.starts_with(root) {
            return Err(format!(
                "source frame {} escapes the manifest directory",
                frame.frame_id
            ));
        }
        if fs::metadata(&source)
            .map_err(|error| error.to_string())?
            .len()
            != expected_size as u64
        {
            return Err(format!(
                "source frame {} has an incorrect byte count",
                frame.frame_id
            ));
        }
        if sha256(&source)? != frame.sha256.to_ascii_lowercase() {
            return Err(format!("source hash mismatch for frame {}", frame.frame_id));
        }
    }
    if !manifest
        .sequence
        .reset_frame_ids
        .contains(&manifest.sequence.frames[0].frame_id)
    {
        return Err("the first frame must reset history".into());
    }
    if manifest
        .sequence
        .reset_frame_ids
        .iter()
        .any(|frame_id| !frame_ids.contains(frame_id))
    {
        return Err("reset_frame_ids references a missing frame".into());
    }
    let mut variant_names = std::collections::BTreeSet::new();
    for variant in &manifest.variants {
        if variant.name.is_empty()
            || !variant_names.insert(variant.name.as_str())
            || variant
                .name
                .bytes()
                .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')))
        {
            return Err("variant names may contain only ASCII letters, digits, '_' and '-'".into());
        }
        if variant.settings.jitter {
            return Err("experimental jitter is not supported by stable captured replay".into());
        }
        if !variant.settings.sharpening.is_finite()
            || !(0.0..=1.0).contains(&variant.settings.sharpening)
        {
            return Err("variant sharpening must be finite and in [0, 1]".into());
        }
        if !matches!(
            variant.settings.motion_quality.as_str(),
            "ultra" | "high" | "balanced" | "performance"
        ) {
            return Err("variant motion_quality is unsupported".into());
        }
        if !matches!(variant.backend.as_str(), "off" | "fsr_3_1_4")
            || !matches!(variant.guidance.as_str(), "none" | "zero" | "estimated")
            || (variant.backend == "off" && variant.guidance != "none")
            || (variant.backend == "fsr_3_1_4" && variant.guidance == "none")
        {
            return Err(format!(
                "variant {} has an unsupported backend/guidance pair",
                variant.name
            ));
        }
    }
    let manifest_hash = sha256(&path)?;
    Ok((manifest, manifest_hash))
}

unsafe fn create_images(
    gpu: &Gpu,
    source_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    source_format: vk::Format,
) -> Result<ReplayImages, String> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let input_size = u64::from(source_extent.width) * u64::from(source_extent.height) * 4;
    let output_size = u64::from(output_extent.width) * u64::from(output_extent.height) * 4;
    let image_usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::COLOR_ATTACHMENT;
    let buffer = |size, usage| unsafe {
        Buffer::new(&gpu.device, &gpu.memory, size, usage, host)
            .map_err(|error| format!("allocate replay buffer: {error:?}"))
    };
    let image = |extent, format, usage| unsafe {
        Image::new(&gpu.device, &gpu.memory, extent, format, usage)
            .map_err(|error| format!("allocate replay image: {error:?}"))
    };
    Ok(ReplayImages {
        current: image(source_extent, source_format, image_usage)?,
        previous: image(source_extent, source_format, image_usage)?,
        output: image(output_extent, vk::Format::R8G8B8A8_UNORM, image_usage)?,
        upload_current: buffer(input_size, vk::BufferUsageFlags::TRANSFER_SRC)?,
        upload_previous: buffer(input_size, vk::BufferUsageFlags::TRANSFER_SRC)?,
        readback: buffer(output_size, vk::BufferUsageFlags::TRANSFER_DST)?,
    })
}

unsafe fn upload_source(
    gpu: &Gpu,
    image: &Image,
    staging: &Buffer,
    bytes: &[u8],
    old_layout: vk::ImageLayout,
    extent: vk::Extent2D,
) {
    unsafe { staging.write(bytes) }.unwrap();
    unsafe {
        gpu.submit(|command| {
            image_barrier(
                &gpu.device,
                command,
                image.handle,
                old_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            gpu.device.cmd_copy_buffer_to_image(
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
                &gpu.device,
                command,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });
    }
}

fn frame_timing(frames: &[ReplayFrame], index: usize) -> FrameTiming {
    let nominal_ns = Duration::from_micros(16_667).as_nanos() as u64;
    let minimum_ns = Duration::from_micros(250).as_nanos() as u64;
    let maximum_ns = Duration::from_millis(250).as_nanos() as u64;
    let mut last_timestamp = None;
    let mut last_validated_ns = nominal_ns;
    let mut smoothed_ns = nominal_ns;
    for (current_index, frame) in frames.iter().enumerate().take(index + 1) {
        let raw_ns = last_timestamp
            .map(|previous| frame.timestamp_ns.saturating_sub(previous))
            .unwrap_or(nominal_ns);
        last_timestamp = Some(frame.timestamp_ns);
        if (minimum_ns..=maximum_ns).contains(&raw_ns) {
            last_validated_ns = raw_ns;
        }
        smoothed_ns = ((u128::from(smoothed_ns) * 9 + u128::from(last_validated_ns)) / 10) as u64;
        if current_index == index {
            return FrameTiming {
                raw: Duration::from_nanos(raw_ns),
                validated: Duration::from_nanos(last_validated_ns),
                smoothed: Duration::from_nanos(smoothed_ns),
            };
        }
    }
    FrameTiming::default()
}

#[test]
fn replay_frame_timing_matches_runtime_delta_validation_and_ema() {
    let frames = [
        ReplayFrame {
            frame_id: 1,
            timestamp_ns: 10_000_000,
            slot_index: 0,
            motion_slot_index: 0,
            source_file: "a.rgba".into(),
            sha256: "a".repeat(64),
        },
        ReplayFrame {
            frame_id: 2,
            timestamp_ns: 26_666_667,
            slot_index: 0,
            motion_slot_index: 1,
            source_file: "b.rgba".into(),
            sha256: "b".repeat(64),
        },
        ReplayFrame {
            frame_id: 3,
            timestamp_ns: 26_666_767,
            slot_index: 0,
            motion_slot_index: 0,
            source_file: "c.rgba".into(),
            sha256: "c".repeat(64),
        },
    ];

    assert_eq!(frame_timing(&frames, 0).raw, Duration::from_micros(16_667));
    assert_eq!(
        frame_timing(&frames, 1),
        FrameTiming {
            raw: Duration::from_nanos(16_666_667),
            validated: Duration::from_nanos(16_666_667),
            smoothed: Duration::from_nanos(16_666_966),
        }
    );
    assert_eq!(
        frame_timing(&frames, 2),
        FrameTiming {
            raw: Duration::from_nanos(100),
            validated: Duration::from_nanos(16_666_667),
            smoothed: Duration::from_nanos(16_666_936),
        }
    );
}

fn pixel_metrics(bytes: &[u8], previous: Option<&[u8]>) -> (f64, f64) {
    let luma = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            (f64::from(pixel[0]) * 0.2126
                + f64::from(pixel[1]) * 0.7152
                + f64::from(pixel[2]) * 0.0722)
                / 255.0
        })
        .sum::<f64>()
        / (bytes.len() / 4) as f64;
    let delta = previous.map_or(0.0, |previous| {
        bytes
            .iter()
            .zip(previous)
            .map(|(current, previous)| {
                let difference = f64::from(*current) - f64::from(*previous);
                difference * difference
            })
            .sum::<f64>()
            / (bytes.len().max(1) as f64 * 255.0 * 255.0)
    });
    (luma, delta)
}

unsafe fn run_off_frame(
    gpu: &Gpu,
    images: &ReplayImages,
    extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    old_output_layout: vk::ImageLayout,
) -> Vec<u8> {
    let mut bytes =
        vec![0; (u64::from(output_extent.width) * u64::from(output_extent.height) * 4) as usize];
    unsafe {
        gpu.submit(|command| {
            let viewport = content_viewport(extent, output_extent);
            let left = (viewport.offset[0] * output_extent.width as f32).round() as i32;
            let top = (viewport.offset[1] * output_extent.height as f32).round() as i32;
            let right = ((viewport.offset[0] + viewport.size[0]) * output_extent.width as f32)
                .round() as i32;
            let bottom = ((viewport.offset[1] + viewport.size[1]) * output_extent.height as f32)
                .round() as i32;
            image_barrier(
                &gpu.device,
                command,
                images.current.handle,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                &gpu.device,
                command,
                images.output.handle,
                old_output_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            gpu.device.cmd_clear_color_image(
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)],
            );
            gpu.device.cmd_blit_image(
                command,
                images.current.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                images.output.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: extent.width as i32,
                            y: extent.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D {
                            x: left,
                            y: top,
                            z: 0,
                        },
                        vk::Offset3D {
                            x: right,
                            y: bottom,
                            z: 1,
                        },
                    ])],
                vk::Filter::LINEAR,
            );
            image_barrier(
                &gpu.device,
                command,
                images.current.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                &gpu.device,
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            gpu.device.cmd_copy_image_to_buffer(
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                images.readback.handle,
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
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            gpu.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            image_barrier(
                &gpu.device,
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
        });
        images.readback.read(&mut bytes).unwrap();
    }
    bytes
}

unsafe fn run_fsr_frame(
    gpu: &Gpu,
    images: &ReplayImages,
    motion: &mut MotionEstimator,
    guidance: &mut tuxscaling_temporal::GuidanceEstimator,
    backend: &mut Fsr314Upscaler,
    source_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    frame: &ReplayFrame,
    frame_index: usize,
    reset: bool,
    zero_guidance: bool,
    sharpening: f32,
    timing: FrameTiming,
    old_output_layout: vk::ImageLayout,
) -> Vec<u8> {
    let mut bytes =
        vec![0; (u64::from(output_extent.width) * u64::from(output_extent.height) * 4) as usize];
    unsafe {
        gpu.submit(|command| {
            if old_output_layout == vk::ImageLayout::UNDEFINED {
                image_barrier(
                    &gpu.device,
                    command,
                    images.output.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
            }
            let motion_slot = frame.motion_slot_index as usize;
            let slot = frame.slot_index as usize;
            let reset_reason = if frame_index == 0 {
                GuidanceReset::Initialize
            } else if reset {
                GuidanceReset::SceneChange
            } else {
                GuidanceReset::None
            };
            let guidance_view = if zero_guidance {
                guidance.view_with_controls(
                    motion,
                    frame.frame_id,
                    source_extent,
                    !reset,
                    timing,
                    reset_reason,
                    true,
                    GuidanceAblations::NONE,
                )
            } else {
                guidance.view(
                    motion,
                    frame.frame_id,
                    source_extent,
                    !reset,
                    timing,
                    reset_reason,
                )
            };
            motion.record(command, motion_slot, !reset, 0);
            if zero_guidance {
                guidance.record_with_ablation_for_slot(
                    command,
                    !reset,
                    timing,
                    slot,
                    true,
                    GuidanceAblations::NONE,
                );
            } else {
                guidance.record_with_timing_for_slot(command, !reset, timing, slot);
            }
            let backend_frame = BackendFrame {
                command_buffer: command,
                slot,
                source: BackendImage {
                    image: images.current.handle,
                    view: images.current.view,
                    format: images.current.format,
                    extent: source_extent,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                output: BackendImage {
                    image: images.output.handle,
                    view: images.output.view,
                    format: vk::Format::R8G8B8A8_UNORM,
                    extent: output_extent,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                },
                guidance: guidance_view,
                viewport: content_viewport(source_extent, output_extent),
                output_sharpening: if sharpening > 0.0 {
                    OutputSharpening::new(true, sharpening)
                } else {
                    OutputSharpening::disabled()
                },
                frame_id: frame.frame_id,
                reset_history: reset,
                debug_view: 0,
            };
            backend.record(backend_frame).unwrap();
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
                images.readback.handle,
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
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            gpu.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            image_barrier(
                &gpu.device,
                command,
                images.output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        });
        images.readback.read(&mut bytes).unwrap();
    }
    bytes
}

unsafe fn run_variant(
    gpu: &Gpu,
    manifest_root: &Path,
    manifest: &ReplayManifest,
    variant: &ReplayVariant,
    output_dir: &Path,
) -> Result<ReplayVariantReport, String> {
    let sequence = &manifest.sequence;
    let source_extent = vk::Extent2D {
        width: sequence.source_extent[0],
        height: sequence.source_extent[1],
    };
    let output_extent = vk::Extent2D {
        width: sequence.output_extent[0],
        height: sequence.output_extent[1],
    };
    let source_format = source_format(&sequence.source_format)?;
    let source_size = checked_rgba_bytes(sequence.source_extent)?;
    let images = unsafe { create_images(gpu, source_extent, output_extent, source_format)? };
    let mut prior_frame_bytes = None::<Vec<u8>>;
    let mut prior_output = None::<Vec<u8>>;
    let mut output_luma_sum = 0.0;
    let mut temporal_delta_sum = 0.0;

    let mut motion = None;
    let mut guidance = None;
    let mut backend = None;
    if variant.backend == "fsr_3_1_4" {
        let decode_srgb = sequence.numeric_encoding == "srgb_nonlinear"
            && !matches!(
                source_format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            );
        let mut estimator = unsafe {
            MotionEstimator::new(
                &gpu.device,
                &gpu.memory,
                source_extent,
                images.current.view,
                decode_srgb,
            )
        }
        .map_err(|error| format!("create optical-flow estimator: {error:?}"))?;
        estimator.set_quality(match variant.settings.motion_quality.as_str() {
            "ultra" => MotionQuality::Ultra,
            "high" => MotionQuality::High,
            "balanced" => MotionQuality::Balanced,
            "performance" => MotionQuality::Performance,
            _ => return Err("unsupported motion quality preset".into()),
        });
        let guidance_estimator = unsafe {
            tuxscaling_temporal::GuidanceEstimator::new_with_slots(
                &gpu.device,
                &gpu.memory,
                source_extent,
                images.current.view,
                images.previous.view,
                estimator.confidence.view,
                estimator.vectors.view,
                estimator.metadata.handle,
                estimator.stats.handle,
                sequence.slot_count as usize,
            )
        }
        .map_err(|error| format!("create frame-guidance estimator: {error:?}"))?;
        let initial_view = guidance_estimator.view(
            &estimator,
            sequence.frames[0].frame_id,
            source_extent,
            false,
            frame_timing(&sequence.frames, 0),
            GuidanceReset::Initialize,
        );
        let config = BackendConfig {
            game_extent: source_extent,
            output_extent,
            source_format,
            output_format: vk::Format::R8G8B8A8_UNORM,
            color_encoding: BackendColorEncoding::SrgbNonlinear,
            viewport: content_viewport(source_extent, output_extent),
            guidance: initial_view.capabilities(),
        };
        let physical = unsafe { gpu.instance.enumerate_physical_devices() }
            .map_err(|error| format!("enumerate Vulkan devices: {error:?}"))?
            .into_iter()
            .next()
            .ok_or_else(|| "no Vulkan physical device available".to_owned())?;
        let environment = BackendEnvironment::new(&gpu.instance, physical, &gpu.device);
        let mut fsr = unsafe {
            Fsr314Upscaler::new(
                &environment,
                config,
                initial_view,
                sequence.slot_count as usize,
            )
        }
        .map_err(|error| format!("create FSR 3.1.4 backend: {error:?}"))?;
        fsr.configure(config)
            .map_err(|error| format!("configure FSR 3.1.4 backend: {error:?}"))?;
        motion = Some(estimator);
        guidance = Some(guidance_estimator);
        backend = Some(fsr);
    }

    for (index, frame) in sequence.frames.iter().enumerate() {
        let source_path = manifest_root.join(&frame.source_file);
        let source_bytes = fs::read(&source_path)
            .map_err(|error| format!("read source frame {}: {error}", frame.frame_id))?;
        if source_bytes.len() != source_size {
            return Err(format!(
                "source frame {} changed size after validation",
                frame.frame_id
            ));
        }
        let is_reset = sequence.reset_frame_ids.contains(&frame.frame_id);
        let previous_bytes = if index == 0 || is_reset {
            source_bytes.clone()
        } else {
            prior_frame_bytes
                .clone()
                .ok_or_else(|| "replay previous-frame state is missing".to_owned())?
        };
        let current_layout = if index == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let previous_layout = if index == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        unsafe {
            upload_source(
                gpu,
                &images.current,
                &images.upload_current,
                &source_bytes,
                current_layout,
                source_extent,
            );
            upload_source(
                gpu,
                &images.previous,
                &images.upload_previous,
                &previous_bytes,
                previous_layout,
                source_extent,
            );
        }
        let mut output = if variant.backend == "off" {
            let old_layout = if index == 0 {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::TRANSFER_DST_OPTIMAL
            };
            unsafe { run_off_frame(gpu, &images, source_extent, output_extent, old_layout) }
        } else {
            let motion = motion.as_mut().expect("FSR variant owns motion estimator");
            let guidance = guidance
                .as_mut()
                .expect("FSR variant owns guidance estimator");
            let backend = backend.as_mut().expect("FSR variant owns backend");
            let view_layout = if index == 0 {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            };
            unsafe {
                run_fsr_frame(
                    gpu,
                    &images,
                    motion,
                    guidance,
                    backend,
                    source_extent,
                    output_extent,
                    frame,
                    index,
                    is_reset,
                    variant.guidance == "zero",
                    variant.settings.sharpening,
                    frame_timing(&sequence.frames, index),
                    view_layout,
                )
            }
        };
        let (frame_luma, frame_delta) = pixel_metrics(&output, prior_output.as_deref());
        output_luma_sum += frame_luma;
        temporal_delta_sum += frame_delta;
        let frame_path =
            output_dir.join(format!("{}-frame-{:08}.rgba", variant.name, frame.frame_id));
        fs::write(&frame_path, &output).map_err(|error| format!("write replay frame: {error}"))?;
        let _ = frame_path;
        prior_frame_bytes = Some(source_bytes);
        prior_output = Some(std::mem::take(&mut output));
    }

    let frame_count = sequence.frames.len() as f64;
    let mut metrics = BTreeMap::new();
    metrics.insert("encoded_mean_luma".into(), output_luma_sum / frame_count);
    metrics.insert(
        "mean_temporal_delta_mse".into(),
        temporal_delta_sum / frame_count,
    );
    metrics.insert("output_frame_count".into(), frame_count);
    Ok(ReplayVariantReport {
        name: variant.name.clone(),
        backend: variant.backend.clone(),
        guidance: variant.guidance.clone(),
        settings: variant.settings.clone(),
        consumed_frames: sequence.frames.clone(),
        metrics,
    })
}

fn synthetic_manifest(root: &Path) -> ReplayManifest {
    let source_extent = [32, 32];
    let mut frames = Vec::new();
    for frame_id in 0..2_u64 {
        let mut bytes = Vec::with_capacity(32 * 32 * 4);
        for y in 0..32_u32 {
            for x in 0..32_u32 {
                let shift = frame_id as u32;
                let checker = ((x + shift) / 4 + y / 4) % 2;
                bytes.extend_from_slice(&[
                    if checker == 0 { 220 } else { 30 },
                    if checker == 0 { 80 } else { 190 },
                    if (x + shift) % 7 < 3 { 245 } else { 12 },
                    255,
                ]);
            }
        }
        let source_file = format!("synthetic-{frame_id}.rgba");
        fs::write(root.join(&source_file), &bytes).unwrap();
        let hash = sha256(&root.join(&source_file)).unwrap();
        frames.push(ReplayFrame {
            frame_id,
            timestamp_ns: frame_id * 16_666_667,
            slot_index: frame_id as u32 % 3,
            motion_slot_index: frame_id as u32 % 2,
            source_file,
            sha256: hash,
        });
    }
    ReplayManifest {
        schema_version: 1,
        sequence: ReplaySequence {
            id: "synthetic-pan".into(),
            source_extent,
            source_format: "R8G8B8A8_UNORM".into(),
            numeric_encoding: "srgb_nonlinear".into(),
            output_extent: [64, 64],
            content_viewport: [0, 0, 64, 64],
            slot_count: 3,
            reset_frame_ids: vec![0],
            frames,
        },
        variants: [
            ("fsr_estimated", "fsr_3_1_4", "estimated"),
            ("fsr_zero", "fsr_3_1_4", "zero"),
            ("off", "off", "none"),
        ]
        .into_iter()
        .map(|(name, backend, guidance)| ReplayVariant {
            name: name.into(),
            backend: backend.into(),
            guidance: guidance.into(),
            settings: ReplaySettings::default(),
        })
        .collect(),
    }
}

fn replay_from_paths(manifest_path: &Path, output_dir: &Path) -> Result<(), String> {
    let (manifest, manifest_hash) = read_manifest(manifest_path)?;
    fs::create_dir_all(output_dir).map_err(|error| format!("create replay output: {error}"))?;
    let manifest_root = manifest_path
        .canonicalize()
        .map_err(|error| format!("canonicalize manifest: {error}"))?
        .parent()
        .ok_or_else(|| "manifest has no parent directory".to_owned())?
        .to_path_buf();
    let gpu = unsafe { Gpu::new() };
    let mut reports = Vec::new();
    for variant in &manifest.variants {
        reports.push(unsafe { run_variant(&gpu, &manifest_root, &manifest, variant, output_dir) }?);
    }
    let report = ReplayReport {
        schema_version: 1,
        manifest_sha256: manifest_hash,
        implementation_verified: true,
        quality_accepted: None,
        variants: reports,
    };
    let report_path = output_dir.join("report.json");
    let report_bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize replay report: {error}"))?;
    fs::write(&report_path, report_bytes)
        .map_err(|error| format!("write replay report {}: {error}", report_path.display()))?;
    println!(
        "captured replay sequence={} variants={}",
        manifest.sequence.id,
        manifest.variants.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan GPU and the FidelityFX 3.1.4 provider"]
fn replay_manifest() {
    match (
        std::env::var_os("TUXSCALING_REPLAY_MANIFEST"),
        std::env::var_os("TUXSCALING_REPLAY_OUTPUT"),
    ) {
        (Some(manifest), Some(output)) => {
            replay_from_paths(Path::new(&manifest), Path::new(&output)).unwrap()
        }
        (None, None) => {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "tuxscaling-replay-fixture-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            let manifest = synthetic_manifest(&root);
            let manifest_path = root.join("manifest.json");
            fs::write(
                &manifest_path,
                serde_json::to_vec(&manifest_to_json(&manifest)).unwrap(),
            )
            .unwrap();
            let output = root.join("output");
            replay_from_paths(&manifest_path, &output).unwrap();
            fs::remove_dir_all(root).unwrap();
        }
        _ => panic!("both TUXSCALING_REPLAY_MANIFEST and TUXSCALING_REPLAY_OUTPUT are required"),
    }
}

fn manifest_to_json(manifest: &ReplayManifest) -> serde_json::Value {
    serde_json::json!({
        "schema_version": manifest.schema_version,
        "sequence": {
            "id": manifest.sequence.id,
            "source_extent": manifest.sequence.source_extent,
            "source_format": manifest.sequence.source_format,
            "numeric_encoding": manifest.sequence.numeric_encoding,
            "output_extent": manifest.sequence.output_extent,
            "content_viewport": manifest.sequence.content_viewport,
            "slot_count": manifest.sequence.slot_count,
            "reset_frame_ids": manifest.sequence.reset_frame_ids,
            "frames": manifest.sequence.frames,
        },
        "variants": manifest.variants.iter().map(|variant| serde_json::json!({
            "name": variant.name,
            "backend": variant.backend,
            "guidance": variant.guidance,
            "settings": variant.settings,
        })).collect::<Vec<_>>(),
    })
}
