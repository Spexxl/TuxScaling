use super::CaptureManifest;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Deserialize, Serialize)]
struct ReplayManifest {
    schema_version: u32,
    sequence: ReplaySequence,
    variants: Vec<ReplayVariant>,
}

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ReplayFrame {
    frame_id: u64,
    timestamp_ns: u64,
    slot_index: u32,
    motion_slot_index: u32,
    source_file: String,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ReplayVariant {
    name: String,
    backend: String,
    guidance: String,
    settings: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ReplayReport {
    schema_version: u32,
    manifest_sha256: String,
    implementation_verified: bool,
    quality_accepted: Option<bool>,
    variants: Vec<ReplayVariantReport>,
}

#[derive(Debug, Deserialize)]
struct ReplayVariantReport {
    name: String,
    backend: String,
    guidance: String,
    settings: serde_json::Map<String, serde_json::Value>,
    consumed_frames: Vec<ReplayFrame>,
    metrics: std::collections::BTreeMap<String, f64>,
}

struct LoadedManifest {
    path: PathBuf,
    hash: String,
    manifest: ReplayManifest,
}

#[derive(Default)]
struct ReplayOptions {
    manifest: Option<PathBuf>,
    output: Option<PathBuf>,
    validate_only: bool,
    validate_report: Option<PathBuf>,
    require_quality: bool,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let result = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|error| format!("run sha256sum for {}: {error}", path.display()))?;
    if !result.status.success() {
        return Err(format!(
            "sha256sum failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    let hash = String::from_utf8_lossy(&result.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !is_sha256(&hash) {
        return Err(format!(
            "sha256sum returned an invalid digest for {}",
            path.display()
        ));
    }
    Ok(hash)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_pixels(extent: [u32; 2]) -> Result<usize, String> {
    if extent.contains(&0) {
        return Err("source and output extents must be nonzero".into());
    }
    (extent[0] as usize)
        .checked_mul(extent[1] as usize)
        .ok_or_else(|| "image extent overflows addressable memory".into())
}

fn bytes_per_pixel(format: &str) -> Option<usize> {
    match format {
        "R8G8B8A8_UNORM" | "R8G8B8A8_SRGB" | "B8G8R8A8_UNORM" | "B8G8R8A8_SRGB" => Some(4),
        _ => None,
    }
}

fn load_manifest(path: &Path) -> Result<LoadedManifest, String> {
    let path = path
        .canonicalize()
        .map_err(|error| format!("open replay manifest {}: {error}", path.display()))?;
    let manifest_bytes = fs::read(&path)
        .map_err(|error| format!("read replay manifest {}: {error}", path.display()))?;
    let manifest: ReplayManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("decode replay manifest: {error}"))?;
    validate_manifest_fields(&manifest)?;
    let root = path
        .parent()
        .ok_or_else(|| "replay manifest has no parent directory".to_owned())?
        .canonicalize()
        .map_err(|error| format!("canonicalize replay directory: {error}"))?;
    for frame in &manifest.sequence.frames {
        validate_frame_file(&root, frame, &manifest.sequence)?;
    }
    let hash = sha256_file(&path)?;
    Ok(LoadedManifest {
        path,
        hash,
        manifest,
    })
}

fn validate_manifest_fields(manifest: &ReplayManifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported replay manifest schema_version {}",
            manifest.schema_version
        ));
    }
    let sequence = &manifest.sequence;
    if sequence.id.trim().is_empty() {
        return Err("sequence id must not be empty".into());
    }
    let source_pixels = checked_pixels(sequence.source_extent)?;
    checked_pixels(sequence.output_extent)?;
    if sequence.slot_count == 0 {
        return Err("replay sequence slot_count must be nonzero".into());
    }
    if bytes_per_pixel(&sequence.source_format).is_none() {
        return Err(format!(
            "unsupported source format {}",
            sequence.source_format
        ));
    }
    let expected_viewport = aspect_fit_viewport(sequence.source_extent, sequence.output_extent)?;
    if sequence.content_viewport != expected_viewport {
        return Err(format!(
            "content_viewport {:?} does not match aspect-fit viewport {:?}",
            sequence.content_viewport, expected_viewport
        ));
    }
    if !matches!(
        sequence.numeric_encoding.as_str(),
        "srgb_nonlinear" | "linear_sdr"
    ) {
        return Err(format!(
            "unsupported numeric_encoding {}",
            sequence.numeric_encoding
        ));
    }
    if sequence.numeric_encoding != "srgb_nonlinear" {
        return Err(
            "captured replay currently supports only srgb_nonlinear input; linear_sdr requires the Q2 color adapter"
                .into(),
        );
    }
    let [x, y, width, height] = sequence.content_viewport;
    if width == 0
        || height == 0
        || x.checked_add(width)
            .is_none_or(|right| right > sequence.output_extent[0])
        || y.checked_add(height)
            .is_none_or(|bottom| bottom > sequence.output_extent[1])
    {
        return Err("content_viewport must be a nonempty rectangle inside output_extent".into());
    }
    if sequence.frames.len() < 2 {
        return Err("replay sequence requires at least two source frames".into());
    }
    let mut previous_id = None;
    let mut previous_timestamp = None;
    let mut ids = BTreeSet::new();
    for frame in &sequence.frames {
        if frame.slot_index >= sequence.slot_count {
            return Err(format!(
                "frame {} slot_index {} is outside slot_count {}",
                frame.frame_id, frame.slot_index, sequence.slot_count
            ));
        }
        if frame.motion_slot_index >= 2 {
            return Err(format!(
                "frame {} motion_slot_index {} is outside the two-slot motion history",
                frame.frame_id, frame.motion_slot_index
            ));
        }
        if previous_id.is_some_and(|previous| frame.frame_id <= previous)
            || previous_timestamp.is_some_and(|previous| frame.timestamp_ns <= previous)
        {
            return Err("source frames must have strictly increasing IDs and timestamps".into());
        }
        if !is_sha256(&frame.sha256) {
            return Err(format!(
                "frame {} has an invalid source SHA-256",
                frame.frame_id
            ));
        }
        previous_id = Some(frame.frame_id);
        previous_timestamp = Some(frame.timestamp_ns);
        ids.insert(frame.frame_id);
        let expected = source_pixels
            .checked_mul(bytes_per_pixel(&sequence.source_format).unwrap())
            .ok_or_else(|| "source frame byte size overflows addressable memory".to_owned())?;
        if expected == 0 {
            return Err("source frame must contain pixel data".into());
        }
    }
    if sequence.reset_frame_ids.is_empty()
        || sequence
            .reset_frame_ids
            .iter()
            .any(|frame_id| !ids.contains(frame_id))
    {
        return Err("reset_frame_ids must refer to frames in this sequence".into());
    }
    if !sequence
        .reset_frame_ids
        .contains(&sequence.frames[0].frame_id)
    {
        return Err("the first replay frame must reset temporal history".into());
    }
    if manifest.variants.is_empty() {
        return Err("replay manifest must define at least one variant".into());
    }
    let mut names = BTreeSet::new();
    for variant in &manifest.variants {
        if variant.name.trim().is_empty() || !names.insert(variant.name.as_str()) {
            return Err("variant names must be nonempty and unique".into());
        }
        if !matches!(variant.backend.as_str(), "fsr_3_1_4" | "off") {
            return Err(format!("unsupported replay backend {}", variant.backend));
        }
        if !matches!(variant.guidance.as_str(), "estimated" | "zero" | "none") {
            return Err(format!("unsupported guidance mode {}", variant.guidance));
        }
        if variant.backend == "off" && variant.guidance != "none" {
            return Err("the off backend requires guidance=none".into());
        }
        if variant.backend == "fsr_3_1_4" && variant.guidance == "none" {
            return Err("fsr_3_1_4 requires estimated or zero guidance".into());
        }
        if variant
            .name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')))
        {
            return Err("variant names may contain only ASCII letters, digits, '_' and '-'".into());
        }
        validate_variant_settings(&variant.settings)?;
    }
    Ok(())
}

pub fn manifest_from_capture(capture_dir: &Path, output_path: &Path) -> Result<(), String> {
    let capture_dir = capture_dir
        .canonicalize()
        .map_err(|error| format!("open live capture {}: {error}", capture_dir.display()))?;
    let mut captures = fs::read_dir(&capture_dir)
        .map_err(|error| format!("read live capture directory: {error}"))?
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("frame-") && entry.path().extension().is_some_and(|ext| ext == "json")
        })
        .filter_map(|entry| {
            fs::read(entry.path())
                .ok()
                .and_then(|bytes| serde_json::from_slice::<CaptureManifest>(&bytes).ok())
        })
        .collect::<Vec<_>>();
    captures.sort_by_key(|capture| capture.frame_id);
    if captures.len() < 2 {
        return Err(format!(
            "live capture needs at least two metadata frames, found {}",
            captures.len()
        ));
    }
    if captures.windows(2).any(|pair| {
        pair[0].frame_id >= pair[1].frame_id || pair[0].timestamp_ns >= pair[1].timestamp_ns
    }) {
        return Err("live capture frame IDs and timestamps must increase".into());
    }
    let first = &captures[0];
    if first.backend != "FSR 3.1.4"
        || first.guidance_mode != "Estimated"
        || first.numeric_encoding != "srgb_nonlinear"
        || first.guidance_scale != 1.0
        || first.slot_count == 0
        || first.ablations.post_capture_jitter
    {
        return Err(
            "live replay source must use FSR 3.1.4, Estimated guidance, guidance_scale=1.0, sRGB SDR and zero jitter".into(),
        );
    }
    let source_resource = first
        .resources
        .iter()
        .find(|resource| resource.name == "source")
        .ok_or_else(|| "live capture is missing the source resource".to_owned())?;
    if bytes_per_pixel(&source_resource.format).is_none()
        || first.sharpening.enabled && !first.sharpening.sharpness.is_finite()
        || !matches!(
            first.motion_quality.as_str(),
            "ultra" | "high" | "balanced" | "performance"
        )
    {
        return Err("live capture has unsupported source format or active settings".into());
    }
    for capture in &captures {
        let source = capture
            .resources
            .iter()
            .find(|resource| resource.name == "source")
            .ok_or_else(|| format!("live capture frame {} is missing source", capture.frame_id))?;
        if capture.generation_id != first.generation_id
            || capture.game_extent != first.game_extent
            || capture.output_extent != first.output_extent
            || capture.numeric_encoding != first.numeric_encoding
            || source.extent != source_resource.extent
            || source.format != source_resource.format
            || capture.guidance_scale != first.guidance_scale
            || capture.slot_count != first.slot_count
            || capture.slot_index >= first.slot_count
            || capture.motion_slot_index >= 2
            || capture.motion_quality != first.motion_quality
            || capture.sharpening.enabled != first.sharpening.enabled
            || capture.sharpening.sharpness != first.sharpening.sharpness
            || capture.ablations.post_capture_jitter
        {
            return Err(format!(
                "live capture frame {} changed generation, extents, format, encoding or replay settings",
                capture.frame_id
            ));
        }
        if capture.frame_delta_ns.raw == 0
            || capture.frame_delta_ns.validated == 0
            || capture.frame_delta_ns.smoothed == 0
        {
            return Err(format!(
                "live capture frame {} has invalid frame-delta metadata",
                capture.frame_id
            ));
        }
    }
    let output_parent = output_path
        .parent()
        .ok_or_else(|| "replay manifest output has no parent directory".to_owned())?;
    let frames_dir = output_parent.join("frames");
    fs::create_dir_all(&frames_dir)
        .map_err(|error| format!("create replay source directory: {error}"))?;
    let mut frames = Vec::with_capacity(captures.len());
    let mut reset_frame_ids = vec![captures[0].frame_id];
    for capture in &captures {
        let source = capture
            .resources
            .iter()
            .find(|resource| resource.name == "source")
            .expect("source resources validated above");
        let file_name = Path::new(&source.file)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        if file_name.as_deref() != Some(source.file.as_str()) {
            return Err(format!(
                "live source file path is not a plain file name for frame {}",
                capture.frame_id
            ));
        }
        let capture_path = capture_dir.join(&source.file);
        let copied_name = format!("frame-{:08}-source.bin", capture.frame_id);
        let copied_path = frames_dir.join(&copied_name);
        fs::copy(&capture_path, &copied_path)
            .map_err(|error| format!("copy live source frame {}: {error}", capture.frame_id))?;
        let expected_size = checked_pixels(first.game_extent)?
            .checked_mul(bytes_per_pixel(&source.format).unwrap())
            .ok_or_else(|| "live source image byte count overflows".to_owned())?;
        if fs::metadata(&copied_path)
            .map_err(|error| format!("inspect copied source frame: {error}"))?
            .len()
            != expected_size as u64
        {
            return Err(format!(
                "live source frame {} has an unexpected byte count",
                capture.frame_id
            ));
        }
        if capture.frame_id != captures[0].frame_id && capture.reset_reason != "None" {
            reset_frame_ids.push(capture.frame_id);
        }
        frames.push(ReplayFrame {
            frame_id: capture.frame_id,
            timestamp_ns: capture.timestamp_ns,
            slot_index: capture.slot_index,
            motion_slot_index: capture.motion_slot_index,
            source_file: format!("frames/{copied_name}"),
            sha256: sha256_file(&copied_path)?,
        });
    }
    let sharpening = if first.sharpening.enabled {
        first.sharpening.sharpness
    } else {
        0.0
    };
    let settings = |sharpening| {
        serde_json::from_value(serde_json::json!({
            "jitter": false,
            "sharpening": sharpening,
            "motion_quality": first.motion_quality,
        }))
        .expect("replay settings are JSON object")
    };
    let manifest = ReplayManifest {
        schema_version: 1,
        sequence: ReplaySequence {
            id: format!(
                "runtime-capture-{}x{}-to-{}x{}",
                first.game_extent[0],
                first.game_extent[1],
                first.output_extent[0],
                first.output_extent[1]
            ),
            source_extent: first.game_extent,
            source_format: source_resource.format.clone(),
            numeric_encoding: first.numeric_encoding.clone(),
            output_extent: first.output_extent,
            content_viewport: aspect_fit_viewport(first.game_extent, first.output_extent)?,
            slot_count: first.slot_count,
            reset_frame_ids,
            frames,
        },
        variants: vec![
            ReplayVariant {
                name: "fsr_estimated".into(),
                backend: "fsr_3_1_4".into(),
                guidance: "estimated".into(),
                settings: settings(sharpening),
            },
            ReplayVariant {
                name: "fsr_zero".into(),
                backend: "fsr_3_1_4".into(),
                guidance: "zero".into(),
                settings: settings(sharpening),
            },
            ReplayVariant {
                name: "off".into(),
                backend: "off".into(),
                guidance: "none".into(),
                settings: settings(0.0),
            },
        ],
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize runtime replay manifest: {error}"))?;
    fs::write(output_path, manifest_bytes).map_err(|error| {
        format!(
            "write runtime replay manifest {}: {error}",
            output_path.display()
        )
    })
}

fn aspect_fit_viewport(source: [u32; 2], output: [u32; 2]) -> Result<[u32; 4], String> {
    checked_pixels(source)?;
    checked_pixels(output)?;
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
    Ok([left, top, right - left, bottom - top])
}

fn validate_variant_settings(
    settings: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), String> {
    for required in ["jitter", "sharpening", "motion_quality"] {
        if !settings.contains_key(required) {
            return Err(format!(
                "replay variant settings must explicitly include {required}"
            ));
        }
    }
    for key in settings.keys() {
        if !matches!(key.as_str(), "jitter" | "sharpening" | "motion_quality") {
            return Err(format!("unsupported replay variant setting {key}"));
        }
    }
    if settings
        .get("jitter")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err("experimental jitter is not supported by stable captured replay".into());
    }
    if settings
        .get("jitter")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err("jitter setting must be boolean".into());
    }
    if let Some(sharpening) = settings.get("sharpening") {
        let value = sharpening
            .as_f64()
            .ok_or_else(|| "sharpening setting must be a finite number in [0, 1]".to_owned())?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err("sharpening setting must be a finite number in [0, 1]".into());
        }
    }
    if let Some(quality) = settings.get("motion_quality") {
        let quality = quality
            .as_str()
            .ok_or_else(|| "motion_quality setting must be a string".to_owned())?;
        if !matches!(quality, "ultra" | "high" | "balanced" | "performance") {
            return Err(format!("unsupported motion_quality setting {quality}"));
        }
    }
    Ok(())
}

fn settings_match(
    expected: &serde_json::Map<String, serde_json::Value>,
    actual: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    expected.len() == actual.len()
        && expected.iter().all(|(key, expected_value)| {
            let Some(actual_value) = actual.get(key) else {
                return false;
            };
            if key == "sharpening" {
                return expected_value
                    .as_f64()
                    .zip(actual_value.as_f64())
                    .is_some_and(|(expected, actual)| {
                        (expected as f32).to_bits() == (actual as f32).to_bits()
                    });
            }
            expected_value == actual_value
        })
}

fn validate_frame_file(
    root: &Path,
    frame: &ReplayFrame,
    sequence: &ReplaySequence,
) -> Result<(), String> {
    let source_relative = Path::new(&frame.source_file);
    if source_relative.as_os_str().is_empty()
        || source_relative.is_absolute()
        || source_relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "frame {} source_file must be a safe relative path",
            frame.frame_id
        ));
    }
    let source = root.join(source_relative);
    let source = source.canonicalize().map_err(|error| {
        format!(
            "missing source frame {} ({}): {error}",
            frame.frame_id,
            source.display()
        )
    })?;
    if !source.starts_with(root) {
        return Err(format!(
            "frame {} source path escapes the manifest directory",
            frame.frame_id
        ));
    }
    let expected_size = checked_pixels(sequence.source_extent)?
        .checked_mul(bytes_per_pixel(&sequence.source_format).unwrap())
        .ok_or_else(|| "source frame byte size overflows addressable memory".to_owned())?;
    let actual_size = fs::metadata(&source)
        .map_err(|error| format!("inspect source frame {}: {error}", frame.frame_id))?
        .len();
    if actual_size != expected_size as u64 {
        return Err(format!(
            "source frame {} has {actual_size} bytes, expected {expected_size}",
            frame.frame_id
        ));
    }
    let actual_hash = sha256_file(&source)?;
    if actual_hash != frame.sha256.to_ascii_lowercase() {
        return Err(format!("source hash mismatch for frame {}", frame.frame_id));
    }
    Ok(())
}

fn validate_report(path: &Path) -> Result<ReplayReport, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("read replay report {}: {error}", path.display()))?;
    let report: ReplayReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("replay report metrics/metadata invalid: {error}"))?;
    if report.schema_version != 1 || !is_sha256(&report.manifest_sha256) {
        return Err("replay report has an unsupported schema or invalid manifest hash".into());
    }
    if !report.implementation_verified {
        return Err("replay report does not verify the GPU implementation".into());
    }
    if report.variants.is_empty() {
        return Err("replay report has no variant metrics".into());
    }
    let mut names = BTreeSet::new();
    for variant in &report.variants {
        if variant.name.trim().is_empty()
            || !names.insert(variant.name.as_str())
            || variant.consumed_frames.len() < 2
        {
            return Err(
                "replay report is missing a unique variant name or two consumed frames".into(),
            );
        }
        if !matches!(variant.backend.as_str(), "fsr_3_1_4" | "off")
            || !matches!(variant.guidance.as_str(), "estimated" | "zero" | "none")
            || (variant.backend == "off" && variant.guidance != "none")
            || (variant.backend == "fsr_3_1_4" && variant.guidance == "none")
        {
            return Err(format!(
                "replay report has invalid backend/guidance for {}",
                variant.name
            ));
        }
        validate_variant_settings(&variant.settings)?;
        if variant
            .consumed_frames
            .iter()
            .any(|frame| frame.source_file.trim().is_empty() || !is_sha256(&frame.sha256))
        {
            return Err(format!(
                "replay report has invalid source frame metadata for {}",
                variant.name
            ));
        }
        if variant.metrics.is_empty() || variant.metrics.values().any(|metric| !metric.is_finite())
        {
            return Err(format!(
                "replay metrics are missing or non-finite for {}",
                variant.name
            ));
        }
        if variant.consumed_frames.windows(2).any(|pair| {
            pair[0].frame_id >= pair[1].frame_id || pair[0].timestamp_ns >= pair[1].timestamp_ns
        }) {
            return Err(format!(
                "replay report frame order is invalid for {}",
                variant.name
            ));
        }
    }
    Ok(report)
}

fn validate_report_matches(report: &ReplayReport, loaded: &LoadedManifest) -> Result<(), String> {
    if report.manifest_sha256 != loaded.hash {
        return Err("replay report manifest hash does not match the input manifest".into());
    }
    if report.variants.len() != loaded.manifest.variants.len() {
        return Err("replay report variant count differs from the manifest".into());
    }
    for (expected, actual) in loaded.manifest.variants.iter().zip(&report.variants) {
        if expected.name != actual.name
            || expected.backend != actual.backend
            || expected.guidance != actual.guidance
            || !settings_match(&expected.settings, &actual.settings)
            || actual.consumed_frames != loaded.manifest.sequence.frames
        {
            return Err(format!(
                "variant {} did not consume the identical ordered source frames",
                expected.name
            ));
        }
    }
    Ok(())
}

fn parse_options(args: &[String]) -> Result<ReplayOptions, String> {
    let mut options = ReplayOptions::default();
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        index += 1;
        match argument {
            "--manifest" => {
                let value = args.get(index).ok_or("--manifest requires a path")?;
                options.manifest = Some(PathBuf::from(value));
                index += 1;
            }
            "--output" => {
                let value = args.get(index).ok_or("--output requires a directory")?;
                options.output = Some(PathBuf::from(value));
                index += 1;
            }
            "--validate-only" => options.validate_only = true,
            "--require-quality" => options.require_quality = true,
            "--validate-report" => {
                let value = args.get(index).ok_or("--validate-report requires a path")?;
                options.validate_report = Some(PathBuf::from(value));
                index += 1;
            }
            value => return Err(format!("unknown quality-replay argument: {value}")),
        }
    }
    if options.validate_report.is_some() {
        if options.output.is_some() || options.validate_only {
            return Err(
                "--validate-report cannot be combined with --output or --validate-only".into(),
            );
        }
    } else if options.manifest.is_none() {
        return Err("--manifest is required".into());
    } else if options.validate_only {
        if options.output.is_some() || options.require_quality {
            return Err("--validate-only does not accept --output or --require-quality".into());
        }
    } else if options.output.is_none() {
        return Err("--output is required unless --validate-only is set".into());
    }
    Ok(options)
}

pub fn execute(root: &Path, args: &[String]) -> bool {
    match execute_result(root, args) {
        Ok(()) => true,
        Err(error) => {
            eprintln!("cargo xtask quality-replay: {error}");
            false
        }
    }
}

fn execute_result(root: &Path, args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let require_quality = options.require_quality;
    if let Some(path) = options.validate_report {
        let report = validate_report(&path)?;
        if let Some(manifest_path) = options.manifest {
            let loaded = load_manifest(&manifest_path)?;
            validate_report_matches(&report, &loaded)?;
        }
        let quality = match report.quality_accepted {
            Some(true) => "accepted",
            Some(false) => "rejected",
            None => "not-evaluated",
        };
        println!(
            "validated {} replay variants; quality={quality}",
            report.variants.len()
        );
        if require_quality && report.quality_accepted != Some(true) {
            return Err("replay quality was not accepted".into());
        }
        return Ok(());
    }
    let manifest_path = options
        .manifest
        .expect("validated options include manifest");
    let loaded = load_manifest(&manifest_path)?;
    if options.validate_only {
        println!(
            "validated sequence={} frames={} variants={} manifest_sha256={}",
            loaded.manifest.sequence.id,
            loaded.manifest.sequence.frames.len(),
            loaded.manifest.variants.len(),
            loaded.hash
        );
        return Ok(());
    }
    let output = options.output.expect("validated options include output");
    fs::create_dir_all(&output)
        .map_err(|error| format!("create output directory {}: {error}", output.display()))?;
    let output = output
        .canonicalize()
        .map_err(|error| format!("canonicalize output directory: {error}"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
        .as_nanos();
    let run_dir = output.join(format!("run-{}-{nonce}", std::process::id()));
    fs::create_dir(&run_dir).map_err(|error| format!("create replay run directory: {error}"))?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let result = Command::new(cargo)
        .current_dir(root)
        .args([
            "test",
            "--manifest-path",
            root.join("Cargo.toml").to_str().unwrap_or("Cargo.toml"),
            "-p",
            "tuxscaling-upscaler",
            "--test",
            "captured_replay_gpu",
            "replay_manifest",
            "--",
            "--ignored",
            "--exact",
            "--test-threads=1",
        ])
        .env("TUXSCALING_REPLAY_MANIFEST", &loaded.path)
        .env("TUXSCALING_REPLAY_OUTPUT", &run_dir)
        .output()
        .map_err(|error| format!("start captured replay GPU worker: {error}"))?;
    print!("{}", String::from_utf8_lossy(&result.stdout));
    eprint!("{}", String::from_utf8_lossy(&result.stderr));
    if !result.status.success() {
        return Err(format!(
            "captured replay GPU worker exited with {}",
            result.status
        ));
    }
    let report_path = run_dir.join("report.json");
    let report = validate_report(&report_path)?;
    validate_report_matches(&report, &loaded)?;
    println!("quality replay report: {}", report_path.display());
    if require_quality && report.quality_accepted != Some(true) {
        return Err("replay quality was not accepted".into());
    }
    Ok(())
}
