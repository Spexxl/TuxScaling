use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const RGBA_HASH: &str = "44bb653b69f11ee73b1b879227e4eddf4fd2b3f1c28fec475aac7d1f342624a6";

struct TestDirectory(PathBuf, bool);

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let name = format!("tuxscaling-quality-replay-{}-{nonce}", std::process::id());
        let (path, keep) = match std::env::var_os("TUXSCALING_QUALITY_REPLAY_EVIDENCE_DIR") {
            Some(root) => (PathBuf::from(root).join(name), true),
            None => (std::env::temp_dir().join(name), false),
        };
        fs::create_dir_all(&path).unwrap();
        Self(path, keep)
    }

    fn write_manifest(&self, value: &Value) -> PathBuf {
        let path = self.0.join("manifest.json");
        fs::write(&path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        if !self.1 {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn valid_manifest(directory: &Path) -> Value {
    fs::write(directory.join("frame-0.rgba"), b"RGBA").unwrap();
    fs::write(directory.join("frame-1.rgba"), b"RGBA").unwrap();
    json!({
        "schema_version": 1,
        "sequence": {
            "id": "two-frame-pan",
            "source_extent": [1, 1],
            "source_format": "R8G8B8A8_UNORM",
            "numeric_encoding": "srgb_nonlinear",
            "output_extent": [2, 2],
            "content_viewport": [0, 0, 2, 2],
            "slot_count": 1,
            "reset_frame_ids": [0],
            "frames": [
                {"frame_id": 0, "timestamp_ns": 0, "slot_index": 0, "motion_slot_index": 0, "source_file": "frame-0.rgba", "sha256": RGBA_HASH},
                {"frame_id": 1, "timestamp_ns": 16_666_667, "slot_index": 0, "motion_slot_index": 1, "source_file": "frame-1.rgba", "sha256": RGBA_HASH}
            ]
        },
        "variants": [
            {"name": "fsr_estimated", "backend": "fsr_3_1_4", "guidance": "estimated", "settings": {"jitter": false, "sharpening": 0.30000001192092896, "motion_quality": "balanced"}},
            {"name": "fsr_zero", "backend": "fsr_3_1_4", "guidance": "zero", "settings": {"jitter": false, "sharpening": 0.30000001192092896, "motion_quality": "balanced"}},
            {"name": "off", "backend": "off", "guidance": "none", "settings": {"jitter": false, "sharpening": 0.0, "motion_quality": "balanced"}}
        ]
    })
}

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .output()
        .unwrap()
}

fn validate_manifest(directory: &TestDirectory, manifest: &Value) -> Output {
    let path = directory.write_manifest(manifest);
    invoke(&[
        "quality-replay",
        "--manifest",
        path.to_str().unwrap(),
        "--validate-only",
    ])
}

fn hash_file(path: &Path) -> String {
    let output = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}

fn matching_report(directory: &TestDirectory, manifest: &Value) -> Value {
    let manifest_path = directory.write_manifest(manifest);
    let frames = manifest["sequence"]["frames"].clone();
    let variants = manifest["variants"]
        .as_array()
        .unwrap()
        .iter()
        .map(|variant| {
            json!({
                "name": variant["name"],
                "backend": variant["backend"],
                "guidance": variant["guidance"],
                "settings": variant["settings"],
                "consumed_frames": frames,
                "metrics": {"sample": 0.25}
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "manifest_sha256": hash_file(&manifest_path),
        "implementation_verified": true,
        "quality_accepted": null,
        "variants": variants
    })
}

fn validate_report_against_manifest(
    directory: &TestDirectory,
    manifest: &Value,
    report: &Value,
) -> Output {
    let manifest_path = directory.write_manifest(manifest);
    let report_path = directory.0.join("report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    invoke(&[
        "quality-replay",
        "--manifest",
        manifest_path.to_str().unwrap(),
        "--validate-report",
        report_path.to_str().unwrap(),
    ])
}

fn require_quality_against_manifest(
    directory: &TestDirectory,
    manifest: &Value,
    report: &Value,
) -> Output {
    let manifest_path = directory.write_manifest(manifest);
    let report_path = directory.0.join("report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    invoke(&[
        "quality-replay",
        "--manifest",
        manifest_path.to_str().unwrap(),
        "--validate-report",
        report_path.to_str().unwrap(),
        "--require-quality",
    ])
}

#[test]
fn replay_requires_verified_quality_when_gate_is_requested() {
    let directory = TestDirectory::new();
    let manifest = valid_manifest(&directory.0);
    let mut report = matching_report(&directory, &manifest);

    let missing = require_quality_against_manifest(&directory, &manifest, &report);
    assert!(!missing.status.success());

    report["quality_accepted"] = json!(false);
    let rejected = require_quality_against_manifest(&directory, &manifest, &report);
    assert!(!rejected.status.success());

    report["quality_accepted"] = json!(true);
    let accepted = require_quality_against_manifest(&directory, &manifest, &report);
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
}

#[test]
fn replay_rejects_source_mismatch_before_quality_status() {
    let directory = TestDirectory::new();
    let manifest = valid_manifest(&directory.0);
    let mut report = matching_report(&directory, &manifest);
    report["quality_accepted"] = json!(true);
    report["variants"][0]["consumed_frames"][0]["sha256"] = json!("0".repeat(64));

    let result = require_quality_against_manifest(&directory, &manifest, &report);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("identical ordered source frames"));
}

fn gpu_manifest(directory: &TestDirectory) -> Value {
    gpu_manifest_for(directory, [32, 32], [64, 64], "captured-pan")
}

fn gpu_manifest_for(
    directory: &TestDirectory,
    source_extent: [u32; 2],
    output_extent: [u32; 2],
    sequence_id: &str,
) -> Value {
    let mut frames = Vec::new();
    for frame_id in 0..2_u64 {
        let mut bytes =
            Vec::with_capacity(source_extent[0] as usize * source_extent[1] as usize * 4);
        for y in 0..source_extent[1] {
            for x in 0..source_extent[0] {
                let checker = ((x + frame_id as u32) / 4 + y / 4) % 2;
                bytes.extend_from_slice(&[
                    if checker == 0 { 220 } else { 30 },
                    if checker == 0 { 80 } else { 190 },
                    if (x + frame_id as u32) % 7 < 3 {
                        245
                    } else {
                        12
                    },
                    255,
                ]);
            }
        }
        let source_file = format!("gpu-frame-{frame_id}.rgba");
        let source_path = directory.0.join(&source_file);
        fs::write(&source_path, bytes).unwrap();
        frames.push(json!({
            "frame_id": frame_id,
            "timestamp_ns": frame_id * 16_666_667,
            "slot_index": frame_id as u32 % 3,
            "motion_slot_index": frame_id as u32 % 2,
            "source_file": source_file,
            "sha256": hash_file(&source_path),
        }));
    }
    let source_aspect = source_extent[0] as f32 / source_extent[1] as f32;
    let output_aspect = output_extent[0] as f32 / output_extent[1] as f32;
    let size = if output_aspect > source_aspect {
        [source_aspect / output_aspect, 1.0]
    } else {
        [1.0, output_aspect / source_aspect]
    };
    let offset = [(1.0 - size[0]) * 0.5, (1.0 - size[1]) * 0.5];
    let left = (offset[0] * output_extent[0] as f32).round() as u32;
    let top = (offset[1] * output_extent[1] as f32).round() as u32;
    let right = ((offset[0] + size[0]) * output_extent[0] as f32).round() as u32;
    let bottom = ((offset[1] + size[1]) * output_extent[1] as f32).round() as u32;
    json!({
        "schema_version": 1,
        "sequence": {
            "id": sequence_id,
            "source_extent": source_extent,
            "source_format": "R8G8B8A8_UNORM",
            "numeric_encoding": "srgb_nonlinear",
            "output_extent": output_extent,
            "content_viewport": [left, top, right - left, bottom - top],
            "slot_count": 3,
            "reset_frame_ids": [0],
            "frames": frames
        },
        "variants": [
            {"name":"fsr_estimated","backend":"fsr_3_1_4","guidance":"estimated","settings":{"jitter":false,"sharpening":0.0,"motion_quality":"balanced"}},
            {"name":"fsr_zero","backend":"fsr_3_1_4","guidance":"zero","settings":{"jitter":false,"sharpening":0.0,"motion_quality":"balanced"}},
            {"name":"off","backend":"off","guidance":"none","settings":{"jitter":false,"sharpening":0.0,"motion_quality":"balanced"}}
        ]
    })
}

#[test]
fn replay_accepts_valid_manifest() {
    let directory = TestDirectory::new();
    let result = validate_manifest(&directory, &valid_manifest(&directory.0));

    assert!(
        result.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn replay_requires_a_valid_motion_history_slot() {
    let directory = TestDirectory::new();
    let mut missing = valid_manifest(&directory.0);
    missing["sequence"]["frames"][0]
        .as_object_mut()
        .unwrap()
        .remove("motion_slot_index");
    let result = validate_manifest(&directory, &missing);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("motion_slot_index"));

    let mut out_of_range = valid_manifest(&directory.0);
    out_of_range["sequence"]["frames"][1]["motion_slot_index"] = json!(2);
    let result = validate_manifest(&directory, &out_of_range);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("two-slot motion history"));
}

#[test]
fn replay_rejects_mixed_source_hashes() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]["frames"][1]["sha256"] = json!("0".repeat(64));
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("source hash mismatch"));
}

#[test]
fn replay_rejects_missing_frames() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]["frames"][1]["source_file"] = json!("missing.rgba");
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("missing source frame"));
}

#[test]
fn replay_requires_encoding() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]
        .as_object_mut()
        .unwrap()
        .remove("numeric_encoding");
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("numeric_encoding"));
}

#[test]
fn replay_rejects_format_without_gpu_worker_support() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]["source_format"] = json!("R16G16B16A16_SFLOAT");
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported source format"));
}

#[test]
fn replay_rejects_linear_encoding_until_the_adapter_supports_it() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]["numeric_encoding"] = json!("linear_sdr");
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("linear_sdr requires the Q2 color adapter")
    );
}

#[test]
fn replay_requires_explicit_settings_without_hidden_defaults() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["variants"][0]["settings"]
        .as_object_mut()
        .unwrap()
        .remove("motion_quality");
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("explicitly include motion_quality"));
}

#[test]
fn replay_rejects_unsupported_jitter_variant() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["variants"][0]["settings"]["jitter"] = json!(true);
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("jitter"));
}

#[test]
fn replay_rejects_invalid_sharpening_setting() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["variants"][0]["settings"]["sharpening"] = json!(1.5);
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("sharpening"));
}

#[test]
fn replay_rejects_content_viewport_that_stretches_the_source() {
    let directory = TestDirectory::new();
    let mut manifest = valid_manifest(&directory.0);
    manifest["sequence"]["content_viewport"] = json!([0, 0, 2, 1]);
    let result = validate_manifest(&directory, &manifest);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("content_viewport"));
}

#[test]
fn replay_rejects_non_finite_metrics() {
    let directory = TestDirectory::new();
    let report = directory.0.join("report.json");
    fs::write(
        &report,
        br#"{"schema_version":1,"manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","implementation_verified":true,"quality_accepted":null,"variants":[{"name":"off","backend":"off","guidance":"none","settings":{"jitter":false,"sharpening":0.0,"motion_quality":"balanced"},"consumed_frames":[{"frame_id":0,"timestamp_ns":1,"slot_index":0,"motion_slot_index":0,"source_file":"f0.rgba","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"frame_id":1,"timestamp_ns":2,"slot_index":0,"motion_slot_index":1,"source_file":"f1.rgba","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"metrics":{"mse":1e999}}]}"#,
    )
    .unwrap();
    let result = invoke(&[
        "quality-replay",
        "--validate-report",
        report.to_str().unwrap(),
    ]);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("metrics"));
}

#[test]
fn replay_rejects_report_without_verified_gpu_implementation() {
    let directory = TestDirectory::new();
    let report = directory.0.join("report.json");
    fs::write(
        &report,
        br#"{"schema_version":1,"manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","implementation_verified":false,"quality_accepted":null,"variants":[{"name":"off","backend":"off","guidance":"none","settings":{"jitter":false,"sharpening":0.0,"motion_quality":"balanced"},"consumed_frames":[{"frame_id":0,"timestamp_ns":1,"slot_index":0,"motion_slot_index":0,"source_file":"f0.rgba","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"frame_id":1,"timestamp_ns":2,"slot_index":0,"motion_slot_index":1,"source_file":"f1.rgba","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"metrics":{"mse":0.0}}]}"#,
    )
    .unwrap();
    let result = invoke(&[
        "quality-replay",
        "--validate-report",
        report.to_str().unwrap(),
    ]);

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("does not verify"));
}

#[test]
fn replay_report_must_match_variant_contract_and_ordered_source_frames() {
    let directory = TestDirectory::new();
    let manifest = valid_manifest(&directory.0);
    let mut report = matching_report(&directory, &manifest);
    report["variants"][0]["settings"]["sharpening"] = json!(0.3);
    let result = validate_report_against_manifest(&directory, &manifest, &report);
    assert!(
        result.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );

    let mut mismatched_backend = report.clone();
    mismatched_backend["variants"][0]["backend"] = json!("off");
    mismatched_backend["variants"][0]["guidance"] = json!("none");
    let result = validate_report_against_manifest(&directory, &manifest, &mismatched_backend);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("identical ordered source frames"));

    let mut mismatched_settings = report.clone();
    mismatched_settings["variants"][0]["settings"]["sharpening"] = json!(0.5);
    let result = validate_report_against_manifest(&directory, &manifest, &mismatched_settings);
    assert!(!result.status.success());

    let mut mismatched_frame = report;
    mismatched_frame["variants"][0]["consumed_frames"][1]["timestamp_ns"] = json!(99);
    let result = validate_report_against_manifest(&directory, &manifest, &mismatched_frame);
    assert!(!result.status.success());
}

#[test]
#[ignore = "requires a Vulkan GPU and FidelityFX 3.1.4"]
fn replay_dispatches_identical_frames_through_all_gpu_variants() {
    let directory = TestDirectory::new();
    let manifest = gpu_manifest(&directory);
    let manifest_path = directory.write_manifest(&manifest);
    let output_dir = directory.0.join("replay-output");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "quality-replay",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let run_dir = fs::read_dir(&output_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let report: Value =
        serde_json::from_slice(&fs::read(run_dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["implementation_verified"], true);
    assert!(report["quality_accepted"].is_null());
    let variants = report["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 3);
    assert_eq!(
        variants[0]["consumed_frames"],
        variants[1]["consumed_frames"]
    );
    assert_eq!(
        variants[1]["consumed_frames"],
        variants[2]["consumed_frames"]
    );
    for variant in variants {
        assert_eq!(variant["consumed_frames"].as_array().unwrap().len(), 2);
        assert!(
            run_dir
                .join(format!(
                    "{}-frame-00000000.rgba",
                    variant["name"].as_str().unwrap()
                ))
                .is_file()
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU and FidelityFX 3.1.4; covers upscale, ultrawide fit and native AA"]
fn replay_accepts_the_required_resolution_matrix() {
    println!(
        "GPU resolution matrix evidence root: {}",
        std::env::var("TUXSCALING_QUALITY_REPLAY_EVIDENCE_DIR")
            .unwrap_or_else(|_| "temporary (deleted after test)".into())
    );
    for (source_extent, output_extent, sequence_id) in [
        ([1280, 720], [1920, 1080], "1080p-upscale"),
        ([1280, 720], [3440, 1440], "ultrawide-letterbox"),
        ([1920, 1080], [1920, 1080], "native-aa"),
    ] {
        let directory = TestDirectory::new();
        let manifest = gpu_manifest_for(&directory, source_extent, output_extent, sequence_id);
        let manifest_path = directory.write_manifest(&manifest);
        let output_dir = directory.0.join("replay-output");
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args([
                "quality-replay",
                "--manifest",
                manifest_path.to_str().unwrap(),
                "--output",
                output_dir.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "sequence={sequence_id}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        let run_dir = fs::read_dir(&output_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let report: Value =
            serde_json::from_slice(&fs::read(run_dir.join("report.json")).unwrap()).unwrap();
        assert_eq!(report["implementation_verified"], true);
        assert_eq!(report["variants"].as_array().unwrap().len(), 3);
        for variant in report["variants"].as_array().unwrap() {
            assert_eq!(variant["consumed_frames"].as_array().unwrap().len(), 2);
        }
    }
}
