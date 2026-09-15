use png::{BitDepth, ColorType, Encoder};
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Output, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tuxscaling_motion::MotionQuality;

const VKCUBE_LAYER_EVIDENCE_MARKER: &str = "TuxScaling swapchain:";
const PORTABLE_WSI_SCENARIOS: [&str; 5] = [
    "mutable_format",
    "present_wait_generation",
    "hdr_replacement",
    "display_timing",
    "incompatible_direct",
];

#[derive(Debug, Default, PartialEq, Eq)]
struct MaintenanceEvidence {
    device_enabled: bool,
    virtual_swapchain: bool,
    present_fences: bool,
    present_modes: bool,
    released_images: bool,
    overlay_submitted: bool,
    reconstructed_present: bool,
    validation_error: bool,
}

fn maintenance_evidence_text(stdout: &str, stderr: &str) -> String {
    format!("{stdout}\n{stderr}").to_ascii_lowercase()
}

fn has_positive_field(output: &str, field: &str) -> bool {
    output.split_whitespace().any(|token| {
        token
            .strip_prefix(field)
            .and_then(|value| value.parse::<u32>().ok())
            .is_some_and(|value| value > 0)
    })
}

fn parse_maintenance_evidence(stdout: &str, stderr: &str) -> MaintenanceEvidence {
    let output = maintenance_evidence_text(stdout, stderr);
    MaintenanceEvidence {
        device_enabled: output.contains("event=maintenance1_device enabled=1"),
        virtual_swapchain: output.contains("event=logical_swapchain_created")
            && output.contains("virtual=1"),
        present_fences: output.contains("event=maintenance1_present")
            && has_positive_field(&output, "fences="),
        present_modes: output.contains("event=maintenance1_present")
            && has_positive_field(&output, "modes="),
        released_images: output.contains("event=maintenance1_release")
            && output.contains("result=success")
            && has_positive_field(&output, "logical_count=")
            && has_positive_field(&output, "physical_count="),
        overlay_submitted: output.contains("event=overlay_submitted maintenance1=1"),
        reconstructed_present: output.contains("event=reconstructed_present"),
        validation_error: output.contains("validation error")
            || output.contains("vuid-")
            || output.contains("panic"),
    }
}

fn maintenance_evidence_complete(evidence: &MaintenanceEvidence) -> bool {
    evidence.device_enabled
        && evidence.virtual_swapchain
        && evidence.present_fences
        && evidence.present_modes
        && evidence.released_images
        && evidence.overlay_submitted
        && evidence.reconstructed_present
        && !evidence.validation_error
}

fn maintenance_output_is_valid(stdout: &str, stderr: &str, backend: BackendSelection) -> bool {
    let output = maintenance_evidence_text(stdout, stderr);
    let evidence = parse_maintenance_evidence(stdout, stderr);
    let backend_complete = match backend {
        BackendSelection::Reference => output.contains("backend=reference"),
        BackendSelection::Fsr314 => {
            output.contains("event=fsr_dispatch backend=fsr_3_1_4")
                && output.contains("backend=fsr_3_1_4")
        }
    };
    maintenance_evidence_complete(&evidence)
        && backend_complete
        && output.contains("logical=1280x720")
        && output.contains("physical=3440x1440")
        && output.contains("event=logical_recreation_translated")
        && !output.contains("maintenance1_fallback")
        && !output.contains("virtual=0")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BackendSelection {
    #[default]
    Reference,
    Fsr314,
}

impl BackendSelection {
    const fn config_value(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Fsr314 => "fsr_3_1_4",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WsiExtent {
    width: u32,
    height: u32,
}

fn parse_wsi_extent(value: &str) -> Option<WsiExtent> {
    let (width, height) = value.split_once('x')?;
    Some(WsiExtent {
        width: width.parse().ok()?,
        height: height.parse().ok()?,
    })
}

fn wsi_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|token| {
        token
            .strip_prefix(field)
            .and_then(|value| value.strip_prefix('='))
    })
}

fn wsi_positive_field(line: &str, field: &str) -> bool {
    wsi_field(line, field)
        .is_some_and(|value| value == "1" || value.parse::<u32>().is_ok_and(|v| v > 0))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct WsiCompatibilityEvidence {
    requested: Option<WsiExtent>,
    native_target: Option<WsiExtent>,
    logical_created: Option<WsiExtent>,
    published_logical: Option<WsiExtent>,
    published_physical: Option<WsiExtent>,
    virtual_created: Option<bool>,
    virtual_active: bool,
    mutable_format: bool,
    view_formats: u32,
    native_published: bool,
    presenter_published: bool,
    fsr_dispatch: bool,
    reconstructed_present: bool,
    overlay_submitted: bool,
    scenario_verified: bool,
    alternate_views: bool,
    present_wait_current: bool,
    present_wait_old: bool,
    present_wait_recreated: bool,
    present_wait_generations: Vec<u64>,
    hdr_before: bool,
    hdr_after: bool,
    status_query: bool,
    counter_query: bool,
    refresh_query: bool,
    timing_count: bool,
    timing_data: bool,
    direct_fallback: bool,
    direct_fallback_reason: Option<String>,
    validation_error: bool,
    unverified: bool,
    panic: bool,
    virtual_zero: bool,
    post_publish_recreation: bool,
}

fn parse_wsi_compatibility_evidence(stdout: &str, stderr: &str) -> WsiCompatibilityEvidence {
    let mut evidence = WsiCompatibilityEvidence::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let lowercase = line.to_ascii_lowercase();
        if lowercase.contains("validation error")
            || lowercase.contains("vuid-")
            || lowercase.contains("validation failed")
        {
            evidence.validation_error = true;
        }
        if lowercase.contains("panic") || lowercase.contains("panicked") {
            evidence.panic = true;
        }
        if lowercase.contains("result=unverified") {
            evidence.unverified = true;
        }
        if lowercase
            .split_whitespace()
            .any(|field| field == "virtual=0")
        {
            evidence.virtual_zero = true;
        }
        let Some(event) = wsi_field(&lowercase, "event") else {
            continue;
        };
        match event {
            "wsi_scenario_request" => {
                evidence.requested = wsi_field(&lowercase, "requested").and_then(parse_wsi_extent);
            }
            "borderless_target" => {
                evidence.native_target = wsi_field(&lowercase, "extent").and_then(parse_wsi_extent);
            }
            "logical_swapchain_created" => {
                evidence.logical_created =
                    wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.virtual_created =
                    wsi_field(&lowercase, "virtual").and_then(|value| match value {
                        "0" => Some(false),
                        "1" => Some(true),
                        _ => None,
                    });
                evidence.virtual_zero |= wsi_field(&lowercase, "virtual") == Some("0");
                evidence.mutable_format = wsi_field(&lowercase, "mutable_format") == Some("1");
                evidence.view_formats = wsi_field(&lowercase, "view_formats")
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_default();
            }
            "virtual_swapchain_active" => {
                evidence.virtual_active = true;
                evidence.published_logical =
                    wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.published_physical =
                    wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
            }
            "native_generation_published" => {
                evidence.published_logical =
                    wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.published_physical =
                    wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
                evidence.native_published = true;
            }
            "presenter_generation_published" => {
                evidence.published_logical =
                    wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.published_physical =
                    wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
                evidence.presenter_published = true;
            }
            "direct_swapchain_created" => {
                evidence.virtual_created = Some(false);
                evidence.virtual_zero = true;
            }
            "fsr_dispatch" => {
                evidence.fsr_dispatch = wsi_field(&lowercase, "backend") == Some("fsr_3_1_4");
            }
            "present_wait" => {
                if wsi_field(&lowercase, "translated") == Some("1")
                    && let Some(generation) =
                        wsi_field(&lowercase, "generation").and_then(|value| value.parse().ok())
                {
                    evidence.present_wait_generations.push(generation);
                }
            }
            "reconstructed_present" => evidence.reconstructed_present = true,
            "overlay_submitted" => evidence.overlay_submitted = true,
            "virtualization_preflight" => {
                if wsi_field(&lowercase, "result") == Some("direct") {
                    evidence.direct_fallback = true;
                    evidence.direct_fallback_reason =
                        wsi_field(&lowercase, "reason").map(str::to_owned);
                }
            }
            "wsi_scenario" => {
                evidence.scenario_verified |= wsi_field(&lowercase, "result") == Some("verified");
                evidence.alternate_views |= wsi_positive_field(&lowercase, "alternate_views");
                evidence.present_wait_current |=
                    wsi_positive_field(&lowercase, "present_wait_current");
                evidence.present_wait_old |= wsi_positive_field(&lowercase, "present_wait_old");
                evidence.present_wait_recreated |=
                    wsi_positive_field(&lowercase, "present_wait_recreated");
                evidence.hdr_before |= wsi_positive_field(&lowercase, "hdr_before");
                evidence.hdr_after |= wsi_positive_field(&lowercase, "hdr_after");
                let queries = wsi_field(&lowercase, "queries").unwrap_or_default();
                evidence.status_query |= queries.split(',').any(|query| query == "status");
                evidence.counter_query |= queries.split(',').any(|query| query == "counter");
                evidence.refresh_query |= queries.split(',').any(|query| query == "refresh");
                let timing = wsi_field(&lowercase, "timing").unwrap_or_default();
                evidence.timing_count |= timing.split(',').any(|value| value == "count");
                evidence.timing_data |= timing.split(',').any(|value| value == "data");
                evidence.post_publish_recreation |=
                    wsi_positive_field(&lowercase, "recreations_after_publish");
            }
            _ => {}
        }
    }
    evidence
}

fn wsi_compatibility_output_is_valid(
    stdout: &str,
    stderr: &str,
    scenario: &str,
    backend: BackendSelection,
) -> bool {
    let evidence = parse_wsi_compatibility_evidence(stdout, stderr);
    if evidence.validation_error || evidence.panic || evidence.unverified {
        return false;
    }
    if !evidence.scenario_verified {
        return false;
    }
    if scenario == "incompatible_direct" {
        return evidence.requested.is_some()
            && evidence.direct_fallback
            && matches!(
                evidence.direct_fallback_reason.as_deref(),
                Some("incompatible_wsi_extension" | "unsupported_pnext")
            )
            && evidence.virtual_created == Some(false)
            && evidence.virtual_zero;
    }
    if evidence.virtual_created != Some(true)
        || evidence.virtual_zero
        || evidence.direct_fallback
        || !evidence.virtual_active
        || evidence.requested.is_none()
        || evidence.logical_created != evidence.requested
        || evidence.published_logical != evidence.requested
        || evidence.native_target != evidence.published_physical
        || !(evidence.native_published || evidence.presenter_published)
        || !evidence.reconstructed_present
        || !evidence.overlay_submitted
        || evidence.post_publish_recreation
    {
        return false;
    }
    if backend == BackendSelection::Fsr314 && !evidence.fsr_dispatch {
        return false;
    }
    match scenario {
        "mutable_format" => {
            evidence.mutable_format && evidence.view_formats >= 2 && evidence.alternate_views
        }
        "present_wait_generation" => {
            evidence.present_wait_current
                && evidence.present_wait_old
                && evidence.present_wait_recreated
                && evidence.present_wait_generations.contains(&0)
                && evidence
                    .present_wait_generations
                    .iter()
                    .any(|generation| *generation > 0)
        }
        "hdr_replacement" => evidence.hdr_before && evidence.hdr_after,
        "display_timing" => {
            evidence.status_query
                && evidence.counter_query
                && evidence.refresh_query
                && evidence.timing_count
                && evidence.timing_data
        }
        _ => false,
    }
}

fn parse_backend(value: &str) -> Result<BackendSelection, String> {
    match value {
        "reference" => Ok(BackendSelection::Reference),
        "fsr_3_1_4" => Ok(BackendSelection::Fsr314),
        value => Err(format!("unknown backend: {value}")),
    }
}

fn parse_backend_args(args: &[&str]) -> Result<BackendSelection, String> {
    let mut backend = BackendSelection::Reference;
    let mut seen = false;
    let mut index = 0;
    while index < args.len() {
        if args[index] != "--backend" {
            return Err(format!("unknown argument: {}", args[index]));
        }
        if seen {
            return Err("--backend may only be specified once".into());
        }
        seen = true;
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| "--backend requires reference or fsr_3_1_4".to_owned())?;
        backend = parse_backend(value)?;
        index += 1;
        if index < args.len() && args[index] == "--backend" {
            return Err("--backend may only be specified once".into());
        }
    }
    Ok(backend)
}

fn parse_wsi_compatibility_args(args: &[&str]) -> Result<(BackendSelection, Vec<String>), String> {
    let mut backend = BackendSelection::Reference;
    let mut backend_seen = false;
    let mut allowed: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--backend" => {
                if backend_seen {
                    return Err("--backend may only be specified once".into());
                }
                backend_seen = true;
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--backend requires reference or fsr_3_1_4".to_owned())?;
                backend = parse_backend(value)?;
                index += 1;
            }
            "--allow-unverified" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--allow-unverified requires a scenario name".to_owned())?;
                for scenario in value.split(',') {
                    if !PORTABLE_WSI_SCENARIOS.contains(&scenario) {
                        return Err(format!("unknown wsi-compatibility scenario: {scenario}"));
                    }
                    if !allowed.iter().any(|allowed| allowed == scenario) {
                        allowed.push(scenario.to_owned());
                    }
                }
                index += 1;
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }
    Ok((backend, allowed))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VkcubeOptions {
    seconds: u64,
    release: bool,
    backend: BackendSelection,
    sharpening_enabled: bool,
    sharpness: f32,
    control_sequence: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VkcubeLaunch {
    profile_dir: PathBuf,
    config_path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VkcubeExit {
    Success,
    MissingExecutable,
    BuildFailure,
    EarlyExit(i32),
    ValidationError,
    MissingStartupEvidence,
    UnexpectedExit,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ControlApplied {
    setting: String,
    value: String,
    generation: Option<u64>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct VkcubeEvidence {
    layer_startup: bool,
    native_published: bool,
    native_publishes: u32,
    presenter_publishes: u32,
    virtual_active: bool,
    virtual_zero: bool,
    fallback: bool,
    validation_error: bool,
    overlay_submitted: u32,
    fsr_dispatches: u32,
    reconstructed_presents: u32,
    logical: Option<WsiExtent>,
    physical: Option<WsiExtent>,
    control_applied: Vec<ControlApplied>,
}

fn parse_vkcube_evidence(stdout: &str, stderr: &str) -> VkcubeEvidence {
    let mut evidence = VkcubeEvidence::default();
    for line in format!("{stdout}\n{stderr}").lines() {
        let lowercase = line.to_ascii_lowercase();
        evidence.layer_startup |=
            lowercase.contains(&VKCUBE_LAYER_EVIDENCE_MARKER.to_ascii_lowercase());
        evidence.validation_error |= lowercase.contains("validation error")
            || lowercase.contains("vuid-")
            || lowercase.contains("panic")
            || lowercase.contains("panicked");
        evidence.virtual_zero |= lowercase
            .split_whitespace()
            .any(|field| field == "virtual=0");
        let Some(event) = wsi_field(&lowercase, "event") else {
            continue;
        };
        match event {
            "native_generation_published" => {
                evidence.native_published = true;
                evidence.native_publishes = evidence.native_publishes.saturating_add(1);
                evidence.logical = wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.physical = wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
            }
            "presenter_generation_published" => {
                evidence.native_published = true;
                evidence.presenter_publishes = evidence.presenter_publishes.saturating_add(1);
                evidence.logical = wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.physical = wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
            }
            "control_applied" => {
                if let (Some(setting), Some(value)) = (
                    wsi_field(&lowercase, "setting").map(str::to_owned),
                    wsi_field(&lowercase, "value").map(str::to_owned),
                ) {
                    evidence.control_applied.push(ControlApplied {
                        setting,
                        value,
                        generation: wsi_field(&lowercase, "generation")
                            .and_then(|value| value.parse().ok()),
                    });
                }
            }
            "virtual_swapchain_active" => {
                evidence.virtual_active = true;
                evidence.logical = wsi_field(&lowercase, "logical").and_then(parse_wsi_extent);
                evidence.physical = wsi_field(&lowercase, "physical").and_then(parse_wsi_extent);
            }
            "fsr_dispatch" => evidence.fsr_dispatches = evidence.fsr_dispatches.saturating_add(1),
            "reconstructed_present" => {
                evidence.reconstructed_presents = evidence.reconstructed_presents.saturating_add(1)
            }
            "overlay_submitted" => {
                evidence.overlay_submitted = evidence.overlay_submitted.saturating_add(1)
            }
            "presenter_fallback"
            | "presenter_fallback_to_direct"
            | "presenter_instance_unavailable" => evidence.fallback = true,
            _ => {}
        }
    }
    evidence
}

fn vkcube_output_is_valid(stdout: &str, stderr: &str, backend: BackendSelection) -> bool {
    let evidence = parse_vkcube_evidence(stdout, stderr);
    let output = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    if [
        "recreation_loop",
        "recreations_after_publish=1",
        "logical_takeover",
        "overlay=hidden",
        "cursor=missing",
        "grab=stuck",
    ]
    .iter()
    .any(|marker| output.contains(marker))
    {
        return false;
    }
    let continuous_dispatch = match backend {
        BackendSelection::Reference => true,
        BackendSelection::Fsr314 => evidence.fsr_dispatches >= 3,
    };
    evidence.layer_startup
        && evidence.native_published
        && evidence.virtual_active
        && !evidence.virtual_zero
        && !evidence.fallback
        && !evidence.validation_error
        && evidence.overlay_submitted >= 3
        && evidence.reconstructed_presents >= 3
        && continuous_dispatch
        && evidence.logical.is_some_and(|extent| {
            extent
                == (WsiExtent {
                    width: 1280,
                    height: 720,
                })
        })
        && evidence.physical.is_some_and(|extent| {
            extent.width > 0 && extent.height > 0 && Some(extent) != evidence.logical
        })
}

fn vkcube_control_sequence_is_valid(stdout: &str, stderr: &str) -> bool {
    let evidence = parse_vkcube_evidence(stdout, stderr);
    let required = [
        ("upscaler", "off"),
        ("upscaler", "fsr_3_1_4"),
        ("guidance_mode", "zero"),
        ("guidance_mode", "estimated"),
        ("quality", "performance"),
        ("quality", "balanced"),
        ("sharpening_enabled", "false"),
        ("sharpening_enabled", "true"),
    ];
    let generations = evidence
        .control_applied
        .iter()
        .filter_map(|applied| applied.generation)
        .collect::<BTreeSet<_>>();
    let published = evidence
        .native_publishes
        .saturating_add(evidence.presenter_publishes);
    evidence.native_publishes <= 1
        && evidence.presenter_publishes <= 1
        && published >= 1
        && generations.len() == 1
        && required.iter().all(|(setting, value)| {
            evidence
                .control_applied
                .iter()
                .any(|applied| applied.setting == *setting && applied.value == *value)
        })
}

impl VkcubeExit {
    const fn success(self) -> bool {
        matches!(self, Self::Success)
    }
}

fn parse_vkcube_args(args: &[&str]) -> Result<VkcubeOptions, String> {
    let mut options = VkcubeOptions {
        seconds: 10,
        release: false,
        backend: BackendSelection::Reference,
        sharpening_enabled: true,
        sharpness: 0.3,
        control_sequence: false,
    };
    let mut backend_seen = false;
    let mut sharpening_seen = false;
    let mut index = 0;
    while index < args.len() {
        match args[index] {
            "--release" if !options.release => options.release = true,
            "--release" => return Err("--release may only be specified once".into()),
            "--seconds" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--seconds requires a positive integer".to_owned())?;
                options.seconds = value
                    .parse::<u64>()
                    .ok()
                    .filter(|seconds| *seconds > 0)
                    .ok_or_else(|| "--seconds requires a positive integer".to_owned())?;
            }
            "--backend" => {
                if backend_seen {
                    return Err("--backend may only be specified once".into());
                }
                backend_seen = true;
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--backend requires reference or fsr_3_1_4".to_owned())?;
                options.backend = parse_backend(value)?;
            }
            "--disable-sharpening" if !sharpening_seen => {
                sharpening_seen = true;
                options.sharpening_enabled = false;
            }
            "--disable-sharpening" => {
                return Err("sharpening options may only be specified once".into());
            }
            "--control-sequence" if !options.control_sequence => {
                options.control_sequence = true;
            }
            "--control-sequence" => {
                return Err("--control-sequence may only be specified once".into());
            }
            "--sharpness" if !sharpening_seen => {
                sharpening_seen = true;
                index += 1;
                let value = args.get(index).ok_or_else(|| {
                    "--sharpness requires a finite value in [0.0, 1.0]".to_owned()
                })?;
                options.sharpness = value
                    .parse::<f32>()
                    .ok()
                    .filter(|sharpness| sharpness.is_finite() && (0.0..=1.0).contains(sharpness))
                    .ok_or_else(|| {
                        "--sharpness requires a finite value in [0.0, 1.0]".to_owned()
                    })?;
            }
            "--sharpness" => {
                return Err("sharpening options may only be specified once".into());
            }
            value => return Err(format!("unknown vkcube argument: {value}")),
        }
        index += 1;
    }
    Ok(options)
}

fn vkcube_launch(root: &Path, options: VkcubeOptions) -> VkcubeLaunch {
    let cargo_target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    vkcube_launch_in(root, options, cargo_target_dir.as_deref())
}

fn vkcube_launch_in(
    root: &Path,
    options: VkcubeOptions,
    cargo_target_dir: Option<&Path>,
) -> VkcubeLaunch {
    let profile = if options.release { "release" } else { "debug" };
    let target = cargo_target_dir
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("target"));
    VkcubeLaunch {
        profile_dir: target.join(profile),
        config_path: root.join("target").join(format!("vkcube-{profile}.toml")),
    }
}

fn classify_vkcube_exit(
    exit_code: Option<i32>,
    timed_out: bool,
    startup_evidence: bool,
    validation_error: bool,
    missing_executable: bool,
) -> VkcubeExit {
    if missing_executable {
        return VkcubeExit::MissingExecutable;
    }
    if validation_error {
        return VkcubeExit::ValidationError;
    }
    if let Some(code) = exit_code {
        if code != 0 {
            return VkcubeExit::EarlyExit(code);
        }
        return if startup_evidence {
            VkcubeExit::Success
        } else {
            VkcubeExit::MissingStartupEvidence
        };
    }
    if !startup_evidence {
        return VkcubeExit::MissingStartupEvidence;
    }
    if timed_out {
        VkcubeExit::Success
    } else if startup_evidence {
        VkcubeExit::EarlyExit(-1)
    } else {
        VkcubeExit::UnexpectedExit
    }
}

fn classify_vkcube_output(stdout: &[u8], stderr: &[u8]) -> (bool, bool) {
    let output = format!(
        "{}\n{}",
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    )
    .to_ascii_lowercase();
    (
        output.contains(&VKCUBE_LAYER_EVIDENCE_MARKER.to_ascii_lowercase()),
        output.contains("validation error"),
    )
}

fn configure_vkcube_command(root: &Path, options: VkcubeOptions) -> Command {
    let launch = vkcube_launch(root, options);
    let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let libraries = std::iter::once(launch.profile_dir.clone())
        .chain(std::env::split_paths(&inherited))
        .collect::<Vec<_>>();
    let mut command = Command::new("vkcube");
    validation(&mut command)
        .args(["--wsi", "xcb", "--width", "1280", "--height", "720"])
        .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
        .env("LD_LIBRARY_PATH", std::env::join_paths(libraries).unwrap())
        .env(
            "VK_INSTANCE_LAYERS",
            "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
        )
        .env("TUXSCALING_VIEW", "reconstructed")
        .env("TUXSCALING_CONFIG", launch.config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if options.control_sequence {
        command.env("TUXSCALING_TEST_CONTROL_SEQUENCE", "1");
    }
    command
}

fn wait_for_vkcube(
    mut child: Child,
    seconds: u64,
    backend: BackendSelection,
    control_sequence: bool,
) -> VkcubeExit {
    let Some(mut stdout) = child.stdout.take() else {
        return VkcubeExit::UnexpectedExit;
    };
    let Some(mut stderr) = child.stderr.take() else {
        return VkcubeExit::UnexpectedExit;
    };
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.read_to_end(&mut bytes);
        (result, bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr.read_to_end(&mut bytes);
        (result, bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status, false),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let status = child.wait();
                let Some(status) = status.ok() else {
                    return VkcubeExit::UnexpectedExit;
                };
                break (status, true);
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return VkcubeExit::UnexpectedExit;
            }
        }
    };
    let (stdout_result, stdout) = stdout_reader.join().ok().unwrap_or((
        Err(std::io::Error::other("vkcube stdout reader panicked")),
        Vec::new(),
    ));
    let (stderr_result, stderr) = stderr_reader.join().ok().unwrap_or((
        Err(std::io::Error::other("vkcube stderr reader panicked")),
        Vec::new(),
    ));
    print!("{}", String::from_utf8_lossy(&stdout));
    eprint!("{}", String::from_utf8_lossy(&stderr));
    if stdout_result.is_err() || stderr_result.is_err() {
        return VkcubeExit::UnexpectedExit;
    }
    let (startup_evidence, validation_error) = classify_vkcube_output(&stdout, &stderr);
    let classification = classify_vkcube_exit(
        status.code(),
        timed_out,
        startup_evidence,
        validation_error,
        false,
    );
    let stdout_text = String::from_utf8_lossy(&stdout);
    let stderr_text = String::from_utf8_lossy(&stderr);
    if classification.success()
        && (!vkcube_output_is_valid(&stdout_text, &stderr_text, backend)
            || (control_sequence && !vkcube_control_sequence_is_valid(&stdout_text, &stderr_text)))
    {
        VkcubeExit::MissingStartupEvidence
    } else {
        classification
    }
}

fn run_vkcube(root: &Path, options: VkcubeOptions) -> VkcubeExit {
    let mut build = Command::new("cargo");
    build.args(["build", "-p", "tuxscaling-layer", "--lib"]);
    if options.release {
        build.arg("--release");
    }
    if !build.status().is_ok_and(|status| status.success()) {
        return VkcubeExit::BuildFailure;
    }
    let launch = vkcube_launch(root, options);
    if std::fs::write(
        &launch.config_path,
        generated_config_with_backend_and_sharpening(
            "native",
            1.0,
            None,
            options.backend,
            options.sharpening_enabled,
            options.sharpness,
        ),
    )
    .is_err()
    {
        return VkcubeExit::BuildFailure;
    }
    match configure_vkcube_command(root, options).spawn() {
        Ok(child) => wait_for_vkcube(
            child,
            options.seconds,
            options.backend,
            options.control_sequence,
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => VkcubeExit::MissingExecutable,
        Err(_) => VkcubeExit::UnexpectedExit,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisualQualityOptions {
    display: String,
    input: (u32, u32),
    output: (u32, u32),
    warmup: usize,
    frames: usize,
    skip_gpu_evidence: bool,
}

fn parse_extent_argument(name: &str, value: &str) -> Result<(u32, u32), String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("{name} requires WIDTHxHEIGHT"))?;
    let width = width
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} width must be positive"))?;
    let height = height
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} height must be positive"))?;
    Ok((width, height))
}

fn parse_visual_quality_args(args: &[&str]) -> Result<VisualQualityOptions, String> {
    let mut display = "nested-xwayland".to_owned();
    let mut input = None;
    let mut output = None;
    let mut warmup = 180_usize;
    let mut frames = 120_usize;
    let mut skip_gpu_evidence = false;
    let mut index = 0;
    while index < args.len() {
        let argument = args[index];
        index += 1;
        let value = |index: &mut usize, flag: &str| {
            let value = args
                .get(*index)
                .copied()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            *index += 1;
            Ok::<_, String>(value)
        };
        match argument {
            "--display" => {
                display = value(&mut index, "--display")?.to_owned();
                if display != "nested-xwayland" {
                    return Err("--display currently supports only nested-xwayland".into());
                }
            }
            "--input" => {
                input = Some(parse_extent_argument(
                    "--input",
                    value(&mut index, "--input")?,
                )?)
            }
            "--output" => {
                output = Some(parse_extent_argument(
                    "--output",
                    value(&mut index, "--output")?,
                )?);
            }
            "--warmup" => {
                warmup = value(&mut index, "--warmup")?
                    .parse()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "--warmup requires a positive integer".to_owned())?;
            }
            "--frames" => {
                frames = value(&mut index, "--frames")?
                    .parse()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "--frames requires a positive integer".to_owned())?;
            }
            "--skip-gpu-evidence" => {
                skip_gpu_evidence = true;
            }
            value => return Err(format!("unknown visual-quality argument: {value}")),
        }
    }
    Ok(VisualQualityOptions {
        display,
        input: input.ok_or_else(|| "--input is required".to_owned())?,
        output: output.ok_or_else(|| "--output is required".to_owned())?,
        warmup,
        frames,
        skip_gpu_evidence,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct VisualQualityMetrics {
    mse: f32,
    psnr: f32,
    ssim: f32,
    flicker_mse: f32,
    ghost_trail: f32,
    shimmer: f32,
    difference_map: Vec<[f32; 4]>,
}

fn luma_value(pixel: [f32; 4]) -> f32 {
    pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
}

fn image_mse(reference: &[[f32; 4]], estimate: &[[f32; 4]]) -> f32 {
    let count = reference.len().min(estimate.len());
    if count == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for index in 0..count {
        for (reference_value, estimate_value) in reference[index].iter().zip(estimate[index]) {
            sum += (reference_value - estimate_value).powi(2);
        }
    }
    sum / (count * 4) as f32
}

fn visual_quality_metrics(
    reference: &[[f32; 4]],
    estimate: &[[f32; 4]],
    previous_estimate: &[[f32; 4]],
) -> VisualQualityMetrics {
    let count = reference.len().min(estimate.len());
    let previous_count = estimate.len().min(previous_estimate.len());
    if count == 0 {
        return VisualQualityMetrics {
            mse: 0.0,
            psnr: 120.0,
            ssim: 0.0,
            flicker_mse: 0.0,
            ghost_trail: 0.0,
            shimmer: 0.0,
            difference_map: Vec::new(),
        };
    }

    let mut mse_sum = 0.0_f32;
    let mut flicker_sum = 0.0_f32;
    let mut ghost_sum = 0.0_f32;
    let mut shimmer_sum = 0.0_f32;
    let mut reference_luma_sum = 0.0_f32;
    let mut estimate_luma_sum = 0.0_f32;
    let mut difference_map = Vec::with_capacity(count);
    for index in 0..count.max(previous_count) {
        if index < count {
            let reference_pixel = reference[index];
            let estimate_pixel = estimate[index];
            for (reference_value, estimate_value) in reference_pixel.iter().zip(estimate_pixel) {
                mse_sum += (reference_value - estimate_value).powi(2);
            }
            reference_luma_sum += luma_value(reference_pixel);
            estimate_luma_sum += luma_value(estimate_pixel);
            for (reference_value, estimate_value) in
                reference_pixel[..3].iter().zip(estimate_pixel[..3].iter())
            {
                ghost_sum += (reference_value - estimate_value).abs();
            }
            difference_map.push([
                (reference_pixel[0] - estimate_pixel[0]).abs(),
                (reference_pixel[1] - estimate_pixel[1]).abs(),
                (reference_pixel[2] - estimate_pixel[2]).abs(),
                1.0,
            ]);
        }
        if index < previous_count {
            let estimate_pixel = estimate[index];
            let previous_pixel = previous_estimate[index];
            for (estimate_value, previous_value) in estimate_pixel.iter().zip(previous_pixel) {
                flicker_sum += (estimate_value - previous_value).powi(2);
            }
            for (estimate_value, previous_value) in
                estimate_pixel[..3].iter().zip(previous_pixel[..3].iter())
            {
                shimmer_sum += (estimate_value - previous_value).abs();
            }
        }
    }

    let reference_mean = reference_luma_sum / count as f32;
    let estimate_mean = estimate_luma_sum / count as f32;
    let mut reference_variance = 0.0;
    let mut estimate_variance = 0.0;
    let mut covariance = 0.0;
    for index in 0..count {
        let reference_delta = luma_value(reference[index]) - reference_mean;
        let estimate_delta = luma_value(estimate[index]) - estimate_mean;
        reference_variance += reference_delta * reference_delta;
        estimate_variance += estimate_delta * estimate_delta;
        covariance += reference_delta * estimate_delta;
    }
    let denominator = (count.saturating_sub(1).max(1)) as f32;
    reference_variance /= denominator;
    estimate_variance /= denominator;
    covariance /= denominator;
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);
    let ssim = ((2.0 * reference_mean * estimate_mean + c1) * (2.0 * covariance + c2))
        / ((reference_mean.powi(2) + estimate_mean.powi(2) + c1)
            * (reference_variance + estimate_variance + c2));
    let mse = mse_sum / (count * 4) as f32;
    let psnr = if mse <= f32::EPSILON {
        120.0
    } else {
        10.0 * (1.0 / mse).log10()
    };
    let flicker_mse = flicker_sum / (previous_count.max(1) * 4) as f32;
    let ghost_trail = ghost_sum / (count.max(1) * 3) as f32;
    let shimmer = shimmer_sum / (previous_count.max(1) * 3) as f32;
    VisualQualityMetrics {
        mse,
        psnr,
        ssim,
        flicker_mse,
        ghost_trail,
        shimmer,
        difference_map,
    }
}

#[derive(Debug, Deserialize)]
struct CaptureManifest {
    frame_id: u64,
    generation_id: u64,
    game_extent: [u32; 2],
    guidance_extent: [u32; 2],
    output_extent: [u32; 2],
    viewport: [f32; 4],
    reset_reason: String,
    backend: String,
    guidance_mode: String,
    ablations: CaptureAblations,
    sharpening: CaptureSharpening,
    fsr_inputs: CaptureFsrInputs,
    history_age: u64,
    gpu_timings_ms: Vec<f32>,
    resources: Vec<CaptureResource>,
}

#[derive(Debug, Deserialize)]
struct CaptureAblations {
    motion: bool,
    relative_depth: bool,
    reactive: bool,
    composition: bool,
    exposure: bool,
    confidence_disocclusion: bool,
    post_capture_jitter: bool,
}

#[derive(Debug, Deserialize)]
struct CaptureSharpening {
    enabled: bool,
    sharpness: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct CaptureFsrInputs {
    motion: String,
    confidence: String,
    depth: String,
    exposure: String,
    reactive: String,
    composition: String,
    jitter: String,
}

#[derive(Debug, Deserialize)]
struct CaptureResource {
    name: String,
    file: String,
    format: String,
    extent: [u32; 2],
    bytes: usize,
}

fn read_capture_resource(
    directory: &Path,
    manifest: &CaptureManifest,
    name: &str,
) -> Result<(Vec<[f32; 4]>, [u32; 2]), String> {
    let resource = manifest
        .resources
        .iter()
        .find(|resource| resource.name == name)
        .ok_or_else(|| format!("capture is missing resource {name}"))?;
    let bytes = fs::read(directory.join(&resource.file))
        .map_err(|error| format!("read {}: {error}", resource.file))?;
    if bytes.len() != resource.bytes {
        return Err(format!(
            "capture resource {} has {} bytes, metadata declares {}",
            resource.name,
            bytes.len(),
            resource.bytes
        ));
    }
    let expected_pixels = resource.extent[0] as usize * resource.extent[1] as usize;
    if !matches!(
        resource.format.as_str(),
        "R8G8B8A8_UNORM" | "B8G8R8A8_UNORM" | "R8G8B8A8_SRGB" | "B8G8R8A8_SRGB"
    ) {
        return Err(format!(
            "visual-quality currently requires an RGBA8 color resource, got {} for {name}",
            resource.format
        ));
    }
    if bytes.len() != expected_pixels.saturating_mul(4) {
        return Err(format!(
            "capture resource {name} has an invalid RGBA8 extent/byte count"
        ));
    }
    let image = bytes
        .chunks(4)
        .map(|pixel| {
            if resource.format.starts_with('B') {
                [
                    f32::from(pixel[2]) / 255.0,
                    f32::from(pixel[1]) / 255.0,
                    f32::from(pixel[0]) / 255.0,
                    f32::from(pixel[3]) / 255.0,
                ]
            } else {
                [
                    f32::from(pixel[0]) / 255.0,
                    f32::from(pixel[1]) / 255.0,
                    f32::from(pixel[2]) / 255.0,
                    f32::from(pixel[3]) / 255.0,
                ]
            }
        })
        .collect();
    Ok((image, resource.extent))
}

fn read_capture_series(
    directory: &Path,
    options: &VisualQualityOptions,
    expected_backend: &str,
) -> Result<Vec<CaptureManifest>, String> {
    let mut manifests = fs::read_dir(directory)
        .map_err(|error| format!("read capture directory {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|value| value == "json")
        })
        .filter_map(|entry| {
            fs::read_to_string(entry.path())
                .ok()
                .and_then(|text| serde_json::from_str::<CaptureManifest>(&text).ok())
        })
        .collect::<Vec<_>>();
    manifests.sort_by_key(|manifest| manifest.frame_id);
    let end_frame = options
        .warmup
        .checked_add(options.frames)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "visual-quality frame range overflowed".to_owned())?
        as u64;
    let selected = manifests
        .into_iter()
        .filter(|manifest| {
            manifest.frame_id >= options.warmup as u64 && manifest.frame_id <= end_frame
        })
        .collect::<Vec<_>>();
    if selected.len() != options.frames {
        return Err(format!(
            "expected {} captured frames in {}..={}, found {}",
            options.frames,
            options.warmup,
            end_frame,
            selected.len()
        ));
    }
    let mut stable_estimated_motion_seen = false;
    for manifest in &selected {
        if manifest.game_extent != [options.input.0, options.input.1]
            || manifest.output_extent != [options.output.0, options.output.1]
            || manifest.guidance_extent != manifest.game_extent
        {
            return Err(format!(
                "frame {} has incorrect capture extents",
                manifest.frame_id
            ));
        }
        if manifest.backend != expected_backend {
            return Err(format!(
                "frame {} active backend is {}, expected {expected_backend}",
                manifest.frame_id, manifest.backend
            ));
        }
        if manifest.generation_id == u64::MAX
            || manifest.gpu_timings_ms.len() != 15
            || manifest.guidance_mode.is_empty()
            || !manifest.viewport.iter().all(|value| value.is_finite())
            || !manifest
                .gpu_timings_ms
                .iter()
                .all(|value| value.is_finite())
            || !manifest.sharpening.sharpness.is_finite()
        {
            return Err(format!(
                "frame {} has invalid diagnostic metadata",
                manifest.frame_id
            ));
        }
        if expected_backend != "Off" && manifest.history_age == 0 {
            return Err(format!(
                "frame {} has stale history after warmup",
                manifest.frame_id
            ));
        }
        if expected_backend == "Off" && manifest.sharpening.enabled {
            return Err(format!(
                "frame {} unexpectedly has sharpening enabled in the Off capture",
                manifest.frame_id
            ));
        }
        if expected_backend == "FSR 3.1.4"
            && !fsr_input_contract_is_safe_for_frame(
                &manifest.guidance_mode,
                &manifest.fsr_inputs,
                manifest.history_age,
                &manifest.reset_reason,
            )
        {
            return Err(format!(
                "frame {} reports an unsafe FSR input contract",
                manifest.frame_id
            ));
        }
        if expected_backend == "FSR 3.1.4"
            && manifest.guidance_mode == "Estimated"
            && manifest.fsr_inputs.motion == "Estimated"
            && manifest.fsr_inputs.confidence == "Estimated"
        {
            stable_estimated_motion_seen = true;
        }
        if manifest.reset_reason == "ProviderFailure" {
            return Err(format!(
                "frame {} reports active provider fallback",
                manifest.frame_id
            ));
        }
    }
    if expected_backend == "FSR 3.1.4"
        && selected
            .iter()
            .any(|manifest| manifest.guidance_mode == "Estimated")
        && !stable_estimated_motion_seen
    {
        return Err("FSR Estimated capture contains no stable estimated-motion frame".into());
    }
    Ok(selected)
}

#[cfg(test)]
fn fsr_input_contract_is_safe(guidance_mode: &str, inputs: &CaptureFsrInputs) -> bool {
    fsr_input_contract_is_safe_for_frame(guidance_mode, inputs, u64::MAX, "None")
}

fn fsr_input_contract_is_safe_for_frame(
    guidance_mode: &str,
    inputs: &CaptureFsrInputs,
    history_age: u64,
    reset_reason: &str,
) -> bool {
    let motion_is_estimated = inputs.motion == "Estimated" && inputs.confidence == "Estimated";
    let motion_is_neutral = inputs.motion == "Neutral" && inputs.confidence == "Neutral";
    let (motion, confidence) = match guidance_mode {
        "Estimated" if motion_is_estimated => ("Estimated", "Estimated"),
        "Estimated" if motion_is_neutral && (history_age <= 1 || reset_reason != "None") => {
            ("Neutral", "Neutral")
        }
        "Zero" => ("Neutral", "Neutral"),
        _ => return false,
    };
    inputs.motion == motion
        && inputs.confidence == confidence
        && inputs.depth == "SuppressedIncompatible"
        && inputs.exposure == "SuppressedIncompatible"
        && inputs.reactive == "Neutral"
        && inputs.composition == "Neutral"
        && inputs.jitter == "Neutral"
}

fn guidance_comparison_gate_passed(
    estimated_psnr: f32,
    zero_psnr: f32,
    estimated_ssim: f32,
    zero_ssim: f32,
    estimated_flicker: f32,
    zero_flicker: f32,
) -> bool {
    estimated_psnr.is_finite()
        && zero_psnr.is_finite()
        && estimated_ssim.is_finite()
        && zero_ssim.is_finite()
        && estimated_flicker.is_finite()
        && zero_flicker.is_finite()
        && estimated_flicker <= zero_flicker + 0.002
        && estimated_psnr + 0.25 >= zero_psnr
        && estimated_ssim + 0.005 >= zero_ssim
}

fn scene_category(frame_id: u64) -> &'static str {
    [
        "translation",
        "rotation_scaling",
        "thin_geometry_vegetation",
        "hud_text",
        "transparency_particles",
        "emissive_noise",
        "occlusion_disocclusion",
        "pause_resume_scene_cut",
    ][frame_id as usize % 8]
}

#[derive(Debug, Clone, Serialize)]
struct CategoryReport {
    category: String,
    frames: usize,
    psnr_db: f32,
    ssim: f32,
    flicker_mse: f32,
    ghost_trail: f32,
    shimmer: f32,
    gate_passed: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct AblationReport {
    signal: String,
    estimated: String,
    fallback: String,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct PresetReport {
    preset: String,
    quality_baseline: bool,
    work_units: u64,
    timing_status: String,
}

#[derive(Debug, Clone, Serialize)]
struct GuidanceComparisonReport {
    frames: usize,
    source_frames_match: bool,
    estimated_psnr_db: f32,
    zero_psnr_db: f32,
    estimated_ssim: f32,
    zero_ssim: f32,
    estimated_flicker_mse: f32,
    zero_flicker_mse: f32,
    estimated_zero_mse: f32,
    temporal_gate_passed: bool,
    status: String,
}

#[derive(Debug, Serialize)]
struct DiagnosticReport {
    guidance_mode: String,
    frames: usize,
    history_age_min: u64,
    history_age_max: u64,
    gpu_timing_mean_ms: Vec<f32>,
    guidance_resources: Vec<String>,
    guidance_resource_bytes_valid: bool,
}

#[derive(Debug, Serialize)]
struct VisualQualityReport {
    command: String,
    display: String,
    input: [u32; 2],
    output: [u32; 2],
    warmup: usize,
    frames: usize,
    source_frames_match: bool,
    fsr_backend: String,
    off_backend: String,
    categories: Vec<CategoryReport>,
    guidance_comparison: GuidanceComparisonReport,
    ablations: Vec<AblationReport>,
    presets: Vec<PresetReport>,
    diagnostics: DiagnosticReport,
    artifacts: Vec<String>,
    gate: String,
}

fn finite_image(image: &[[f32; 4]]) -> bool {
    image.iter().flatten().all(|value| value.is_finite())
}

fn write_png(path: &Path, image: &[[f32; 4]], width: u32, height: u32) -> Result<(), String> {
    if image.len() != width as usize * height as usize || !finite_image(image) {
        return Err(format!("cannot encode invalid image at {}", path.display()));
    }
    let mut bytes = Vec::with_capacity(image.len() * 4);
    for pixel in image {
        for value in pixel {
            bytes.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    let file =
        fs::File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("write {} header: {error}", path.display()))?;
    writer
        .write_image_data(&bytes)
        .map_err(|error| format!("write {}: {error}", path.display()))
}

fn side_by_side(left: &[[f32; 4]], right: &[[f32; 4]], width: u32, height: u32) -> Vec<[f32; 4]> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    for row in 0..height as usize {
        let start = row * width as usize;
        result.extend_from_slice(&left[start..start + width as usize]);
        result.extend_from_slice(&right[start..start + width as usize]);
    }
    result
}

fn vertical_wipe(left: &[[f32; 4]], right: &[[f32; 4]], width: u32, height: u32) -> Vec<[f32; 4]> {
    left.iter()
        .zip(right.iter())
        .enumerate()
        .map(|(index, (left, right))| {
            if index % (width as usize) < width as usize / 2 {
                *left
            } else {
                *right
            }
        })
        .take(width as usize * height as usize)
        .collect()
}

fn magnified_crop(image: &[[f32; 4]], width: u32, height: u32) -> (Vec<[f32; 4]>, u32, u32) {
    let crop_width = (width / 4).max(1);
    let crop_height = (height / 4).max(1);
    let left = (width - crop_width) / 2;
    let top = (height - crop_height) / 2;
    let scale = 4_u32;
    let mut crop = Vec::with_capacity((crop_width * scale * crop_height * scale) as usize);
    for y in 0..crop_height {
        for _ in 0..scale {
            for x in 0..crop_width {
                for _ in 0..scale {
                    crop.push(image[(top + y) as usize * width as usize + (left + x) as usize]);
                }
            }
        }
    }
    (crop, crop_width * scale, crop_height * scale)
}

fn amplified_difference(
    reference: &[[f32; 4]],
    estimate: &[[f32; 4]],
    multiplier: f32,
) -> Vec<[f32; 4]> {
    reference
        .iter()
        .zip(estimate.iter())
        .map(|(reference, estimate)| {
            [
                ((reference[0] - estimate[0]).abs() * multiplier).clamp(0.0, 1.0),
                ((reference[1] - estimate[1]).abs() * multiplier).clamp(0.0, 1.0),
                ((reference[2] - estimate[2]).abs() * multiplier).clamp(0.0, 1.0),
                1.0,
            ]
        })
        .collect()
}

struct NestedMutter {
    mutter: Child,
    runtime_dir: PathBuf,
    wayland_display: String,
    display: String,
    xauthority: PathBuf,
}

impl NestedMutter {
    fn environment(&self) -> [(&str, &str); 4] {
        [
            (
                "XDG_RUNTIME_DIR",
                self.runtime_dir.to_str().unwrap_or("/tmp"),
            ),
            ("WAYLAND_DISPLAY", &self.wayland_display),
            ("DISPLAY", &self.display),
            (
                "XAUTHORITY",
                self.xauthority.to_str().unwrap_or("/dev/null"),
            ),
        ]
    }
}

impl Drop for NestedMutter {
    fn drop(&mut self) {
        if self.mutter.try_wait().ok().flatten().is_none() {
            let _ = self.mutter.kill();
            let _ = self.mutter.wait();
        }
    }
}

fn wait_for_path(path: &Path, child: &mut Child, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return Ok(());
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("poll compositor: {error}"))?
        {
            return Err(format!("nested compositor exited with {status}"));
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("timed out waiting for {}", path.display()))
}

fn parse_public_x11_display(log: &str) -> Option<String> {
    log.lines().find_map(|line| {
        let (_, remainder) = line.split_once("Using public X11 display ")?;
        let display = remainder.split(',').next()?.trim();
        (!display.is_empty()).then(|| display.to_owned())
    })
}

fn private_xauthority(runtime_dir: &Path) -> Option<PathBuf> {
    fs::read_dir(runtime_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".mutter-Xwaylandauth."))
        })
}

fn wait_for_nested_x11(
    log_path: &Path,
    runtime_dir: &Path,
    child: &mut Child,
    timeout: Duration,
) -> Result<(String, PathBuf), String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let log = fs::read_to_string(log_path).unwrap_or_default();
        if let (Some(display), Some(xauthority)) = (
            parse_public_x11_display(&log),
            private_xauthority(runtime_dir),
        ) {
            return Ok((display, xauthority));
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("poll nested Mutter: {error}"))?
        {
            return Err(format!(
                "nested Mutter exited with {status}; see {}",
                log_path.display()
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!(
        "timed out waiting for a public X11 display; see {}",
        log_path.display()
    ))
}

fn start_nested_mutter(width: u32, height: u32) -> Result<NestedMutter, String> {
    let runtime_dir =
        std::env::temp_dir().join(format!("tuxscaling-wayland-{}", std::process::id()));
    fs::create_dir_all(&runtime_dir)
        .map_err(|error| format!("create private Wayland runtime directory: {error}"))?;
    #[cfg(unix)]
    fs::set_permissions(&runtime_dir, fs::Permissions::from_mode(0o700))
        .map_err(|error| format!("secure private Wayland runtime directory: {error}"))?;
    let wayland_display = format!("tuxscaling-visual-{}", std::process::id());
    let wayland_socket = runtime_dir.join(&wayland_display);
    let log_path = runtime_dir.join("mutter.log");
    let log = fs::File::create(&log_path)
        .map_err(|error| format!("create nested Mutter log: {error}"))?;
    let log_stdout = log
        .try_clone()
        .map_err(|error| format!("clone nested Mutter log: {error}"))?;
    let mut mutter = Command::new("mutter")
        .args([
            "--wayland",
            "--headless",
            "--virtual-monitor",
            &format!("{width}x{height}"),
            "--wayland-display",
            &wayland_display,
        ])
        .env("XDG_RUNTIME_DIR", &runtime_dir)
        .env("XDG_CONFIG_HOME", &runtime_dir)
        .env("GSETTINGS_BACKEND", "memory")
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("XAUTHORITY")
        .stdout(Stdio::from(log_stdout))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|error| format!("start nested Mutter: {error}"))?;
    if let Err(error) = wait_for_path(&wayland_socket, &mut mutter, Duration::from_secs(5)) {
        let _ = mutter.kill();
        let _ = mutter.wait();
        return Err(error);
    }
    let (display, xauthority) =
        match wait_for_nested_x11(&log_path, &runtime_dir, &mut mutter, Duration::from_secs(5)) {
            Ok(value) => value,
            Err(error) => {
                let _ = mutter.kill();
                let _ = mutter.wait();
                return Err(error);
            }
        };
    Ok(NestedMutter {
        mutter,
        runtime_dir,
        wayland_display,
        display,
        xauthority,
    })
}

fn capture_command(
    root: &Path,
    binary: &Path,
    config: &Path,
    capture_dir: &Path,
    guard: &NestedMutter,
    options: &VisualQualityOptions,
    libraries: &std::ffi::OsString,
) -> Command {
    let end_frame = options
        .warmup
        .saturating_add(options.frames)
        .saturating_sub(1);
    let mut command = Command::new(binary);
    validation(&mut command)
        .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
        .env("LD_LIBRARY_PATH", libraries)
        .env(
            "VK_INSTANCE_LAYERS",
            "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
        )
        .env("TUXSCALING_VIEW", "reconstructed")
        .env("TUXSCALING_CONFIG", config)
        .env("TUXSCALING_TEST_SCENARIO", "upscale")
        .env("TUXSCALING_TEST_RESIZE_INTERVAL", "0")
        .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
        .env(
            "TUXSCALING_TEST_FRAMES",
            end_frame.saturating_add(1).to_string(),
        )
        .env("TUXSCALING_TEST_SINGLE_WINDOW", "1")
        .env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "0")
        .env("TUXSCALING_TEST_FORCE_RESIZE_FAILURE", "0")
        .env("TUXSCALING_CAPTURE_DIR", capture_dir)
        .env("TUXSCALING_CAPTURE_MAX_FRAMES", options.frames.to_string())
        .env("TUXSCALING_CAPTURE_START_FRAME", options.warmup.to_string())
        .env("TUXSCALING_CAPTURE_END_FRAME", end_frame.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (name, value) in guard.environment() {
        command.env(name, value);
    }
    command
}

#[allow(clippy::too_many_arguments)]
fn run_visual_capture(
    root: &Path,
    binary: &Path,
    config: &Path,
    capture_dir: &Path,
    guard: &NestedMutter,
    options: &VisualQualityOptions,
    libraries: &std::ffi::OsString,
    log_path: &Path,
) -> Result<(), String> {
    let result = command_output_with_timeout(
        capture_command(root, binary, config, capture_dir, guard, options, libraries),
        90,
    )
    .map_err(|error| format!("visual scene could not start: {error}"))?;
    let (output, timed_out) = result;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    fs::write(log_path, format!("{stdout}\n{stderr}"))
        .map_err(|error| format!("write visual scene log {}: {error}", log_path.display()))?;
    if timed_out {
        return Err("visual scene timed out".into());
    }
    if !output.status.success() {
        return Err(format!(
            "visual scene exited with {} (see {})",
            output.status,
            log_path.display()
        ));
    }
    if stdout.to_ascii_lowercase().contains("validation error")
        || stderr.to_ascii_lowercase().contains("validation error")
    {
        return Err(format!(
            "visual scene emitted validation errors (see {})",
            log_path.display()
        ));
    }
    Ok(())
}

fn update_category(
    categories: &mut BTreeMap<String, (usize, f32, f32, f32, f32, f32)>,
    category: &str,
    metrics: &VisualQualityMetrics,
) {
    let entry = categories
        .entry(category.to_owned())
        .or_insert((0, 0.0, 0.0, 0.0, 0.0, 0.0));
    entry.0 += 1;
    entry.1 += metrics.psnr;
    entry.2 += metrics.ssim;
    entry.3 += metrics.flicker_mse;
    entry.4 += metrics.ghost_trail;
    entry.5 += metrics.shimmer;
}

fn validate_diagnostic_resources(
    directory: &Path,
    manifest: &CaptureManifest,
) -> Result<Vec<String>, String> {
    let required = [
        "motion",
        "confidence",
        "disocclusion",
        "reactive",
        "composition",
        "relative_depth",
        "exposure",
    ];
    for name in required {
        let resource = manifest
            .resources
            .iter()
            .find(|resource| resource.name == name)
            .ok_or_else(|| format!("capture is missing diagnostic resource {name}"))?;
        let bytes = fs::read(directory.join(&resource.file))
            .map_err(|error| format!("read diagnostic resource {}: {error}", resource.file))?;
        if bytes.len() != resource.bytes || resource.extent.contains(&0) {
            return Err(format!(
                "diagnostic resource {name} has invalid metadata or bytes"
            ));
        }
    }
    Ok(required.into_iter().map(String::from).collect())
}

fn category_gate_passed(psnr: f32, ssim: f32, flicker: f32, ghost: f32, shimmer: f32) -> bool {
    // These are conservative engineering gates for an 8-bit Off-baseline
    // comparison. They reject broken output without treating the baseline as
    // ground truth.
    psnr >= 10.0 && ssim >= 0.75 && flicker <= 0.10 && ghost <= 0.50 && shimmer <= 0.50
}

#[allow(clippy::too_many_arguments)]
fn visual_quality_report(
    options: &VisualQualityOptions,
    fsr_directory: &Path,
    zero_directory: &Path,
    off_directory: &Path,
    fsr: &[CaptureManifest],
    zero: &[CaptureManifest],
    off: &[CaptureManifest],
    report_dir: &Path,
    evidence: Option<&CollectedQualityEvidence>,
) -> Result<VisualQualityReport, String> {
    if fsr.len() != zero.len() || fsr.len() != off.len() || fsr.is_empty() {
        return Err("FSR, Zero, and Off capture series have different or empty lengths".into());
    }
    let width = options.output.0;
    let height = options.output.1;
    let mut categories = BTreeMap::new();
    let (mut previous, _) = read_capture_resource(off_directory, &off[0], "spatial_off")?;
    let (mut previous_zero, _) = read_capture_resource(zero_directory, &zero[0], "reconstructed")?;
    let mut source_frames_match = true;
    let mut all_finite = true;
    let mut estimated_psnr_total = 0.0;
    let mut zero_psnr_total = 0.0;
    let mut estimated_ssim_total = 0.0;
    let mut zero_ssim_total = 0.0;
    let mut estimated_flicker_total = 0.0;
    let mut zero_flicker_total = 0.0;
    let mut estimated_zero_mse_total = 0.0;
    for ((fsr_manifest, zero_manifest), off_manifest) in fsr.iter().zip(zero.iter()).zip(off.iter())
    {
        if fsr_manifest.frame_id != zero_manifest.frame_id
            || fsr_manifest.frame_id != off_manifest.frame_id
        {
            source_frames_match = false;
            break;
        }
        let (fsr_source, _) = read_capture_resource(fsr_directory, fsr_manifest, "source")?;
        let (fsr_output, fsr_output_extent) =
            read_capture_resource(fsr_directory, fsr_manifest, "reconstructed")?;
        let (zero_source, _) = read_capture_resource(zero_directory, zero_manifest, "source")?;
        let (zero_output, zero_output_extent) =
            read_capture_resource(zero_directory, zero_manifest, "reconstructed")?;
        let (off_source, _) = read_capture_resource(off_directory, off_manifest, "source")?;
        let (off_output, off_output_extent) =
            read_capture_resource(off_directory, off_manifest, "spatial_off")?;
        source_frames_match &= fsr_source == off_source && fsr_source == zero_source;
        if !source_frames_match {
            break;
        }
        if fsr_output_extent != [width, height]
            || zero_output_extent != [width, height]
            || off_output_extent != [width, height]
            || fsr_output.len() != zero_output.len()
            || fsr_output.len() != off_output.len()
            || fsr_output.len() != width as usize * height as usize
        {
            return Err(format!(
                "frame {} output extent does not match requested output",
                fsr_manifest.frame_id
            ));
        }
        let metrics = visual_quality_metrics(&off_output, &fsr_output, &previous);
        let zero_metrics = visual_quality_metrics(&off_output, &zero_output, &previous_zero);
        let estimated_zero_mse = image_mse(&fsr_output, &zero_output);
        all_finite &= metrics.mse.is_finite()
            && metrics.psnr.is_finite()
            && metrics.ssim.is_finite()
            && metrics.flicker_mse.is_finite()
            && metrics.ghost_trail.is_finite()
            && metrics.shimmer.is_finite()
            && finite_image(&metrics.difference_map)
            && zero_metrics.mse.is_finite()
            && zero_metrics.psnr.is_finite()
            && zero_metrics.ssim.is_finite()
            && zero_metrics.flicker_mse.is_finite()
            && zero_metrics.ghost_trail.is_finite()
            && zero_metrics.shimmer.is_finite()
            && finite_image(&zero_metrics.difference_map)
            && estimated_zero_mse.is_finite();
        estimated_psnr_total += metrics.psnr;
        zero_psnr_total += zero_metrics.psnr;
        estimated_ssim_total += metrics.ssim;
        zero_ssim_total += zero_metrics.ssim;
        estimated_flicker_total += metrics.flicker_mse;
        zero_flicker_total += zero_metrics.flicker_mse;
        estimated_zero_mse_total += estimated_zero_mse;
        update_category(
            &mut categories,
            scene_category(fsr_manifest.frame_id),
            &metrics,
        );
        previous = fsr_output;
        previous_zero = zero_output;
    }
    if !source_frames_match {
        return Err("FSR, Zero, and Off captures do not share identical source frames".into());
    }
    if !all_finite {
        return Err("visual-quality metrics contain non-finite values".into());
    }

    let first_fsr = &fsr[0];
    let first_off = &off[0];
    let guidance_resources = validate_diagnostic_resources(fsr_directory, first_fsr)?;
    let (first_fsr_output, first_fsr_output_extent) =
        read_capture_resource(fsr_directory, first_fsr, "reconstructed")?;
    let first_zero = &zero[0];
    let (first_zero_output, first_zero_output_extent) =
        read_capture_resource(zero_directory, first_zero, "reconstructed")?;
    let (first_off_output, first_off_output_extent) =
        read_capture_resource(off_directory, first_off, "spatial_off")?;
    if first_fsr_output_extent != [width, height]
        || first_zero_output_extent != [width, height]
        || first_off_output_extent != [width, height]
    {
        return Err("first visual-quality output extent is invalid".into());
    }
    write_png(
        &report_dir.join("fsr.png"),
        &first_fsr_output,
        width,
        height,
    )?;
    write_png(
        &report_dir.join("off.png"),
        &first_off_output,
        width,
        height,
    )?;
    write_png(
        &report_dir.join("zero.png"),
        &first_zero_output,
        width,
        height,
    )?;
    write_png(
        &report_dir.join("side-by-side.png"),
        &side_by_side(&first_off_output, &first_fsr_output, width, height),
        width.saturating_mul(2),
        height,
    )?;
    write_png(
        &report_dir.join("wipe.png"),
        &vertical_wipe(&first_off_output, &first_fsr_output, width, height),
        width,
        height,
    )?;
    let (crop, crop_width, crop_height) = magnified_crop(&first_fsr_output, width, height);
    write_png(
        &report_dir.join("magnified.png"),
        &crop,
        crop_width,
        crop_height,
    )?;
    write_png(
        &report_dir.join("amplified-diff.png"),
        &amplified_difference(&first_off_output, &first_fsr_output, 8.0),
        width,
        height,
    )?;
    write_png(
        &report_dir.join("estimated-zero-side-by-side.png"),
        &side_by_side(&first_zero_output, &first_fsr_output, width, height),
        width.saturating_mul(2),
        height,
    )?;
    write_png(
        &report_dir.join("estimated-zero-diff.png"),
        &amplified_difference(&first_zero_output, &first_fsr_output, 8.0),
        width,
        height,
    )?;

    let categories = categories
        .into_iter()
        .map(
            |(category, (frames, psnr, ssim, flicker, ghost, shimmer))| CategoryReport {
                category,
                frames,
                psnr_db: psnr / frames as f32,
                ssim: ssim / frames as f32,
                flicker_mse: flicker / frames as f32,
                ghost_trail: ghost / frames as f32,
                shimmer: shimmer / frames as f32,
                gate_passed: category_gate_passed(
                    psnr / frames as f32,
                    ssim / frames as f32,
                    flicker / frames as f32,
                    ghost / frames as f32,
                    shimmer / frames as f32,
                ),
                status: "measured against the captured Off baseline".into(),
            },
        )
        .collect::<Vec<_>>();
    let category_gates_passed = categories.iter().all(|category| category.gate_passed);
    let frame_count = fsr.len() as f32;
    let estimated_psnr = estimated_psnr_total / frame_count;
    let zero_psnr = zero_psnr_total / frame_count;
    let estimated_ssim = estimated_ssim_total / frame_count;
    let zero_ssim = zero_ssim_total / frame_count;
    let estimated_flicker = estimated_flicker_total / frame_count;
    let zero_flicker = zero_flicker_total / frame_count;
    let estimated_zero_mse = estimated_zero_mse_total / frame_count;
    let temporal_gate_passed = guidance_comparison_gate_passed(
        estimated_psnr,
        zero_psnr,
        estimated_ssim,
        zero_ssim,
        estimated_flicker,
        zero_flicker,
    );
    let guidance_comparison = GuidanceComparisonReport {
        frames: fsr.len(),
        source_frames_match,
        estimated_psnr_db: estimated_psnr,
        zero_psnr_db: zero_psnr,
        estimated_ssim,
        zero_ssim,
        estimated_flicker_mse: estimated_flicker,
        zero_flicker_mse: zero_flicker,
        estimated_zero_mse,
        temporal_gate_passed,
        status: if temporal_gate_passed {
            "Estimated guidance is no worse than Zero within the deterministic tolerance; values are measured against the captured Off baseline".into()
        } else {
            "Estimated guidance regressed against Zero; keep the captured evidence and investigate estimator quality before enabling a stronger policy".into()
        },
    };
    let ablations = match evidence {
        Some(evidence) => quality_ablation_rows(evidence),
        None => [
            ("motion", first_fsr.ablations.motion),
            ("relative_depth", first_fsr.ablations.relative_depth),
            ("reactive", first_fsr.ablations.reactive),
            ("composition", first_fsr.ablations.composition),
            ("exposure", first_fsr.ablations.exposure),
            (
                "confidence_disocclusion",
                first_fsr.ablations.confidence_disocclusion,
            ),
            (
                "post_capture_jitter",
                first_fsr.ablations.post_capture_jitter,
            ),
        ]
        .into_iter()
        .map(|(signal, disabled)| AblationReport {
            signal: signal.into(),
            estimated: (!disabled).to_string(),
            fallback: disabled.to_string(),
            status: "default Estimated run; GPU evidence skipped (--skip-gpu-evidence), run without the flag for the measured ablation matrix".into(),
        })
        .collect(),
    };
    let presets = match evidence {
        Some(evidence) => quality_preset_rows(evidence, options.input),
        None => [MotionQuality::Balanced, MotionQuality::Performance]
            .into_iter()
            .map(|quality| PresetReport {
                preset: format!("{quality:?}"),
                quality_baseline: quality == MotionQuality::Balanced,
                work_units: quality
                    .dispatch_plan(options.input.0, options.input.1)
                    .candidate_evaluations,
                timing_status:
                    "capture GPU timings are present; GPU evidence skipped (--skip-gpu-evidence), run without the flag for the measured preset comparison".into(),
            })
            .collect(),
    };
    let mut timing_sum = vec![0.0_f32; first_fsr.gpu_timings_ms.len()];
    for frame in fsr {
        for (sum, value) in timing_sum.iter_mut().zip(frame.gpu_timings_ms.iter()) {
            *sum += *value;
        }
    }
    let diagnostics = DiagnosticReport {
        guidance_mode: first_fsr.guidance_mode.clone(),
        frames: fsr.len(),
        history_age_min: fsr.iter().map(|frame| frame.history_age).min().unwrap_or(0),
        history_age_max: fsr.iter().map(|frame| frame.history_age).max().unwrap_or(0),
        gpu_timing_mean_ms: timing_sum
            .into_iter()
            .map(|value| value / fsr.len() as f32)
            .collect(),
        guidance_resources,
        guidance_resource_bytes_valid: true,
    };
    let artifacts = [
        "fsr.png",
        "zero.png",
        "off.png",
        "side-by-side.png",
        "wipe.png",
        "magnified.png",
        "amplified-diff.png",
        "estimated-zero-side-by-side.png",
        "estimated-zero-diff.png",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    Ok(VisualQualityReport {
        command: format!(
            "cargo xtask visual-quality --display {} --input {}x{} --output {}x{} --warmup {} --frames {}",
            options.display,
            options.input.0,
            options.input.1,
            options.output.0,
            options.output.1,
            options.warmup,
            options.frames
        ),
        display: options.display.clone(),
        input: [options.input.0, options.input.1],
        output: [options.output.0, options.output.1],
        warmup: options.warmup,
        frames: options.frames,
        source_frames_match,
        fsr_backend: first_fsr.backend.clone(),
        off_backend: first_off.backend.clone(),
        categories,
        guidance_comparison,
        ablations,
        presets,
        diagnostics,
        artifacts,
        gate: if category_gates_passed && temporal_gate_passed {
            "passed: captures, extents, history, validation, finite metrics, source-frame identity, category thresholds, and Estimated-versus-Zero guidance checks are valid; values are an Off-baseline comparison, not a ground-truth claim".into()
        } else {
            "failed: one or more category thresholds or Estimated-versus-Zero guidance checks did not pass".into()
        },
    })
}

fn write_visual_quality_markdown(report: &VisualQualityReport, path: &Path) -> Result<(), String> {
    let mut markdown = format!(
        "# Temporal visual quality\n\n- Command: `{}`\n- Display: `{}`\n- Input: `{}x{}`\n- Output: `{}x{}`\n- Warmup: `{}` frames\n- Measured: `{}` frames\n- Source frames match: `{}`\n- Gate: {}\n\n",
        report.command,
        report.display,
        report.input[0],
        report.input[1],
        report.output[0],
        report.output[1],
        report.warmup,
        report.frames,
        report.source_frames_match,
        report.gate
    );
    markdown.push_str("## Categories\n\n| Category | Frames | PSNR (dB) | SSIM | Flicker MSE | Ghost trail | Shimmer | Gate | Status |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n");
    for category in &report.categories {
        markdown.push_str(&format!(
            "| {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} | {} |\n",
            category.category,
            category.frames,
            category.psnr_db,
            category.ssim,
            category.flicker_mse,
            category.ghost_trail,
            category.shimmer,
            category.gate_passed,
            category.status
        ));
    }
    let comparison = &report.guidance_comparison;
    markdown.push_str(&format!(
        "\n## Guidance comparison\n\n- Frames: `{}`\n- Source frames match across Estimated, Zero, and Off: `{}`\n- Estimated PSNR (dB): `{:.6}`\n- Zero PSNR (dB): `{:.6}`\n- Estimated SSIM: `{:.6}`\n- Zero SSIM: `{:.6}`\n- Estimated flicker MSE: `{:.6}`\n- Zero flicker MSE: `{:.6}`\n- Estimated-versus-Zero output MSE: `{:.6}`\n- Temporal gate: `{}`\n- Status: {}\n",
        comparison.frames,
        comparison.source_frames_match,
        comparison.estimated_psnr_db,
        comparison.zero_psnr_db,
        comparison.estimated_ssim,
        comparison.zero_ssim,
        comparison.estimated_flicker_mse,
        comparison.zero_flicker_mse,
        comparison.estimated_zero_mse,
        comparison.temporal_gate_passed,
        comparison.status
    ));
    markdown.push_str(
        "\n## Ablation matrix\n\n| Signal | Estimated | Fallback | Status |\n|---|---|---|---|\n",
    );
    for ablation in &report.ablations {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            ablation.signal, ablation.estimated, ablation.fallback, ablation.status
        ));
    }
    markdown.push_str("\n## Presets\n\n| Preset | Balanced baseline | Candidate work units | Timing status |\n|---|---|---:|---|\n");
    for preset in &report.presets {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            preset.preset, preset.quality_baseline, preset.work_units, preset.timing_status
        ));
    }
    markdown.push_str(&format!(
        "\n## Diagnostics\n\n- Guidance mode: `{}`\n- History age: `{}`..`{}`\n- GPU timing means (ms): `{}`\n- Guidance resources: `{}`\n- Diagnostic bytes valid: `{}`\n",
        report.diagnostics.guidance_mode,
        report.diagnostics.history_age_min,
        report.diagnostics.history_age_max,
        report
            .diagnostics
            .gpu_timing_mean_ms
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
        report.diagnostics.guidance_resources.join(", "),
        report.diagnostics.guidance_resource_bytes_valid
    ));
    markdown.push_str("\n## Artifacts\n\n");
    for artifact in &report.artifacts {
        markdown.push_str(&format!("- `{artifact}`\n"));
    }
    fs::write(path, markdown).map_err(|error| format!("write {}: {error}", path.display()))
}

fn run_visual_quality(root: &Path, options: &VisualQualityOptions) -> bool {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_secs());
    let report_dir = root.join(format!("target/visual-quality/{timestamp}"));
    let fsr_capture = report_dir.join("capture-fsr");
    let zero_capture = report_dir.join("capture-zero");
    let off_capture = report_dir.join("capture-off");
    if fs::create_dir_all(&fsr_capture).is_err()
        || fs::create_dir_all(&zero_capture).is_err()
        || fs::create_dir_all(&off_capture).is_err()
    {
        eprintln!(
            "cargo xtask visual-quality: unable to create {}",
            report_dir.display()
        );
        return false;
    }
    // Fail fast on the measured ablation/preset evidence before the long
    // scene captures; the GPU suites take seconds, the captures take minutes.
    let evidence = if options.skip_gpu_evidence {
        None
    } else {
        let gpu_dir = report_dir.join("gpu-evidence");
        if fs::create_dir_all(&gpu_dir).is_err() {
            eprintln!(
                "cargo xtask visual-quality: unable to create {}",
                gpu_dir.display()
            );
            return false;
        }
        match collect_quality_evidence(root, &gpu_dir, "cargo xtask visual-quality") {
            Some(evidence) => Some(evidence),
            None => {
                eprintln!("cargo xtask visual-quality: GPU evidence stage failed");
                return false;
            }
        }
    };
    if !run(
        "cargo",
        &[
            "build",
            "-p",
            "tuxscaling-layer",
            "--lib",
            "--example",
            "wsi",
            "--release",
        ],
    ) {
        return false;
    }
    let guard = match start_nested_mutter(options.output.0, options.output.1) {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: unverified nested display: {error}");
            return false;
        }
    };
    let fsr_config = report_dir.join("fsr.toml");
    let zero_config = report_dir.join("zero.toml");
    let off_config = report_dir.join("off.toml");
    let fsr_config_source = generated_config_with_backend_and_sharpening(
        "native",
        1.0,
        Some("balanced"),
        BackendSelection::Fsr314,
        true,
        0.3,
    );
    let zero_config_source = format!("guidance_mode = \"zero\"\n{fsr_config_source}");
    if fs::write(
        &fsr_config,
        fsr_config_source,
    )
    .is_err()
        || fs::write(&zero_config, zero_config_source).is_err()
        || fs::write(
            &off_config,
            "output_resolution = \"native\"\nguidance_scale = 1.0\nmotion_quality = \"balanced\"\nsharpening_enabled = false\nsharpness = 0.0\nupscaler = \"off\"\n",
        )
        .is_err()
    {
        eprintln!("cargo xtask visual-quality: unable to write capture configs");
        return false;
    }
    let binary = root.join("target/release/examples/wsi");
    let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let libraries = std::env::join_paths(
        std::iter::once(root.join("target/release")).chain(std::env::split_paths(&inherited)),
    )
    .unwrap_or(inherited);
    let fsr_result = run_visual_capture(
        root,
        &binary,
        &fsr_config,
        &fsr_capture,
        &guard,
        options,
        &libraries,
        &report_dir.join("fsr-run.log"),
    );
    if let Err(error) = fsr_result {
        eprintln!("cargo xtask visual-quality: FSR capture failed: {error}");
        return false;
    }
    let zero_result = run_visual_capture(
        root,
        &binary,
        &zero_config,
        &zero_capture,
        &guard,
        options,
        &libraries,
        &report_dir.join("zero-run.log"),
    );
    if let Err(error) = zero_result {
        eprintln!("cargo xtask visual-quality: Zero capture failed: {error}");
        return false;
    }
    let off_result = run_visual_capture(
        root,
        &binary,
        &off_config,
        &off_capture,
        &guard,
        options,
        &libraries,
        &report_dir.join("off-run.log"),
    );
    if let Err(error) = off_result {
        eprintln!("cargo xtask visual-quality: Off capture failed: {error}");
        return false;
    }
    let fsr = match read_capture_series(&fsr_capture, options, "FSR 3.1.4") {
        Ok(frames) => frames,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: invalid FSR capture: {error}");
            return false;
        }
    };
    let zero = match read_capture_series(&zero_capture, options, "FSR 3.1.4") {
        Ok(frames) => frames,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: invalid Zero capture: {error}");
            return false;
        }
    };
    let off = match read_capture_series(&off_capture, options, "Off") {
        Ok(frames) => frames,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: invalid Off capture: {error}");
            return false;
        }
    };
    let report = match visual_quality_report(
        options,
        &fsr_capture,
        &zero_capture,
        &off_capture,
        &fsr,
        &zero,
        &off,
        &report_dir,
        evidence.as_ref(),
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: quality gate failed: {error}");
            return false;
        }
    };
    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("cargo xtask visual-quality: serialize report: {error}");
            return false;
        }
    };
    let passed = report.gate.starts_with("passed:");
    if fs::write(report_dir.join("report.json"), json).is_err()
        || write_visual_quality_markdown(&report, &report_dir.join("report.md")).is_err()
    {
        eprintln!("cargo xtask visual-quality: unable to write report artifacts");
        return false;
    }
    println!("TuxScaling visual-quality report: {}", report_dir.display());
    passed
}

#[derive(Debug, PartialEq, Eq)]
struct ProtonAcceptanceOptions {
    evidence_dir: PathBuf,
    game_command: Vec<String>,
    preflight_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProtonLauncherProbe {
    proton: bool,
    wine: bool,
    wine64: bool,
}

impl ProtonLauncherProbe {
    const fn ready(self) -> bool {
        self.proton || self.wine || self.wine64
    }
}

fn proton_launchers_on_path(path: &std::ffi::OsStr) -> ProtonLauncherProbe {
    let directories = std::env::split_paths(path).collect::<Vec<_>>();
    let present = |name: &str| {
        directories.iter().any(|directory| {
            let candidate = directory.join(name);
            candidate.is_file() && executable_file(&candidate)
        })
    };
    ProtonLauncherProbe {
        proton: present("proton"),
        wine: present("wine"),
        wine64: present("wine64"),
    }
}

fn executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn parse_proton_acceptance_args(args: &[String]) -> Result<ProtonAcceptanceOptions, String> {
    let mut evidence_dir = None;
    let mut game_command = None;
    let mut preflight_only = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--evidence-dir" => {
                index += 1;
                let path = args
                    .get(index)
                    .ok_or_else(|| "--evidence-dir requires a path".to_owned())?;
                if evidence_dir.replace(PathBuf::from(path)).is_some() {
                    return Err("--evidence-dir may only be specified once".into());
                }
            }
            "--preflight-only" if !preflight_only => preflight_only = true,
            "--preflight-only" => {
                return Err("--preflight-only may only be specified once".into());
            }
            "--game-command" => {
                if game_command.is_some() {
                    return Err("--game-command may only be specified once".into());
                }
                let mut command = args[index + 1..].to_vec();
                if command.first().is_some_and(|value| value == "--") {
                    command.remove(0);
                }
                if command.is_empty() {
                    return Err("--game-command requires an executable and arguments".into());
                }
                game_command = Some(command);
                break;
            }
            value => return Err(format!("unknown proton-acceptance argument: {value}")),
        }
        index += 1;
    }
    if preflight_only && game_command.is_some() {
        return Err("--preflight-only cannot be combined with --game-command".into());
    }
    Ok(ProtonAcceptanceOptions {
        evidence_dir: evidence_dir.ok_or_else(|| "--evidence-dir is required".to_owned())?,
        game_command: if preflight_only {
            Vec::new()
        } else {
            game_command.ok_or_else(|| "--game-command is required".to_owned())?
        },
        preflight_only,
    })
}

fn writable_directory(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("create evidence directory {}: {error}", path.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    let probe = path.join(format!(
        ".tuxscaling-proton-probe-{}-{stamp}",
        std::process::id()
    ));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|error| format!("evidence directory is not writable: {error}"))?;
    fs::remove_file(&probe).map_err(|error| format!("remove evidence directory probe: {error}"))
}

fn validation_layer_is_available() -> bool {
    let Ok(output) = Command::new("vulkaninfo")
        .args(["--summary"])
        .env("VK_LOADER_LAYERS_ENABLE", "VK_LAYER_KHRONOS_validation")
        .output()
    else {
        return false;
    };
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    text.contains("VK_LAYER_KHRONOS_validation")
}

fn x11_display_is_available() -> bool {
    x11_display_is_available_with(std::env::var_os("DISPLAY"), |program, args, display| {
        Command::new(program)
            .args(args)
            .env("DISPLAY", display)
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

fn x11_display_is_available_with(
    display: Option<std::ffi::OsString>,
    probe: impl Fn(&str, &[&str], &std::ffi::OsStr) -> bool,
) -> bool {
    let Some(display) = display.filter(|value| !value.is_empty()) else {
        return false;
    };
    [
        ("xdpyinfo", &[][..]),
        ("xprop", &["-root"][..]),
        ("xwininfo", &["-root"][..]),
    ]
    .into_iter()
    .any(|(program, args)| probe(program, args, &display))
}

fn fidelityfx_abi_symbols_are_available(root: &Path) -> bool {
    let library = std::env::var_os("TUXSCALING_FIDELITYFX_LIBRARY")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .unwrap_or_else(|| root.join("lib/libtuxscaling_fidelityfx_vk.so"));
    let Ok(output) = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(library)
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    fidelityfx_symbols_are_complete(&text)
}

fn run_gate_command(root: &Path, evidence_dir: &Path, name: &str, args: &[&str]) -> bool {
    let Ok(output) = Command::new(args[0])
        .args(&args[1..])
        .current_dir(root)
        .output()
    else {
        return false;
    };
    let log = format!(
        "command={:?}\nstatus={}\n{}{}",
        args,
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = fs::write(evidence_dir.join(format!("pre-{name}.log")), log);
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    output.status.success()
}

fn proton_acceptance_gate_commands() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("fmt", vec!["cargo", "fmt", "--all", "--", "--check"]),
        ("test", vec!["cargo", "test", "--workspace"]),
        ("build", vec!["cargo", "build", "--workspace"]),
        (
            "clippy",
            vec![
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "fidelityfx-check",
            vec!["cargo", "xtask", "fidelityfx-check"],
        ),
        (
            "gpu-check",
            vec!["cargo", "xtask", "gpu-check", "--backend", "fsr_3_1_4"],
        ),
        (
            "wsi-compatibility",
            vec![
                "cargo",
                "xtask",
                "wsi-compatibility",
                "--backend",
                "fsr_3_1_4",
                "--allow-unverified",
                "display_timing",
            ],
        ),
        (
            "vkcube",
            vec![
                "cargo",
                "xtask",
                "vkcube",
                "--release",
                "--seconds",
                "20",
                "--backend",
                "fsr_3_1_4",
                "--control-sequence",
            ],
        ),
        (
            "visual-quality",
            vec![
                "cargo",
                "xtask",
                "visual-quality",
                "--display",
                "nested-xwayland",
                "--input",
                "1280x720",
                "--output",
                "2160x1440",
                "--warmup",
                "180",
                "--frames",
                "120",
            ],
        ),
    ]
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ProtonSessionEvidence {
    logical_extent: Option<WsiExtent>,
    physical_extent: Option<WsiExtent>,
    presenter_published: bool,
    virtual_active: bool,
    fsr_dispatches: u32,
    reconstructed_presents: u32,
    overlay_visible: bool,
    overlay_hidden: bool,
    software_cursor_visible: bool,
    input_route_active: bool,
    failure: bool,
}

fn proton_session_log_text(stdout: &str, stderr: &str) -> String {
    format!("{stdout}\n{stderr}").to_ascii_lowercase()
}

fn proton_session_event_has(line: &str, event: &str) -> bool {
    line.split_whitespace()
        .any(|token| token == format!("event={event}"))
}

fn proton_session_event_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|token| {
        token
            .strip_prefix(field)
            .and_then(|value| value.strip_prefix('='))
    })
}

fn proton_session_extent(line: &str, field: &str) -> Option<WsiExtent> {
    proton_session_event_field(line, field).and_then(parse_wsi_extent)
}

fn parse_proton_session_evidence(stdout: &str, stderr: &str) -> ProtonSessionEvidence {
    let output = proton_session_log_text(stdout, stderr);
    let mut evidence = ProtonSessionEvidence::default();
    for line in output.lines() {
        if proton_session_event_has(line, "logical_swapchain_created") {
            let virtual_active = proton_session_event_field(line, "virtual") == Some("1");
            let logical = proton_session_extent(line, "logical");
            let physical = proton_session_extent(line, "physical");
            evidence.virtual_active |= virtual_active;
            if virtual_active && logical.is_some() && physical.is_some() && logical != physical {
                evidence.logical_extent = logical;
                evidence.physical_extent = physical;
            } else {
                evidence.logical_extent = evidence.logical_extent.or(logical);
                evidence.physical_extent = evidence.physical_extent.or(physical);
            }
        }
        if proton_session_event_has(line, "presenter_generation_published") {
            evidence.presenter_published = true;
            evidence.logical_extent = evidence
                .logical_extent
                .or_else(|| proton_session_extent(line, "logical"));
            evidence.physical_extent = evidence
                .physical_extent
                .or_else(|| proton_session_extent(line, "physical"));
        }
        if proton_session_event_has(line, "fsr_dispatch")
            && proton_session_event_field(line, "backend") == Some("fsr_3_1_4")
        {
            evidence.fsr_dispatches = evidence.fsr_dispatches.saturating_add(1);
        }
        if proton_session_event_has(line, "reconstructed_present")
            && (proton_session_event_field(line, "backend") == Some("fsr_3_1_4")
                || line.contains("backend=fsr 3.1.4"))
        {
            evidence.reconstructed_presents = evidence.reconstructed_presents.saturating_add(1);
        }
        if proton_session_event_has(line, "overlay_toggled") {
            match proton_session_event_field(line, "visible") {
                Some("1") => evidence.overlay_visible = true,
                Some("0") => evidence.overlay_hidden = true,
                _ => {}
            }
        }
        if proton_session_event_has(line, "software_cursor")
            && proton_session_event_field(line, "visible") == Some("1")
        {
            evidence.software_cursor_visible = true;
        }
        if proton_session_event_has(line, "input_route_active") {
            evidence.input_route_active = true;
        }
    }
    evidence.failure = [
        "validation error",
        "vuid-",
        "panic",
        "event=presenter_fallback_to_direct",
        "event=presenter_fallback ",
    ]
    .iter()
    .any(|marker| output.contains(marker));
    evidence
}

fn proton_session_evidence_is_valid(
    evidence: &ProtonSessionEvidence,
    capture_count: usize,
) -> bool {
    let Some(logical) = evidence.logical_extent else {
        return false;
    };
    let Some(physical) = evidence.physical_extent else {
        return false;
    };
    evidence.presenter_published
        && evidence.virtual_active
        && logical
            == WsiExtent {
                width: 1280,
                height: 720,
            }
        && physical != logical
        && physical.width >= logical.width
        && physical.height >= logical.height
        && evidence.fsr_dispatches >= 3
        && evidence.reconstructed_presents >= 3
        && evidence.overlay_visible
        && evidence.overlay_hidden
        && evidence.software_cursor_visible
        && evidence.input_route_active
        && capture_count >= 3
        && !evidence.failure
}

fn proton_capture_files_are_valid(root: &Path) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return false;
    };
    let mut valid_frames = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file()
            || path.extension().and_then(|extension| extension.to_str()) != Some("json")
            || !path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.starts_with("frame-"))
        {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let extent = |name: &str| {
            value
                .get(name)
                .and_then(serde_json::Value::as_array)
                .and_then(|values| {
                    (values.len() == 2).then(|| {
                        [
                            values[0]
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok()),
                            values[1]
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok()),
                        ]
                    })
                })
        };
        let Some([Some(game_width), Some(game_height)]) = extent("game_extent") else {
            continue;
        };
        let Some([Some(output_width), Some(output_height)]) = extent("output_extent") else {
            continue;
        };
        let resources = value
            .get("resources")
            .and_then(serde_json::Value::as_array)
            .filter(|resources| !resources.is_empty());
        let Some(resources) = resources else {
            continue;
        };
        let resources_complete = resources.iter().all(|resource| {
            resource
                .get("file")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|file| root.join(file).is_file())
        });
        if value
            .get("frame_id")
            .and_then(serde_json::Value::as_u64)
            .is_none()
            || value.get("backend").and_then(serde_json::Value::as_str) != Some("FSR 3.1.4")
            || game_width != 1280
            || game_height != 720
            || output_width < game_width
            || output_height < game_height
            || [output_width, output_height] == [game_width, game_height]
            || !resources_complete
        {
            continue;
        }
        valid_frames += 1;
    }
    valid_frames >= 3
}

fn run_proton_acceptance(root: &Path, options: &ProtonAcceptanceOptions) -> bool {
    let checks = [
        (
            "display",
            x11_display_is_available(),
            "an accessible X11/XWayland DISPLAY is required",
        ),
        (
            "validation",
            validation_layer_is_available(),
            "VK_LAYER_KHRONOS_validation is unavailable",
        ),
        (
            "layer",
            root.join("target/release/libtuxscaling_layer.so").is_file()
                || root.join("target/debug/libtuxscaling_layer.so").is_file(),
            "the built TuxScaling layer is unavailable",
        ),
        (
            "fidelityfx",
            fidelityfx_abi_symbols_are_available(root),
            "the FidelityFX ABI 2 companion is unavailable",
        ),
    ];
    if let Err(error) = writable_directory(&options.evidence_dir) {
        eprintln!("cargo xtask proton-acceptance: {error}");
        return false;
    }
    let mut preflight_ok = true;
    let mut preflight = String::new();
    for (name, passed, reason) in checks {
        preflight_ok &= passed;
        preflight.push_str(&format!("{name}={passed}"));
        if !passed {
            preflight.push_str(&format!(" reason={reason}"));
        }
        preflight.push('\n');
        if !passed {
            eprintln!("cargo xtask proton-acceptance: {reason}");
        }
    }
    let launchers = proton_launchers_on_path(&std::env::var_os("PATH").unwrap_or_default());
    preflight.push_str(&format!(
        "proton_on_path={}\nwine_on_path={}\nwine64_on_path={}\nlauncher_ready={}\n",
        launchers.proton,
        launchers.wine,
        launchers.wine64,
        launchers.ready()
    ));
    if !launchers.ready() {
        eprintln!(
            "cargo xtask proton-acceptance: no proton, wine, or wine64 executable on PATH; game launch remains blocked"
        );
    }
    let _ = fs::write(options.evidence_dir.join("preflight.txt"), preflight);
    if !preflight_ok {
        return false;
    }
    if options.preflight_only {
        return true;
    }

    for (name, args) in proton_acceptance_gate_commands() {
        if !run_gate_command(root, &options.evidence_dir, name, &args) {
            eprintln!("cargo xtask proton-acceptance: pre-Proton gate failed: {name}");
            return false;
        }
    }

    let log = match fs::File::create(options.evidence_dir.join("proton-session.log")) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("cargo xtask proton-acceptance: create session log: {error}");
            return false;
        }
    };
    let mut command = Command::new(&options.game_command[0]);
    command
        .args(&options.game_command[1..])
        .current_dir(root)
        .env("TUXSCALING_VIEW", "reconstructed")
        .env(
            "TUXSCALING_CAPTURE_DIR",
            options.evidence_dir.join("capture"),
        )
        .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
        .env(
            "VK_INSTANCE_LAYERS",
            "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
        )
        .stdout(Stdio::from(log.try_clone().expect("session log clone")))
        .stderr(Stdio::from(log));
    let status = match command.status() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("cargo xtask proton-acceptance: game command failed to start: {error}");
            return false;
        }
    };
    let _ = fs::write(
        options.evidence_dir.join("proton-session-status.txt"),
        format!("status={status}\nrestarts=0\n"),
    );
    let session_log =
        fs::read_to_string(options.evidence_dir.join("proton-session.log")).unwrap_or_default();
    let evidence = parse_proton_session_evidence(&session_log, "");
    let capture_dir = options.evidence_dir.join("capture");
    let capture_count = fs::read_dir(&capture_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    let path = entry.path();
                    path.is_file()
                        && path.extension().and_then(|extension| extension.to_str()) == Some("json")
                        && path
                            .file_stem()
                            .and_then(|stem| stem.to_str())
                            .is_some_and(|stem| stem.starts_with("frame-"))
                })
                .count()
        })
        .unwrap_or(0);
    let evidence_valid = proton_session_evidence_is_valid(&evidence, capture_count);
    let captures_valid = proton_capture_files_are_valid(&capture_dir);
    let _ = fs::write(
        options.evidence_dir.join("proton-session-evidence.txt"),
        format!(
            "evidence={evidence:?}\ncapture_count={capture_count}\ncaptures_valid={captures_valid}\n"
        ),
    );
    if !status.success() {
        return false;
    }
    if !evidence_valid || !captures_valid {
        eprintln!(
            "cargo xtask proton-acceptance: session evidence incomplete (evidence_valid={evidence_valid}, captures_valid={captures_valid})"
        );
        return false;
    }
    true
}

#[cfg(test)]
fn visual_quality_metrics_for_test(
    reference: &[[f32; 4]],
    estimate: &[[f32; 4]],
    previous_estimate: &[[f32; 4]],
) -> VisualQualityMetrics {
    visual_quality_metrics(reference, estimate, previous_estimate)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BenchmarkCase {
    scenario: &'static str,
    quality: &'static str,
    guidance_scale_percent: u32,
}

fn benchmark_cases() -> Vec<BenchmarkCase> {
    ["upscale", "native_aa"]
        .into_iter()
        .flat_map(|scenario| {
            [100, 75, 50]
                .into_iter()
                .flat_map(move |guidance_scale_percent| {
                    ["ultra", "high", "balanced", "performance"]
                        .into_iter()
                        .map(move |quality| BenchmarkCase {
                            scenario,
                            quality,
                            guidance_scale_percent,
                        })
                })
        })
        .collect()
}

const BENCHMARK_SAMPLE_COUNT: usize = 600;

fn benchmark_output_is_operationally_valid(
    status_success: bool,
    stdout: &str,
    stderr: &str,
) -> bool {
    if !status_success {
        return false;
    }
    let output = format!("{stdout}\n{stderr}");
    let lowercase = output.to_ascii_lowercase();
    for marker in [
        "validation error",
        "error_device_lost",
        "queue idle",
        "synchronous readback",
        "non-finite",
        "incomplete",
        "incoherent metadata",
        "cleanup failure",
        "window restoration failure",
        "budget=",
    ] {
        if lowercase.contains(marker) {
            return false;
        }
    }
    if lowercase.contains("nan") || lowercase.contains("infinity") {
        return false;
    }
    let Some(samples) = output
        .split("GPU samples=")
        .nth(1)
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return false;
    };
    samples == BENCHMARK_SAMPLE_COUNT
}

fn benchmark_report(result: std::io::Result<Output>) -> bool {
    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            print!("{stdout}");
            eprint!("{stderr}");
            benchmark_output_is_operationally_valid(output.status.success(), &stdout, &stderr)
        }
        Err(error) => {
            eprintln!("{error}");
            false
        }
    }
}

#[cfg(test)]
fn quality_fixture_passes(value: f32, minimum: f32) -> bool {
    value.is_finite() && value >= minimum
}

fn report(result: std::io::Result<Output>) -> bool {
    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            print!("{stdout}");
            eprint!("{stderr}");
            output.status.success()
                && !stdout.contains("Validation Error")
                && !stderr.contains("Validation Error")
        }
        Err(error) => {
            eprintln!("{error}");
            false
        }
    }
}

fn command_output_with_timeout(
    mut command: Command,
    seconds: u64,
) -> std::io::Result<(Output, bool)> {
    let mut child = command.spawn()?;
    let Some(mut stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(std::io::Error::other(
            "WSI scenario stdout was not captured",
        ));
    };
    let Some(mut stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(std::io::Error::other(
            "WSI scenario stderr was not captured",
        ));
    };
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.read_to_end(&mut bytes);
        (result, bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stderr.read_to_end(&mut bytes);
        (result, bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let (status, timed_out) = loop {
        match child.try_wait()? {
            Some(status) => break (status, false),
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                break (child.wait()?, true);
            }
            None => thread::sleep(Duration::from_millis(25)),
        }
    };
    let (stdout_result, stdout) = stdout_reader
        .join()
        .map_err(|_| std::io::Error::other("WSI scenario stdout reader panicked"))?;
    let (stderr_result, stderr) = stderr_reader
        .join()
        .map_err(|_| std::io::Error::other("WSI scenario stderr reader panicked"))?;
    stdout_result?;
    stderr_result?;
    Ok((
        Output {
            status,
            stdout,
            stderr,
        },
        timed_out,
    ))
}

fn run_wsi_compatibility(
    root: &Path,
    backend: BackendSelection,
    allowed_unverified: &[String],
) -> bool {
    if !run(
        "cargo",
        &[
            "build",
            "-p",
            "tuxscaling-layer",
            "--lib",
            "--example",
            "wsi",
            "--release",
        ],
    ) {
        return false;
    }
    let config = root.join("target/wsi-compatibility.toml");
    if std::fs::write(
        &config,
        generated_config_with_backend("native", 1.0, Some("ultra"), backend),
    )
    .is_err()
    {
        return false;
    }
    let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let libraries = std::iter::once(root.join("target/release"))
        .chain(std::env::split_paths(&inherited))
        .collect::<Vec<_>>();
    let mut all_passed = true;
    for scenario in PORTABLE_WSI_SCENARIOS {
        let mut command = Command::new(root.join("target/release/examples/wsi"));
        validation(&mut command)
            .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
            .env("LD_LIBRARY_PATH", std::env::join_paths(&libraries).unwrap())
            .env(
                "VK_INSTANCE_LAYERS",
                "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
            )
            .env("TUXSCALING_VIEW", "reconstructed")
            .env("TUXSCALING_CONFIG", &config)
            .env("TUXSCALING_TEST_SCENARIO", scenario)
            .env("TUXSCALING_TEST_RESIZE_INTERVAL", "0")
            .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
            .env("TUXSCALING_TEST_SECONDS", "3")
            .env("TUXSCALING_TEST_SINGLE_WINDOW", "1")
            .env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "0")
            .env("TUXSCALING_TEST_FORCE_RESIZE_FAILURE", "0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let result = command_output_with_timeout(command, 10);
        let (output, timed_out) = match result {
            Ok(result) => result,
            Err(error) => {
                eprintln!("cargo xtask wsi-compatibility: {scenario} could not start: {error}");
                all_passed = false;
                continue;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        print!("{stdout}");
        eprint!("{stderr}");
        if timed_out {
            eprintln!("cargo xtask wsi-compatibility: {scenario} result=unverified reason=timeout");
            all_passed = false;
            continue;
        }
        let valid = output.status.success()
            && wsi_compatibility_output_is_valid(&stdout, &stderr, scenario, backend);
        if !valid {
            let evidence = parse_wsi_compatibility_evidence(&stdout, &stderr);
            let excused = allowed_unverified.iter().any(|allowed| allowed == scenario)
                && evidence.unverified
                && !evidence.validation_error
                && !evidence.panic;
            if excused {
                eprintln!(
                    "cargo xtask wsi-compatibility: {scenario} result=allowed-unverified reason=declared-environment-exception"
                );
            } else {
                eprintln!("cargo xtask wsi-compatibility: {scenario} evidence gate failed");
                all_passed = false;
            }
        }
    }
    all_passed
}

fn generated_config(output_resolution: &str, guidance_scale: f32, quality: Option<&str>) -> String {
    generated_config_with_backend(
        output_resolution,
        guidance_scale,
        quality,
        BackendSelection::Reference,
    )
}

fn generated_config_with_backend(
    output_resolution: &str,
    guidance_scale: f32,
    quality: Option<&str>,
    backend: BackendSelection,
) -> String {
    generated_config_with_backend_and_sharpening(
        output_resolution,
        guidance_scale,
        quality,
        backend,
        true,
        0.3,
    )
}

fn generated_config_with_backend_and_sharpening(
    output_resolution: &str,
    guidance_scale: f32,
    quality: Option<&str>,
    backend: BackendSelection,
    sharpening_enabled: bool,
    sharpness: f32,
) -> String {
    let quality = quality.map_or_else(String::new, |quality| {
        format!("motion_quality = \"{quality}\"\n")
    });
    format!(
        "output_resolution = \"{output_resolution}\"\nguidance_scale = {guidance_scale}\n{quality}sharpening_enabled = {sharpening_enabled}\nsharpness = {sharpness}\nupscaler = \"{}\"\n",
        backend.config_value(),
    )
}

fn run(program: &str, args: &[&str]) -> bool {
    report(Command::new(program).args(args).output())
}
fn validation(command: &mut Command) -> &mut Command {
    command
        .env("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
        .env("VK_LAYER_VALIDATE_SYNC", "1")
        .env("MANGOHUD", "0")
        .env("DISABLE_MANGOHUD", "1")
        .env("DISABLE_LSFG", "1")
}

struct QualityEvidenceSuite {
    name: &'static str,
    args: &'static [&'static str],
    timeout_seconds: u64,
}

fn quality_evidence_suites() -> Vec<QualityEvidenceSuite> {
    vec![
        QualityEvidenceSuite {
            name: "motion-presets",
            args: &[
                "test",
                "-p",
                "tuxscaling-motion",
                "--test",
                "gpu",
                "--",
                "performance_flow_is_faster_than_balanced_after_warmup",
                "balanced_quality_regression_gate",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            timeout_seconds: 600,
        },
        QualityEvidenceSuite {
            name: "upscaler-ablations",
            args: &[
                "test",
                "-p",
                "tuxscaling-upscaler",
                "--features",
                "fidelityfx",
                "--test",
                "fidelityfx_sequence_quality_gpu",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            timeout_seconds: 600,
        },
        QualityEvidenceSuite {
            name: "temporal-fallbacks",
            args: &[
                "test",
                "-p",
                "tuxscaling-temporal",
                "--test",
                "gpu",
                "--",
                "zero_guidance_dispatches_coherent_fallback_resources",
                "provider_failure_writes_fallback_guidance_and_labels_the_view",
                "provider_failure_resets_history_before_the_next_valid_frame",
                "relative_depth_orders_independent_parallax_planes",
                "relative_depth_uses_flat_fallback_below_global_motion_threshold",
                "relative_depth_uses_flat_fallback_below_affine_inlier_threshold",
                "guidance_disocclusion_uses_flow_holes_and_boundaries",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            timeout_seconds: 600,
        },
        QualityEvidenceSuite {
            name: "capture-jitter",
            args: &[
                "test",
                "-p",
                "tuxscaling-capture",
                "--test",
                "gpu",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            timeout_seconds: 600,
        },
    ]
}

/// Keep only the measured metric lines printed by the GPU suites; harness
/// chatter (warnings, test names, result summaries) is dropped.
fn quality_metric_lines(output: &str) -> Vec<String> {
    const PREFIXES: [&str; 16] = [
        "guidance_scale=",
        "baseline fixture=",
        "FSR sequence fixture:",
        "FSR sequence aggregate:",
        "FSR guidance modes:",
        "disocclusion counts",
        "disocclusion F1=",
        "disocclusion mask:",
        "transparency F1=",
        "transparency mask:",
        "disoccluded reconstruction error:",
        "provider reset exposure:",
        "relative depth parallax:",
        "translated history:",
        "guidance interior ",
        "capture jitter:",
    ];
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            PREFIXES
                .iter()
                .any(|prefix| trimmed.starts_with(prefix))
                .then(|| trimmed.to_owned())
        })
        .collect()
}

fn number_after_marker(line: &str, marker: &str) -> Option<f32> {
    let start = line.find(marker)? + marker.len();
    line[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse()
        .ok()
}

/// Parse the `guidance_scale=... balanced_forward_backward_ms=.. performance_forward_backward_ms=..`
/// line printed by the motion preset timing test.
fn parse_preset_medians(output: &str) -> Option<(f32, f32)> {
    const BALANCED: &str = "balanced_forward_backward_ms=";
    const PERFORMANCE: &str = "performance_forward_backward_ms=";
    let line = output
        .lines()
        .map(str::trim)
        .find(|line| line.contains(BALANCED) && line.contains(PERFORMANCE))?;
    Some((
        number_after_marker(line, BALANCED)?,
        number_after_marker(line, PERFORMANCE)?,
    ))
}

struct QualitySuiteResult {
    name: String,
    metrics: Vec<String>,
    log_file: String,
}

fn run_quality_evidence_suite(
    root: &Path,
    suite: &QualityEvidenceSuite,
    dir: &Path,
) -> Option<QualitySuiteResult> {
    let mut command = Command::new("cargo");
    command
        .args(suite.args)
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let (output, timed_out) = command_output_with_timeout(command, suite.timeout_seconds).ok()?;
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let log_file = format!("{}.log", suite.name);
    if fs::write(dir.join(&log_file), combined.as_bytes()).is_err() {
        eprintln!("cargo xtask quality-evidence: unable to write {log_file}");
        return None;
    }
    if timed_out || !output.status.success() {
        eprintln!("cargo xtask quality-evidence: suite {} failed", suite.name);
        return None;
    }
    for metric in quality_metric_lines(&combined) {
        println!("quality-evidence {}: {metric}", suite.name);
    }
    Some(QualitySuiteResult {
        name: suite.name.to_owned(),
        metrics: quality_metric_lines(&combined),
        log_file,
    })
}

#[derive(Serialize)]
struct QualityEvidenceSuiteReport {
    name: String,
    passed: bool,
    metrics: Vec<String>,
    log: String,
}

#[derive(Serialize)]
struct QualityEvidenceReport {
    command: String,
    timestamp: u64,
    balanced_forward_backward_ms: f32,
    performance_forward_backward_ms: f32,
    performance_cheaper: bool,
    suites: Vec<QualityEvidenceSuiteReport>,
}

/// Freshly measured GPU evidence shared by `quality-evidence` and the
/// `visual-quality` pre-stage: per-suite metric lines plus the parsed
/// Balanced-versus-Performance medians.
struct CollectedQualityEvidence {
    suites: Vec<QualityEvidenceSuiteReport>,
    balanced_median: f32,
    performance_median: f32,
}

fn collect_quality_evidence(
    root: &Path,
    dir: &Path,
    command: &str,
) -> Option<CollectedQualityEvidence> {
    let mut suites = Vec::new();
    for suite in quality_evidence_suites() {
        let result = run_quality_evidence_suite(root, &suite, dir)?;
        suites.push(QualityEvidenceSuiteReport {
            name: result.name,
            passed: true,
            metrics: result.metrics,
            log: result.log_file,
        });
    }
    let preset_medians = suites
        .iter()
        .find(|suite| suite.name == "motion-presets")
        .and_then(|suite| parse_preset_medians(&suite.metrics.join("\n")));
    let Some((balanced_median, performance_median)) = preset_medians else {
        eprintln!("{command}: motion-presets suite did not print preset medians");
        return None;
    };
    Some(CollectedQualityEvidence {
        suites,
        balanced_median,
        performance_median,
    })
}

fn suite_metrics<'a>(evidence: &'a CollectedQualityEvidence, name: &str) -> Vec<&'a str> {
    evidence
        .suites
        .iter()
        .filter(|suite| suite.name == name)
        .flat_map(|suite| suite.metrics.iter().map(String::as_str))
        .collect()
}

/// Build the ablation matrix from freshly measured suite evidence. Every row
/// names the covering suites (all freshly passed) plus the signal-specific
/// measured lines; the full suite logs sit next to the report.
fn quality_ablation_rows(evidence: &CollectedQualityEvidence) -> Vec<AblationReport> {
    const SIGNALS: [(&str, &str, &[&str]); 7] = [
        (
            "motion",
            "zero_vs_off_mse",
            &["upscaler-ablations", "temporal-fallbacks"],
        ),
        (
            "relative_depth",
            "relative depth parallax:",
            &["upscaler-ablations", "temporal-fallbacks"],
        ),
        ("reactive", "", &["upscaler-ablations"]),
        ("composition", "", &["upscaler-ablations"]),
        (
            "exposure",
            "provider reset exposure:",
            &["upscaler-ablations", "temporal-fallbacks"],
        ),
        (
            "confidence_disocclusion",
            "disocclusion F1=",
            &["upscaler-ablations", "temporal-fallbacks"],
        ),
        (
            "post_capture_jitter",
            "capture jitter:",
            &["upscaler-ablations", "capture-jitter"],
        ),
    ];
    SIGNALS
        .into_iter()
        .map(|(signal, marker, suites)| {
            let mut parts = Vec::new();
            for suite in suites {
                for metric in suite_metrics(evidence, suite) {
                    if marker.is_empty() || metric.contains(marker) {
                        parts.push(format!("{suite}: {metric}"));
                    }
                }
            }
            // Every covering suite passed or collection would have failed; the
            // one-hot ablation assertions inside the suites are the evidence
            // even where no single printed line names the signal.
            let mut seen = std::collections::BTreeSet::new();
            for suite in suites {
                seen.insert((*suite).to_owned());
            }
            AblationReport {
                signal: signal.into(),
                estimated: "true".into(),
                fallback: "true".into(),
                status: format!(
                    "measured: one-hot ablation distinct and coherent; covering suites passed [{}]; {}",
                    seen.into_iter().collect::<Vec<_>>().join(", "),
                    if parts.is_empty() {
                        "see suite logs for the one-hot assertions".to_owned()
                    } else {
                        parts.join("; ")
                    }
                ),
            }
        })
        .collect()
}

/// Build the preset comparison from the measured medians. Balanced stays the
/// quality baseline; Performance is the cheaper explicit choice.
fn quality_preset_rows(
    evidence: &CollectedQualityEvidence,
    input: (u32, u32),
) -> Vec<PresetReport> {
    let cheaper = evidence.performance_median < evidence.balanced_median;
    [MotionQuality::Balanced, MotionQuality::Performance]
        .into_iter()
        .map(|quality| {
            let timing_status = match quality {
                MotionQuality::Balanced => format!(
                    "measured median {:.4} ms (motion-presets suite at guidance_scale=1.0); quality hashes unchanged, see motion-presets.log",
                    evidence.balanced_median
                ),
                _ => format!(
                    "measured median {:.4} ms (motion-presets suite at guidance_scale=1.0); cheaper than Balanced on the same scene: {cheaper}",
                    evidence.performance_median
                ),
            };
            PresetReport {
                preset: format!("{quality:?}"),
                quality_baseline: quality == MotionQuality::Balanced,
                work_units: quality.dispatch_plan(input.0, input.1).candidate_evaluations,
                timing_status,
            }
        })
        .collect()
}

/// Run the GPU ablation/preset suites fresh, require all of them to pass, and
/// record the measured evidence under `target/quality-evidence/<timestamp>/`.
/// The suites are the Task 10/11 GPU tests: per-signal one-hot ablations with
/// Estimated/Zero/Off coherence, fallback resources, and Balanced-versus-
/// Performance timing with the Balanced no-regression hash gate.
fn run_quality_evidence(root: &Path) -> bool {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_secs());
    let dir = root.join(format!("target/quality-evidence/{timestamp}"));
    if fs::create_dir_all(&dir).is_err() {
        eprintln!(
            "cargo xtask quality-evidence: unable to create {}",
            dir.display()
        );
        return false;
    }
    let Some(evidence) = collect_quality_evidence(root, &dir, "cargo xtask quality-evidence")
    else {
        return false;
    };
    let report = QualityEvidenceReport {
        command: "cargo xtask quality-evidence".to_owned(),
        timestamp,
        balanced_forward_backward_ms: evidence.balanced_median,
        performance_forward_backward_ms: evidence.performance_median,
        performance_cheaper: evidence.performance_median < evidence.balanced_median,
        suites: evidence.suites,
    };
    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("cargo xtask quality-evidence: unable to encode report: {error}");
            return false;
        }
    };
    if fs::write(dir.join("report.json"), json).is_err() {
        eprintln!("cargo xtask quality-evidence: unable to write report.json");
        return false;
    }
    let mut markdown = format!(
        "# Temporal quality evidence\n\n- Command: `{}`\n- Timestamp: `{timestamp}`\n\n",
        report.command
    );
    markdown
        .push_str("## Suites\n\n| Suite | Result | Measured metrics | Log |\n|---|---|---|---|\n");
    for suite in &report.suites {
        markdown.push_str(&format!(
            "| {} | passed | {} | {} |\n",
            suite.name,
            suite.metrics.join("; "),
            suite.log
        ));
    }
    markdown.push_str(&format!(
        "\n## Presets\n\n- Balanced forward/backward median: `{:.4} ms`\n- Performance forward/backward median: `{:.4} ms`\n- Performance cheaper on the same scene at guidance_scale=1.0: `{}`\n- Balanced quality hashes: unchanged (see the `baseline fixture=` lines in `motion-presets.log`)\n",
        report.balanced_forward_backward_ms,
        report.performance_forward_backward_ms,
        report.performance_cheaper
    ));
    if fs::write(dir.join("report.md"), markdown).is_err() {
        eprintln!("cargo xtask quality-evidence: unable to write report.md");
        return false;
    }
    println!("cargo xtask quality-evidence: report in {}", dir.display());
    true
}

fn run_gpu_check(backend: BackendSelection) -> bool {
    let base = report(
        validation(Command::new("cargo").args([
            "test",
            "-p",
            "tuxscaling-capture",
            "-p",
            "tuxscaling-motion",
            "-p",
            "tuxscaling-temporal",
            "-p",
            "tuxscaling-upscaler",
            "--test",
            "gpu",
            "--",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]))
        .output(),
    );
    if !base || backend != BackendSelection::Fsr314 {
        return base;
    }
    let input = report(
        validation(Command::new("cargo").args([
            "test",
            "-p",
            "tuxscaling-upscaler",
            "--test",
            "fidelityfx_input_gpu",
            "--features",
            "fidelityfx",
            "--",
            "--nocapture",
            "--test-threads=1",
        ]))
        .output(),
    );
    input
        && report(
            validation(Command::new("cargo").args([
                "test",
                "-p",
                "tuxscaling-upscaler",
                "--test",
                "fsr314_gpu",
                "--features",
                "fidelityfx",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]))
            .output(),
        )
        && report(
            validation(Command::new("cargo").args([
                "test",
                "-p",
                "tuxscaling-upscaler",
                "--test",
                "fidelityfx_quality_gpu",
                "--features",
                "fidelityfx",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]))
            .output(),
        )
        && report(
            validation(Command::new("cargo").args([
                "test",
                "-p",
                "tuxscaling-upscaler",
                "--test",
                "fidelityfx_sequence_quality_gpu",
                "--features",
                "fidelityfx",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]))
            .output(),
        )
}

fn fidelityfx_check(root: &Path) -> bool {
    let library = std::env::var_os("TUXSCALING_FIDELITYFX_LIBRARY")
        .map(PathBuf::from)
        .map_or_else(
            || root.join("lib/libtuxscaling_fidelityfx_vk.so"),
            |path| {
                if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                }
            },
        );
    if !library.is_file() {
        eprintln!(
            "cargo xtask fidelityfx-check: missing companion library {}",
            library.display()
        );
        return false;
    }
    let Ok(file) = Command::new("file").arg(&library).output() else {
        return false;
    };
    let file_text = String::from_utf8_lossy(&file.stdout);
    if !file.status.success()
        || !fidelityfx_elf_architecture_is_valid(&file_text, std::env::consts::ARCH)
    {
        eprintln!(
            "cargo xtask fidelityfx-check: companion has an unsupported ELF architecture: {}",
            library.display()
        );
        return false;
    }
    let Ok(symbols) = Command::new("nm").args(["-D"]).arg(&library).output() else {
        return false;
    };
    let symbol_text = String::from_utf8_lossy(&symbols.stdout);
    if !symbols.status.success() || !fidelityfx_symbols_are_complete(&symbol_text) {
        eprintln!("cargo xtask fidelityfx-check: companion symbols are incomplete");
        return false;
    }
    if !fidelityfx_generated_hashes_match(root) {
        eprintln!("cargo xtask fidelityfx-check: generated FidelityFX hashes do not match");
        return false;
    }
    report(
        validation(Command::new("cargo").args([
            "test",
            "-p",
            "tuxscaling-upscaler",
            "--test",
            "fidelityfx_library",
            "--features",
            "fidelityfx",
            "--",
            "--nocapture",
        ]))
        .output(),
    )
}

fn fidelityfx_elf_architecture_is_valid(file_text: &str, architecture: &str) -> bool {
    if !file_text.contains("ELF 64-bit") {
        return false;
    }
    match architecture {
        "x86_64" => file_text.contains("x86-64"),
        "aarch64" => file_text.contains("ARM aarch64"),
        "riscv64" => file_text.contains("UCB RISC-V"),
        _ => true,
    }
}

fn fidelityfx_symbols_are_complete(symbol_text: &str) -> bool {
    [
        "tux_ffx_abi_version",
        "tux_ffx_version",
        "tux_ffx_create",
        "tux_ffx_dispatch",
        "tux_ffx_reset",
        "tux_ffx_destroy",
    ]
    .iter()
    .all(|symbol| {
        symbol_text.lines().any(|line| {
            line.split_whitespace()
                .last()
                .is_some_and(|last| last == *symbol)
        })
    })
}

fn fidelityfx_generated_hashes_match(root: &Path) -> bool {
    let generated = root.join("crates/upscaler/native/fidelityfx/generated");
    let manifest = generated.join("SHA256SUMS");
    let Ok(result) = Command::new("sha256sum")
        .args(["--check", "SHA256SUMS"])
        .current_dir(&generated)
        .output()
    else {
        return false;
    };
    if result.status.success() {
        true
    } else {
        eprintln!(
            "cargo xtask fidelityfx-check: unable to verify {}:\n{}{}",
            manifest.display(),
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        false
    }
}

fn main() -> ExitCode {
    let command = std::env::args().nth(1).unwrap_or_default();
    let success = match command.as_str() {
        "fidelityfx-check" => {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            fidelityfx_check(root)
        }
        "gpu-check" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let arguments = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            let backend = match parse_backend_args(&arguments) {
                Ok(backend) => backend,
                Err(error) => {
                    eprintln!("cargo xtask gpu-check: {error}");
                    return ExitCode::from(2);
                }
            };
            run_gpu_check(backend)
        }
        "quality-evidence" => {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            run_quality_evidence(root)
        }
        "smoke" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let arguments = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            let backend = match parse_backend_args(&arguments) {
                Ok(backend) => backend,
                Err(error) => {
                    eprintln!("cargo xtask smoke: {error}");
                    return ExitCode::from(2);
                }
            };
            if !run(
                "cargo",
                &[
                    "build",
                    "-p",
                    "tuxscaling-layer",
                    "--lib",
                    "--example",
                    "wsi",
                ],
            ) {
                return ExitCode::FAILURE;
            }
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap();
            let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
            let libraries = std::iter::once(root.join("target/debug"))
                .chain(std::env::split_paths(&inherited))
                .collect::<Vec<_>>();
            let smoke_config = root.join("target/native-output-smoke.toml");
            std::fs::write(
                &smoke_config,
                generated_config_with_backend("native", 1.0, None, backend),
            )
            .unwrap();
            let direct_config = root.join("target/native-aa-smoke.toml");
            std::fs::write(
                &direct_config,
                generated_config_with_backend("swapchain", 1.0, None, backend),
            )
            .unwrap();
            let guidance_config = root.join("target/guidance-resolve-smoke.toml");
            std::fs::write(
                &guidance_config,
                generated_config_with_backend("native", 0.5, None, backend),
            )
            .unwrap();
            let run_wsi = |scenario: &str, force_temporal_failure, resize_failure| {
                let mut command = Command::new(root.join("target/debug/examples/wsi"));
                validation(&mut command)
                    .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
                    .env("LD_LIBRARY_PATH", std::env::join_paths(&libraries).unwrap())
                    .env(
                        "VK_INSTANCE_LAYERS",
                        "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
                    )
                    .env("TUXSCALING_VIEW", "reconstructed")
                    .env(
                        "TUXSCALING_CONFIG",
                        if scenario == "native" || scenario == "native_aa" {
                            &direct_config
                        } else if scenario == "guidance_resolve" {
                            &guidance_config
                        } else {
                            &smoke_config
                        },
                    )
                    .env("TUXSCALING_TEST_SCENARIO", scenario)
                    .env("TUXSCALING_TEST_RESIZE_INTERVAL", "0")
                    .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
                    .env(
                        "TUXSCALING_TEST_SECONDS",
                        if scenario == "maintenance1" { "8" } else { "3" },
                    )
                    .env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "0")
                    .env("TUXSCALING_TEST_FORCE_RESIZE_FAILURE", "0");
                if force_temporal_failure {
                    command.env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "1");
                }
                if resize_failure {
                    command.env("TUXSCALING_TEST_FORCE_RESIZE_FAILURE", "1");
                }
                if scenario == "resize" || scenario == "promotion_failure" {
                    command.env("TUXSCALING_TEST_RESIZE_INTERVAL", "4");
                }
                let output = command.output();
                if scenario == "maintenance1" {
                    match output {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            print!("{stdout}");
                            eprint!("{stderr}");
                            let valid = output.status.success()
                                && maintenance_output_is_valid(&stdout, &stderr, backend);
                            if !valid {
                                eprintln!(
                                    "cargo xtask smoke: maintenance1 evidence gate failed for {}",
                                    backend.config_value()
                                );
                            }
                            valid
                        }
                        Err(error) => {
                            eprintln!("{error}");
                            false
                        }
                    }
                } else {
                    report(output)
                }
            };
            run_wsi("upscale", false, false)
                && run_wsi("windowed_promote", false, false)
                && run_wsi("already_borderless", false, false)
                && run_wsi("native_aa", false, false)
                && run_wsi("guidance_resolve", false, false)
                && run_wsi("aspect", false, false)
                && run_wsi("resize", false, false)
                && run_wsi("monitor_origin", false, false)
                && run_wsi("promotion_failure", false, true)
                && run_wsi("temporal_failure", true, false)
                && run_wsi("maintenance1", false, false)
        }
        "benchmark" => {
            if !run(
                "cargo",
                &[
                    "build",
                    "-p",
                    "tuxscaling-layer",
                    "--lib",
                    "--example",
                    "wsi",
                    "--release",
                ],
            ) {
                return ExitCode::FAILURE;
            }
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap();
            let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
            let libraries = std::iter::once(root.join("target/release"))
                .chain(std::env::split_paths(&inherited))
                .collect::<Vec<_>>();
            benchmark_cases().into_iter().all(|case| {
                let config = root.join(format!(
                    "target/benchmark-{}-{}-{}.toml",
                    case.scenario,
                    case.quality,
                    case.guidance_scale_percent
                ));
                let output_resolution = if case.scenario == "native_aa" {
                    "swapchain"
                } else {
                    "native"
                };
                if std::fs::write(
                    &config,
                    generated_config(
                        output_resolution,
                        case.guidance_scale_percent as f32 / 100.0,
                        Some(case.quality),
                    ),
                )
                .is_err()
                {
                    return false;
                }
                eprintln!(
                    "TuxScaling benchmark: scenario={} quality={} guidance_scale={}% warmup=180 samples=600",
                    case.scenario, case.quality, case.guidance_scale_percent
                );
                let mut command = Command::new(root.join("target/release/examples/wsi"));
                validation(&mut command)
                    .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
                    .env("LD_LIBRARY_PATH", std::env::join_paths(&libraries).unwrap())
                    .env(
                        "VK_INSTANCE_LAYERS",
                        "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
                    )
                    .env("TUXSCALING_VIEW", "reconstructed")
                    .env("TUXSCALING_CONFIG", config)
                    .env("TUXSCALING_TEST_SCENARIO", case.scenario)
                    .env("TUXSCALING_TEST_RESIZE_INTERVAL", "0")
                    .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
                    .env("TUXSCALING_TEST_FRAMES", "780")
                    .env("TUXSCALING_TEST_SINGLE_WINDOW", "1")
                    .env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "0")
                    .env("TUXSCALING_TEST_FORCE_RESIZE_FAILURE", "0");
                benchmark_report(command.output())
            })
        }
        "vkcube" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let arguments = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            let options = match parse_vkcube_args(&arguments) {
                Ok(options) => options,
                Err(error) => {
                    eprintln!("cargo xtask vkcube: {error}");
                    return ExitCode::from(2);
                }
            };
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            let result = run_vkcube(root, options);
            if !result.success() {
                eprintln!("cargo xtask vkcube failed: {result:?}");
            }
            result.success()
        }
        "wsi-compatibility" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let arguments = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            let (backend, allowed_unverified) = match parse_wsi_compatibility_args(&arguments) {
                Ok(parsed) => parsed,
                Err(error) => {
                    eprintln!("cargo xtask wsi-compatibility: {error}");
                    return ExitCode::from(2);
                }
            };
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            run_wsi_compatibility(root, backend, &allowed_unverified)
        }
        "visual-quality" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let arguments = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            let options = match parse_visual_quality_args(&arguments) {
                Ok(options) => options,
                Err(error) => {
                    eprintln!("cargo xtask visual-quality: {error}");
                    return ExitCode::from(2);
                }
            };
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            run_visual_quality(root, &options)
        }
        "proton-acceptance" => {
            let arguments = std::env::args().skip(2).collect::<Vec<_>>();
            let options = match parse_proton_acceptance_args(&arguments) {
                Ok(options) => options,
                Err(error) => {
                    eprintln!("cargo xtask proton-acceptance: {error}");
                    return ExitCode::from(2);
                }
            };
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            run_proton_acceptance(root, &options)
        }
        "check" => {
            run("cargo", &["fmt", "--all", "--", "--check"])
                && run("cargo", &["test", "--workspace"])
                && run(
                    "cargo",
                    &[
                        "clippy",
                        "--workspace",
                        "--all-targets",
                        "--",
                        "-D",
                        "warnings",
                    ],
                )
        }
        _ => {
            eprintln!(
                "Usage: cargo xtask <benchmark|check|fidelityfx-check|gpu-check|quality-evidence|smoke|vkcube|wsi-compatibility|visual-quality|proton-acceptance> [command options]"
            );
            return ExitCode::from(2);
        }
    };
    if success {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BENCHMARK_SAMPLE_COUNT, BackendSelection, CaptureFsrInputs, CollectedQualityEvidence,
        ProtonAcceptanceOptions, ProtonLauncherProbe, QualityEvidenceSuiteReport, VkcubeExit,
        benchmark_cases, benchmark_output_is_operationally_valid, classify_vkcube_exit,
        classify_vkcube_output, fidelityfx_elf_architecture_is_valid,
        fidelityfx_symbols_are_complete, fsr_input_contract_is_safe,
        fsr_input_contract_is_safe_for_frame, generated_config, guidance_comparison_gate_passed,
        maintenance_evidence_complete, maintenance_output_is_valid, parse_backend_args,
        parse_maintenance_evidence, parse_preset_medians, parse_proton_acceptance_args,
        parse_proton_session_evidence, parse_public_x11_display, parse_visual_quality_args,
        parse_vkcube_args, parse_vkcube_evidence, parse_wsi_compatibility_args,
        proton_acceptance_gate_commands, proton_capture_files_are_valid, proton_launchers_on_path,
        proton_session_evidence_is_valid, quality_ablation_rows, quality_fixture_passes,
        quality_metric_lines, quality_preset_rows, visual_quality_metrics_for_test,
        vkcube_control_sequence_is_valid, vkcube_launch_in, vkcube_output_is_valid,
        wsi_compatibility_output_is_valid, x11_display_is_available_with,
    };
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn proton_acceptance_requires_an_evidence_directory_and_explicit_command() {
        assert!(parse_proton_acceptance_args(&[]).is_err());
        assert!(
            parse_proton_acceptance_args(&["--evidence-dir".into(), "target/proton".into(),])
                .is_err()
        );
        assert!(parse_proton_acceptance_args(&["--game-command".into(), "true".into(),]).is_err());
    }

    #[test]
    fn proton_preflight_only_does_not_require_a_game_command() {
        let options = parse_proton_acceptance_args(&[
            "--evidence-dir".into(),
            "target/proton".into(),
            "--preflight-only".into(),
        ])
        .unwrap();

        assert_eq!(
            options,
            ProtonAcceptanceOptions {
                evidence_dir: Path::new("target/proton").to_path_buf(),
                game_command: Vec::new(),
                preflight_only: true,
            }
        );
    }

    #[test]
    fn proton_preflight_only_rejects_a_game_command() {
        assert!(
            parse_proton_acceptance_args(&[
                "--evidence-dir".into(),
                "target/proton".into(),
                "--preflight-only".into(),
                "--game-command".into(),
                "true".into(),
            ])
            .is_err()
        );
    }

    #[test]
    fn x11_display_probe_accepts_xprop_when_xdpyinfo_is_missing() {
        assert!(x11_display_is_available_with(
            Some(":0".into()),
            |program, _, _| program == "xprop",
        ));
        assert!(!x11_display_is_available_with(
            Some(":0".into()),
            |_, _, _| false,
        ));
        assert!(!x11_display_is_available_with(None, |_, _, _| true));
        assert!(!x11_display_is_available_with(
            Some(std::ffi::OsString::new()),
            |_, _, _| true,
        ));
    }

    #[test]
    fn proton_launcher_probe_reads_path_without_inventing_a_game() {
        let root = std::env::temp_dir().join(format!(
            "tuxscaling-proton-probe-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |value| value.as_nanos())
        ));
        std::fs::create_dir_all(&root).unwrap();
        for name in ["proton", "wine"] {
            std::fs::write(root.join(name), []).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut permissions = std::fs::metadata(root.join(name)).unwrap().permissions();
                permissions.set_mode(0o755);
                std::fs::set_permissions(root.join(name), permissions).unwrap();
            }
        }

        let probe = proton_launchers_on_path(root.as_os_str());
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            probe,
            ProtonLauncherProbe {
                proton: true,
                wine: true,
                wine64: false,
            }
        );
        assert!(!proton_launchers_on_path(std::ffi::OsStr::new("")).ready());
    }

    #[test]
    fn proton_acceptance_preserves_the_explicit_game_command() {
        let options = parse_proton_acceptance_args(&[
            "--evidence-dir".into(),
            "target/proton".into(),
            "--game-command".into(),
            "--".into(),
            "steam".into(),
            "-applaunch".into(),
            "123".into(),
        ])
        .unwrap();

        assert_eq!(
            options,
            ProtonAcceptanceOptions {
                evidence_dir: Path::new("target/proton").to_path_buf(),
                game_command: vec!["steam".into(), "-applaunch".into(), "123".into()],
                preflight_only: false,
            }
        );
    }

    #[test]
    fn proton_acceptance_gates_pin_fsr_and_allow_radv_timing_exception() {
        let gates = proton_acceptance_gate_commands();
        let args_for = |name: &str| {
            gates
                .iter()
                .find(|(gate_name, _)| *gate_name == name)
                .map(|(_, args)| args.as_slice())
                .unwrap_or_else(|| panic!("missing gate {name}"))
        };

        assert_eq!(
            args_for("gpu-check"),
            ["cargo", "xtask", "gpu-check", "--backend", "fsr_3_1_4"]
        );
        assert_eq!(
            args_for("wsi-compatibility"),
            [
                "cargo",
                "xtask",
                "wsi-compatibility",
                "--backend",
                "fsr_3_1_4",
                "--allow-unverified",
                "display_timing"
            ]
        );
        assert_eq!(
            args_for("vkcube"),
            [
                "cargo",
                "xtask",
                "vkcube",
                "--release",
                "--seconds",
                "20",
                "--backend",
                "fsr_3_1_4",
                "--control-sequence"
            ]
        );
    }

    #[test]
    fn proton_session_evidence_requires_one_complete_interactive_run() {
        let complete = concat!(
            "TuxScaling evidence event=logical_swapchain_created logical=1280x720 physical=2160x1440 virtual=1\n",
            "TuxScaling evidence event=presenter_generation_published logical=1280x720 physical=2160x1440 generation=0\n",
            "TuxScaling evidence event=input_route_active mode=absolute\n",
            "TuxScaling evidence event=software_cursor visible=1\n",
            "TuxScaling evidence event=overlay_toggled visible=1 owner=Overlay\n",
            "TuxScaling evidence event=overlay_toggled visible=0 owner=Native\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR 3.1.4 frame=1\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR 3.1.4 frame=2\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR 3.1.4 frame=3\n",
        );
        let evidence = parse_proton_session_evidence(complete, "");
        assert!(proton_session_evidence_is_valid(&evidence, 3));

        let mut incomplete = complete.replace("visible=1 owner=Overlay", "visible=0 owner=Native");
        incomplete.push_str(
            "TuxScaling evidence event=presenter_fallback_to_direct reason=presenter_lost\n",
        );
        let evidence = parse_proton_session_evidence(&incomplete, "");
        assert!(!proton_session_evidence_is_valid(&evidence, 3));
    }

    #[test]
    fn proton_capture_validation_requires_complete_fsr_frames() {
        let root = std::env::temp_dir().join(format!(
            "tuxscaling-proton-capture-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |value| value.as_nanos())
        ));
        std::fs::create_dir_all(&root).unwrap();
        for frame in 0..3 {
            let prefix = format!("frame-{frame:08}");
            std::fs::write(root.join(format!("{prefix}-source.bin")), [0_u8]).unwrap();
            std::fs::write(
                root.join(format!("{prefix}.json")),
                format!(
                    "{{\"frame_id\":{frame},\"backend\":\"FSR 3.1.4\",\"game_extent\":[1280,720],\"output_extent\":[2160,1440],\"resources\":[{{\"file\":\"{prefix}-source.bin\"}}]}}"
                ),
            )
            .unwrap();
        }
        assert!(proton_capture_files_are_valid(&root));
        std::fs::remove_file(root.join("frame-00000001-source.bin")).unwrap();
        assert!(!proton_capture_files_are_valid(&root));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn proton_acceptance_rejects_duplicate_or_unknown_options() {
        assert!(
            parse_proton_acceptance_args(&[
                "--evidence-dir".into(),
                "one".into(),
                "--evidence-dir".into(),
                "two".into(),
                "--game-command".into(),
                "true".into(),
            ])
            .is_err()
        );
        assert!(parse_proton_acceptance_args(&["--unknown".into()]).is_err());
    }

    fn portable_wsi_fixture(scenario: &str) -> String {
        format!(
            concat!(
                "TuxScaling evidence event=wsi_scenario_request scenario={scenario} requested=1280x720\n",
                "TuxScaling evidence event=borderless_target extent=3440x1440\n",
                "TuxScaling evidence event=logical_swapchain_created logical_handle=0x1 physical_handle=0x2 logical=1280x720 physical=1280x720 virtual=1 mutable_format=1 view_formats=2 negotiation=negotiating\n",
                "TuxScaling evidence event=virtual_swapchain_active logical_handle=0x1 logical=1280x720 physical=3440x1440 generation=1\n",
                "TuxScaling evidence event=presenter_generation_published logical=1280x720 physical=3440x1440 presenter_surface=0x4\n",
                "TuxScaling evidence event=fsr_dispatch backend=fsr_3_1_4 logical=1280x720 physical=3440x1440\n",
                "TuxScaling evidence event=present_wait translated=1 generation=0\n",
                "TuxScaling evidence event=present_wait translated=1 generation=1\n",
                "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=1\n",
                "TuxScaling evidence event=overlay_submitted scenario={scenario}\n",
                "TuxScaling evidence event=wsi_scenario scenario={scenario} result=verified alternate_views=1 present_wait_current=1 present_wait_old=1 present_wait_recreated=1 hdr_before=1 hdr_after=1 queries=status,counter,refresh timing=count,data direct=0 recreations_after_publish=0\n",
            ),
            scenario = scenario,
        )
    }

    #[test]
    fn benchmark_matrix_covers_each_quality_and_presentation_mode() {
        let cases = benchmark_cases();
        assert_eq!(cases.len(), 24);
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "upscale")
                .count(),
            12
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "native_aa")
                .count(),
            12
        );
        assert!(cases.iter().any(|case| case.guidance_scale_percent == 100));
        assert!(cases.iter().any(|case| case.guidance_scale_percent == 75));
        assert!(cases.iter().any(|case| case.guidance_scale_percent == 50));
    }

    #[test]
    fn benchmark_accepts_slow_but_finite_complete_samples() {
        let stderr = format!(
            "TuxScaling 1920x1080 GPU samples={BENCHMARK_SAMPLE_COUNT} temporal_median=99.000 temporal_p95=140.000"
        );
        assert!(benchmark_output_is_operationally_valid(true, "", &stderr));
    }

    #[test]
    fn benchmark_rejects_non_finite_or_incomplete_samples() {
        assert!(!benchmark_output_is_operationally_valid(
            true,
            "",
            "GPU samples=600 temporal_median=NaN"
        ));
        assert!(!benchmark_output_is_operationally_valid(
            true,
            "",
            "GPU samples=incomplete temporal_median=1.000"
        ));
    }

    #[test]
    fn quality_fixture_below_threshold_fails_independently_of_timing() {
        assert!(quality_fixture_passes(1.0, 0.90));
        assert!(!quality_fixture_passes(0.89, 0.90));
        assert!(!quality_fixture_passes(f32::NAN, 0.90));
    }

    #[test]
    fn benchmark_output_has_no_latency_budget_result() {
        let stderr = format!(
            "GPU samples={BENCHMARK_SAMPLE_COUNT} temporal_median=99.000 temporal_p95=140.000"
        );
        assert!(!stderr.contains(&format!("{}false", "budget=")));
        assert!(benchmark_output_is_operationally_valid(true, "", &stderr));
    }

    #[test]
    fn vkcube_defaults_to_debug_profile_and_ten_seconds() {
        let options = parse_vkcube_args(&[]).unwrap();
        let launch = vkcube_launch_in(Path::new("/workspace"), options, None);

        assert_eq!(options.seconds, 10);
        assert!(!options.release);
        assert_eq!(launch.profile_dir, Path::new("/workspace/target/debug"));
    }

    #[test]
    fn vkcube_accepts_release_and_positive_seconds() {
        let options = parse_vkcube_args(&["--release", "--seconds", "27"]).unwrap();
        let launch = vkcube_launch_in(Path::new("/workspace"), options, None);

        assert_eq!(options.seconds, 27);
        assert!(options.release);
        assert_eq!(launch.profile_dir, Path::new("/workspace/target/release"));
    }

    #[test]
    fn vkcube_loads_the_layer_from_cargo_target_dir_when_set() {
        let options = parse_vkcube_args(&["--release"]).unwrap();
        let launch = vkcube_launch_in(
            Path::new("/workspace"),
            options,
            Some(Path::new("/tmp/cargo-target")),
        );
        assert_eq!(launch.profile_dir, Path::new("/tmp/cargo-target/release"));
    }

    #[test]
    fn vkcube_accepts_output_sharpening_ablation_values() {
        let disabled = parse_vkcube_args(&["--disable-sharpening"]).unwrap();
        assert!(!disabled.sharpening_enabled);

        let maximum = parse_vkcube_args(&["--sharpness", "1.0"]).unwrap();
        assert!(maximum.sharpening_enabled);
        assert_eq!(maximum.sharpness, 1.0);
    }

    #[test]
    fn vkcube_accepts_a_single_control_sequence_flag() {
        let options = parse_vkcube_args(&["--control-sequence"]).unwrap();
        assert!(options.control_sequence);
        assert!(parse_vkcube_args(&["--control-sequence", "--control-sequence"]).is_err());
        assert!(!parse_vkcube_args(&[]).unwrap().control_sequence);
    }

    #[test]
    fn vkcube_rejects_invalid_or_conflicting_sharpening_options() {
        for args in [
            vec!["--sharpness"],
            vec!["--sharpness", "-0.1"],
            vec!["--sharpness", "1.1"],
            vec!["--sharpness", "NaN"],
            vec!["--sharpness", "0.2", "--disable-sharpening"],
            vec!["--disable-sharpening", "--sharpness", "0.2"],
        ] {
            assert!(parse_vkcube_args(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn vkcube_accepts_the_fidelityfx_backend() {
        let options = parse_vkcube_args(&["--backend", "fsr_3_1_4"]).unwrap();

        assert_eq!(options.backend, BackendSelection::Fsr314);
    }

    #[test]
    fn fidelityfx_check_requires_the_native_host_architecture() {
        assert!(fidelityfx_elf_architecture_is_valid(
            "ELF 64-bit LSB shared object, x86-64, version 1 (SYSV)",
            "x86_64"
        ));
        assert!(!fidelityfx_elf_architecture_is_valid(
            "ELF 32-bit LSB shared object, Intel 80386",
            "x86_64"
        ));
        assert!(!fidelityfx_elf_architecture_is_valid(
            "ELF 64-bit LSB shared object, ARM aarch64",
            "x86_64"
        ));
    }

    #[test]
    fn fidelityfx_check_requires_all_versioned_abi_symbols() {
        let symbols = concat!(
            "00000000 T tux_ffx_abi_version\n",
            "00000000 T tux_ffx_version\n",
            "00000000 T tux_ffx_create\n",
            "00000000 T tux_ffx_dispatch\n",
            "00000000 T tux_ffx_reset\n",
            "00000000 T tux_ffx_destroy\n",
        );
        assert!(fidelityfx_symbols_are_complete(symbols));
        assert!(!fidelityfx_symbols_are_complete(
            symbols
                .replace("tux_ffx_abi_version", "tux_ffx_old_abi")
                .as_str()
        ));
    }

    #[test]
    fn backend_parser_rejects_unknown_names_before_launch() {
        assert!(parse_backend_args(&["--backend", "unknown"]).is_err());
        assert!(parse_backend_args(&["--backend"]).is_err());
        assert!(parse_backend_args(&["--backend", "reference", "--backend", "fsr_3_1_4"]).is_err());
    }

    #[test]
    fn vkcube_rejects_invalid_arguments() {
        for args in [
            vec!["--seconds", "0"],
            vec!["--seconds", "not-a-number"],
            vec!["--unknown"],
            vec!["--release", "--release"],
            vec!["--seconds"],
        ] {
            assert!(parse_vkcube_args(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn vkcube_does_not_treat_a_live_child_as_startup_evidence() {
        assert_eq!(
            classify_vkcube_exit(None, true, false, false, false),
            VkcubeExit::MissingStartupEvidence
        );
    }

    #[test]
    fn vkcube_classifies_missing_executable_as_failure() {
        assert_eq!(
            classify_vkcube_exit(None, false, false, false, true),
            VkcubeExit::MissingExecutable
        );
    }

    #[test]
    fn vkcube_classifies_early_nonzero_exit_as_failure() {
        assert_eq!(
            classify_vkcube_exit(Some(17), false, true, false, false),
            VkcubeExit::EarlyExit(17)
        );
    }

    #[test]
    fn vkcube_classifies_validation_errors_as_failure() {
        assert_eq!(
            classify_vkcube_exit(None, true, true, true, false),
            VkcubeExit::ValidationError
        );
    }

    #[test]
    fn vkcube_reads_layer_evidence_and_validation_errors_from_both_streams() {
        let (startup, validation) = classify_vkcube_output(
            b"TuxScaling swapchain: format=B8G8R8A8_UNORM",
            b"Validation Error: injected",
        );
        assert!(startup);
        assert!(validation);
    }

    #[test]
    fn vkcube_does_not_accept_unrelated_output_as_layer_evidence() {
        let (startup, validation) =
            classify_vkcube_output(b"TuxScaling vkcube startup: layer enabled", b"");
        assert!(!startup);
        assert!(!validation);
    }

    #[test]
    fn vkcube_requires_explicit_startup_evidence() {
        assert_eq!(
            classify_vkcube_exit(None, true, false, false, false),
            VkcubeExit::MissingStartupEvidence
        );
    }

    #[test]
    fn vkcube_requires_continuous_end_to_end_evidence() {
        let positive = concat!(
            "TuxScaling swapchain: format=B8G8R8A8_UNORM\n",
            "TuxScaling evidence event=virtual_swapchain_active logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=native_generation_published logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=1\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=2\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=3\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
        );
        let evidence = parse_vkcube_evidence(positive, "");
        assert_eq!(evidence.fsr_dispatches, 3);
        assert_eq!(evidence.reconstructed_presents, 3);
        assert!(vkcube_output_is_valid(
            positive,
            "",
            BackendSelection::Fsr314
        ));
        assert!(!vkcube_output_is_valid(
            positive.replace("fsr_dispatch", "layer_loaded").as_str(),
            "",
            BackendSelection::Fsr314
        ));
    }

    #[test]
    fn vkcube_control_sequence_requires_all_toggles_on_one_generation() {
        let positive = concat!(
            "TuxScaling swapchain: format=B8G8R8A8_UNORM\n",
            "TuxScaling evidence event=virtual_swapchain_active logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=native_generation_published logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=presenter_generation_published logical=1280x720 physical=2160x1440 generation=1\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=2160x1440\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=1\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=2\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4 frame=3\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
            "TuxScaling evidence event=overlay_submitted virtual=1\n",
            "TuxScaling evidence event=control_applied generation=1 setting=upscaler value=Off\n",
            "TuxScaling evidence event=control_applied generation=1 setting=upscaler value=FSR_3_1_4\n",
            "TuxScaling evidence event=control_applied generation=1 setting=guidance_mode value=Zero\n",
            "TuxScaling evidence event=control_applied generation=1 setting=guidance_mode value=Estimated\n",
            "TuxScaling evidence event=control_applied generation=1 setting=quality value=Performance\n",
            "TuxScaling evidence event=control_applied generation=1 setting=quality value=Balanced\n",
            "TuxScaling evidence event=control_applied generation=1 setting=sharpening_enabled value=false\n",
            "TuxScaling evidence event=control_applied generation=1 setting=sharpening_enabled value=true\n",
        );
        assert!(vkcube_output_is_valid(
            positive,
            "",
            BackendSelection::Fsr314
        ));
        assert!(vkcube_control_sequence_is_valid(positive, ""));
        let presenter_only = positive.replace(
            "TuxScaling evidence event=native_generation_published logical=1280x720 physical=2160x1440\n",
            "",
        );
        assert!(vkcube_control_sequence_is_valid(&presenter_only, ""));
        assert!(!vkcube_control_sequence_is_valid(
            &positive.replace(
                "setting=quality value=Performance\n",
                "setting=quality value=Ultra\n"
            ),
            ""
        ));
        assert!(!vkcube_control_sequence_is_valid(
            &positive.replace(
                "generation=1 setting=quality",
                "generation=2 setting=quality"
            ),
            ""
        ));
        let republished = positive.replace(
            "event=native_generation_published logical=1280x720 physical=2160x1440\n",
            "event=native_generation_published logical=1280x720 physical=2160x1440\nevent=native_generation_published logical=1280x720 physical=2160x1440\n",
        );
        assert!(!vkcube_control_sequence_is_valid(&republished, ""));
    }

    #[test]
    fn vkcube_failure_fixtures_reject_partial_or_unhealthy_runs() {
        let fixtures = include_str!("../../crates/layer/tests/fixtures/vkcube-gate-failures.txt");
        for fixture in fixtures.split("\n---\n") {
            assert!(
                !vkcube_output_is_valid(fixture, "", BackendSelection::Fsr314),
                "fixture unexpectedly passed:\n{fixture}"
            );
        }
    }

    #[test]
    fn generated_configuration_emits_only_the_canonical_guidance_key() {
        let source = generated_config("native", 1.0, Some("ultra"));

        assert!(source.contains("guidance_scale = 1"));
        assert!(source.contains("sharpening_enabled = true"));
        assert!(source.contains("sharpness = 0.3"));
        assert!(!source.contains("processing_scale"));
        assert!(!source.contains("render_scale"));
    }

    #[test]
    fn visual_quality_requires_the_safe_fsr_input_contract() {
        let safe = CaptureFsrInputs {
            motion: "Estimated".into(),
            confidence: "Estimated".into(),
            depth: "SuppressedIncompatible".into(),
            exposure: "SuppressedIncompatible".into(),
            reactive: "Neutral".into(),
            composition: "Neutral".into(),
            jitter: "Neutral".into(),
        };
        assert!(fsr_input_contract_is_safe("Estimated", &safe));

        let mut unsafe_depth = safe.clone();
        unsafe_depth.depth = "Estimated".into();
        assert!(!fsr_input_contract_is_safe("Estimated", &unsafe_depth));

        let zero = CaptureFsrInputs {
            motion: "Neutral".into(),
            confidence: "Neutral".into(),
            ..safe.clone()
        };
        assert!(fsr_input_contract_is_safe("Zero", &zero));
        assert!(!fsr_input_contract_is_safe("Zero", &safe));
        assert!(!fsr_input_contract_is_safe("Unknown", &safe));
    }

    #[test]
    fn visual_quality_accepts_only_initial_neutral_fsr_motion_before_stable_history() {
        let estimated = CaptureFsrInputs {
            motion: "Estimated".into(),
            confidence: "Estimated".into(),
            depth: "SuppressedIncompatible".into(),
            exposure: "SuppressedIncompatible".into(),
            reactive: "Neutral".into(),
            composition: "Neutral".into(),
            jitter: "Neutral".into(),
        };
        let neutral = CaptureFsrInputs {
            motion: "Neutral".into(),
            confidence: "Neutral".into(),
            ..estimated.clone()
        };
        assert!(fsr_input_contract_is_safe_for_frame(
            "Estimated",
            &neutral,
            1,
            "None",
        ));
        assert!(fsr_input_contract_is_safe_for_frame(
            "Estimated",
            &neutral,
            8,
            "SceneCut",
        ));
        assert!(!fsr_input_contract_is_safe_for_frame(
            "Estimated",
            &neutral,
            8,
            "None",
        ));
        assert!(fsr_input_contract_is_safe_for_frame(
            "Estimated",
            &estimated,
            8,
            "None",
        ));
    }

    #[test]
    fn guidance_comparison_gate_uses_finite_quality_tolerances() {
        assert!(guidance_comparison_gate_passed(
            20.0, 20.0, 0.90, 0.90, 0.01, 0.01
        ));
        assert!(guidance_comparison_gate_passed(
            19.75, 20.0, 0.895, 0.90, 0.012, 0.01
        ));
        assert!(!guidance_comparison_gate_passed(
            19.74, 20.0, 0.90, 0.90, 0.01, 0.01
        ));
        assert!(!guidance_comparison_gate_passed(
            20.0, 20.0, 0.894, 0.90, 0.01, 0.01
        ));
        assert!(!guidance_comparison_gate_passed(
            20.0, 20.0, 0.90, 0.90, 0.0121, 0.01
        ));
        assert!(!guidance_comparison_gate_passed(
            f32::NAN,
            20.0,
            0.90,
            0.90,
            0.01,
            0.01,
        ));
    }

    #[test]
    fn maintenance_evidence_accepts_complete_interleaved_output() {
        let stdout = concat!(
            "TuxScaling evidence event=maintenance1_device enabled=1 flavor=ext\n",
            "TuxScaling evidence event=maintenance1_present fences=2 modes=2 virtual=1\n",
            "TuxScaling evidence event=overlay_submitted maintenance1=1\n",
            "TuxScaling evidence event=reconstructed_present backend=FSR_3_1_4\n",
        );
        let stderr = concat!(
            "TuxScaling evidence event=logical_swapchain_created logical_handle=0x1 physical_handle=0x2 logical=1280x720 physical=1280x720 virtual=1 negotiation=negotiating\n",
            "TuxScaling evidence event=virtual_swapchain_active logical_handle=0x1 logical=1280x720 physical=3440x1440 generation=1\n",
            "TuxScaling evidence event=logical_recreation_translated old_logical=0x1 old_physical=0x2 downstream_extent=3440x1440\n",
            "TuxScaling evidence event=maintenance1_release logical_count=1 physical_count=1 result=SUCCESS\n",
            "TuxScaling evidence event=fsr_dispatch backend=FSR_3_1_4 logical=1280x720 physical=3440x1440\n",
        );
        let evidence = parse_maintenance_evidence(stdout, stderr);

        assert!(maintenance_evidence_complete(&evidence));
        assert!(maintenance_output_is_valid(
            stdout,
            stderr,
            BackendSelection::Fsr314
        ));
    }

    #[test]
    fn maintenance_evidence_rejects_missing_fields_and_validation_errors() {
        let incomplete = parse_maintenance_evidence(
            "TuxScaling evidence event=maintenance1_device enabled=1 flavor=khr\n",
            "TuxScaling evidence event=maintenance1_present fences=1 modes=1 virtual=1\n",
        );
        assert!(!maintenance_evidence_complete(&incomplete));

        let validation = parse_maintenance_evidence(
            "TuxScaling evidence event=maintenance1_release logical_count=1 physical_count=1 result=SUCCESS\n",
            "Validation Error: VUID-VkSwapchainCreateInfoKHR-pNext-07781\n",
        );
        assert!(validation.validation_error);
        assert!(!maintenance_evidence_complete(&validation));
    }

    #[test]
    fn wsi_compatibility_accepts_each_portable_scenario() {
        for scenario in [
            "mutable_format",
            "present_wait_generation",
            "hdr_replacement",
            "display_timing",
        ] {
            let output = portable_wsi_fixture(scenario);
            assert!(
                wsi_compatibility_output_is_valid(&output, "", scenario, BackendSelection::Fsr314,),
                "{scenario}"
            );
        }
    }

    #[test]
    fn wsi_compatibility_parser_accepts_allow_unverified() {
        let (backend, allowed) = parse_wsi_compatibility_args(&[
            "--backend",
            "fsr_3_1_4",
            "--allow-unverified",
            "display_timing",
        ])
        .unwrap();
        assert_eq!(backend, BackendSelection::Fsr314);
        assert_eq!(allowed, vec!["display_timing".to_owned()]);
        let (_, none) = parse_wsi_compatibility_args(&["--backend", "reference"]).unwrap();
        assert!(none.is_empty());
        assert!(parse_wsi_compatibility_args(&["--allow-unverified", "bogus"]).is_err());
        assert!(parse_wsi_compatibility_args(&["--allow-unverified"]).is_err());
        assert!(parse_wsi_compatibility_args(&["--unknown"]).is_err());
    }

    #[test]
    fn quality_evidence_extracts_measured_metric_lines() {
        let output = "WARNING: radv is not a conformant Vulkan implementation, testing use only.\n\
            guidance_scale=1.0 balanced_forward_backward_ms=1.9960 performance_forward_backward_ms=0.8521\n\
            test performance_flow_is_faster_than_balanced_after_warmup ... ok\n\
            baseline fixture=translation hash=80b2a154a13c0084 mean_epe=1.593801 p95_epe=10.440307\n\
            FSR guidance modes: zero_vs_off_mse=0.002711920\n\
            FSR sequence aggregate: psnr=17.637dB bilinear=16.635dB reference=17.385dB ssim=0.87448\n\
            relative depth parallax: order=1.000 foreground=1.000 background=0.121\n\
            test result: ok. 2 passed; 0 failed\n";
        let lines = quality_metric_lines(output);
        assert_eq!(
            lines,
            vec![
                "guidance_scale=1.0 balanced_forward_backward_ms=1.9960 performance_forward_backward_ms=0.8521".to_owned(),
                "baseline fixture=translation hash=80b2a154a13c0084 mean_epe=1.593801 p95_epe=10.440307".to_owned(),
                "FSR guidance modes: zero_vs_off_mse=0.002711920".to_owned(),
                "FSR sequence aggregate: psnr=17.637dB bilinear=16.635dB reference=17.385dB ssim=0.87448".to_owned(),
                "relative depth parallax: order=1.000 foreground=1.000 background=0.121".to_owned(),
            ]
        );
    }

    #[test]
    fn quality_evidence_parses_preset_medians() {
        let (balanced, performance) = parse_preset_medians(
            "guidance_scale=1.0 balanced_forward_backward_ms=1.9960 performance_forward_backward_ms=0.8521",
        )
        .unwrap();
        assert!((balanced - 1.9960).abs() < 1e-4);
        assert!((performance - 0.8521).abs() < 1e-4);
        assert!(parse_preset_medians("no metrics here").is_none());
    }

    #[test]
    fn quality_ablation_rows_use_measured_suite_evidence() {
        let evidence = CollectedQualityEvidence {
            suites: vec![
                QualityEvidenceSuiteReport {
                    name: "upscaler-ablations".to_owned(),
                    passed: true,
                    metrics: vec![
                        "FSR guidance modes: zero_vs_off_mse=0.002711920".to_owned(),
                        "FSR sequence aggregate: psnr=17.637dB".to_owned(),
                    ],
                    log: "upscaler-ablations.log".to_owned(),
                },
                QualityEvidenceSuiteReport {
                    name: "temporal-fallbacks".to_owned(),
                    passed: true,
                    metrics: vec![
                        "relative depth parallax: order=1.000 foreground=1.000 background=0.121"
                            .to_owned(),
                    ],
                    log: "temporal-fallbacks.log".to_owned(),
                },
            ],
            balanced_median: 2.0956,
            performance_median: 0.8992,
        };
        let rows = quality_ablation_rows(&evidence);
        assert_eq!(rows.len(), 7);
        assert!(
            !rows
                .iter()
                .any(|row| row.status.contains("follow-up dimension"))
        );
        let depth = rows
            .iter()
            .find(|row| row.signal == "relative_depth")
            .unwrap();
        assert!(depth.status.contains("temporal-fallbacks"));
        assert!(depth.status.contains("order=1.000"));
        let motion = rows.iter().find(|row| row.signal == "motion").unwrap();
        assert!(motion.status.contains("zero_vs_off_mse=0.002711920"));
    }

    #[test]
    fn quality_preset_rows_report_measured_medians() {
        let evidence = CollectedQualityEvidence {
            suites: Vec::new(),
            balanced_median: 2.0956,
            performance_median: 0.8992,
        };
        let rows = quality_preset_rows(&evidence, (1280, 720));
        assert_eq!(rows.len(), 2);
        assert!(rows[0].quality_baseline);
        assert!(!rows[1].quality_baseline);
        assert!(rows[0].timing_status.contains("2.0956"));
        assert!(rows[1].timing_status.contains("0.8992"));
        assert!(rows[1].timing_status.contains("true"));
    }

    #[test]
    fn visual_quality_parser_supports_skipping_gpu_evidence() {
        let base = ["--input", "1280x720", "--output", "2160x1440"];
        assert!(!parse_visual_quality_args(&base).unwrap().skip_gpu_evidence);
        let with_flag = [
            "--input",
            "1280x720",
            "--output",
            "2160x1440",
            "--skip-gpu-evidence",
        ];
        assert!(
            parse_visual_quality_args(&with_flag)
                .unwrap()
                .skip_gpu_evidence
        );
    }

    #[test]
    fn visual_quality_parser_accepts_the_nested_xwayland_contract() {
        let options = parse_visual_quality_args(&[
            "--display",
            "nested-xwayland",
            "--input",
            "1280x720",
            "--output",
            "2160x1440",
            "--warmup",
            "180",
            "--frames",
            "120",
        ])
        .unwrap();
        assert_eq!(options.display, "nested-xwayland");
        assert_eq!(options.input, (1280, 720));
        assert_eq!(options.output, (2160, 1440));
        assert_eq!(options.warmup, 180);
        assert_eq!(options.frames, 120);
    }

    #[test]
    fn visual_quality_parser_rejects_incomplete_or_invalid_extents() {
        for args in [
            vec!["--input", "1280x720"],
            vec!["--output", "2160x1440"],
            vec!["--input", "0x720", "--output", "2160x1440"],
            vec![
                "--input",
                "1280x720",
                "--output",
                "2160x1440",
                "--frames",
                "0",
            ],
        ] {
            assert!(parse_visual_quality_args(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn nested_mutter_parser_extracts_public_x11_display() {
        assert_eq!(
            parse_public_x11_display(
                "libmutter-Message: Using public X11 display :42, (using unix:/tmp/.X11-unix/X43 for managed services)"
            ),
            Some(":42".to_owned())
        );
        assert_eq!(
            parse_public_x11_display("Using Wayland display name 'nested'"),
            None
        );
    }

    #[test]
    fn visual_quality_metrics_match_tiny_known_images_exactly() {
        let reference = vec![[0.0, 0.25, 0.5, 1.0], [1.0, 0.75, 0.5, 1.0]];
        let estimate = vec![[0.0, 0.25, 0.25, 1.0], [0.5, 0.75, 0.5, 1.0]];
        let metrics = visual_quality_metrics_for_test(&reference, &estimate, &estimate);
        assert_eq!(metrics.mse, 0.0390625);
        assert_eq!(metrics.psnr, 14.082399);
        assert_eq!(metrics.ssim, 0.9774446);
        assert_eq!(metrics.flicker_mse, 0.0);
        assert_eq!(metrics.ghost_trail, 0.125);
        assert_eq!(metrics.shimmer, 0.0);
        assert_eq!(metrics.difference_map.len(), 2);
    }

    #[test]
    fn wsi_compatibility_rejects_missing_semantic_fields() {
        let output = portable_wsi_fixture("hdr_replacement");
        let output = output.replace("hdr_after=1", "hdr_after=0");
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "hdr_replacement",
            BackendSelection::Fsr314,
        ));
    }

    #[test]
    fn wsi_compatibility_rejects_virtual_fallback_for_compatible_scenarios() {
        let output = portable_wsi_fixture("mutable_format").replace("virtual=1", "virtual=0");
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "mutable_format",
            BackendSelection::Fsr314,
        ));

        let output = format!(
            "{}TuxScaling evidence event=virtualization_preflight result=direct reason=unexpected\n",
            portable_wsi_fixture("mutable_format")
        );
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "mutable_format",
            BackendSelection::Fsr314,
        ));
    }

    #[test]
    fn wsi_compatibility_accepts_only_audited_direct_fallback_reasons() {
        let output = concat!(
            "TuxScaling evidence event=wsi_scenario_request scenario=incompatible_direct requested=1280x720\n",
            "TuxScaling evidence event=virtualization_preflight result=direct reason=unsupported_pnext stype=DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR\n",
            "TuxScaling evidence event=logical_swapchain_created logical_handle=0x2 physical_handle=0x3 logical=3440x1440 physical=3440x1440 virtual=0 mutable_format=0 view_formats=0 negotiation=direct\n",
            "TuxScaling evidence event=wsi_scenario scenario=incompatible_direct result=verified direct=1\n",
        );
        assert!(wsi_compatibility_output_is_valid(
            output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));

        let output = output.replace("reason=unsupported_pnext", "reason=unexpected");
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));
    }

    #[test]
    fn wsi_compatibility_derives_native_extent_instead_of_hard_coding_it() {
        let output = portable_wsi_fixture("display_timing").replace(
            "event=borderless_target extent=3440x1440",
            "event=borderless_target extent=2560x1440",
        );
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "display_timing",
            BackendSelection::Fsr314,
        ));
    }

    #[test]
    fn wsi_compatibility_rejects_validation_errors_and_panics() {
        let output = portable_wsi_fixture("present_wait_generation");
        for diagnostic in [
            "Validation Error: VUID-VkSwapchainCreateInfoKHR-pNext-07781",
            "panic: assertion failed",
            "VUID-VkPresentInfoKHR-pSwapchains-01296",
        ] {
            assert!(!wsi_compatibility_output_is_valid(
                &output,
                diagnostic,
                "present_wait_generation",
                BackendSelection::Fsr314,
            ));
        }
    }

    #[test]
    fn wsi_compatibility_accepts_only_an_explicit_incompatible_direct_fallback() {
        let output = concat!(
            "TuxScaling evidence event=wsi_scenario_request scenario=incompatible_direct requested=1280x720\n",
            "TuxScaling evidence event=virtualization_preflight result=direct reason=incompatible_wsi_extension extension=VK_NV_present_barrier\n",
            "TuxScaling evidence event=logical_swapchain_created logical_handle=0x2 physical_handle=0x3 logical=3440x1440 physical=3440x1440 virtual=0 mutable_format=0 view_formats=0 negotiation=direct\n",
            "TuxScaling evidence event=wsi_scenario scenario=incompatible_direct result=verified direct=1\n",
        );
        assert!(wsi_compatibility_output_is_valid(
            output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));

        let output = output.replace("incompatible_wsi_extension", "unexpected");
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));
    }

    #[test]
    fn wsi_compatibility_requires_direct_creation_evidence_without_a_logical_event() {
        let output = concat!(
            "TuxScaling evidence event=wsi_scenario_request scenario=incompatible_direct requested=1280x720\n",
            "TuxScaling evidence event=virtualization_preflight result=direct reason=unsupported_pnext stype=DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR\n",
            "TuxScaling evidence event=direct_swapchain_created surface=0x1 extent=1280x720 reason=unsupported_pnext\n",
            "TuxScaling evidence event=wsi_scenario scenario=incompatible_direct result=verified direct=1\n",
        );
        assert!(wsi_compatibility_output_is_valid(
            output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));

        let output = output.replace(
            "TuxScaling evidence event=direct_swapchain_created surface=0x1 extent=1280x720 reason=unsupported_pnext\n",
            "",
        );
        assert!(!wsi_compatibility_output_is_valid(
            &output,
            "",
            "incompatible_direct",
            BackendSelection::Fsr314,
        ));
    }
}
