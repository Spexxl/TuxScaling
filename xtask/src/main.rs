use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const VKCUBE_LAYER_EVIDENCE_MARKER: &str = "TuxScaling swapchain:";

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VkcubeOptions {
    seconds: u64,
    release: bool,
    backend: BackendSelection,
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
    };
    let mut backend_seen = false;
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
            value => return Err(format!("unknown vkcube argument: {value}")),
        }
        index += 1;
    }
    Ok(options)
}

fn vkcube_launch(root: &Path, options: VkcubeOptions) -> VkcubeLaunch {
    let profile = if options.release { "release" } else { "debug" };
    VkcubeLaunch {
        profile_dir: root.join("target").join(profile),
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
    command
}

fn wait_for_vkcube(mut child: Child, seconds: u64) -> VkcubeExit {
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
    classify_vkcube_exit(
        status.code(),
        timed_out,
        startup_evidence,
        validation_error,
        false,
    )
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
        generated_config_with_backend("swapchain", 1.0, None, options.backend),
    )
    .is_err()
    {
        return VkcubeExit::BuildFailure;
    }
    match configure_vkcube_command(root, options).spawn() {
        Ok(child) => wait_for_vkcube(child, options.seconds),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => VkcubeExit::MissingExecutable,
        Err(_) => VkcubeExit::UnexpectedExit,
    }
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
    let quality = quality.map_or_else(String::new, |quality| {
        format!("motion_quality = \"{quality}\"\n")
    });
    format!(
        "output_resolution = \"{output_resolution}\"\nguidance_scale = {guidance_scale}\n{quality}upscaler = \"{}\"\n",
        backend.config_value()
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
    if !file.status.success() || !file_text.contains("ELF") {
        eprintln!(
            "cargo xtask fidelityfx-check: companion is not an ELF shared library: {}",
            library.display()
        );
        return false;
    }
    let Ok(symbols) = Command::new("nm").args(["-D"]).arg(&library).output() else {
        return false;
    };
    let symbol_text = String::from_utf8_lossy(&symbols.stdout);
    let symbols_ok = symbols.status.success()
        && [
            "tux_ffx_version",
            "tux_ffx_create",
            "tux_ffx_dispatch",
            "tux_ffx_reset",
            "tux_ffx_destroy",
        ]
        .iter()
        .all(|symbol| symbol_text.lines().any(|line| line.ends_with(symbol)));
    if !symbols_ok {
        eprintln!("cargo xtask fidelityfx-check: companion symbols are incomplete");
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
                "Usage: cargo xtask <benchmark|check|fidelityfx-check|gpu-check|smoke|vkcube>"
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
        BENCHMARK_SAMPLE_COUNT, BackendSelection, VkcubeExit, benchmark_cases,
        benchmark_output_is_operationally_valid, classify_vkcube_exit, classify_vkcube_output,
        generated_config, maintenance_evidence_complete, maintenance_output_is_valid,
        parse_backend_args, parse_maintenance_evidence, parse_vkcube_args, quality_fixture_passes,
        vkcube_launch,
    };
    use std::path::Path;

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
        let launch = vkcube_launch(Path::new("/workspace"), options);

        assert_eq!(options.seconds, 10);
        assert!(!options.release);
        assert_eq!(launch.profile_dir, Path::new("/workspace/target/debug"));
    }

    #[test]
    fn vkcube_accepts_release_and_positive_seconds() {
        let options = parse_vkcube_args(&["--release", "--seconds", "27"]).unwrap();
        let launch = vkcube_launch(Path::new("/workspace"), options);

        assert_eq!(options.seconds, 27);
        assert!(options.release);
        assert_eq!(launch.profile_dir, Path::new("/workspace/target/release"));
    }

    #[test]
    fn vkcube_accepts_the_fidelityfx_backend() {
        let options = parse_vkcube_args(&["--backend", "fsr_3_1_4"]).unwrap();

        assert_eq!(options.backend, BackendSelection::Fsr314);
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
    fn generated_configuration_emits_only_the_canonical_guidance_key() {
        let source = generated_config("native", 1.0, Some("ultra"));

        assert!(source.contains("guidance_scale = 1"));
        assert!(!source.contains("processing_scale"));
        assert!(!source.contains("render_scale"));
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
}
