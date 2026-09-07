use std::{
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const VKCUBE_STARTUP_MARKER: &str = "TuxScaling vkcube startup: layer enabled";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VkcubeOptions {
    seconds: u64,
    release: bool,
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
    };
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
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    command
}

fn wait_for_vkcube(mut child: Child, seconds: u64) -> VkcubeExit {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut startup_evidence = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return classify_vkcube_exit(status.code(), false, startup_evidence, false, false);
            }
            Ok(None) => {
                if !startup_evidence {
                    println!("{VKCUBE_STARTUP_MARKER}");
                    startup_evidence = true;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let status = child.wait().ok();
                    return classify_vkcube_exit(
                        status.and_then(|status| status.code()),
                        true,
                        startup_evidence,
                        false,
                        false,
                    );
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return VkcubeExit::UnexpectedExit;
            }
        }
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
        "output_resolution = \"swapchain\"\nguidance_scale = 1.0\n",
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
    processing_scale_percent: u32,
}

fn benchmark_cases() -> Vec<BenchmarkCase> {
    ["upscale", "native_aa"]
        .into_iter()
        .flat_map(|scenario| {
            [100, 50]
                .into_iter()
                .flat_map(move |processing_scale_percent| {
                    ["ultra", "high", "balanced", "performance"]
                        .into_iter()
                        .map(move |quality| BenchmarkCase {
                            scenario,
                            quality,
                            processing_scale_percent,
                        })
                })
        })
        .collect()
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
fn run(program: &str, args: &[&str]) -> bool {
    report(Command::new(program).args(args).output())
}
fn validation(command: &mut Command) -> &mut Command {
    command
        .env("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
        .env("VK_LAYER_VALIDATE_SYNC", "1")
        .env("DISABLE_MANGOHUD", "1")
        .env("DISABLE_LSFG", "1")
}
fn main() -> ExitCode {
    let command = std::env::args().nth(1).unwrap_or_default();
    let success = match command.as_str() {
        "gpu-check" => report(
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
        ),
        "smoke" => {
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
                "output_resolution = \"native\"\nprocessing_scale = 1.0\n",
            )
            .unwrap();
            let direct_config = root.join("target/native-aa-smoke.toml");
            std::fs::write(
                &direct_config,
                "output_resolution = \"swapchain\"\nprocessing_scale = 1.0\n",
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
                        } else {
                            &smoke_config
                        },
                    )
                    .env("TUXSCALING_TEST_SCENARIO", scenario)
                    .env("TUXSCALING_TEST_RESIZE_INTERVAL", "0")
                    .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
                    .env("TUXSCALING_TEST_SECONDS", "3")
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
                report(command.output())
            };
            run_wsi("upscale", false, false)
                && run_wsi("windowed_promote", false, false)
                && run_wsi("already_borderless", false, false)
                && run_wsi("native_aa", false, false)
                && run_wsi("aspect", false, false)
                && run_wsi("resize", false, false)
                && run_wsi("monitor_origin", false, false)
                && run_wsi("promotion_failure", false, true)
                && run_wsi("temporal_failure", true, false)
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
            benchmark_cases().into_iter().all(|case| {
                let config = root.join(format!(
                    "target/benchmark-{}-{}-{}.toml",
                    case.scenario,
                    case.quality,
                    case.processing_scale_percent
                ));
                let output_resolution = if case.scenario == "native_aa" {
                    "swapchain"
                } else {
                    "native"
                };
                if std::fs::write(
                    &config,
                    format!(
                        "output_resolution = \"{output_resolution}\"\nprocessing_scale = {}\nmotion_quality = \"{}\"\n",
                        case.processing_scale_percent as f32 / 100.0,
                        case.quality
                    ),
                )
                .is_err()
                {
                    return false;
                }
                eprintln!(
                    "TuxScaling benchmark: scenario={} quality={} processing_scale={}% warmup=180 samples=600",
                    case.scenario, case.quality, case.processing_scale_percent
                );
                let mut command = Command::new(root.join("target/debug/examples/wsi"));
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
                report(command.output())
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
            eprintln!("Usage: cargo xtask <benchmark|check|gpu-check|smoke|vkcube>");
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
        VkcubeExit, benchmark_cases, classify_vkcube_exit, parse_vkcube_args, vkcube_launch,
    };
    use std::path::Path;

    #[test]
    fn benchmark_matrix_covers_each_quality_and_presentation_mode() {
        let cases = benchmark_cases();
        assert_eq!(cases.len(), 16);
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "upscale")
                .count(),
            8
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "native_aa")
                .count(),
            8
        );
        assert!(
            cases
                .iter()
                .any(|case| case.processing_scale_percent == 100)
        );
        assert!(cases.iter().any(|case| case.processing_scale_percent == 50));
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
    fn vkcube_classifies_controlled_timeout_after_startup_as_success() {
        assert_eq!(
            classify_vkcube_exit(None, true, true, false, false),
            VkcubeExit::Success
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
    fn vkcube_requires_explicit_startup_evidence() {
        assert_eq!(
            classify_vkcube_exit(None, true, false, false, false),
            VkcubeExit::MissingStartupEvidence
        );
    }
}
