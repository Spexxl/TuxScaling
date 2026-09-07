use std::process::{Command, ExitCode, Output};

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
            eprintln!("Usage: cargo xtask <benchmark|check|gpu-check|smoke>");
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
    use super::benchmark_cases;

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
}
