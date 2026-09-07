use std::process::{Command, ExitCode, Output};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BenchmarkCase {
    scenario: &'static str,
    quality: &'static str,
}

fn benchmark_cases() -> Vec<BenchmarkCase> {
    ["upscale", "native"]
        .into_iter()
        .flat_map(|scenario| {
            ["ultra", "high", "balanced", "performance"]
                .into_iter()
                .map(move |quality| BenchmarkCase { scenario, quality })
        })
        .collect()
}

fn benchmark_processing_scale() -> f32 {
    0.5
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
                "output_resolution = \"1920x1080\"\nprocessing_scale = 1.0\n",
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
                        if scenario == "native" {
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
                if scenario == "resize" {
                    command.env("TUXSCALING_TEST_RESIZE_INTERVAL", "4");
                }
                report(command.output())
            };
            run_wsi("upscale", false, false)
                && run_wsi("native", false, false)
                && run_wsi("aspect", false, false)
                && run_wsi("upscale", true, false)
                && run_wsi("upscale", false, true)
                && run_wsi("resize", false, false)
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
                    "target/benchmark-{}-{}.toml",
                    case.scenario, case.quality
                ));
                let output_resolution = if case.scenario == "native" {
                    "swapchain"
                } else {
                    "1920x1080"
                };
                if std::fs::write(
                    &config,
                    format!(
                        "output_resolution = \"{output_resolution}\"\nprocessing_scale = {}\nmotion_quality = \"{}\"\n",
                        benchmark_processing_scale(),
                        case.quality
                    ),
                )
                .is_err()
                {
                    return false;
                }
                eprintln!(
                    "TuxScaling benchmark: scenario={} quality={} warmup=180 samples=600",
                    case.scenario, case.quality
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
        assert_eq!(cases.len(), 8);
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "upscale")
                .count(),
            4
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.scenario == "native")
                .count(),
            4
        );
    }

    #[test]
    fn benchmark_uses_the_explicit_low_latency_processing_scale() {
        assert_eq!(super::benchmark_processing_scale(), 0.5);
    }
}
