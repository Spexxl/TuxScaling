use std::process::{Command, ExitCode, Output};

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
            let run_wsi = |force_temporal_failure| {
                let mut command = Command::new(root.join("target/debug/examples/wsi"));
                validation(&mut command)
                    .env("VK_ADD_LAYER_PATH", root.join("assets/vulkan-layer"))
                    .env("LD_LIBRARY_PATH", std::env::join_paths(&libraries).unwrap())
                    .env(
                        "VK_INSTANCE_LAYERS",
                        "VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation",
                    )
                    .env("TUXSCALING_VIEW", "motion")
                    .env("TUXSCALING_TEST_FORCE_VIRTUAL", "1")
                    .env("TUXSCALING_TEST_SECONDS", "12");
                if force_temporal_failure {
                    command.env("TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE", "1");
                }
                report(command.output())
            };
            run_wsi(false) && run_wsi(true)
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
            eprintln!("Usage: cargo xtask <check|gpu-check|smoke>");
            return ExitCode::from(2);
        }
    };
    if success {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
