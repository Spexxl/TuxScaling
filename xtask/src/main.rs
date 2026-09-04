use std::process::{Command, ExitCode};

fn run(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn main() -> ExitCode {
    let command = std::env::args().nth(1).unwrap_or_default();
    if command != "check" {
        eprintln!("Usage: cargo xtask check");
        return ExitCode::from(2);
    }

    let checks = [
        ("cargo", ["fmt", "--all", "--", "--check"].as_slice()),
        ("cargo", ["test", "--workspace"].as_slice()),
        (
            "cargo",
            [
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ]
            .as_slice(),
        ),
    ];

    for (program, args) in checks {
        if !run(program, args) {
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
