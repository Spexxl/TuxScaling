//! Command-line diagnostics for TuxScaling.

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "tuxscaling", version, about = "TuxScaling diagnostics")]
struct Args {}

fn main() -> Result<()> {
    let _ = Args::parse();
    Ok(())
}
