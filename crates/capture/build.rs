use std::{env, path::PathBuf, process::Command};

fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../shaders/capture");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("resample.spv");
    let source = root.join("resample.comp");
    println!("cargo:rerun-if-changed={}", source.display());
    let status = Command::new("glslc")
        .args(["--target-env=vulkan1.0", "-O"])
        .arg(&source)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc is required to build capture resampling");
    assert!(status.success(), "failed to compile {}", source.display());
}
