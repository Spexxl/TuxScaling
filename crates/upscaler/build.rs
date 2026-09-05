use std::{env, path::PathBuf, process::Command};

fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../shaders/upscaler");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let source = root.join("reconstruct.comp");
    println!("cargo:rerun-if-changed={}", source.display());
    let status = Command::new("glslc")
        .args(["--target-env=vulkan1.0", "-O"])
        .arg(&source)
        .arg("-o")
        .arg(out.join("reconstruct.spv"))
        .status()
        .expect("glslc is required to build the reference upscaler");
    assert!(status.success(), "failed to compile {}", source.display());
}
