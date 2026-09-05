use std::{env, path::PathBuf, process::Command};

fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../shaders/temporal");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let source = root.join("guidance.comp");
    println!("cargo:rerun-if-changed={}", source.display());
    let status = Command::new("glslc")
        .arg("--target-env=vulkan1.0")
        .arg("-O")
        .arg(&source)
        .arg("-o")
        .arg(out.join("guidance.spv"))
        .status()
        .expect("glslc is required; install the shader compiler before building");
    assert!(status.success(), "shader compilation failed: guidance");
}
