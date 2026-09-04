use std::{env, path::PathBuf, process::Command};
fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../shaders/motion");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("common.glsl").display()
    );
    for name in [
        "luma",
        "downsample",
        "flow",
        "confidence",
        "scene",
        "invalidate",
        "visualize",
    ] {
        let source = root.join(format!("{name}.comp"));
        println!("cargo:rerun-if-changed={}", source.display());
        let status = Command::new("glslc")
            .arg("--target-env=vulkan1.0")
            .arg("-O")
            .arg(&source)
            .arg("-o")
            .arg(out.join(format!("{name}.spv")))
            .status()
            .expect("glslc is required; install the shader compiler before building");
        assert!(status.success(), "shader compilation failed: {name}");
    }
}
