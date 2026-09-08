use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");
    let shader_root = repo_root.join("shaders/upscaler");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    compile_shader(
        &shader_root.join("reconstruct.comp"),
        &out.join("reconstruct.spv"),
    );
    for name in ["fidelityfx_input", "fidelityfx_output"] {
        let source = shader_root.join(format!("{name}.comp"));
        println!("cargo:rerun-if-changed={}", source.display());
        if source.is_file() {
            compile_shader(&source, &out.join(format!("{name}.spv")));
        }
    }

    println!("cargo:rerun-if-env-changed=TUXSCALING_FIDELITYFX_SDK");
    if env::var_os("CARGO_FEATURE_FIDELITYFX").is_some() {
        if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
            panic!("the fidelityfx feature is supported only on Linux");
        }

        let driver = repo_root.join("scripts/fidelityfx/build-linux.sh");
        println!("cargo:rerun-if-changed={}", driver.display());
        println!(
            "cargo:rerun-if-changed={}",
            repo_root
                .join("scripts/fidelityfx/patches/linux-build.patch")
                .display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            repo_root.join("third_party/fidelityfx-sdk").display()
        );

        let status = Command::new(&driver)
            .arg(&out)
            .status()
            .expect("failed to run the FidelityFX Linux build driver");
        assert!(
            status.success(),
            "failed to build the FidelityFX companion library"
        );
        println!(
            "cargo:rustc-env=TUXSCALING_BUILT_FIDELITYFX={}",
            out.join("libtuxscaling_fidelityfx_vk.so").display()
        );
    }
}

fn compile_shader(source: &std::path::Path, output: &std::path::Path) {
    println!("cargo:rerun-if-changed={}", source.display());
    let status = Command::new("glslc")
        .args(["--target-env=vulkan1.0", "-O"])
        .arg(source)
        .arg("-o")
        .arg(output)
        .status()
        .expect("glslc is required to build the upscaler shaders");
    assert!(status.success(), "failed to compile {}", source.display());
}
