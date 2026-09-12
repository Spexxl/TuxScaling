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
    for name in ["compare", "fidelityfx_input", "fidelityfx_output"] {
        let source = shader_root.join(format!("{name}.comp"));
        println!("cargo:rerun-if-changed={}", source.display());
        if source.is_file() {
            compile_shader(&source, &out.join(format!("{name}.spv")));
        }
    }

    println!("cargo:rerun-if-env-changed=TUXSCALING_FIDELITYFX_LIBRARY");
    if env::var_os("CARGO_FEATURE_FIDELITYFX").is_some() {
        if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
            panic!("the fidelityfx feature is supported only on Linux");
        }

        let configured = env::var_os("TUXSCALING_FIDELITYFX_LIBRARY").map(PathBuf::from);
        let library = configured.map_or_else(
            || repo_root.join("lib/libtuxscaling_fidelityfx_vk.so"),
            |path| {
                if path.is_absolute() {
                    path
                } else {
                    repo_root.join(path)
                }
            },
        );
        println!("cargo:rerun-if-changed={}", library.display());
        assert!(
            library.is_file(),
            "FidelityFX companion library is missing: {} (copy the prebuilt .so there or set TUXSCALING_FIDELITYFX_LIBRARY)",
            library.display()
        );
        println!(
            "cargo:rustc-env=TUXSCALING_BUILT_FIDELITYFX={}",
            library.display()
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
