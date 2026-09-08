# FidelityFX FSR 3.1.4

TuxScaling's optional vendor backend uses AMD's official FidelityFX SDK
1.1.4, tag `v1.1.4`, at commit
`c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`. The selected component is FSR
3.1.4 Super Resolution; frame generation is not included.

## Why 3.1.4

The milestone pins the SDK revision that provides the audited FSR 3.1.4
source, generated Vulkan permutations, and stable project-owned ABI. FSR
3.1.5 is intentionally not substituted: moving to it requires re-auditing
the source, shader hashes, native ABI behavior, validation results, and
license inventory.

## Prerequisites

Normal Linux builds require CMake, Ninja or Make, a C++17 compiler, Vulkan
headers and loader libraries, and `glslc`. Wine is required only for the
maintainer-only shader-header regeneration flow.

Initialize and verify the dependency:

```bash
git submodule update --init --recursive
scripts/fidelityfx/verify-source.sh
scripts/fidelityfx/verify-vulkan-shaders.sh
```

Build and validate the native companion library:

```bash
cargo xtask fidelityfx-check
```

The command verifies the source revision and shader manifest, builds
`libtuxscaling_fidelityfx_vk.so` natively without Wine, checks its five ABI
symbols, and runs the Rust version-loader test. A regular Cargo build with
the default `fidelityfx` feature performs the same native build into Cargo's
`OUT_DIR`; `--no-default-features` omits it.

The generated headers are committed under
`crates/upscaler/native/fidelityfx/generated/vk/`. To regenerate them, use
`scripts/fidelityfx/generate-vulkan-shaders.sh` on a maintainer machine with
Wine and the SDK-provided shader compiler, then run the verifier. The
regeneration script applies the `rgba16f` luma-history correction before
compilation and rewrites the SHA-256 manifest.

## Runtime selection and packaging

`Reference` is the default. Select FSR manually with:

```toml
upscaler = "fsr_3_1_4"
```

The egui overlay exposes the same selector and reports both the requested and
active backend. A failed load, context creation, dispatch, or secondary
recording leaves presentation alive and falls back without submitting the
partial secondary command buffer.

Install `libtuxscaling_fidelityfx_vk.so` beside the TuxScaling Vulkan layer,
or point the loader at an explicit path:

```bash
export TUXSCALING_FIDELITYFX_LIBRARY=/path/to/libtuxscaling_fidelityfx_vk.so
```

The loader rejects every native library whose reported version is not exactly
3.1.4. The runtime uses one queue submission per presented frame and no
per-frame queue idle operation.

## Validation

Run the ignored adapter lifecycle and WSI scenarios with validation enabled:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  cargo xtask gpu-check --backend fsr_3_1_4
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  cargo xtask smoke --backend fsr_3_1_4
cargo xtask vkcube --seconds 10 --release --backend fsr_3_1_4
```

FSR currently receives zero camera jitter and synthetic flat/relative depth
derived from estimated color guidance. These are explicit limitations:
TuxScaling does not intercept native game jitter or native motion/depth
semantics, so output-quality comparisons must not be described as native FSR
integration.

## Upgrade checklist

1. Verify the new SDK tag and commit, version macros, and license files.
2. Re-audit the luma-history patch and regenerate all Vulkan headers.
3. Rebuild the Linux shim and validate all ABI symbols and reported version.
4. Run Rust lifecycle, adapter, resize, Native AA, backend-switch, fallback,
   and validation scenarios.
5. Re-run the deterministic quality comparison before changing the default.
6. Update `THIRD_PARTY_NOTICES.md` and review the complete license inventory.
