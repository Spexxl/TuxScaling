# FidelityFX FSR 3.1.4

TuxScaling's optional vendor backend consumes the packaged Linux companion
library built from AMD's official FidelityFX SDK 1.1.4, tag `v1.1.4`, at commit
`c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`. The selected component is FSR
3.1.4 Super Resolution; frame generation is not included. The SDK source is
not stored in this repository; normal builds need only the packaged `.so`.

## Why 3.1.4

The milestone pins the SDK revision that provides the audited FSR 3.1.4
source, generated Vulkan permutations, and stable project-owned ABI. FSR
3.1.5 is intentionally not substituted: moving to it requires re-auditing
the source, shader hashes, native ABI behavior, validation results, and
license inventory.

## Prerequisites

Normal Linux builds require the packaged companion library, Vulkan headers and
loader libraries, and `glslc`. CMake, Ninja or Make, a C++17 compiler, Wine,
and an external SDK checkout are required only for maintainer rebuilds or
shader regeneration.

The packaged companion is `lib/libtuxscaling_fidelityfx_vk.so`. To select a
different local build:

```bash
cp /path/to/libtuxscaling_fidelityfx_vk.so lib/
# or:
export TUXSCALING_FIDELITYFX_LIBRARY=/path/to/libtuxscaling_fidelityfx_vk.so
```

Validate the packaged companion library:

```bash
cargo xtask fidelityfx-check
```

The command checks that the selected prebuilt file is an ELF shared library,
contains its five ABI symbols, and passes the Rust version-loader test. A
regular Cargo build with the default `fidelityfx` feature uses the same local
file; `--no-default-features` omits the optional backend.

The generated headers are retained for the optional external rebuild flow.
To regenerate them, set `TUXSCALING_FIDELITYFX_SDK` to an external exact SDK
checkout and use `scripts/fidelityfx/generate-vulkan-shaders.sh` on a
maintainer machine with Wine, then run the verifier. The regeneration script
applies the `rgba16f` luma-history correction before compilation and rewrites
the SHA-256 manifest.

## Runtime selection and packaging

`Reference` is the default. Select FSR manually with:

```toml
upscaler = "fsr_3_1_4"
```

Selecting `Off` (config `upscaler = "off"` or the overlay selector) disables
temporal upscaling and every simulation stage: frames take a plain
aspect-fit blit plus the overlay composite until another upscaler is
selected, which resumes the full pipeline.

The egui overlay exposes the same selector and reports both the requested and
active backend. A failed load, context creation, dispatch, or secondary
recording leaves presentation alive and falls back without submitting the
partial secondary command buffer.

The output-sharpening slider is a perceptual strength control in the range
`0.0..=1.0`, enabled by default at `0.3`. The runtime maps that control through
an explicit square-root response before passing it to FSR so useful changes are
visible below the maximum without changing the temporal input contract. The
checkbox disables output sharpening independently of temporal reconstruction.

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

The GPU check replays twelve deterministic capture fixtures covering
translation, rotation, scaling, camera pan, thin geometry, HUD, transparency,
particles, occlusion/disocclusion, noise, scene cuts, and pause/resume. It
reports FSR, bilinear, and Reference PSNR/SSIM plus motion-compensated
temporal flicker, and fails unless FSR improves the aggregate bilinear result.

### FSR input contract

The adapter deliberately forwards only guidance that satisfies the canonical
contract. Estimated motion and confidence are used when the frame is valid,
non-zero, measured in source/game pixels, directed `CurrentToPrevious`, and not
marked for a history reset. Every other case uses neutral motion and confidence;
the runtime resets temporal history whenever the guidance metadata requires it.

The remaining FSR inputs are explicit compatibility fallbacks until native
semantics can be obtained safely:

- depth is a flat zero image, with the inverted/infinite depth creation flags;
- exposure is a persistent 1x1 image containing unity and `preExposure = 1.0`;
- reactive and transparency/composition masks are zero images;
- camera jitter is zero in the stable default. The experimental Halton mode is
  forwarded only when the captured color and guidance images share the game
  input extent; in that mode FSR receives the paired jittered color and the
  matching jitter-cancellation flag. Post-capture jitter remains an estimate,
  not native engine projection jitter.

These values are not estimates of engine depth, exposure, reactive regions, or
jitter. The overlay and diagnostic capture metadata expose the per-input states
(`Estimated`, `Neutral`, or `SuppressedIncompatible`) so a quality report can
reject an unsafe adapter contract instead of silently treating it as native
data. The `visual-quality` command captures FSR with Estimated guidance, FSR
with Zero guidance, and an Off spatial baseline, then reports both comparisons
and fails when Estimated guidance regresses against Zero beyond the documented
deterministic tolerance.

## Upgrade checklist

1. Verify the new SDK tag and commit, version macros, and license files.
2. Re-audit the luma-history patch and regenerate all Vulkan headers.
3. Rebuild the Linux shim and validate all ABI symbols and reported version.
4. Run Rust lifecycle, adapter, resize, Native AA, backend-switch, fallback,
   and validation scenarios.
5. Re-run the deterministic quality comparison before changing the default.
6. Update `THIRD_PARTY_NOTICES.md` and review the complete license inventory.
