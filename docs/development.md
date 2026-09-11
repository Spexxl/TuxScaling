# Development Guide

## Requirements

- Linux x86_64.
- Rust stable with Rust 1.88 or newer, including `rustfmt` and `clippy`.
- Cargo.
- A prebuilt `lib/libtuxscaling_fidelityfx_vk.so` for the optional FidelityFX backend.
- CMake 3.20 or newer, a C++17 compiler, and Ninja or Make only when rebuilding
  the companion library from an external SDK checkout.
- Vulkan loader and Vulkan headers.
- `glslc` for project shader compilation. Wine and the external SDK are needed
  only when regenerating the committed FidelityFX shader headers.
- Mesa RADV for the primary validation target.
- X11/XWayland runtime libraries for interactive overlay input; native Wayland input is not enabled yet.

On Fedora, the system packages normally used for local development are `cmake`, `gcc`, `vulkan-loader`, `vulkan-headers`, `mesa-vulkan-drivers`, and `pkgconf-pkg-config`. The exact package names may vary by Fedora release.

## Build and test

From the repository root:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo xtask check
cargo xtask gpu-check
cargo xtask smoke
cargo xtask fidelityfx-check
```

`cargo xtask check` is the preferred host-only command. `gpu-check` runs ignored GPU tests. `smoke` exercises multiple swapchains, grouped presents, resize, capture, optical flow, virtual-output fallback, and validation synchronization.

Pass `--backend fsr_3_1_4` to `gpu-check` or `smoke` to exercise the experimental FidelityFX path. The FSR GPU check includes the adapter, lifecycle, deterministic captured-sequence quality, and static quality gates. `Reference` remains the default.

The debug WSI harness sets `TUXSCALING_TEST_FORCE_VIRTUAL=1` so it can validate small logical game images against the discovered monitor output without depending on a window manager fullscreen transition. Release builds ignore this test-only override.

The harness accepts `TUXSCALING_TEST_SCENARIO=upscale|windowed_promote|already_borderless|native_aa|aspect|resize|monitor_origin|promotion_failure|temporal_failure|guidance_resolve|maintenance1`. These scenarios cover ordinary window promotion, already-borderless ownership, equal-extent Native AA, centered aspect-fit bars, swapchain recreation, negative monitor origins, promotion failure, temporal failure, and revision-1 swapchain-maintenance compatibility. `TUXSCALING_TEST_FORCE_RESIZE_FAILURE=1` verifies the direct-mode transaction fallback. Every other frame uses `vkAcquireNextImage2KHR` so both acquisition entry points stay covered. Scenario runs snapshot and verify X11 geometry and fullscreen state after cleanup.

The `maintenance1` scenario requires revision 1 of either `VK_EXT_surface_maintenance1` or `VK_KHR_surface_maintenance1`, the matching swapchain-maintenance extension, `VK_KHR_get_surface_capabilities2`, and the `swapchainMaintenance1` feature. It creates a virtual swapchain from the first application-visible handle, preserves the logical extent and image contract, and forwards driver-provided compatible present modes and scaling metadata to each physical generation. Application present fences and dynamic present modes are forwarded unchanged on individual and grouped presents. Release-without-present calls translate logical indices to the active physical generation and commit the logical mapping only after downstream success. The deferred-memory-allocation flag is accepted and forwarded as a permitted hint.

Set `TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE=1` in a debug run to exercise the mandatory spatial bilinear fallback used when temporal recording fails after virtual output has started.

Set `TUXSCALING_TEST_RESIZE_INTERVAL=0` only for a warmed-up timing run; the default interval is four frames and keeps resize coverage enabled.

## Native boundary

The core layer remains SDK-free. The optional FidelityFX boundary loads the
packaged companion at runtime:

```bash
cp /path/to/libtuxscaling_fidelityfx_vk.so lib/
# or:
export TUXSCALING_FIDELITYFX_LIBRARY=/path/to/libtuxscaling_fidelityfx_vk.so
```

The legacy project-owned boundary remains available under
`native/include/backend.h`; optional maintainer rebuilds and packaging are
documented in [fidelityfx.md](fidelityfx.md).

## Local Vulkan layer inspection

The development manifest is located at `assets/vulkan-layer/tuxscaling.json`. After building a shared layer, set `VK_LAYER_PATH` to a directory containing the manifest and the matching `libtuxscaling_layer.so`, then launch a controlled Vulkan test application.

Use `VK_LOADER_DEBUG=all` to inspect loader and layer discovery. Use `RUST_LOG=info` or `RUST_LOG=debug` for structured runtime diagnostics once the layer logging path is active.

The layer currently supports SDR swapchains with transfer and sampling usage. On fullscreen X11/XWayland, `output_resolution = "native"` uses the active monitor mode as the output while preserving the game's requested swapchain extent as its logical input; `swapchain` disables this virtual output path, and `WIDTHxHEIGHT` selects a fixed output extent. Native Wayland remains direct presentation. Maintenance create metadata is owned by the layer and rebuilt for every downstream physical generation, so application pointers are never retained. Superseded physical generations stay owned by the logical swapchain until logical teardown, avoiding premature destruction while an application present fence may still refer to the old generation. Native-generation publication is a two-phase transaction: the lifecycle is marked reconfiguring under the state mutex, downstream Vulkan/runtime work runs without that mutex, and the new physical handle, images, mapping, contract, and negotiation are published together only when the captured generation is still current.

Unsupported formats and unsupported maintenance create chains select direct presentation before a logical handle is published. Unknown or malformed maintenance data on an already-published logical swapchain returns an explicit Vulkan error and is never silently removed from the forwarded chain. Backend failures remain fail-open through the spatial bypass, and borderless-window cleanup is centralized and idempotent. The maintenance path does not add a queued frame or per-frame idle/fence wait, and the existing egui controls and persistence behavior are unchanged.

### Vulkan WSI compatibility contract

Virtualization eligibility is decided by an audited, semantic WSI registry. Unrelated device extensions are ignored. Translated contracts include mutable-format swapchains and owned view-format lists, incremental present, present IDs and waits, HDR metadata, display timing, display control, swapchain/surface maintenance aliases, and swapchain colorspace. The logical image contract preserves the application format, usage, sharing mode, mutable-image flag, and view-format list; every physical generation receives the owned downstream create contract.

Present IDs are routed to the physical generation that accepted them, including retained generations after native-output replacement. HDR metadata is copied as plain owned data and reapplied before a replacement generation is published. Display-timing queries merge retained-generation results by present ID and preserve Vulkan count/data and `VK_INCOMPLETE` semantics. Application synchronization objects remain application-owned.

Contracts that cannot be translated select direct presentation before a logical handle is published. This includes shared-presentable and display swapchains, external full-screen ownership, low-latency or present-barrier contracts, device-group swapchain structures, unsupported flags, malformed or unknown swapchain-facing `pNext` nodes, and unsupported image contracts. Diagnostics use stable reasons such as `malformed_extension_names`, `incompatible_wsi_extension`, `unsupported_flags`, `unsupported_pnext`, or `unsupported_image_contract`; no valid flag or `pNext` node is silently stripped.

Run `cargo xtask wsi-compatibility --backend fsr_3_1_4` for the portable mutable-format, present-wait, HDR-replacement, display-timing, and incompatible-direct scenarios. It requires a display-backed X11/XWayland session and Vulkan validation; the native extent is read from the active monitor at runtime. If the required display-timing or display-control extension is unavailable, that scenario is reported as `unverified` rather than passed. The current milestone covers X11/XWayland only; native Wayland remains direct presentation.

The default profile is equivalent to:

```toml
motion_quality = "ultra"
output_resolution = "native"
guidance_scale = 1.0
debug_view = "original"
upscaler = "reference"
```

`guidance_scale` is optional internal work reduction for estimator guidance after full-resolution source capture. It does not change the game resolution or the output resolution; its default is `1.0` and its manual range is `0.5..=1.0`. The legacy `processing_scale` and `render_scale` keys are accepted as deserialization aliases during the migration.

Timing is telemetry, not a correctness gate. The overlay reports capture, luma, pyramid, forward/backward flow, confidence, scene, invalidate, reactive, exposure, depth, resolve, guidance total, reconstruction, overlay, and full injected timings. Logs report median and p95 for every phase plus total guidance preparation and total GPU work. The quality gates are deterministic fixture metrics shared by all presets and guidance scales; timing values do not change the selected preset.

Run the acceptance benchmark with `cargo xtask benchmark`. It builds the release WSI harness and uses one swapchain, Vulkan validation, 180 warm-up frames, 600 measured frames, `guidance_scale = 1.0`, `0.75`, and `0.5`, all four motion qualities, and both upscale and Native AA scenarios. On a 1920x1080 monitor this is the 1280x720-to-1920x1080 matrix; the harness follows the active RandR monitor elsewhere. Total guidance preparation, every phase, reconstruction, and total GPU-work median/p95 telemetry are emitted separately. The benchmark fails only for operational errors or missing/non-finite samples; slow but finite timing remains informational.

The deterministic acceptance gates are shared across presets and tested scales: mean motion EPE `<= 1.0 px`, p95 EPE `<= 2.0 px`, confidence AUROC `>= 0.90`, disocclusion F1 `>= 0.75`, reactive F1 `>= 0.70`, composition F1 `>= 0.65`, exposure error `<= 0.15 EV`, and depth ordering `>= 0.85`. `tests/fixtures/quality-baselines.txt` lists the stable metric names; it does not replace the actual estimator gate.

Run a controlled visible `vkcube` session with `cargo xtask vkcube --seconds 10`. The runner builds the layer in the selected debug or release profile, enables the TuxScaling and Vulkan validation layers, captures both child output streams, requires the layer's `TuxScaling swapchain:` evidence, rejects validation errors, and terminates and reaps `vkcube` after the requested interval. Use `--release` for the release profile and `--seconds N` for a positive duration. For a finite normal-exit maintenance regression, use the display-backed environment and command below after the maintenance smoke gates:

```bash
env \
  DISPLAY=:0 \
  XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.WOAIV3 \
  MANGOHUD=0 \
  VK_ADD_LAYER_PATH="$PWD/assets/vulkan-layer" \
  VK_INSTANCE_LAYERS=VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation \
  VK_LAYER_VALIDATE_SYNC=1 \
  DISABLE_MANGOHUD=1 \
  DISABLE_LSFG=1 \
  TUXSCALING_VIEW=reconstructed \
  TUXSCALING_CONFIG="$PWD/target/native-output-smoke.toml" \
  LD_LIBRARY_PATH="$PWD/target/debug:$PWD/lib" \
  vkcube --wsi xcb --width 1280 --height 720 --c 120
```

For the display-backed maintenance smoke, refresh `DISPLAY` and `XAUTHORITY`, verify `/tmp/.X11-unix/X0` and the authority file are readable, disable any ambient MangoHud implicit layer with `MANGOHUD=0`, and run `cargo xtask smoke --backend reference` followed by `cargo xtask smoke --backend fsr_3_1_4`. The maintenance gate requires logical `1280x720`, physical `3440x1440` on the validation workstation, `virtual=1`, successful release and logical recreation, present fences and modes, overlay submission, reconstructed presentation, FSR dispatch for the FSR run, and zero validation or maintenance-fallback events.

The following is a historical RX 9060 XT/RADV telemetry snapshot at 1920x1080. It is informational and is not a latency-quality gate; rerun the release benchmark for current values:

| Presentation | Quality | Median | p95 |
| --- | --- | ---: | ---: |
| 1280x720 to 1920x1080 | Ultra | 9.751 ms | 10.413 ms |
| 1280x720 to 1920x1080 | High | 5.372 ms | 5.578 ms |
| 1280x720 to 1920x1080 | Balanced | 2.311 ms | 2.331 ms |
| 1280x720 to 1920x1080 | Performance | 1.477 ms | 1.522 ms |
| Native AA 1920x1080 | Ultra | 11.937 ms | 12.355 ms |
| Native AA 1920x1080 | High | 6.598 ms | 6.685 ms |
| Native AA 1920x1080 | Balanced | 3.125 ms | 3.160 ms |
| Native AA 1920x1080 | Performance | 2.094 ms | 2.126 ms |

## Language and naming policy

All source code, identifiers, comments, documentation, configuration keys, logs, errors, tests, UI text, and commit messages are written in English. Directory names remain short; Cargo package names use the `tuxscaling-` namespace.
