# Development Guide

## Requirements

- Linux x86_64.
- Rust stable with Rust 1.88 or newer, including `rustfmt` and `clippy`.
- Cargo.
- CMake 3.20 or newer and a C compiler.
- Vulkan loader and Vulkan headers.
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
```

`cargo xtask check` is the preferred host-only command. `gpu-check` runs ignored GPU tests. `smoke` exercises multiple swapchains, grouped presents, resize, capture, optical flow, virtual-output fallback, and validation synchronization.

The debug WSI harness sets `TUXSCALING_TEST_FORCE_VIRTUAL=1` so it can validate small logical game images against the discovered monitor output without depending on a window manager fullscreen transition. Release builds ignore this test-only override.

The harness accepts `TUXSCALING_TEST_SCENARIO=upscale|native|aspect|resize`. These scenarios exercise 1280x720 to native output, equal-extent Native AA, centered aspect-fit bars, and swapchain recreation. `TUXSCALING_TEST_FORCE_RESIZE_FAILURE=1` verifies the direct-mode transaction fallback. Every other frame uses `vkAcquireNextImage2KHR` so both acquisition entry points stay covered.

Set `TUXSCALING_TEST_FORCE_TEMPORAL_FAILURE=1` in a debug run to exercise the mandatory spatial bilinear fallback used when temporal recording fails after virtual output has started.

Set `TUXSCALING_TEST_RESIZE_INTERVAL=0` only for a warmed-up timing run; the default interval is four frames and keeps resize coverage enabled.

## Native boundary

The native boundary is intentionally SDK-free in the initial milestone:

```bash
cmake -S native -B target/native-configure
```

Vendor runtimes must not be copied into the repository. Future adapters will load optional libraries at runtime and communicate through the versioned header in `native/include/backend.h`.

## Local Vulkan layer inspection

The development manifest is located at `assets/vulkan-layer/tuxscaling.json`. After building a shared layer, set `VK_LAYER_PATH` to a directory containing the manifest and the matching `libtuxscaling_layer.so`, then launch a controlled Vulkan test application.

Use `VK_LOADER_DEBUG=all` to inspect loader and layer discovery. Use `RUST_LOG=info` or `RUST_LOG=debug` for structured runtime diagnostics once the layer logging path is active.

The layer currently supports SDR swapchains with transfer and sampling usage. On fullscreen X11/XWayland, `output_resolution = "native"` uses the active monitor mode as the output while preserving the game's requested swapchain extent as its logical input; `swapchain` disables this virtual output path, and `WIDTHxHEIGHT` selects a fixed output extent. Native Wayland remains direct presentation. Unsupported formats, application-managed presentation fences, and unsupported present data use pass-through.

The default profile is equivalent to:

```toml
motion_quality = "balanced"
output_resolution = "native"
processing_scale = 1.0
debug_view = "original"
```

`processing_scale` is optional internal work reduction after capture. It does not change the game resolution or the output resolution; its default is `1.0`. The legacy `render_scale` key is accepted during the migration.

The fixed quality targets are Ultra (12 ms), High (8 ms), Balanced (4 ms), and Performance (2.5 ms) for temporal work at 1080p. The overlay reports capture, luma, pyramid, forward/backward flow, confidence, scene, invalidate, reactive, exposure, reconstruction, and overlay timings. It warns when measured work exceeds the selected target; it never changes the preset automatically.

## Language and naming policy

All source code, identifiers, comments, documentation, configuration keys, logs, errors, tests, UI text, and commit messages are written in English. Directory names remain short; Cargo package names use the `tuxscaling-` namespace.
