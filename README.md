# TuxScaling

TuxScaling is a Linux-only Vulkan layer for experimenting with temporal upscaling in games that do not provide a native temporal-upscaling integration.

The project targets native Vulkan applications games translated to Vulkan through Proton.

## Current status

The repository captures supported swapchains, estimates optical flow and temporal guidance on the GPU, optionally resamples the captured game image, runs a vendor-neutral reference reconstruction or the experimental AMD FidelityFX FSR 3.1.4 Super Resolution backend, and renders an egui diagnostic overlay. On fullscreen X11/XWayland, the layer can present a lower-resolution game swapchain through a monitor-sized output swapchain. The reference backend remains the default until the FSR quality gates pass.

## Design boundaries

- Rust owns the runtime, Vulkan layer, frame pipeline, synthetic temporal inputs, configuration, diagnostics, and egui interface.
- Native C or C++ code is isolated behind a small C ABI for optional SDK adapters.
- Inputs are always estimated from captured color frames; game-native temporal inputs are not intercepted.
- Guidance includes source-pixel current-to-previous motion, confidence, temporal reactive and disocclusion masks, log-luminance exposure, and a flat depth fallback. Jitter is always zero.
- The default `Balanced` preset is fixed for the session. `Ultra`, `High`, and `Performance` trade optical-flow work for precision; the overlay applies changes at a frame boundary and reports per-pass GPU timings.
- The reference backend uses an optional 50–100% internal guidance scale (default 100%), neighborhood clamping, confidence-weighted accumulation, and history reset on first frame, long pause, presentation failure, resize, or preset change.
- `Reference` is the default upscaler. `FSR 3.1.4` is selectable through `upscaler = "fsr_3_1_4"` or the egui overlay; its native Linux companion library is loaded only when the FidelityFX feature is enabled and failures fall back to the reference/spatial path.
- FSR consumes estimated full-game-resolution guidance, dispatches zero jitter, and records through the runtime's single-submit secondary command buffer path. It does not change the game's internal resolution or provide frame generation.
- X11/XWayland input is optional at runtime. `Insert` toggles the overlay and pointer/keyboard grabs are released on close and destruction. Native Wayland remains configuration-only in this milestone.
- The injected runtime uses `ash` directly and does not use `wgpu`, `vulkano`, `eframe`, or `winit`.
- The initial milestone does not include Windows, DirectX, OpenGL, frame generation, a standalone GUI, or a required FSR 4 runtime.

## Repository layout

```text
crates/       Rust workspace packages with short directory names
native/       C ABI and optional native adapter build boundary
shaders/      Motion, temporal guidance, reconstruction, and shared shader sources
assets/       Vulkan layer manifests
tests/        Deterministic fixtures and capture replay inputs
docs/         Architecture and development documentation
xtask/        Workspace validation commands
```

## Development

See [docs/development.md](docs/development.md) for toolchain setup and validation commands. Run `cargo xtask gpu-check` for GPU motion/capture tests and `cargo xtask gpu-check --backend fsr_3_1_4` for the FidelityFX adapter, lifecycle, captured-sequence quality, and output-quality checks. Run `cargo xtask smoke --backend fsr_3_1_4` for the multi-swapchain WSI test and `cargo xtask fidelityfx-check` for the prebuilt companion's ELF, ABI-symbol, and loader checks. The runtime logs per-phase GPU medians and p95 values after warm-up when timestamp queries are available.

The packaged Linux companion library lives under `lib/`. To replace it with
another local build:

```bash
cp /path/to/libtuxscaling_fidelityfx_vk.so lib/
```

Alternatively set `TUXSCALING_FIDELITYFX_LIBRARY` to the companion path. See
[docs/fidelityfx.md](docs/fidelityfx.md) for packaging and optional maintainer
rebuild instructions.
