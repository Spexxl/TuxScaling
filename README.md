# TuxScaling

TuxScaling is a Linux-only Vulkan layer for experimenting with temporal upscaling in games that do not provide a native temporal-upscaling integration.

The project targets native Vulkan applications and Windows games translated to Vulkan through Proton. The first hardware validation target is an AMD Radeon RX 9060 XT with Mesa RADV.

## Current status

The repository captures supported swapchains, estimates optical flow and temporal guidance on the GPU, downsamples the captured frame, runs a vendor-neutral reference reconstruction with ping-pong history, and renders an egui diagnostic overlay. Vendor SDK adapters and production FSR/DLSS/XeSS backends remain future work.

## Design boundaries

- Rust owns the runtime, Vulkan layer, frame pipeline, synthetic temporal inputs, configuration, diagnostics, and egui interface.
- Native C or C++ code is isolated behind a small C ABI for optional SDK adapters.
- Inputs are always estimated from captured color frames; game-native temporal inputs are not intercepted.
- Guidance includes source-pixel current-to-previous motion, confidence, temporal reactive and disocclusion masks, log-luminance exposure, and a flat depth fallback. Jitter is always zero.
- The default `Balanced` preset is fixed for the session. `Ultra`, `High`, and `Performance` trade optical-flow work for precision; the overlay reports capture, flow, guidance, reconstruction, and overlay timings.
- The reference backend uses a configurable 50–100% internal scale (default 67%), neighborhood clamping, confidence-weighted accumulation, and history reset on first frame, long pause, presentation failure, resize, or preset change.
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

See [docs/development.md](docs/development.md) for toolchain setup and validation commands. Run `cargo xtask gpu-check` for GPU motion/capture tests and `cargo xtask smoke` for the multi-swapchain WSI test. The runtime logs per-phase GPU medians and p95 values after warm-up when timestamp queries are available.
