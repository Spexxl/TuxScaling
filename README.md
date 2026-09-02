# TuxScaling

TuxScaling is a Linux-only Vulkan layer for experimenting with temporal upscaling in games that do not provide a native temporal-upscaling integration.

The project targets native Vulkan applications and Windows games translated to Vulkan through Proton. The first hardware validation target is an AMD Radeon RX 9060 XT with Mesa RADV.

## Current status

The repository currently contains the initial compiling scaffold. The Vulkan layer is pass-through, configuration and overlay state have typed shells, and the backend boundary is vendor-neutral. Temporal capture, optical flow, reconstruction, interactive overlay rendering, and vendor SDK adapters are not implemented yet.

## Design boundaries

- Rust owns the runtime, Vulkan layer, frame pipeline, synthetic temporal inputs, configuration, diagnostics, and egui interface.
- Native C or C++ code is isolated behind a small C ABI for optional SDK adapters.
- Inputs are always estimated from captured color frames; game-native temporal inputs are not intercepted.
- The injected runtime uses `ash` directly and does not use `wgpu`, `vulkano`, `eframe`, or `winit`.
- The initial milestone does not include Windows, DirectX, OpenGL, frame generation, a standalone GUI, or a required FSR 4 runtime.

## Repository layout

```text
crates/       Rust workspace packages with short directory names
native/       C ABI and optional native adapter build boundary
shaders/      Motion, temporal, and shared shader sources
assets/       Vulkan layer manifests
tests/        Deterministic fixtures and capture replay inputs
docs/         Architecture and development documentation
xtask/        Workspace validation commands
```

## Development

See [docs/development.md](docs/development.md) for toolchain setup and validation commands.
