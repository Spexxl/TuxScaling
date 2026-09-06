# Architecture

TuxScaling is a Linux-only Vulkan implicit layer for temporal upscaling in games without native temporal-upscaling integration. The first validation target is an AMD Radeon RX 9060 XT with Mesa RADV. Native Linux Vulkan applications and Proton applications are in scope; native Windows and DirectX hooks are not.

## Runtime flow

```text
Vulkan present
  -> color capture
  -> optical-flow estimation
  -> guidance validation and estimated masks/exposure/depth
  -> low-resolution simulation
  -> reference temporal reconstruction
  -> egui overlay
  -> downstream presentation
```

Inputs are always estimated from captured color frames. TuxScaling does not intercept native DLSS, FSR, or XeSS calls and does not consume game-provided motion vectors, depth, exposure, camera matrices, or UI masks.

## Workspace boundaries

- `runtime`: per-swapchain orchestration and fail-open decisions.
- `vulkan`: `ash`-based handles, dispatch, synchronization, and resource helpers.
- `layer`: Vulkan loader negotiation and present interception.
- `capture`: internal color copies and format validation.
- `motion`: GPU optical-flow estimation and confidence.
- `temporal`: history, reset detection, disocclusion, and synthetic inputs.
- `upscaler`: vendor-neutral backend contract and runtime registry.
- `overlay`: `egui` state and widgets without graphics dependencies.
- `overlay-vulkan`: custom `egui` renderer over `ash`.
- `input`: X11/XWayland input translation; native Wayland is configuration-only for now.
- `config`: validated versioned TOML profiles.
- `cli`: diagnostics and benchmarks outside the injected library.

## Native boundary

Rust owns the project. Optional SDK adapters live under `native/` and expose a versioned C ABI. C++ types, exceptions, allocation ownership, and standard-library types never cross that boundary. Optional backends are loaded dynamically and unavailable backends must not prevent pass-through presentation.

## Overlay boundary

The overlay uses `egui` directly. It does not use `eframe`, `winit`, or `wgpu`, because TuxScaling does not own the application window or create a separate graphics device. The overlay is drawn after reconstruction so its pixels remain sharp and do not enter temporal history.

## Failure policy

Unsupported formats, unavailable backends, allocation failures, swapchain changes, and internal initialization failures disable processing for the affected swapchain and forward the original presentation whenever safe. Exported C ABI callbacks catch Rust panics, and unsafe code is isolated to Vulkan, FFI, mapped memory, and platform-input boundaries.

## Current milestone

The current workspace captures supported swapchains, computes estimated optical flow and guidance on RADV, tracks temporal history, optionally reduces processing resolution, runs a reference reconstruction, and injects an egui diagnostic panel with X11/XWayland input. Fullscreen X11/XWayland virtualization separates the game's logical swapchain images from monitor-sized output images; native Wayland remains direct. `vkcube` and the WSI harness are the validation targets for grouped presents, resize, and synchronization. Vendor SDK adapters and production FSR/DLSS/XeSS integrations remain separate milestones. See [Vulkan Layer Runtime](vulkan-layer-runtime.md) for its maintenance contract.
