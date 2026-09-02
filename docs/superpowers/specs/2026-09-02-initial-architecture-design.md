# TuxScaling Initial Architecture Design

**Date:** 2026-09-02
**Status:** Approved for implementation planning

## Purpose

TuxScaling is a Linux-only Vulkan layer that adds temporal upscaling to games that do not provide a native temporal-upscaling integration. It captures presented color frames, estimates the missing temporal inputs, dispatches a selectable upscaler, and presents the reconstructed image. It supports native Linux games and Windows games translated to Vulkan by Proton.

The first development and validation target is an AMD Radeon RX 9060 XT using Mesa RADV. The architecture remains vendor-neutral, but the first usable backend will follow a Vulkan-capable FSR path rather than making the project depend on the current Windows-only FSR 4 runtime.

## Product Boundaries

TuxScaling always uses synthetic inputs. It does not intercept native DLSS, FSR, or XeSS calls and does not consume game-provided motion vectors, depth, exposure, camera matrices, or UI masks.

The initial project includes:

- A Linux Vulkan implicit layer.
- Swapchain and presentation interception.
- Color-frame capture and GPU-side processing.
- Optical-flow motion estimation.
- Synthetic temporal inputs and history management.
- A vendor-neutral upscaler contract and backend registry.
- An in-game `egui` overlay rendered into the intercepted Vulkan output.
- X11 and Wayland input integration for overlay controls.
- TOML configuration, diagnostics, and benchmark tooling.

The initial project excludes:

- Native Windows support.
- DirectX hooks or Windows proxy DLLs.
- Native temporal-input extraction from games.
- Frame generation.
- A standalone desktop GUI.
- FSR 4 as a required backend.
- Shipping DLSS or XeSS proprietary runtime binaries.
- OpenGL support.

## Architectural Approach

The repository is a Cargo-first Rust workspace. Rust owns the Vulkan layer, runtime, frame pipeline, input estimation, overlay, configuration, diagnostics, and backend abstraction. C or C++ is allowed only inside isolated native adapters when a vendor SDK cannot be integrated through a practical C ABI.

Native adapters expose a small, versioned C ABI. C++ types, exceptions, allocation ownership, and standard-library types must not cross that boundary. The core runtime discovers backend capabilities at runtime and continues operating when an optional backend or vendor runtime is unavailable.

`ash` provides low-level Vulkan bindings. Higher-level graphics frameworks such as `wgpu` and `vulkano` are not used in the injected runtime because the layer must preserve direct control over handles, function dispatch, synchronization, image layouts, queue submission, and swapchain lifetime.

## Repository Structure

```text
TuxScaling/
├── Cargo.toml
├── rust-toolchain.toml
├── crates/
│   ├── runtime/
│   ├── vulkan/
│   ├── layer/
│   ├── capture/
│   ├── motion/
│   ├── temporal/
│   ├── upscaler/
│   ├── overlay/
│   ├── overlay-vulkan/
│   ├── input/
│   ├── config/
│   └── cli/
├── native/
│   ├── CMakeLists.txt
│   ├── include/
│   └── adapters/
├── shaders/
│   ├── motion/
│   ├── temporal/
│   └── common/
├── assets/
│   └── vulkan-layer/
├── tests/
│   ├── fixtures/
│   └── captures/
├── docs/
└── xtask/
```

Directory names remain short. Cargo package names use the `tuxscaling-` namespace, such as `tuxscaling-runtime`, to make dependency graphs and diagnostics unambiguous. No vendor-specific backend directory is part of the initial public repository structure. Vendor implementations remain internal to the neutral `upscaler` boundary until isolation into a separate package is technically justified.

All repository artifacts are written in English, including code, identifiers, comments, documentation, configuration keys, logs, errors, tests, UI text, commit messages, and release notes.

## Crate Responsibilities

### `runtime`

Owns per-swapchain runtime state and coordinates each frame. It consumes capture results, advances temporal history, requests motion estimation, selects an upscaler backend, schedules overlay rendering, and provides a fail-open path to the original presentation call.

### `vulkan`

Contains focused wrappers for borrowed and owned Vulkan resources, dispatch tables, command pools, descriptor management, barriers, timeline synchronization, and deferred destruction. It does not contain product policy or upscaling logic.

### `layer`

Exports the Vulkan layer ABI, negotiates with the loader, maintains instance and device dispatch chains, intercepts the required swapchain functions, and delegates processing to `runtime`. Exported callbacks contain panic boundaries so Rust unwinding never crosses the Vulkan C ABI.

### `capture`

Identifies presentable images, validates supported image properties, and schedules GPU-side copies or format conversions into internal images. It reports unsupported paths without changing application-owned resources permanently.

### `motion`

Produces dense screen-space motion estimates from consecutive color frames using compute shaders. It exposes confidence and validity information alongside the vector field so downstream stages can reject unreliable history.

### `temporal`

Owns color history, motion history, reset detection, disocclusion estimates, reactive masks, and synthetic exposure values. It converts captured frames and motion estimates into a canonical input frame for the upscaler API.

### `upscaler`

Defines backend capability queries, context creation, resize/reset behavior, frame dispatch, and failure reporting. The public contract uses neutral Rust types and borrowed Vulkan resource descriptions. Optional SDK bridges remain runtime-selectable and do not leak vendor types into callers.

### `overlay`

Owns `egui` state, screens, widgets, theme, and editable settings. It consumes immutable runtime snapshots and emits explicit configuration commands. It does not depend on Vulkan, windowing frameworks, or backend implementations.

### `overlay-vulkan`

Converts `egui` paint output into Vulkan vertex/index uploads, texture updates, scissor state, and draw commands using `ash`. It renders after temporal reconstruction so the overlay remains sharp and does not enter temporal history.

### `input`

Converts X11 and Wayland keyboard, pointer, text, focus, and scale information into `egui::RawInput`. With the overlay closed, it observes only configured toggle shortcuts. With the overlay open, it consumes input only when `egui` requests keyboard or pointer ownership.

### `config`

Defines versioned TOML schemas, defaults, validation, per-application profiles, and atomic persistence. Runtime code receives validated immutable snapshots rather than reading configuration files directly.

### `cli`

Provides layer discovery checks, configuration validation, capability inspection, log collection, and benchmark entry points. It is not part of the injected shared library.

## Frame Data Flow

For each eligible presentation:

1. `layer` resolves the device, queue, swapchain, image index, and downstream dispatch function.
2. `runtime` checks whether the swapchain is enabled, supported, initialized, and ready for processing.
3. `capture` transitions or copies the application image into an internal color resource without taking ownership of application objects.
4. `motion` compares the current captured image with the previous accepted image and produces motion plus confidence.
5. `temporal` detects resets and constructs canonical synthetic inputs, including validity masks and history metadata.
6. `upscaler` records reconstruction work into a project-owned command buffer and output image.
7. `overlay` builds the current `egui` frame, and `overlay-vulkan` draws it over the reconstructed output.
8. `runtime` submits the work with explicit synchronization and delegates the final call to the next `vkQueuePresentKHR` implementation.
9. Successfully consumed resources become history for the next frame. Failed or skipped frames invalidate history when reuse would be unsafe.

Swapchain creation, resize, format change, device loss, queue changes, discontinuous frame timing, and configuration changes trigger explicit lifecycle transitions. The runtime never assumes that a single process has only one device or one swapchain.

## Overlay Behavior

The initial overlay provides:

- Enable and bypass controls.
- Active backend and capability status.
- Input and output resolutions.
- Quality preset selection.
- Motion and history confidence diagnostics.
- Frame-time and stage-time metrics.
- Debug-view selection.
- Configuration reload and persistence status.

The overlay uses `egui` directly. It does not use `eframe`, `winit`, or `wgpu`, because TuxScaling neither owns the application window nor creates a separate rendering device. A project-owned Vulkan renderer integrates `egui` with the intercepted command stream.

## Dependency Policy

Workspace dependencies are declared centrally and pinned deliberately. The initial Rust dependency set is:

- `ash` for Vulkan bindings.
- `egui` for the immediate-mode overlay.
- `libloading` for optional runtime libraries.
- `serde` and `toml` for configuration.
- `thiserror` for library error types.
- `anyhow` for CLI and build tooling only.
- `tracing` and `tracing-subscriber` for structured diagnostics.
- `clap` for command-line parsing.
- `bitflags` for capability and state flags.
- `bytemuck` for validated plain-data GPU transfers.
- `parking_lot` for short, non-async synchronization where a lock is necessary.
- `proptest` as a development dependency for state-machine and configuration properties.

The injected path does not use an async runtime. Dependencies that spawn threads, install signal handlers, initialize a global logger, or assume ownership of the process event loop require explicit review before adoption.

Native adapters use CMake as isolated build islands. Cargo remains the top-level build and developer interface. Shader compilation is orchestrated by `xtask`, and release builds embed validated SPIR-V rather than compiling shaders inside a game process.

## Error Handling and Safety

The layer is fail-open whenever presentation can safely continue. Unsupported formats, unavailable backends, missing optional libraries, transient allocation failures, and internal initialization failures disable TuxScaling for the affected swapchain and forward the original presentation operation.

Device loss and invalid synchronization are propagated according to Vulkan semantics. TuxScaling does not conceal an error returned by the downstream driver. Errors are rate-limited per swapchain to prevent log storms inside games.

Every exported C ABI entry point catches Rust panics. Unsafe code is concentrated in Vulkan dispatch, FFI adapters, mapped GPU memory, and platform input boundaries. Each unsafe module documents the invariants its safe callers must uphold.

Backend failures are isolated. A backend cannot leave the shared runtime believing an output or history image is valid unless submission completed according to its declared synchronization contract.

## Testing and Verification

The initial scaffold must pass formatting, compilation, linting, unit tests, and documentation checks through one workspace command exposed by `xtask`.

Testing is divided into:

- Unit tests for configuration, capability negotiation, lifecycle transitions, history reset rules, and overlay commands.
- Property tests for generated swapchain events and malformed configuration values.
- Shader compilation and reflection checks.
- Headless Vulkan tests using a controlled test application when a compatible ICD is available.
- Capture-replay tests using repository fixtures with deterministic frame pairs.
- A minimal Vulkan smoke application that exercises swapchain creation, recreation, multiple present modes, resize, and clean shutdown under the layer.
- Manual validation on Mesa RADV with the RX 9060 XT, followed by Proton validation after the native Vulkan path is stable.

Performance measurements report GPU time for capture, motion, temporal preparation, upscaling, overlay, and total added frame latency. Correctness is evaluated with debug views for captured color, motion, confidence, disocclusion, and reconstructed output.

## Initial Implementation Milestone

The first milestone creates a compiling workspace and architectural skeleton. It includes dependency policy, crate boundaries, typed interfaces, a loadable pass-through Vulkan layer, configuration parsing, structured logging, a non-rendering overlay state model, build tooling, and basic tests.

The first milestone does not claim working temporal upscaling. GPU capture, optical flow, reconstruction, interactive overlay rendering, and vendor SDK integration are separate subsequent milestones built behind the approved interfaces.
