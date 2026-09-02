# TuxScaling Initial Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a compiling English-only Rust workspace for TuxScaling with the approved crate boundaries, initial dependencies, Vulkan layer manifest, native adapter boundary, shader folders, and baseline tests.

**Architecture:** Cargo is the top-level build. The workspace contains short directory names with namespaced Cargo package names. The first layer is a pass-through Vulkan implicit layer; runtime processing, upscaling, and overlay rendering are represented by typed interfaces but are not claimed as implemented in this milestone.

**Tech Stack:** Rust stable 2024 edition, `ash` 0.38, `egui` 0.33, `serde` 1, `toml` 0.9, `thiserror` 2, `anyhow` 1, `tracing` 0.1, `tracing-subscriber` 0.3, `clap` 4, `bitflags` 2, `bytemuck` 1, `parking_lot` 0.12, `proptest` 1, CMake 3.20+, SPIR-V shader assets.

**Spec:** `docs/superpowers/specs/2026-09-02-initial-architecture-design.md`

## Global Constraints

- Linux-only; target native Vulkan applications and Proton applications after Vulkan translation.
- First validation target is AMD Radeon RX 9060 XT with Mesa RADV.
- All repository artifacts, identifiers, UI strings, logs, and documentation are English.
- Inputs are synthetic; no native DLSS/FSR/XeSS input interception is added.
- No Windows, DirectX, OpenGL, frame generation, standalone GUI, or required FSR 4 runtime support.
- `ash` is used directly; do not add `wgpu`, `vulkano`, `eframe`, or `winit`.
- Native code is isolated under `native/` and communicates through a C ABI.
- The injected path has no async runtime and must fail open when optional processing is unavailable.

---

### Task 1: Create the Cargo workspace and toolchain policy

**Files:**
- Create: `Cargo.toml`
- Create: `rust-toolchain.toml`
- Create: `.cargo/config.toml`
- Create: `.gitignore`

**Interfaces:**
- Produces a workspace that includes every directory listed in Task 2 and centralizes dependency versions under `[workspace.dependencies]`.

- [ ] **Step 1: Write the workspace manifest**

Create a virtual workspace with `resolver = "3"`, `edition = "2024"`, all crate members, and these centralized dependencies:

```toml
[workspace]
resolver = "3"
members = ["crates/*", "xtask"]

[workspace.package]
edition = "2024"
license = "MIT OR Apache-2.0"
rust-version = "1.88"

[workspace.dependencies]
ash = "0.38"
egui = "0.33"
libloading = "0.8"
serde = { version = "1", features = ["derive"] }
toml = "0.9"
thiserror = "2"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }
clap = { version = "4", features = ["derive"] }
bitflags = "2"
bytemuck = { version = "1", features = ["derive"] }
parking_lot = "0.12"
proptest = "1"
```

- [ ] **Step 2: Pin the stable toolchain**

Set `rust-toolchain.toml` to the stable channel with `rustfmt` and `clippy` components. Do not pin a nightly toolchain.

- [ ] **Step 3: Add Cargo defaults**

Configure `.cargo/config.toml` to use Cargo's incremental compilation for development and keep linker behavior platform-default. Do not add target-specific flags before a smoke build demonstrates the need.

- [ ] **Step 4: Add repository ignores**

Ignore `target/`, generated SPIR-V files, local logs, and IDE metadata while preserving checked-in shader source and layer manifests.

- [ ] **Step 5: Verify the workspace shell**

Run: `cargo metadata --no-deps --format-version 1`

Expected: Cargo parses the workspace after Task 2 directories exist; before Task 2, a missing-member error is acceptable and must be re-run after Task 2.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml rust-toolchain.toml .cargo/config.toml .gitignore
git commit -m "build: initialize Rust workspace policy"
```

### Task 2: Create the Rust crate boundaries

**Files:**
- Create: `crates/{runtime,vulkan,layer,capture,motion,temporal,upscaler,overlay,overlay-vulkan,input,config,cli}/Cargo.toml`
- Create: `crates/{runtime,vulkan,layer,capture,motion,temporal,upscaler,overlay,overlay-vulkan,input,config,cli}/src/lib.rs`
- Create: `crates/layer/src/lib.rs`
- Create: `crates/cli/src/main.rs`

**Interfaces:**
- `tuxscaling-config::Config` parses validated TOML.
- `tuxscaling-upscaler::UpscalerBackend` defines capability discovery without vendor types.
- `tuxscaling-overlay::OverlayState` owns egui state but no Vulkan handles.
- `tuxscaling-runtime::Runtime` exposes a pass-through-safe lifecycle shell.

- [ ] **Step 1: Add one manifest per crate**

Use namespaced package names with short directory paths. Add only dependencies required by each boundary: `ash` to `vulkan`, `layer`, and `overlay-vulkan`; `egui` to `overlay`; `serde`/`toml` to `config`; `clap`/`anyhow` to `cli`; `tracing` to `runtime`, `layer`, and `cli`; and internal path dependencies only where the interface requires them.

- [ ] **Step 2: Add compiling library shells**

Each `lib.rs` must contain a module-level English documentation comment and a public marker function:

```rust
/// Returns the crate name used by workspace diagnostics.
pub const CRATE_NAME: &str = "tuxscaling-runtime";
```

Use the matching package name in every crate. Do not add empty private modules or speculative implementation code.

- [ ] **Step 3: Define the neutral upscaler contract**

In `crates/upscaler/src/lib.rs`, define serializable-free neutral types for `BackendId`, `BackendCapabilities`, `InputResolution`, and a trait with these exact methods:

```rust
pub trait UpscalerBackend {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn resize(&mut self, input: InputResolution, output: InputResolution) -> Result<(), BackendError>;
    fn reset(&mut self) -> Result<(), BackendError>;
}
```

Include `BackendError` with `Unavailable`, `InvalidConfiguration`, and `Internal` variants using `thiserror`.

- [ ] **Step 4: Define configuration and overlay state shells**

`config` must expose a `Config` struct with `enabled: bool`, `quality: String`, and `toggle_key: String`, plus `Config::parse(&str) -> Result<Self, ConfigError>`. `overlay` must expose `OverlayState { visible: bool }` and `OverlayState::new()`. The overlay crate may construct an `egui::Context`, but it must not create a window or renderer.

- [ ] **Step 5: Define the runtime lifecycle shell**

`runtime` must expose `RuntimeState` with `Disabled`, `Ready`, and `Bypassed` variants and `Runtime::new() -> Self`. `Runtime::process_present()` must return a decision enum that currently always selects pass-through and documents why processing is not enabled in this milestone.

- [ ] **Step 6: Add initial tests before implementation completion**

Test `Config::parse` for defaults and invalid TOML, `OverlayState::new` for hidden state, and `Runtime::process_present` for pass-through. Add a compile-time test that a dummy backend implements `UpscalerBackend` without vendor dependencies.

- [ ] **Step 7: Verify and commit**

Run: `cargo fmt --all -- --check && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`

Expected: all crates compile and tests pass with no warnings.

```bash
git add crates
git commit -m "feat: add Rust crate boundaries"
```

### Task 3: Add native, shader, layer-manifest, and test fixture boundaries

**Files:**
- Create: `native/CMakeLists.txt`
- Create: `native/include/backend.h`
- Create: `shaders/{motion,temporal,common}/.gitkeep`
- Create: `assets/vulkan-layer/tuxscaling.json`
- Create: `tests/fixtures/README.md`
- Create: `tests/captures/.gitkeep`
- Create: `xtask/Cargo.toml`
- Create: `xtask/src/main.rs`

**Interfaces:**
- `native/include/backend.h` defines versioned C ABI declarations only.
- `assets/vulkan-layer/tuxscaling.json` is a development manifest pointing at the built layer library.
- `xtask` provides `cargo xtask check` as the single local validation entry point.

- [ ] **Step 1: Write the C ABI header**

Define `TUXSCALING_BACKEND_ABI_VERSION 1`, opaque context and resource handles, a capability query function, and a destroy function. Use fixed-width integer types, explicit ownership comments, and no C++ constructs.

- [ ] **Step 2: Write the native CMake shell**

Create a CMake project that declares an `INTERFACE` target for the ABI include directory and does not build a vendor SDK yet. The default configure must succeed without proprietary SDKs.

- [ ] **Step 3: Write the Vulkan layer manifest**

Create a manifest with the explicit layer name `VK_LAYER_TUXSCALING_overlay`, the development library path `libtuxscaling_layer.so`, `VK_LAYER_API_VERSION`, and the disable environment variable `TUXSCALING_DISABLE`. Keep the JSON valid and use English metadata.

- [ ] **Step 4: Add shader and capture documentation**

Use English README files to state that shader sources and deterministic frame fixtures will be added by later milestones. Do not add generated binaries or fake captures.

- [ ] **Step 5: Add `xtask check`**

Implement `cargo xtask check` to run `cargo fmt --all -- --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings` in order, returning the first non-zero status.

- [ ] **Step 6: Verify and commit**

Run: `cargo xtask check` and `cmake -S native -B target/native-configure`

Expected: both commands succeed without requiring an SDK or Vulkan device.

```bash
git add native shaders assets tests xtask
git commit -m "build: add native and runtime asset boundaries"
```

### Task 4: Validate the complete scaffold and document developer commands

**Files:**
- Modify: `README.md`
- Create: `docs/development.md`

**Interfaces:**
- Documentation defines the exact commands for building, testing, validating the layer manifest, and configuring RADV diagnostics.

- [ ] **Step 1: Document the project status**

Rewrite `README.md` in English with the project purpose, explicit non-goals, current milestone status, and the approved directory map. State clearly that the scaffold is pass-through and does not yet perform temporal upscaling.

- [ ] **Step 2: Document Linux/RADV setup**

In `docs/development.md`, document required tools (`rustup`, stable Rust, Cargo, CMake, a C compiler, Vulkan loader/headers, Mesa RADV), `VK_LAYER_PATH` usage for local testing, and `RUST_LOG` usage. Do not instruct users to install proprietary SDK binaries.

- [ ] **Step 3: Run final verification**

Run:

```bash
cargo xtask check
cmake -S native -B target/native-configure
python3 -m json.tool assets/vulkan-layer/tuxscaling.json >/dev/null
git diff --check HEAD~4..HEAD
git status --short --branch
```

Expected: all checks pass, the working tree is clean, and the branch contains only the scaffold commits plus the already-approved design and plan documents.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/development.md
git commit -m "docs: describe initial development workflow"
```

## Plan Self-Review

- The plan covers every first-milestone deliverable in the approved architecture: workspace policy, crate boundaries, typed interfaces, pass-through layer shell, configuration, logging dependencies, overlay state model, native ABI, assets, shader folders, tests, and developer documentation.
- No vendor-specific public directory is introduced; FSR/DLSS/XeSS implementation is deferred behind `upscaler` and the native ABI.
- The plan intentionally does not claim working capture, optical flow, temporal reconstruction, overlay rendering, or vendor integration.
- All identifiers, paths, documentation, and user-visible text specified by the plan are English.
