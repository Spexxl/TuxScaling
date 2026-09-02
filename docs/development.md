# Development Guide

## Requirements

- Linux x86_64.
- Rust stable with Rust 1.88 or newer, including `rustfmt` and `clippy`.
- Cargo.
- CMake 3.20 or newer and a C compiler.
- Vulkan loader and Vulkan headers.
- Mesa RADV for the primary validation target.

On Fedora, the system packages normally used for local development are `cmake`, `gcc`, `vulkan-loader`, `vulkan-headers`, `mesa-vulkan-drivers`, and `pkgconf-pkg-config`. The exact package names may vary by Fedora release.

## Build and test

From the repository root:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo xtask check
```

`cargo xtask check` is the preferred single command. It runs formatting checks, workspace tests, and Clippy with warnings treated as errors.

## Native boundary

The native boundary is intentionally SDK-free in the initial milestone:

```bash
cmake -S native -B target/native-configure
```

Vendor runtimes must not be copied into the repository. Future adapters will load optional libraries at runtime and communicate through the versioned header in `native/include/backend.h`.

## Local Vulkan layer inspection

The development manifest is located at `assets/vulkan-layer/tuxscaling.json`. After building a shared layer, set `VK_LAYER_PATH` to a directory containing the manifest and the matching `libtuxscaling_layer.so`, then launch a controlled Vulkan test application.

Use `VK_LOADER_DEBUG=all` to inspect loader and layer discovery. Use `RUST_LOG=info` or `RUST_LOG=debug` for structured runtime diagnostics once the layer logging path is active.

The initial layer is pass-through, so it should not be enabled for production games until capture and synchronization milestones are complete.

## Language and naming policy

All source code, identifiers, comments, documentation, configuration keys, logs, errors, tests, UI text, and commit messages are written in English. Directory names remain short; Cargo package names use the `tuxscaling-` namespace.
