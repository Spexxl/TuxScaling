# Progress

## Recovery 1: Vulkan swapchain compatibility

- Status: Tasks 1-9 are implemented locally. The portable WSI gate is partially verified; `display_timing` remains explicitly unverified because the current RADV device does not expose `VK_GOOGLE_display_timing`. Tasks 10 and 11 have not started.
- Scope: semantic WSI compatibility decisions, owned swapchain contracts, equivalent logical images, stable logical tokens, physical generations, generation-aware synchronization and metadata routing, and portable X11/XWayland acceptance scenarios.
- Verified locally: unrelated extensions remain compatible; mutable-format views, present-ID waits across generations, HDR metadata replacement, and audited direct fallback through `DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR` pass under Vulkan validation with FSR 3.1.4. The native-generation failure path now preserves the previous physical generation and runtime when recovery is possible, and destroys only temporary resources.
- Unverified locally: the display-timing scenario reports `result=unverified reason=extension_unavailable` because `VK_GOOGLE_display_timing` is absent on the selected device. This is an environment limitation, not a successful gate.
- Commits: `aea0619`, `74f0803`, `eb1acc5`, `3b1adaf`, `0b2213d`, `37039f2`, and `df5932f` implement the compatibility layers; `9ca5c4f` adds the complete-decision parser tests; `aa4678d` adds the portable WSI scenarios and runtime evidence gate; `15f5f5b` removes workspace warnings; the latest corrective commit preserves the old generation during recoverable native-publication failures.

## Review cycle: Recovery 1 portable WSI acceptance

- RED: `cargo test -p xtask --bin xtask wsi_compatibility -- --nocapture` initially failed because the evidence validator was absent; the focused direct-reason test also failed before the audited-reason parser was implemented.
- GREEN: `cargo test -p xtask --bin xtask wsi_compatibility -- --nocapture`, `cargo test --workspace`, `cargo build --workspace`, `cargo clippy --workspace --lib -- -D warnings`, `cargo xtask fidelityfx-check`, `cargo xtask gpu-check --backend fsr_3_1_4`, formatting, and diff checks pass.
- Display-backed result: `mutable_format`, `present_wait_generation`, `hdr_replacement`, and `incompatible_direct` passed with no validation errors. `display_timing` was executed and returned the required unverified result for the missing driver extension. The exact X11/XWayland `vkcube --wsi xcb --width 1280 --height 720` gate passed its acceptance evidence for 20 seconds after `393c3e1`: logical `1280x720`, physical `3440x1440`, `virtual=1`, FSR 3.1.4 dispatch, reconstructed present, and no validation error or post-publication recreation; exit `124` is the expected timeout.
- Boundary: the real WSI acceptance is not closed while any required scenario is unverified; the egui/Proton task must remain unopened until a compatible display-timing environment is available.
- Remote policy: work remains on `main`; no push or remote mutation was performed.
