# Progress

## Recovery 1: Vulkan swapchain compatibility

- Status: Tasks 1-9 are implemented locally. The portable WSI gate is partially verified; `display_timing` remains explicitly unverified because the current RADV device does not expose `VK_GOOGLE_display_timing`. Tasks 10 and 11 have not started.
- Scope: semantic WSI compatibility decisions, owned swapchain contracts, equivalent logical images, stable logical tokens, physical generations, generation-aware synchronization and metadata routing, and portable X11/XWayland acceptance scenarios.
- Verified locally: unrelated extensions remain compatible; mutable-format views, present-ID waits across generations, HDR metadata replacement, and audited direct fallback through `DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR` pass under Vulkan validation with FSR 3.1.4.
- Unverified locally: the display-timing scenario reports `result=unverified reason=extension_unavailable` because `VK_GOOGLE_display_timing` is absent on the selected device. This is an environment limitation, not a successful gate.
- Commits: `aea0619`, `74f0803`, `eb1acc5`, `3b1adaf`, `0b2213d`, `37039f2`, and `df5932f` implement the compatibility layers; `9ca5c4f` adds the complete-decision parser tests; the current Task 9 commit adds the portable WSI scenarios and runtime evidence gate.

## Review cycle: Recovery 1 portable WSI acceptance

- RED: `cargo test -p xtask --bin xtask wsi_compatibility -- --nocapture` initially failed because the evidence validator was absent; the focused direct-reason test also failed before the audited-reason parser was implemented.
- GREEN: `cargo test -p xtask --bin xtask wsi_compatibility -- --nocapture`, `cargo test -p tuxscaling-layer --lib` (114 tests), formatting, diff checks, and strict layer Clippy pass.
- Display-backed result: `mutable_format`, `present_wait_generation`, `hdr_replacement`, and `incompatible_direct` passed with no validation errors. `display_timing` was executed and returned the required unverified result for the missing driver extension.
- Boundary: the real WSI acceptance is not closed while any required scenario is unverified; the egui/Proton task must remain unopened until a compatible display-timing environment is available.
- Remote policy: work remains on `main`; no push or remote mutation was performed.
