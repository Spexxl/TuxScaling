# Progress

## Recovery 1: Swapchain maintenance virtualization

- Status: implementation and display-backed acceptance complete; final host and finite-vkcube gates are being recorded in the current review cycle.
- Scope: `VK_EXT_swapchain_maintenance1` revision 1 and binary-compatible KHR aliases, owned create metadata, present fences and modes, release-without-present translation, logical swapchain recreation, and retained physical generations.
- Acceptance evidence: the maintenance WSI harness now exposes a logical `1280x720` contract on its first create, publishes physical `3440x1440` generations after exact X11/Vulkan confirmation, exercises grouped and individual presents, release/reacquire, overlay submission, Reference reconstruction, and FSR 3.1.4 dispatch.
- Safety boundary: unsupported create chains remain direct before logical publication; unsupported data after publication returns an explicit error; native publication uses a marked two-phase lifecycle transaction without holding the swapchain mutex across downstream Vulkan, runtime, or window-system calls.
- Commits: `07f1fa9`, `9ad4ef0`, `ac54c46`, `854f9be`, `dee4310`, and `e9a58b5`.

## Review cycle: Recovery 1 maintenance acceptance

- Current gate set: formatter, diff check, workspace build/test/clippy with all targets and features, FidelityFX checks, RADV GPU validation, Reference and FSR X11/XWayland smoke, and a normal-exit `vkcube --wsi xcb --width 1280 --height 720 --c 120` regression.
- Required evidence: no `virtual=0` fallback, no maintenance fallback, no Vulkan validation error, stable logical handles/images/extents across recreation and native-generation replacement, and no egui task started before the gate closes.
- Remote policy: work remains on `main`; no push or GitHub mutation is authorized.
