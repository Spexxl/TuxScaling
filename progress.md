# Progress

## Plan inventory

The Vulkan swapchain compatibility plan contains 11 tasks. Tasks 1-9 are
implemented locally. Task 10 was executed against the real Steam/Proton path
but is not accepted, and Task 11 documentation is committed while the final
audit remains open until every required gate passes.

## Vulkan swapchain compatibility

- Implemented: semantic WSI extension classification, owned swapchain create
  chains, equivalent mutable-format logical images, stable logical tokens,
  physical generations, present-ID and wait routing, HDR and timing query
  translation, maintenance aliases, direct fail-open decisions, and portable
  X11/XWayland WSI scenarios.
- Implemented: two-phase native-generation publication. The lifecycle marker
  blocks Acquire, Present, application recreation, and Destroy while the
  runtime is detached; downstream Vulkan, image queries, runtime
  reconfiguration, and rollback run without a layer swapchain mutex. A
  publication commits only against the captured logical token, surface, and
  generation.
- Verified: workspace tests, all-target/all-feature build and Clippy,
  FidelityFX checks, FSR 3.1.4 GPU checks, formatting, and diff checks.
- Verified: display-backed `mutable_format`, `present_wait_generation`,
  `hdr_replacement`, and `incompatible_direct` scenarios under validation.
- Environment-unverified: `display_timing` reports
  `result=unverified reason=extension_unavailable` because the selected RADV
  device does not expose the required display-timing/display-control support.
- Verified: the exact X11/XWayland vkcube gate on Mutter with logical
  `1280x720`, physical `3440x1440`, `virtual=1`, FSR 3.1.4 dispatch,
  reconstructed present, no validation errors, and no post-publication
  recreation.

## Review cycle: Proton/DXVK acceptance

- The real title was launched through Steam and GE-Proton 11-6 with the
  validation layer, native output, guidance scale 1, ultra motion quality,
  FSR 3.1.4, MangoHud disabled, and LSFG disabled.
- The first session only reached repeated logical `1280x720` virtual
  swapchains followed by a direct contract, without a native-generation
  publication or an upscale dispatch, and without egui interaction.
- Root causes found with surface-identity evidence (no game-specific
  branches): Wine chains host Xlib surface creation through the layer (the
  Win32 wrapper is bypassed); Wine reparents game windows so configure
  coordinates must be parent-relative; Wine-owned windows never retain an
  externally requested EWMH fullscreen flag, so exact monitor geometry now
  counts as native; Unity tracks the HWND size and adopts a promoted window,
  so native requests go direct with the lease kept (a later smaller request
  promotes idempotently) and stale logical overrides are dropped whenever
  virtualization is abandoned with no live virtual swapchain.
- The follow-up session published a native generation (`logical=1280x720`
  `physical=2160x1440`), dispatched FSR 3.1.4, presented reconstructed
  output, translated present waits across the generation, submitted the
  overlay, and kept `virtual=1` with zero validation errors, VUIDs, or
  panics. The user visibly confirmed the game rendering, opened the
  TuxScaling egui with `Insert`, and confirmed the session stable
  fullscreen after the application adopted the promoted window.
- No remote push was performed.

## Local commits

The implementation remains on `main` and preserves the existing local
history. The compatibility commits are `aea0619`, `74f0803`, `eb1acc5`,
`3b1adaf`, `0b2213d`, `37039f2`, `df5932f`, `9ca5c4f`, `aa4678d`, `15f5f5b`,
and `704aa1b`. The Proton follow-ups are `97783f1` (Win32 surface mapping),
`127dc84`, `1652963`, `0c1ff59` (surface-identity diagnostics), `c1f14fe`
(parent-relative configure), `b166086` (geometry-native without EWMH),
`893e8e8` (superseded follow-restore experiment), and `2a85592`
(adopt promoted windows tracked by the application).

## Boundary

The `display_timing` WSI scenario remains environment-unverified on the
selected RADV device (missing display-timing/display-control support); all
other portable scenarios pass. Validation was performed on AMD Radeon
RX 9060 XT / RADV / Mesa with a Mutter XWayland session; portability to
other drivers rests on synthetic contracts, not on claims. No remote
publication is authorized.
