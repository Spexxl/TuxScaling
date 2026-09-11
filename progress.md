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
- The title visibly rendered its menu in a real XWayland window and reached
  repeated logical `1280x720` virtual swapchains. It then destroyed those
  contracts and issued a final `vkCreateSwapchainKHR` with `oldSwapchain=0`
  and requested `3440x1440`, producing `virtual=0` without a native-generation
  publication or an upscale dispatch. The required egui interaction was not
  observed.
- This is a failed acceptance gate, not a production-code excuse for a game
  or executable special case. No egui production task was started.
- The temporary Steam launch configuration was restored byte-for-byte and no
  remote push was performed.

## Local commits

The implementation remains on `main` and preserves the existing local
history. The compatibility commits are `aea0619`, `74f0803`, `eb1acc5`,
`3b1adaf`, `0b2213d`, `37039f2`, `df5932f`, `9ca5c4f`, `aa4678d`, `15f5f5b`,
and `704aa1b`. The documentation/report commit is created only after its
checks are verified.

## Boundary

The plan is not globally complete: Task 10 remains unaccepted and the
display-timing scenario remains environment-unverified. The next valid step
is to fix or re-run those gates with evidence; no remote publication is
authorized.
