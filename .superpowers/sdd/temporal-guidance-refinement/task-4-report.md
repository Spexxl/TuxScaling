# Task 4 report — Reprojected masks, exposure, and scene detection

## Implementation

- Added dense processing-extent motion as a read-only guidance input and reprojected the previous color frame before residual evaluation.
- Implemented reactive scores from exposure-compensated log-luminance and chroma residuals, with finite-value fail-open handling.
- Implemented disocclusion from estimated forward occupancy, explicit holes, flow divergence, confidence as a separate cue, appearance change, and target bounds. It is no longer defined as `1-confidence`.
- Added a persistent R8 transparency history image. Composition scores combine reprojected persistent residual, local variance, alpha-like color changes, and disagreement with dominant motion; the current result is copied back to history in the same command buffer.
- Added clipped (5th–95th percentile) log-luminance exposure estimation over Task 2 compact statistics and temporal adaptation using `FrameTiming.smoothed` delta. Exposure remains a 1x1 `R32_SFLOAT`; `pre_exposure` remains `1.0` and `ConstantFallback` semantics are preserved.
- Hardened scene reduction by comparing histograms after a bounded EV shift and requiring low motion consistency for a cut. Existing configured scene thresholds and invalid-history fail-open behavior remain in place.
- Added explicit provider-failure recording. It writes zero reactive/disocclusion/composition, flat depth, exposure `1.0`, marks guidance resources as `ConstantFallback`, and exposes `GuidanceReset::ProviderFailure`.
- Preserved the existing statistics descriptors, timing query indexes, one command buffer, and one submission flow. Runtime now passes dense motion, compact statistics, and smoothed timing to guidance.

## TDD RED/GREEN evidence

RED was observed before implementing the producers:

```text
$ cargo test -p tuxscaling-temporal --test gpu guidance_shader_contains_reprojected_mask_and_exposure_producers -- --nocapture
test guidance_shader_contains_reprojected_mask_and_exposure_producers ... FAILED
guidance shader is missing dense_motion
```

After the descriptor/shader implementation, the same test passed:

```text
$ cargo test -p tuxscaling-temporal --test gpu guidance_shader_contains_reprojected_mask_and_exposure_producers -- --nocapture
test guidance_shader_contains_reprojected_mask_and_exposure_producers ... ok
test result: ok. 1 passed; 0 failed
```

The focused GPU fixtures were expanded to use independent labels and host readback only in tests. Validation-enabled output:

```text
$ env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 DISABLE_MANGOHUD=1 \
    cargo test -p tuxscaling-temporal --test gpu -- --ignored --nocapture --test-threads=1
guidance_disocclusion_uses_flow_holes_and_boundaries ...
disocclusion F1=0.933
reactive F1=0.959
guidance_masks_separate_transparency_from_a_static_background ...
transparency F1=0.806
normal_guidance_record_updates_the_exposure_image ... ok
test result: ok. 3 passed; 0 failed; 1 filtered out
```

Scene resistance and existing motion fixtures passed on RADV:

```text
$ cargo test -p tuxscaling-motion --test gpu bounded_flash_and_fade_are_not_scene_cuts -- --ignored --nocapture --test-threads=1
exposure multiplier=1.5 estimated=1.497 cut=0
exposure multiplier=0.65 estimated=3.392 cut=0
test result: ok. 1 passed; 0 failed
```

## Verification

```text
$ git diff --check
$ cargo fmt --all -- --check
```

Both completed successfully.

```text
$ cargo xtask check
```

Completed with exit code 0: workspace tests, doc tests, and workspace Clippy passed. GPU/X11 cases marked ignored remained ignored in this non-GPU command.

```text
$ cargo xtask gpu-check
```

Completed with exit code 0 and validation enabled. Capture passed; all 9 motion GPU fixtures passed, including dense extent, repeated confidence, pan, affine zoom, cut detection, occlusion confidence, exposure, and bounded flash/fade resistance. No validation errors were reported.

Validation-enabled `vkcube` was attempted with the currently discovered XAUTHORITY:

```text
$ find /run/user/1000 -maxdepth 1 -type f -name '*auth*' -printf '%p\\n'
/run/user/1000/.mutter-Xwaylandauth.CB5WU3
$ timeout 10s env DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.CB5WU3 \
    VK_ADD_LAYER_PATH="$PWD/assets/vulkan-layer" \
    LD_LIBRARY_PATH="$PWD/target/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_INSTANCE_LAYERS=VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation \
    VK_LAYER_VALIDATE_SYNC=1 DISABLE_MANGOHUD=1 DISABLE_LSFG=1 \
    vkcube --wsi xlib
timeout: the monitored command dumped core
```

The smoke exited 139 in `vkcube` before establishing a usable Xlib run; this is an environment/window-system limitation, not a repository validation error.

## Files changed

- `crates/motion/tests/gpu.rs`
- `crates/runtime/src/pipeline.rs`
- `crates/runtime/src/present.rs`
- `crates/temporal/src/gpu.rs`
- `crates/temporal/tests/gpu.rs`
- `shaders/motion/scene_reduce.comp`
- `shaders/temporal/guidance.comp`

## Self-review

- Descriptor formats and bindings were checked against the shader and validated on RADV; the dense motion source is `R16G16_SFLOAT`, transparency/composition is `R8_UNORM`, and exposure is `R32_SFLOAT` 1x1.
- The transparency history copy and all image layout transitions remain inside the existing guidance command recording. No queue idle, CPU readback, new Vulkan feature, or separate crate was introduced.
- Compact statistics and query indexes from Task 2 remain unchanged. The guidance exposure pass reads persistent records after the existing motion-to-guidance memory barrier.
- Invalid history and explicit provider failure produce coherent finite fallbacks. Non-finite motion, color, confidence, residual, and exposure intermediates are sanitized.
- Deterministic GPU fixture readbacks report disocclusion F1 `0.933`, reactive F1 `0.959`, and transparency F1 `0.806` on the available RADV device. These are fixture measurements, not universal hardware guarantees.

## Concerns

- The forward occupancy is estimated from the available dense current-to-previous field because native bidirectional flow is private to the motion provider; it is intentionally conservative around bounds and confidence holes.
- The clipped exposure estimate reconstructs percentile tails from compact 32-bin statistics while retaining exact compact log sums for sub-bin precision. This is bounded and finite, but it is still an approximation of per-pixel percentile sorting.
- Validation-enabled `vkcube --wsi xlib` remains blocked by the environment's `vkcube` crash. Dedicated validation GPU tests passed.

## Commit

`feat: estimate temporal masks and exposure`

## Fix round 1 — producer/fallback correctness

### RED evidence

The first post-review run exposed the confidence/occupancy coupling. With
confidence removed from occupancy but before the translated fixture and
explicit flow-hole setup were completed, the GPU regression reported no
detected disocclusion for the changed region:

```text
$ VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test -p tuxscaling-temporal --test gpu guidance_disocclusion_uses_flow_holes_and_boundaries -- --ignored --nocapture
disocclusion counts tp=0 fp=0 fn=256
disocclusion F1=0.000
thread 'guidance_disocclusion_uses_flow_holes_and_boundaries' panicked
assertion failed: score >= 0.65
test result: FAILED. 0 passed; 1 failed
```

The review also found that `record_provider_failure` had no runtime call
site, that the history copy source was `transparency`, and that guidance
normalized `metadata.consistent` using the full processing extent instead of
the level-0 sample count. These were source-level RED findings confirmed by
searching the runtime and shader call/binding sites before the fix.

### GREEN implementation and evidence

- `forward_occupancy` now uses only dense forward-flow coverage and projected
  bounds. Confidence remains an independent disocclusion cue. The translated
  flow-hole fixture injects a deterministic `[-5, 0]` current-to-previous
  field and uses generated boundary labels; it reports disocclusion F1 `0.998`
  (threshold `>= 0.75`). A complementary low-confidence/full-coverage fixture
  reports maximum disocclusion `41/255` and average `37.4/255`.
- Guidance owns a separate `transparency_residual` R8 image at binding 12.
  The shader writes the persistent residual there; the existing composition
  image remains the published composition, and only the residual is copied to
  the history sampler. The two-frame translated fixture reports:

  ```text
  translated history: first=0.691 second=0.362 background=0.008 exposure=8.070->7.810
  ```

- The metadata normalization denominator is the level-0 count
  `((width + 1) / 2) * ((height + 1) / 2)`, matching the motion pyramid. The
  source regression checks this expression and the preserved compact stats
  descriptor.
- Runtime spatial fallback now records `guidance.record_provider_failure` on
  its existing command buffer before the spatial blit. Invalid and provider
  failure views label all produced guidance masks/exposure as
  `ConstantFallback`. Provider fallback readback is zero for all R8 masks and
  exposure is exactly `1.000`.
- Invalid/provider failure clears `history_initialized`; successful provider
  attempts clear the failure latch before view construction. The reset fixture
  reports `first=2.030 fallback=1.000 fresh=2.030`, proving the next valid
  frame starts with a direct estimate rather than adapting from fallback.
- The two-frame fixture also covers exposure adaptation. All GPU readback is
  test-only; production keeps one command buffer/submission and no runtime
  readback.

Focused validation command and output:

```text
$ VK_LOADER_LAYERS_ENABLE='~implicit~' VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test -p tuxscaling-temporal --test gpu -- --ignored --nocapture
running 7 tests
low-confidence coverage disocclusion: max=41 average=37.4
provider reset exposure: first=2.030 fallback=1.000 fresh=2.030
translated history: first=0.691 second=0.362 background=0.008 exposure=8.070->7.810
disocclusion counts tp=239 fp=0 fn=1
disocclusion F1=0.998
reactive F1=1.000
test result: ok. 7 passed; 0 failed; 1 filtered out
```

The complete workspace checks passed:

```text
$ cargo xtask check
exit code 0; workspace tests, doc tests, and Clippy passed

$ cargo xtask gpu-check
exit code 0; capture passed and all 9 motion GPU fixtures passed
```

The required validation-enabled smoke was retried with the current XAUTHORITY
(`/run/user/1000/.mutter-Xwaylandauth.CB5WU3`) and again crashed inside the
environment's `vkcube` Xlib path:

```text
$ timeout 10s env DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.CB5WU3 VK_ADD_LAYER_PATH="$PWD/assets/vulkan-layer" LD_LIBRARY_PATH="$PWD/target/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" VK_INSTANCE_LAYERS=VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 DISABLE_MANGOHUD=1 DISABLE_LSFG=1 vkcube --wsi xlib
timeout: the monitored command dumped core
exit code 139
```

`cargo fmt --all -- --check` and `git diff --check` also passed.

### Files and self-review

Fix-round files:

- `crates/runtime/src/present.rs`
- `crates/temporal/src/gpu.rs`
- `crates/temporal/tests/gpu.rs`
- `shaders/temporal/guidance.comp`

The existing stats descriptors and query indexes are unchanged. Binding 12 is
an additional residual output; all image transitions and the residual-to-history
copy remain in the existing guidance recording flow. The fallback path writes
finite reactive/disocclusion/transparency/depth/exposure values before its
spatial output. No neural model, CPU readback, new Vulkan requirement, or
separate crate was added.

### Concerns

- The translated flow test supplies a deterministic dense-flow field so the
  mask producer is evaluated independently from motion-estimator quality; the
  runtime still consumes the real provider-owned dense field.
- Compact-stat percentile exposure remains a bounded 32-bin approximation,
  while two-frame adaptation and provider reset behavior are covered on RADV.
- `vkcube --wsi xlib` remains blocked by the reproducible host crash above;
  validation-enabled temporal GPU fixtures and `cargo xtask gpu-check` pass.

### Fix commit

`fix: harden temporal guidance producers` (this report is included in the fix commit)
