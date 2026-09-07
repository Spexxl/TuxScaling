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
