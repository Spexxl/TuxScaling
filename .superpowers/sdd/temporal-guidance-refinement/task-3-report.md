# Task 3 report — Dense optical flow and calibrated confidence

## Implementation

- Kept the coarse-to-fine pyramid and bidirectional flow buffers private to the motion estimator.
- Changed published `R16G16_SFLOAT` motion and `R8_UNORM` confidence images to the full processing extent, including odd extents.
- Added a dense resolve compute pass. It scales level-0 sub-pixel vectors to source pixels and performs a luminance edge-guided 3x3 weighted median resolve.
- Extended the existing flow cost with Charbonnier-robust luminance and gradient matching while preserving the existing quality presets, search budgets, and half-pixel refinement.
- Reworked confidence to combine photometric residual, search ambiguity, forward/backward consistency, bounds validity, and local spatial stability. Invalid history produces exact zero motion and confidence, and non-finite intermediate values are sanitized.
- Updated temporal guidance, reconstruction, scene statistics, and visualization consumers to sample processing-extent guidance directly. GPU query indexes and the Task 2 persistent statistics/reduction layout remain unchanged.
- Extended ignored RADV GPU fixtures for processing extent, all four presets, odd dimensions, translation, affine rotation/zoom, independent-object confidence, uniform frames, invalid history, mean EPE, p95 EPE, and AUROC.

## TDD RED/GREEN evidence

The first focused fixture was added before changing production allocation:

```text
$ cargo test -p tuxscaling-motion --test gpu dense_outputs_match_processing_extent -- --ignored --nocapture
assertion `left == right` failed
left: Extent2D { width: 33, height: 25 }
right: Extent2D { width: 129, height: 97 }
```

That RED failure demonstrated that the old `ceil(extent/4)` publication was still being observed. After the extent allocation, dense resolve, confidence pass, and consumer updates:

```text
$ cargo test -p tuxscaling-motion --test gpu dense_outputs_match_processing_extent -- --ignored --nocapture
test dense_outputs_match_processing_extent ... ok
```

The expanded GPU fixture suite then passed:

```text
$ cargo xtask gpu-check
Ultra dense translation EPE=0.0000 p95=0.0000
High dense translation EPE=0.0938 p95=0.0000
Balanced dense translation EPE=0.7340 p95=0.0000
Performance dense translation EPE=0.3601 p95=0.0000
Ultra independent object confidence=0.2256 background=0.6176
Ultra occlusion AUROC=1.0000
High independent object confidence=0.1662 background=0.6176
High occlusion AUROC=1.0000
Balanced independent object confidence=0.2366 background=0.6176
Balanced occlusion AUROC=1.0000
Performance independent object confidence=0.2527 background=0.6095
Performance occlusion AUROC=1.0000
128x96 displacement=(0,0) EPE=0.0000 cut=0
128x96 displacement=(8,0) EPE=0.0000 cut=0
128x96 displacement=(-8,4) EPE=0.2889 cut=0
129x97 displacement=(16,-8) EPE=0.0000 cut=0
occlusion confidence: inside=0.0000, outside=0.4757
Ultra affine EPE=0.4117 p95=0.7128
High affine EPE=0.4067 p95=0.6941
Balanced affine EPE=0.4067 p95=0.6941
Performance affine EPE=0.4465 p95=0.7782
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Verification commands and output

```text
$ cargo fmt --all -- --check
$ git diff --check
```

Both completed with exit status 0.

```text
$ cargo xtask check
Finished `dev` profile
workspace tests: all passed; repository-marked GPU/X11 tests remained ignored
workspace Clippy/check: finished successfully
```

The command exited 0 after compiling the changed shaders/crates and running all workspace tests and doc tests.

```text
$ cargo clippy -p tuxscaling-motion -p tuxscaling-temporal -p tuxscaling-upscaler --all-targets -- -D warnings
Finished `dev` profile
```

The command exited 0 with no warnings.

Validation-enabled Xlib smoke, using the isolated `target/debug/libtuxscaling_layer.so` and `assets/vulkan-layer/tuxscaling.json`, was attempted with the requested display credentials:

```text
$ timeout 10s env DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.KZT6U3 \
  VK_ADD_LAYER_PATH="$PWD/assets/vulkan-layer" \
  LD_LIBRARY_PATH="$PWD/target/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_INSTANCE_LAYERS=VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation \
  VK_LAYER_VALIDATE_SYNC=1 DISABLE_MANGOHUD=1 DISABLE_LSFG=1 \
  vkcube --wsi xlib
timeout: the monitored command dumped core
```

The Xlib smoke exited 139 in `vkcube` before establishing a usable window-system run. A matching XCB attempt reported `Cannot connect to XCB`; the validation-enabled `cargo xtask gpu-check` path passed without validation errors.

## Files changed

- `crates/motion/build.rs`
- `crates/motion/src/gpu.rs`
- `crates/motion/tests/gpu.rs`
- `crates/upscaler/src/reference.rs`
- `shaders/motion/confidence.comp`
- `shaders/motion/dense_resolve.comp`
- `shaders/motion/flow.comp`
- `shaders/motion/scene.comp`
- `shaders/motion/visualize.comp`
- `shaders/temporal/guidance.comp`
- `shaders/upscaler/reconstruct.comp`

## Self-review

- The coarse grid is not published: only full processing-extent images are bound to temporal/upscaler consumers.
- Existing four `MotionQuality` variants and their profile values are unchanged.
- The dense pass runs before the existing confidence timestamp, so the established query indexes and Task 2 statistics partial/reduction passes are preserved.
- All shader image accesses clamp odd-edge coordinates; robust losses, bounded confidence factors, and explicit finite checks avoid NaN/Inf propagation.
- Invalid-history tests read back every dense pixel and require exact `[0, 0]` motion and `0.0` confidence.
- The fixture assertions independently calculate EPE/p95 and confidence AUROC from generated labels; no production output is reused as a test oracle.

## Concerns

- The validation-enabled Xlib `vkcube` smoke could not complete because this environment's Xlib path crashes in `vkcube`; XCB/Wayland connection paths were unavailable. The dedicated validation GPU test suite passed.
- The reported EPE/AUROC values are deterministic fixture measurements on the available RADV GPU, not a claim for every vendor or scene distribution. Broader occlusion datasets should be used before shipping acceptance claims.

## Commit

`feat: refine dense optical flow guidance` (final commit hash is reported by `git log -1`)

## Fix round 1 — race-free confidence source and private coarse state

### Findings addressed

- Added a persistent full-resolution `dense_input` storage image. The dense resolve pass writes both the published `motion_image` and this source; confidence declares `dense_motion_input` as `readonly` and reads its 3x3 neighbourhood while writing only the published output. This removes the prior same-dispatch neighbourhood read/write race.
- Shifted the statistics descriptor from binding 9 to binding 10 and added the dense source at binding 9 without changing the existing query indexes or Task 2 partial/reduction algorithm.
- Made `MotionEstimator::levels`, `luma`, `forward`, and `backward` private. Published dense vectors/confidence, metadata, statistics, visualization, and configured cut thresholds remain available to their existing consumers.
- Added a source-path regression test and a repeated odd-extent GPU output stability test. Existing odd-extent finiteness, independent-object AUROC, invalid-history zeroing, and all-preset EPE tests remain enabled.

### TDD RED/GREEN evidence

The new regression test was introduced before the source/descriptors/shader fix:

```text
$ cargo test -p tuxscaling-motion --test gpu confidence_reads_a_separate_dense_motion_source -- --nocapture
test confidence_reads_a_separate_dense_motion_source ... FAILED
assertion failed: shader.contains("dense_motion_input")
test result: FAILED. 0 passed; 1 failed
```

After adding the separate source and read-only confidence declaration:

```text
$ cargo test -p tuxscaling-motion --test gpu confidence_reads_a_separate_dense_motion_source -- --nocapture
test confidence_reads_a_separate_dense_motion_source ... ok
test result: ok. 1 passed; 0 failed
```

The GPU stability regression passed with identical repeated outputs:

```text
$ cargo test -p tuxscaling-motion --test gpu dense_confidence_output_is_stable_across_repeated_dispatches -- --ignored --nocapture --test-threads=1
repeated dense output deltas: motion=0.000000, confidence=0.000000
test dense_confidence_output_is_stable_across_repeated_dispatches ... ok
test result: ok. 1 passed; 0 failed
```

### Fix-round verification

```text
$ cargo xtask check
...
test result: ok. 1 passed; 0 failed; 0 ignored; 8 ignored
...
Finished `dev` profile
```

The workspace check exited 0; all normal workspace tests/doc tests and Clippy/check completed, with the expected repository-marked GPU/X11 tests ignored.

```text
$ cargo xtask gpu-check
...
repeated dense output deltas: motion=0.000000, confidence=0.000000
...
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; 8 ignored
```

The validation-enabled capture and motion GPU suites exited 0. All 8 ignored motion GPU fixtures executed successfully and no validation errors were reported.

```text
$ timeout 10s env DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.KZT6U3 \
  VK_ADD_LAYER_PATH="$PWD/assets/vulkan-layer" \
  LD_LIBRARY_PATH="$PWD/target/debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  VK_INSTANCE_LAYERS=VK_LAYER_TUXSCALING_overlay:VK_LAYER_KHRONOS_validation \
  VK_LAYER_VALIDATE_SYNC=1 DISABLE_MANGOHUD=1 DISABLE_LSFG=1 \
  vkcube --wsi xlib
timeout: the monitored command dumped core
```

The requested validation-enabled Xlib smoke exited 139 in `vkcube` before a usable window-system run, even with the isolated `target/debug` layer and provided XAUTHORITY.

### Fix-round self-review

- `confidence.comp` has no `imageLoad(motion_image, ...)`; its motion and neighbourhood reads come from a separately bound `readonly` image that is fully populated before the confidence dispatch.
- The dense source and published output are both initialized from `UNDEFINED` to `GENERAL`; the dense resolve dispatch is followed by the existing memory barrier before confidence starts.
- Binding 10 is used consistently by `common.glsl`, `scene_reduce.comp`, and the Rust descriptor writes. The timestamp/query sequence and statistics dispatch order are unchanged.
- Coarse pyramid levels, luminance buffers, and bidirectional flow buffers are no longer public fields. Consumer compilation was rechecked by the workspace check.
- The repeated output test runs two complete odd-extent GPU pairs and compares every dense vector/confidence value, while the static regression prevents accidentally reintroducing the old source path.

### Fix-round concerns

- The repeated-dispatch regression cannot prove all possible GPU scheduling races by itself, but the confidence shader now has a separate descriptor declared `readonly` and no longer reads the image it writes. Validation-enabled GPU fixtures pass with exact repeated output equality.
- The validation-enabled Xlib `vkcube` smoke remains blocked by the environment's Xlib crash; this is unchanged from the initial Task 3 verification.
