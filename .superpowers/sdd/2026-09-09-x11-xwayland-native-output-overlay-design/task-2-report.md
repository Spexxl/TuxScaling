# Task 2: Layer swapchain virtualization report

## Scope delivered

The Vulkan layer now separates the application-visible logical swapchain
identity from the downstream physical `VkSwapchainKHR` for every virtualized
swapchain. The direct path remains unchanged: it returns the driver's handle
and uses no logical-image mapping.

## Stable logical identity

Virtualized swapchains receive a layer-generated non-dispatchable-handle token
instead of the driver handle. The state map is keyed by that token and retains
the current physical handle separately. `vkGetSwapchainImagesKHR` returns the
layer-owned game-extent images for the token. `vkAcquireNextImageKHR` and
`vkAcquireNextImage2KHR` translate the token to the physical handle, then map
the acquired physical index to a free logical slot before returning it to the
application. `vkQueuePresentKHR` resolves that logical slot back to the
physical index and replaces the downstream swapchain/index arrays. It releases
the logical slot after the downstream present invocation. `vkDestroySwapchainKHR`
and `oldSwapchain` similarly translate the logical token to the current
physical handle. Consequently, a retired logical token is never forwarded to
the driver.

The runtime frame entry points now accept separate logical and physical indices.
The logical index selects the game image; the physical index selects output
images, command slots, fences, queries, and presentation state. This supports
logical and physical image counts that differ without changing runtime resource
generation construction, which remains Task 3 work.

## Logical-handle allocation invariant

Vulkan non-dispatchable handles are opaque 64-bit values, so the layer uses a
tagged token namespace (`0x8000_0000_0000_0000 | counter`) only at its
interception boundary; it never passes such a token to an ICD. Allocation
checks all currently registered logical and physical handles. If the initial
downstream physical handle occupies the tagged namespace, or if the allocator
cannot find a collision-free token, virtualization is declined and the normal
direct driver handle is returned unchanged.

This is a layer allocation invariant, not a Vulkan guarantee that an ICD will
never emit a value with the tag. Any future physical-generation replacement
must perform the same collision check before publication; if it observes a
collision it must retain the valid current physical generation or fail open to
direct presentation. Task 3 owns that publication path. The Task 2 paths that
exist today never forward the virtual token downstream.

## Mapping and recovery behavior

`Mapping` is a pure generation-scoped ownership module. It assigns only idle
logical slots, rejects duplicate acquired physical indices, resolves the
physical index at present, releases the slot, and refuses a generation change
until all slots are idle. Virtualization preflight/allocation failure preserves
the direct fallback. Existing virtual-swapchain eligibility checks, capture
fallback, damage-region mapping, and feature-chain behavior remain in place.

Task 1's Vulkan-free `PresentationNegotiation` API is preserved. Its exact
X11 rectangle, observed fullscreen, accepted downstream extent, and
`output_recreated` activation gate are covered from the layer's mapping tests;
intermediate extents do not activate its public virtualized state.

## Test-first evidence

The first targeted run for the new mapping module failed because
`AcquireError`, `Mapping`, and `OldSwapchain` did not yet exist. The added
logical-handle test likewise failed before `LogicalSwapchainHandle` existed.
The production mapping and tagged-token implementation were then added.

Fresh checks after implementation:

- `cargo fmt --all -- --check`
- `cargo test -p tuxscaling-layer` (18 tests)
- `cargo test -p tuxscaling-runtime` (19 tests)
- `cargo check -p tuxscaling-layer`
- `git diff --check`

## Remaining concern

No real X11/Xwayland WSI run was available in this environment, so driver-side
handle translation and compositor behavior are unit/compile validated only.
Physical-generation replacement and its runtime-resource reconstruction are
explicitly deferred to Task 3; its implementation must invoke the already
provided idle-generation invalidation and tagged-handle collision policy.

## Round 1 review fixes

The review findings were addressed in the layer integration and runtime
adapter:

1. Creation now performs the Task 1 `PresentationNegotiation` gate after
   borderless promotion. It observes the exact native X11 `Rect`, EWMH
   fullscreen state, and downstream fixed/range `SurfaceExtent`, then calls
   `output_recreated` only when all requirements are accepted. The gate is
   non-blocking and failure restores the borderless lease and keeps direct
   swapchain creation; `vkCreateSwapchainKHR` is never blocked waiting for the
   five-second deadline.
2. Logical image allocation uses the requested logical count (falling back to
   the physical count only when the request is zero). Runtime validation now
   requires both vectors to be non-empty, while logical source selection and
   physical output/resource selection use their respective indices and counts.
3. Retired tagged logical tokens are recorded explicitly and return
   `ERROR_OUT_OF_DATE_KHR` from acquire/present/oldSwapchain translation. They
   are never treated as raw downstream handles. Destroy translates only active
   virtual state to its physical handle, then retires the logical token.
4. Focused hook-level tests cover acquire-to-present translation with distinct
   logical/physical indices and counts, `SUBOPTIMAL_KHR` and
   `ERROR_OUT_OF_DATE_KHR`, retired-token rejection, exact negotiation gating,
   retired oldSwapchain rejection, and independent creation counts.

### Fix-cycle red evidence

Tests were added before their corresponding implementation and were run to
capture the expected failures. The creation count test initially failed to
compile:

```text
$ cargo test -p tuxscaling-layer hooks::creation::tests::creation_keeps_logical_image_count_independent_from_physical_count -- --exact
error[E0432]: unresolved import `super::logical_image_count`
    --> crates/layer/src/hooks/creation.rs:1137:32
     |
1137 |         confirm_native_output, logical_image_count, translate_old_swapchain,
     |                                ^^^^^^^^^^^^^^^^^^^ no `logical_image_count` in `hooks::creation`
error: could not compile `tuxscaling-layer` (lib test) due to 1 previous error
```

The runtime independence test initially failed at the old equal-count guard:

```text
$ cargo test -p tuxscaling-runtime present::tests::accepts_independent_logical_and_physical_image_counts -- --exact
test present::tests::accepts_independent_logical_and_physical_image_counts ... FAILED
assertion failed: swapchain.is_valid()
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 20 filtered out
```

The gate and retired-token tests also first failed against missing integration
helpers, including:

```text
error[E0432]: unresolved import `super::confirm_native_output`
    --> crates/layer/src/hooks/creation.rs:1040:17
     |
1040 |     use super::{confirm_native_output, virtual_swapchain_supported};
     |                 ^^^^^^^^^^^^^^^^^^^^^ no `confirm_native_output` in `hooks::creation`

error[E0425]: cannot find function `reject_retired_swapchain` in this scope
    --> crates/layer/src/hooks/acquire.rs:250:20
```

### Fix-cycle green evidence

After the minimal implementation, the focused tests passed:

```text
$ cargo test -p tuxscaling-layer hooks::creation::tests::creation_keeps_logical_image_count_independent_from_physical_count -- --exact
test hooks::creation::tests::creation_keeps_logical_image_count_independent_from_physical_count ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-layer hooks::creation::tests::layer_gate_requires_exact_observed_native_output_before_recreation -- --exact
test hooks::creation::tests::layer_gate_requires_exact_observed_native_output_before_recreation ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-layer hooks::creation::tests::retired_old_swapchain_is_rejected_instead_of_forwarded -- --exact
test hooks::creation::tests::retired_old_swapchain_is_rejected_instead_of_forwarded ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-layer hooks::acquire::tests::hook_acquire_present_round_trip_supports_distinct_indices_and_statuses -- --exact
test hooks::acquire::tests::hook_acquire_present_round_trip_supports_distinct_indices_and_statuses ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-layer hooks::acquire::tests::retired_logical_tokens_return_a_safe_error_without_driver_fallback -- --exact
test hooks::acquire::tests::retired_logical_tokens_return_a_safe_error_without_driver_fallback ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-layer hooks::presentation::tests::present_hook_rejects_a_retired_logical_token_without_translation_fallback -- --exact
test hooks::presentation::tests::present_hook_rejects_a_retired_logical_token_without_translation_fallback ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 24 filtered out

$ cargo test -p tuxscaling-runtime present::tests::accepts_independent_logical_and_physical_image_counts -- --exact
test present::tests::accepts_independent_logical_and_physical_image_counts ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 20 filtered out

$ cargo test -p tuxscaling-runtime present::tests::selects_source_by_logical_index_and_output_by_physical_index -- --exact
test present::tests::selects_source_by_logical_index_and_output_by_physical_index ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 20 filtered out
```

The complete focused package runs passed with 25 layer tests and 21 runtime
tests. Formatting, workspace tests, Clippy with `-D warnings`, and
`git diff --check` were also run before commit. A real X11/Xwayland WSI run
remains unavailable in this environment.

## Round 2 scoped review fixes

This cycle keeps the Task 2 boundary and does not begin Task 3.

- `PresentationNegotiation` is carried through surface and swapchain state
  instead of being only a local policy value. The five-second monotonic
  deadline remains owned by Task 1. Creation performs a non-blocking
  request/observation preflight; `output_recreated` is called only after the
  physical downstream swapchain has been created. Failed or expired
  negotiation restores the lease and falls back directly. This report makes
  no claim that the create-time observation is asynchronous.
- `vkReleaseSwapchainImagesEXT` and `vkGetSwapchainStatusKHR` are enumerated
  in both device-proc dispatch paths and return
  `ERROR_EXTENSION_NOT_PRESENT` from safe layer stubs. Unknown tagged logical
  handles are rejected as `ERROR_OUT_OF_DATE_KHR` in acquire, present, and
  old-swapchain translation, so they cannot reach downstream as raw tokens.
- `controlled_downstream_double_receives_rewritten_present_array` acquires a
  physical index, rewrites the present pair to the physical swapchain/index,
  and releases the logical slot. The evidence is scoped to this controlled
  translation boundary, not a live ICD call.

### Round 2 verification

```text
$ cargo fmt --all
$ cargo test -p tuxscaling-layer --lib
test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured
$ cargo test -p tuxscaling-runtime --lib
test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured
$ git diff --check
```

No live X11/Xwayland WSI validation was run. The first commit attempt was
blocked because `.git/index.lock` could not be created: `.git` is read-only in
this execution context.

## Async negotiation completion

The prior synchronous create-time activation has been removed. On the first
eligible `vkCreateSwapchainKHR`, the layer requests borderless mode, stores
the `PresentationNegotiation` and lease on the surface, and returns the
application's direct swapchain unchanged. It does not query native geometry
to activate virtualization and does not wait or sleep.

Every `vkQueuePresentKHR` now observes each presented surface at the frame
boundary using the current X11 geometry, EWMH fullscreen state, and fresh
downstream `SurfaceExtent`. A mismatch leaves the persistent negotiation in
`Negotiating`; an exact later observation moves it to
`RecreatingOutput`. The next application swapchain recreation is the
recreation boundary: the layer revalidates the observation, creates the
native physical swapchain, calls `output_recreated`, and only then publishes
`Virtualized`. Any mismatch, unavailable observation, or monotonic five-second
deadline expiry fails open and keeps direct presentation.

Deterministic evidence now covers the first mismatch, later exact observation,
recreation revalidation, the ready predicate, and deadline expiry to
`Failed/Direct` behavior. The extension rejection handlers remain present for
`vkReleaseSwapchainImagesEXT` and `vkGetSwapchainStatusKHR`.
