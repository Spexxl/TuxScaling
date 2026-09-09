# Task 1 Report: Display Policy and Negotiation Control Plane

## Status

DONE_WITH_CONCERNS

## Implementation

- Replaced monitor selection by largest intersection with half-open window-center containment, preserving negative monitor origins through signed coordinate arithmetic.
- Kept the existing Vulkan-independent `display` crate as the only production-code change.
- Preserved the existing `BorderlessLease` snapshot, fullscreen ownership, geometry request, observation, and restoration behavior.
- Added a generic public presentation negotiation API with private detailed stages: `Direct`, `RequestingBorderless`, `WaitingForNativeExtent`, `RecreatingOutput`, `Active`, and `Failed`.
- Exposed the required public states: `Direct`, `Negotiating`, `Virtualized`, and `Failed`.
- Enforced a five-second `Instant`-based deadline, exact target `Rect` geometry, and exact downstream fixed extent or accepting extent range before output recreation can become active.
- Kept the first failure reason immutable for a negotiation lifetime and returned active output to negotiation when the exact native condition is lost.

## Commit

- `7e2af9b914671dd76670ea062a903b1325f7991f feat: add display negotiation policy`

## Test-Driven Development Evidence

The first red run added the new control-plane tests before implementation:

```text
$ cargo test -p tuxscaling-display
error[E0432]: unresolved imports `super::Extent`, `super::NegotiationFailure`,
`super::PresentationNegotiation`, `super::PresentationState`, `super::SurfaceExtent`
error: could not compile `tuxscaling-display` (lib test)
```

The first green run passed all display tests:

```text
running 11 tests
test result: ok. 11 passed; 0 failed; 0 ignored
```

The second red run covered loss of exact native output after activation:

```text
test tests::active_negotiation_returns_to_waiting_when_native_extent_is_lost ... FAILED
assertion `left == right` failed
  left: Virtualized
 right: Negotiating
```

The second green run passed after the minimum transition was added:

```text
running 12 tests
test result: ok. 12 passed; 0 failed; 0 ignored
```

## Final Verification

```text
$ cargo fmt --all -- --check
exit 0

$ cargo clippy -p tuxscaling-display --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo]

$ cargo clippy --workspace --all-targets --all-features -- -D warnings
Finished `dev` profile [unoptimized + debuginfo]

$ cargo test --workspace --all-targets --all-features
display: 12 passed; 0 failed
workspace: all executed tests passed; GPU- and desktop-dependent tests were ignored as declared
```

## Concerns

- The control plane is intentionally not wired into layer hooks, runtime, overlay, or xtask; that integration belongs to the next task and was excluded by this task's scope.
- No live X11 or XWayland compositor session was available for a real borderless-promotion test. The X11 desktop-dependent test remains ignored by the workspace test command.

## Fix Round 1

### Status

DONE_WITH_CONCERNS

### Reviewer findings addressed

- `RequestingBorderless` now starts the single five-second monotonic deadline, and that same deadline is carried through `WaitingForNativeExtent` and `RecreatingOutput`. Both an unacknowledged request and stalled output recreation fail with `DeadlineExpired`.
- `observe` now receives the observed EWMH fullscreen state and requires it alongside exact X11 geometry and accepted downstream extent before entering `RecreatingOutput`.
- `RecreatingOutput` now revalidates geometry, fullscreen, surface extent acceptance, and the unchanged deadline on every observation and immediately before `Active`. A failed revalidation returns to `WaitingForNativeExtent`; it cannot publish stale dimensions.
- The existing first-failure-wins behavior remains stable.
- No Vulkan dependency or out-of-scope crate was added.

### TDD red verification

Tests were changed first to require the corrected API and behaviors. Before the production fix, the focused command failed because the old implementation lacked the deadline-aware request, fullscreen observation, and final revalidation signatures:

```text
$ cargo test -p tuxscaling-display
error[E0061]: this method takes 1 argument but 2 arguments were supplied
error[E0061]: this method takes 3 arguments but 4 arguments were supplied
error[E0061]: this method takes 0 arguments but 4 arguments were supplied
error: could not compile `tuxscaling-display` (lib test) due to 30 previous errors
```

### Covering tests and green output

The fix round added or expanded these focused tests:

- `negotiation_times_out_while_borderless_request_is_unacknowledged`
- `negotiation_times_out_during_output_recreation`
- `negotiation_rejects_exact_geometry_with_rejected_surface_extent`
- `negotiation_accepts_a_surface_extent_range_containing_the_target`
- `negotiation_requires_ewmh_fullscreen_before_recreation`
- `negotiation_revalidates_geometry_fullscreen_and_extent_during_recreation`
- `output_recreation_does_not_activate_with_a_stale_surface_extent`

```text
$ cargo test -p tuxscaling-display
running 19 tests
test result: ok. 19 passed; 0 failed; 0 ignored
```

Final verification after the code fix:

```text
$ cargo fmt --all -- --check
exit 0

$ cargo clippy --workspace --all-targets --all-features -- -D warnings
Finished `dev` profile [unoptimized + debuginfo]

$ cargo test --workspace --all-targets --all-features
display: 19 passed; 0 failed
workspace: all executed tests passed; GPU- and desktop-dependent tests were ignored as declared

$ git diff --check
exit 0
```

### Fix commits

- `fae7296d57a919945138a2be1d4315f68ccdde54 fix: harden display negotiation gates`
- `6960a7e830240d14045bd99b380536e8c722f65d docs: record display policy task`

### Fix-round concerns

- The state machine remains intentionally unintegrated with layer hooks, runtime, overlay, and xtask until the next task.
- No live X11 or XWayland compositor validation was available; the deterministic tests cover the policy, but not compositor behavior.
