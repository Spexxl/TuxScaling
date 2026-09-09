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
