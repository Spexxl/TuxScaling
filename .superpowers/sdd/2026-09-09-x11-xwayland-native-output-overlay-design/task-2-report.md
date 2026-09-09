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
