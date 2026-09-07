# Vulkan Layer Runtime

## Purpose

The Vulkan layer injects work immediately before presentation while preserving the application's Vulkan device, queues, and swapchains. The layer must fail open: if TuxScaling cannot process a present, the original call is forwarded unchanged.

The runtime captures supported swapchains, computes estimated optical flow and frame guidance, runs the reference reconstruction when the format supports storage images, and renders the Egui diagnostic panel over `vkcube` on RADV. In fullscreen X11/XWayland virtual-output mode, game-owned layer images remain at the requested game extent while the real swapchain images use the configured output extent. Unsupported formats and allocation failures bypass reconstruction while preserving presentation. If temporal recording fails after virtualization has started, the runtime attempts a spatial bilinear blit into the real output before presenting.

The overlay displays `Virtual upscale`, `Native AA`, `Windowed 1:1`, or a direct fallback reason alongside game, processing, and output extents. It also reports the window mode and selected monitor, for example `Promoted borderless` and `1920x1080 at 0,0`. The logical surface capabilities saved before a resize are restored to the application, while internal layer queries continue to use downstream physical capabilities. Present IDs and Google present timing structures survive incremental-present rectangle remapping.

## Ownership

`layer` owns loader negotiation, Vulkan dispatch forwarding, handle registries, and exported ABI functions. It does not own Egui render resources.

`overlay-vulkan` owns overlay swapchain resources:

- image views and framebuffers;
- capture, previous-frame, motion, guidance, and reconstruction resources for supported formats;
- render pass and Egui renderer;
- one frame slot per swapchain image;
- command buffers, fences, and render-complete semaphores;
- resource creation, recreation, and destruction.

`overlay` remains Vulkan-independent. It produces Egui paint primitives and texture deltas only.

## Present synchronization

For an intercepted present, the layer receives the application's wait semaphores and the selected swapchain image index.

1. Select the frame slot for that image.
2. Reuse the slot only after its previous fence is complete.
3. Record capture, motion, guidance, optional reconstruction, and overlay commands for the selected image in one command buffer.
4. Submit commands waiting on the application's original semaphores.
5. Signal the slot's render-complete semaphore and fence.
6. Forward the present with only the render-complete semaphore as its wait semaphore.

The implementation must not call `vkQueueWaitIdle` per frame. A slot fence may be waited only when the matching slot is reused. A failed overlay record or submit forwards the original present data unchanged.

## Lifetime

Layer-owned objects must be destroyed before their downstream owner:

1. Destroy overlay resources before `vkDestroySwapchainKHR` reaches the next layer.
2. Remove queue and swapchain registry entries during destruction.
3. Destroy remaining device-owned resources before `vkDestroyDevice` reaches the next layer.
4. Remove instance state before `vkDestroyInstance` reaches the next layer.

Swapchain recreation creates a new overlay state. A successful replacement retains the surface's one `BorderlessLease`; failed creation, capability validation, virtual-image allocation, or runtime initialization restores it exactly once before direct fallback. Old state is released through the normal destroy path and never reused for a new format or extent.

## Failure and ABI policy

Unknown queues, unsupported swapchain formats, allocation failures, renderer failures, and invalid present data must bypass processing and preserve downstream behavior. A temporal failure on a virtual swapchain first uses the spatial fallback so the acquired game image is still visible in the physical output; only a fallback failure disables processing.

Every exported Vulkan function catches Rust panics at the ABI boundary. A panic returns a Vulkan error only when forwarding safely is impossible; otherwise the original downstream command is called.

## Validation

The runtime milestone is complete only when all checks pass:

- `cargo test --workspace`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `vkcube` renders an Egui panel with the layer enabled;
- validation layers report no synchronization or lifetime errors during create, resize, present, and destruction;
- the WSI harness exercises two swapchains, grouped presents, resize, and resource destruction without global queue-idle stalls;
- GPU tests cover known motion, scene cuts, occlusion confidence, capture isolation, per-signal mask fallbacks, relative-depth support state, and the deterministic quality fixtures. The acceptance benchmark separates guidance from reconstruction and requires p95 no greater than 135 percent of median for each reported total.

## Scope boundary

The remaining runtime milestone is broader WSI and format coverage. Vendor-specific backends, native Wayland interaction, and neural depth are intentionally separate milestones.
