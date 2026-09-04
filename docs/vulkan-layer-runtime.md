# Vulkan Layer Runtime

## Purpose

The Vulkan layer injects work immediately before presentation while preserving the application's Vulkan device, queues, and swapchains. The layer must fail open: if TuxScaling cannot process a present, the original call is forwarded unchanged.

The validated runtime captures supported SDR swapchains, computes estimated optical flow, and renders the Egui diagnostic panel over `vkcube` on RADV. It remains experimental until frame reconstruction and broader format coverage are added.

## Ownership

`layer` owns loader negotiation, Vulkan dispatch forwarding, handle registries, and exported ABI functions. It does not own Egui render resources.

`overlay-vulkan` owns overlay swapchain resources:

- image views and framebuffers;
- capture and motion resources for supported SDR formats;
- render pass and Egui renderer;
- one frame slot per swapchain image;
- command buffers, fences, and render-complete semaphores;
- resource creation, recreation, and destruction.

`overlay` remains Vulkan-independent. It produces Egui paint primitives and texture deltas only.

## Present synchronization

For an intercepted present, the layer receives the application's wait semaphores and the selected swapchain image index.

1. Select the frame slot for that image.
2. Reuse the slot only after its previous fence is complete.
3. Record capture, motion, and overlay commands for the selected image.
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

Swapchain recreation creates a new overlay state. Old state is released through the normal destroy path and never reused for a new format or extent.

## Failure and ABI policy

Unknown queues, unsupported swapchain formats, allocation failures, renderer failures, and invalid present data must bypass processing and preserve downstream behavior.

Every exported Vulkan function catches Rust panics at the ABI boundary. A panic returns a Vulkan error only when forwarding safely is impossible; otherwise the original downstream command is called.

## Validation

The runtime milestone is complete only when all checks pass:

- `cargo test --workspace`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `vkcube` renders an Egui panel with the layer enabled;
- validation layers report no synchronization or lifetime errors during create, resize, present, and destruction;
- the WSI harness exercises two swapchains, grouped presents, resize, and resource destruction without global queue-idle stalls;
- GPU tests cover known motion, scene cuts, occlusion confidence, and capture isolation.

## Scope boundary

This milestone only makes the injection runtime safe and maintainable. Frame capture, optical flow, temporal history, and upscaling remain separate milestones.
