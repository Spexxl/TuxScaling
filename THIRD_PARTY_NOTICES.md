# Third-party notices

## AMD FidelityFX SDK 1.1.4

TuxScaling includes the official AMD FidelityFX SDK as a Git submodule at
`third_party/fidelityfx-sdk`.

- Repository: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
- Tag: `v1.1.4`
- Pinned commit: `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`
- License: MIT, as provided by the upstream `LICENSE.txt` and the component
  license text in `sdk/LICENSE.txt`.
- Included components: the FSR 3.1.4 Super Resolution component, its Vulkan
  backend and API support, and the generated Vulkan shader permutations used
  by the native companion library.

The complete upstream license text remains in the submodule. Verify the
checkout with `scripts/fidelityfx/verify-source.sh` before building.

The OptiScaler source was not used, copied, or translated.
