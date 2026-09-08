# Third-party notices

## AMD FidelityFX SDK 1.1.4

The optional Linux companion library is based on AMD's official FidelityFX SDK
1.1.4. The SDK source is not vendored in this repository; normal users only
need the packaged `lib/libtuxscaling_fidelityfx_vk.so`.

- Repository: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git
- Tag: `v1.1.4`
- Pinned commit: `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`
- License: MIT, as provided by the upstream `LICENSE.txt` and the component
  license text in `sdk/LICENSE.txt`.
- Included components in the companion: the FSR 3.1.4 Super Resolution
  component, its Vulkan backend and API support, and the generated Vulkan
  shader permutations used by the native companion library.

Maintainers rebuilding the companion must obtain the exact upstream source at
the pinned commit and review its complete MIT license inventory before
packaging. See `docs/fidelityfx.md` for the external-source rebuild flow.

The Linux companion library is a TuxScaling-owned C ABI shim linked against
the selected SDK components. It is not the Windows DLL shipped in upstream
release packages; Linux deployments must install the resulting
`libtuxscaling_fidelityfx_vk.so` beside the Vulkan layer or set
`TUXSCALING_FIDELITYFX_LIBRARY`.

The project intentionally pins SDK 1.1.4 / FSR 3.1.4 for this milestone. A
newer SDK such as 1.1.5 requires a separate source, shader, ABI, and license
audit before it can replace the pinned build.

The OptiScaler source was not used, copied, or translated.
