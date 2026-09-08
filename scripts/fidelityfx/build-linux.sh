#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/../.." && pwd -P)"
output_dir="${1:?usage: build-linux.sh OUTPUT_DIR}"

if [[ "$#" -ne 1 ]]; then
    echo "usage: build-linux.sh OUTPUT_DIR" >&2
    exit 2
fi

sdk_root="${TUXSCALING_FIDELITYFX_SDK:?set TUXSCALING_FIDELITYFX_SDK to an external FidelityFX SDK checkout}"
generated_vk="$repo_root/crates/upscaler/native/fidelityfx/generated/vk"
patch_file="$script_dir/patches/linux-build.patch"

"$script_dir/verify-source.sh" "$sdk_root"
"$repo_root/scripts/fidelityfx/verify-vulkan-shaders.sh"
test -f "$patch_file"
test -d "$generated_vk"

cmake_bin="${CMAKE:-cmake}"
test -x "$(command -v "$cmake_bin")"

work_root="$(mktemp -d -t tuxscaling-fidelityfx-linux.XXXXXX)"
trap 'rm -rf "$work_root"' EXIT

sdk_copy="$work_root/fidelityfx-sdk"
build_dir="$work_root/build"
mkdir -p "$output_dir"
cp -a "$sdk_root/." "$sdk_copy"
patch --quiet --forward --strip=1 --directory="$sdk_copy" < "$patch_file"

cmake_args=(
    -S "$repo_root/crates/upscaler/native/fidelityfx"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DFFX_SDK_ROOT="$sdk_copy"
    -DFFX_GENERATED_SHADER_PATH="$generated_vk"
)

if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
fi

"$cmake_bin" "${cmake_args[@]}"
"$cmake_bin" --build "$build_dir" --parallel

library="$build_dir/output/libtuxscaling_fidelityfx_vk.so"
test -f "$library"
cp "$library" "$output_dir/libtuxscaling_fidelityfx_vk.so"
