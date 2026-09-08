#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/../.." && pwd -P)"
sdk_root="${TUXSCALING_FIDELITYFX_SDK:-$repo_root/third_party/fidelityfx-sdk}"
generated_root="$repo_root/crates/upscaler/native/fidelityfx/generated"
generated_vk="$generated_root/vk"
patch_file="$script_dir/patches/luma-history-rgba16f.patch"

"$script_dir/verify-source.sh" "$sdk_root"
test -d "$generated_vk"
test -f "$generated_root/SHA256SUMS"

passes=(
    ffx_fsr3upscaler_accumulate_pass
    ffx_fsr3upscaler_autogen_reactive_pass
    ffx_fsr3upscaler_debug_view_pass
    ffx_fsr3upscaler_luma_instability_pass
    ffx_fsr3upscaler_luma_pyramid_pass
    ffx_fsr3upscaler_prepare_inputs_pass
    ffx_fsr3upscaler_prepare_reactivity_pass
    ffx_fsr3upscaler_rcas_pass
    ffx_fsr3upscaler_shading_change_pass
    ffx_fsr3upscaler_shading_change_pyramid_pass
)
suffixes=(
    _permutations.h
    _wave64_permutations.h
    _16bit_permutations.h
    _wave64_16bit_permutations.h
)

mapfile -t headers < <(find "$generated_vk" -maxdepth 1 -type f -name '*.h' -printf '%f\n' | LC_ALL=C sort)
test "${#headers[@]}" -eq 40
test -z "$(find "$generated_vk" -maxdepth 1 -type f ! -name '*.h' -print -quit)"
test "$(wc -l < "$generated_root/SHA256SUMS")" -eq 40

for pass_name in "${passes[@]}"; do
    for suffix in "${suffixes[@]}"; do
        expected="${pass_name}${suffix}"
        printf '%s\n' "${headers[@]}" | grep -Fxq "$expected"
    done
done

(cd "$generated_root" && LC_ALL=C sha256sum --check SHA256SUMS)

check_root="$(mktemp -d -t tuxscaling-fidelityfx-shader-check.XXXXXX)"
trap 'rm -rf "$check_root"' EXIT
cp -a "$sdk_root/." "$check_root/sdk"
patch --quiet --directory="$check_root/sdk" --forward --strip=1 < "$patch_file"
grep -Fq 'rgba16f) uniform image2D  rw_luma_history' \
    "$check_root/sdk/sdk/include/FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h"
