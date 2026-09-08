#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/../.." && pwd -P)"
sdk_root="${TUXSCALING_FIDELITYFX_SDK:-$repo_root/third_party/fidelityfx-sdk}"
generated_root="$repo_root/crates/upscaler/native/fidelityfx/generated"
generated_vk="$generated_root/vk"
patch_file="$script_dir/patches/luma-history-rgba16f.patch"

"$script_dir/verify-source.sh" "$sdk_root"

wine_bin="${WINE_BIN:-}"
if [[ -z "$wine_bin" ]]; then
    wine_bin="$(command -v wine 2>/dev/null || command -v wine64 2>/dev/null || true)"
fi
if [[ -z "$wine_bin" ]]; then
    echo "Wine is required to regenerate FidelityFX Vulkan shader headers" >&2
    exit 1
fi
test -x "$wine_bin"

shader_compiler="$sdk_root/sdk/tools/binary_store/FidelityFX_SC.exe"
glslang_validator="$sdk_root/sdk/tools/binary_store/glslangValidator.exe"
test -f "$shader_compiler"
test -f "$glslang_validator"
test -f "$patch_file"

work_root="$(mktemp -d -t tuxscaling-fidelityfx-shaders.XXXXXX)"
trap 'rm -rf "$work_root"' EXIT
sdk_copy="$work_root/fidelityfx-sdk"
cp -a "$sdk_root/." "$sdk_copy"
patch --quiet --directory="$sdk_copy" --forward --strip=1 < "$patch_file"

wine_path() {
    local winepath_bin="$(command -v winepath 2>/dev/null || true)"
    if [[ -n "$winepath_bin" && -x "$winepath_bin" ]]; then
        "$winepath_bin" -w "$1" | tr -d '\r\n'
    else
        printf 'Z:%s' "$1"
    fi
}

shader_compiler_win="$(wine_path "$sdk_copy/sdk/tools/binary_store/FidelityFX_SC.exe")"
glslang_validator_win="$(wine_path "$sdk_copy/sdk/tools/binary_store/glslangValidator.exe")"
output_win="$(wine_path "$work_root/output")"
include_gpu_win="$(wine_path "$sdk_copy/sdk/include/FidelityFX/gpu")"
include_fsr_win="$(wine_path "$sdk_copy/sdk/include/FidelityFX/gpu/fsr3upscaler")"
mkdir -p "$work_root/output" "$generated_vk"
find "$generated_vk" -maxdepth 1 -type f -delete

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

base_args=(
    -reflection
    -deps=gcc
    -DFFX_GPU=1
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
    -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
    -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
    '-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}'
    '-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}'
    '-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}'
    '-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}'
    '-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1}'
    '-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}'
    -compiler=glslang
    "-glslangexe=$glslang_validator_win"
    -e
    CS
    --target-env
    vulkan1.2
    -S
    comp
    -Os
    -DFFX_GLSL=1
    "-I$include_gpu_win"
    "-I$include_fsr_win"
    "-output=$output_win"
)

for pass_name in "${passes[@]}"; do
    shader="$sdk_copy/sdk/src/backends/vk/shaders/fsr3upscaler/$pass_name.glsl"
    shader_win="$(wine_path "$shader")"
    for variant in normal wave64 16bit wave64_16bit; do
        name="$pass_name"
        extra_args=(-DFFX_HALF=0)
        suffix="_permutations"
        case "$variant" in
            wave64)
                name="${pass_name}_wave64"
                suffix="_wave64_permutations"
                ;;
            16bit)
                name="${pass_name}_16bit"
                suffix="_16bit_permutations"
                extra_args=(-DFFX_HALF=1)
                ;;
            wave64_16bit)
                name="${pass_name}_wave64_16bit"
                suffix="_wave64_16bit_permutations"
                extra_args=(-DFFX_HALF=1)
                ;;
        esac

        MANGOHUD=0 "$wine_bin" "$shader_compiler_win" "${base_args[@]}" "${extra_args[@]}" "-name=$name" "$shader_win"
        aggregate="$work_root/output/${name}_permutations.h"
        test -f "$aggregate"
        flattened_file="$work_root/output/${name}.flat.h"
        : > "$flattened_file"
        while IFS= read -r line; do
            if [[ "$line" =~ ^#include[[:space:]]+\"([^\"]+)\"$ ]]; then
                blob="${BASH_REMATCH[1]}"
                cat "$work_root/output/$blob" >> "$flattened_file"
            else
                printf '%s\n' "$line" >> "$flattened_file"
            fi
        done < "$aggregate"
        mv "$flattened_file" "$generated_vk/${pass_name}${suffix}.h"
    done
done

(cd "$generated_root" && sha256sum vk/*.h | LC_ALL=C sort) > "$generated_root/SHA256SUMS"
"$script_dir/verify-vulkan-shaders.sh"
