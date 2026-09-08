#!/usr/bin/env bash
set -euo pipefail

ffx_root="${1:-third_party/fidelityfx-sdk}"
ffx_expected_commit="c6efa6bf7f2027b3ec94f28578bb5965eabb9e55"
ffx_actual_commit="$(git -C "$ffx_root" rev-parse HEAD)"

test "$ffx_actual_commit" = "$ffx_expected_commit"
grep -q '#define FFX_FSR3UPSCALER_VERSION_MAJOR      (3)' "$ffx_root/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h"
grep -q '#define FFX_FSR3UPSCALER_VERSION_MINOR      (1)' "$ffx_root/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h"
grep -q '#define FFX_FSR3UPSCALER_VERSION_PATCH      (4)' "$ffx_root/sdk/include/FidelityFX/host/ffx_fsr3upscaler.h"
grep -q 'Permission is hereby granted, free of charge' "$ffx_root/LICENSE.txt"
