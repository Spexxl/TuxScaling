#include "ffx_provider.h"
#include "ffx_provider_fsr3upscale.h"

#include <cstddef>

static constexpr ffxProvider* providers[] = {
    &ffxProvider_FSR3Upscale::Instance,
};

static constexpr std::size_t provider_count = _countof(providers);

const ffxProvider* GetffxProvider(ffxStructType_t desc_type, uint64_t override_id, void*)
{
    for (std::size_t index = 0; index < provider_count; ++index) {
        if (providers[index]->CanProvide(desc_type) &&
            (override_id == 0 || providers[index]->GetId() == override_id)) {
            return providers[index];
        }
    }

    return nullptr;
}

const ffxProvider* GetAssociatedProvider(ffxContext* context)
{
    if (!context || !*context) {
        return nullptr;
    }

    const auto* header = static_cast<const InternalContextHeader*>(*context);
    return header->provider;
}

uint64_t GetProviderCount(ffxStructType_t desc_type, void* device)
{
    return GetProviderVersions(desc_type, device, UINT64_MAX, nullptr, nullptr);
}

uint64_t GetProviderVersions(
    ffxStructType_t desc_type,
    void* device,
    uint64_t capacity,
    uint64_t* version_ids,
    const char** version_names)
{
    const ffxProvider* provider = GetffxProvider(desc_type, 0, device);
    if (!provider || capacity == 0) {
        return 0;
    }

    if (version_ids) {
        version_ids[0] = provider->GetId();
    }
    if (version_names) {
        version_names[0] = provider->GetVersionName();
    }

    return 1;
}
