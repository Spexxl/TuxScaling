#include "tux_fidelityfx.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_types.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

static_assert(sizeof(TuxFfxDispatchInfo) == 304, "TuxFfxDispatchInfo ABI size changed");
static_assert(alignof(TuxFfxDispatchInfo) == 8, "TuxFfxDispatchInfo ABI alignment changed");
static_assert(offsetof(TuxFfxDispatchInfo, enable_sharpening) == 292,
              "TuxFfxDispatchInfo sharpening flag offset changed");
static_assert(offsetof(TuxFfxDispatchInfo, sharpness) == 296,
              "TuxFfxDispatchInfo sharpness offset changed");

struct TuxFfxContext {
    ffxContext sdk_context = nullptr;
    bool pending_reset = false;
    uint64_t physical_device = 0;
    uint64_t device = 0;
};

static VkImage to_vk_image(uint64_t handle)
{
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
    return reinterpret_cast<VkImage>(static_cast<uintptr_t>(handle));
#else
    return static_cast<VkImage>(handle);
#endif
}

static VkCommandBuffer to_vk_command_buffer(uint64_t handle)
{
    return reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(handle));
}

static VkDevice to_vk_device(uint64_t handle)
{
    return reinterpret_cast<VkDevice>(static_cast<uintptr_t>(handle));
}

static VkPhysicalDevice to_vk_physical_device(uint64_t handle)
{
    return reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(handle));
}

static PFN_vkGetDeviceProcAddr to_vk_get_device_proc_addr(uint64_t handle)
{
    return reinterpret_cast<PFN_vkGetDeviceProcAddr>(static_cast<uintptr_t>(handle));
}

struct PhysicalDispatch {
    PFN_vkGetPhysicalDeviceFeatures get_features;
    PFN_vkGetPhysicalDeviceFeatures2 get_features2;
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties;
    PFN_vkGetPhysicalDeviceProperties get_properties;
    PFN_vkGetPhysicalDeviceProperties2 get_properties2;
    PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions;
    uint32_t vulkan_api_version;
    uint32_t users;
};

static std::mutex physical_dispatch_mutex;
static std::unordered_map<uintptr_t, PhysicalDispatch> physical_dispatches;

static uintptr_t physical_device_key(VkPhysicalDevice physical_device)
{
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
    return reinterpret_cast<uintptr_t>(physical_device);
#else
    return static_cast<uintptr_t>(physical_device);
#endif
}

static void register_physical_dispatch(
    VkPhysicalDevice physical_device,
    PFN_vkGetPhysicalDeviceFeatures get_features,
    PFN_vkGetPhysicalDeviceFeatures2 get_features2,
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties,
    PFN_vkGetPhysicalDeviceProperties get_properties,
    PFN_vkGetPhysicalDeviceProperties2 get_properties2,
    PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions,
    uint32_t vulkan_api_version)
{
    std::lock_guard<std::mutex> lock(physical_dispatch_mutex);
    auto& dispatch = physical_dispatches[physical_device_key(physical_device)];
    dispatch.get_features = get_features;
    dispatch.get_features2 = get_features2;
    dispatch.get_memory_properties = get_memory_properties;
    dispatch.get_properties = get_properties;
    dispatch.get_properties2 = get_properties2;
    dispatch.enumerate_extensions = enumerate_extensions;
    dispatch.vulkan_api_version = vulkan_api_version;
    ++dispatch.users;
}

static void unregister_physical_dispatch(VkPhysicalDevice physical_device)
{
    std::lock_guard<std::mutex> lock(physical_dispatch_mutex);
    const auto key = physical_device_key(physical_device);
    const auto entry = physical_dispatches.find(key);
    if (entry == physical_dispatches.end()) {
        return;
    }
    if (--entry->second.users == 0) {
        physical_dispatches.erase(entry);
    }
}

static bool find_physical_dispatch(VkPhysicalDevice physical_device, PhysicalDispatch& result)
{
    std::lock_guard<std::mutex> lock(physical_dispatch_mutex);
    const auto entry = physical_dispatches.find(physical_device_key(physical_device));
    if (entry == physical_dispatches.end()) {
        return false;
    }
    result = entry->second;
    return true;
}

struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr get_proc_addr;
    uint32_t users;
};

static std::mutex device_dispatch_mutex;
static std::unordered_map<uintptr_t, DeviceDispatch> device_dispatches;

static void register_device_dispatch(VkDevice device, PFN_vkGetDeviceProcAddr get_proc_addr)
{
    std::lock_guard<std::mutex> lock(device_dispatch_mutex);
    auto& dispatch = device_dispatches[reinterpret_cast<uintptr_t>(device)];
    dispatch.get_proc_addr = get_proc_addr;
    ++dispatch.users;
}

static void unregister_device_dispatch(VkDevice device)
{
    std::lock_guard<std::mutex> lock(device_dispatch_mutex);
    const auto entry = device_dispatches.find(reinterpret_cast<uintptr_t>(device));
    if (entry == device_dispatches.end()) {
        return;
    }
    if (--entry->second.users == 0) {
        device_dispatches.erase(entry);
    }
}

static bool find_device_dispatch(VkDevice device, DeviceDispatch& result)
{
    std::lock_guard<std::mutex> lock(device_dispatch_mutex);
    const auto entry = device_dispatches.find(reinterpret_cast<uintptr_t>(device));
    if (entry == device_dispatches.end()) {
        return false;
    }
    result = entry->second;
    return true;
}

static VKAPI_ATTR void VKAPI_CALL get_buffer_memory_requirements2_compat(
    VkDevice device,
    const VkBufferMemoryRequirementsInfo2* info,
    VkMemoryRequirements2* requirements)
{
    if (!info || !requirements) {
        return;
    }

    DeviceDispatch dispatch{};
    if (!find_device_dispatch(device, dispatch) || !dispatch.get_proc_addr) {
        std::memset(&requirements->memoryRequirements, 0, sizeof(requirements->memoryRequirements));
        return;
    }

    const auto proc = dispatch.get_proc_addr(device, "vkGetBufferMemoryRequirements");
    const auto get_buffer_memory_requirements =
        reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(proc);
    if (get_buffer_memory_requirements) {
        get_buffer_memory_requirements(device, info->buffer, &requirements->memoryRequirements);
    } else {
        std::memset(&requirements->memoryRequirements, 0, sizeof(requirements->memoryRequirements));
    }
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc_addr_compat(
    VkDevice device,
    const char* name)
{
    DeviceDispatch dispatch{};
    if (!name || !find_device_dispatch(device, dispatch) || !dispatch.get_proc_addr) {
        return nullptr;
    }
    if (std::strcmp(name, "vkGetBufferMemoryRequirements2KHR") == 0) {
        return reinterpret_cast<PFN_vkVoidFunction>(get_buffer_memory_requirements2_compat);
    }
    return dispatch.get_proc_addr(device, name);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physical_device,
    const char* layer_name,
    uint32_t* property_count,
    VkExtensionProperties* properties)
{
    PhysicalDispatch dispatch{};
    return find_physical_dispatch(physical_device, dispatch) && dispatch.enumerate_extensions
               ? dispatch.enumerate_extensions(physical_device, layer_name, property_count, properties)
                     : VK_ERROR_INITIALIZATION_FAILED;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceFeatures* features)
{
    PhysicalDispatch dispatch{};
    if (find_physical_dispatch(physical_device, dispatch) && dispatch.get_features) {
        dispatch.get_features(physical_device, features);
    } else if (features) {
        std::memset(features, 0, sizeof(*features));
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceFeatures2* features)
{
    PhysicalDispatch dispatch{};
    if (!features) {
        return;
    }
    if (!find_physical_dispatch(physical_device, dispatch)) {
        std::memset(features, 0, sizeof(*features));
        return;
    }
    if (dispatch.vulkan_api_version >= VK_API_VERSION_1_1 && dispatch.get_features2) {
        dispatch.get_features2(physical_device, features);
    } else if (dispatch.get_features) {
        dispatch.get_features(physical_device, &features->features);
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceMemoryProperties* properties)
{
    PhysicalDispatch dispatch{};
    if (find_physical_dispatch(physical_device, dispatch) && dispatch.get_memory_properties) {
        dispatch.get_memory_properties(physical_device, properties);
    } else if (properties) {
        std::memset(properties, 0, sizeof(*properties));
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceProperties* properties)
{
    PhysicalDispatch dispatch{};
    if (find_physical_dispatch(physical_device, dispatch) && dispatch.get_properties) {
        dispatch.get_properties(physical_device, properties);
    } else if (properties) {
        std::memset(properties, 0, sizeof(*properties));
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceProperties2* properties)
{
    PhysicalDispatch dispatch{};
    if (!properties) {
        return;
    }
    if (!find_physical_dispatch(physical_device, dispatch)) {
        std::memset(properties, 0, sizeof(*properties));
        return;
    }
    if (dispatch.vulkan_api_version >= VK_API_VERSION_1_1 && dispatch.get_properties2) {
        dispatch.get_properties2(physical_device, properties);
    } else if (dispatch.get_properties) {
        dispatch.get_properties(physical_device, &properties->properties);
    }
}

static FfxApiResource to_ffx_resource(const TuxFfxImage& image)
{
    if (image.image == 0) {
        return {};
    }

    FfxApiResourceDescription description{};
    description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    description.format = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(image.format));
    description.width = image.width;
    description.height = image.height;
    description.depth = 1;
    description.mipCount = 1;
    description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    description.usage = image.usage;

    return ffxApiGetResourceVK(
        reinterpret_cast<void*>(static_cast<uintptr_t>(image.image)),
        description,
        image.state);
}

static int32_t map_create_status(ffxReturnCode_t status)
{
    if (status == FFX_API_RETURN_OK) {
        return TUX_FFX_OK;
    }
    if (status == FFX_API_RETURN_NO_PROVIDER) {
        return TUX_FFX_UNSUPPORTED_VERSION;
    }
    return TUX_FFX_CREATE_FAILED;
}

extern "C" TUX_FFX_API TuxFfxVersion tux_ffx_version(void)
{
    return TuxFfxVersion{
        FFX_FSR3UPSCALER_VERSION_MAJOR,
        FFX_FSR3UPSCALER_VERSION_MINOR,
        FFX_FSR3UPSCALER_VERSION_PATCH,
    };
}

extern "C" TUX_FFX_API uint32_t tux_ffx_abi_version(void)
{
    return TUX_FFX_ABI_VERSION;
}

extern "C" TUX_FFX_API int32_t tux_ffx_create(
    const TuxFfxCreateInfo* info,
    TuxFfxContext** context)
{
    if (!info || !context || *context || info->physical_device == 0 || info->device == 0 ||
        info->get_device_proc_addr == 0 || info->get_physical_device_features == 0 ||
        info->get_physical_device_features2 == 0 ||
        info->get_physical_device_memory_properties == 0 ||
        info->get_physical_device_properties == 0 ||
        info->get_physical_device_properties2 == 0 ||
        info->enumerate_device_extension_properties == 0 || info->max_render_width == 0 ||
        info->max_render_height == 0 || info->max_output_width == 0 ||
        info->max_output_height == 0) {
        return TUX_FFX_INVALID_ARGUMENT;
    }

    TuxFfxContext* created = new (std::nothrow) TuxFfxContext{};
    if (!created) {
        return TUX_FFX_CREATE_FAILED;
    }

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.header.pNext = nullptr;
    backend.vkDevice = to_vk_device(info->device);
    backend.vkPhysicalDevice = to_vk_physical_device(info->physical_device);
    const auto original_get_device_proc_addr = to_vk_get_device_proc_addr(info->get_device_proc_addr);
    backend.vkDeviceProcAddr = get_device_proc_addr_compat;
    created->physical_device = info->physical_device;
    created->device = info->device;
    register_device_dispatch(backend.vkDevice, original_get_device_proc_addr);
    register_physical_dispatch(
        backend.vkPhysicalDevice,
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(static_cast<uintptr_t>(
            info->get_physical_device_features)),
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(static_cast<uintptr_t>(
            info->get_physical_device_features2)),
        reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(static_cast<uintptr_t>(
            info->get_physical_device_memory_properties)),
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(static_cast<uintptr_t>(
            info->get_physical_device_properties)),
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(static_cast<uintptr_t>(
            info->get_physical_device_properties2)),
        reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(static_cast<uintptr_t>(
            info->enumerate_device_extension_properties)),
        info->vulkan_api_version);

    ffxCreateContextDescUpscale upscale{};
    upscale.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    upscale.header.pNext = &backend.header;
    upscale.flags = info->flags;
    upscale.maxRenderSize = {info->max_render_width, info->max_render_height};
    upscale.maxUpscaleSize = {info->max_output_width, info->max_output_height};
    upscale.fpMessage = nullptr;

    const ffxReturnCode_t status = ffxCreateContext(
        &created->sdk_context,
        &upscale.header,
        nullptr);
    if (status != FFX_API_RETURN_OK) {
        unregister_physical_dispatch(backend.vkPhysicalDevice);
        unregister_device_dispatch(backend.vkDevice);
        delete created;
        return map_create_status(status);
    }

    *context = created;
    return TUX_FFX_OK;
}

extern "C" TUX_FFX_API int32_t tux_ffx_dispatch(
    TuxFfxContext* context,
    const TuxFfxDispatchInfo* info)
{
    if (!context || !info || !context->sdk_context || info->command_buffer == 0 ||
        info->render_width == 0 || info->render_height == 0 || info->output_width == 0 ||
        info->output_height == 0 || info->pre_exposure <= 0.0f ||
        !std::isfinite(info->sharpness) || info->sharpness < 0.0f || info->sharpness > 1.0f) {
        return TUX_FFX_INVALID_ARGUMENT;
    }

    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.header.pNext = nullptr;
    dispatch.commandList = reinterpret_cast<void*>(to_vk_command_buffer(info->command_buffer));
    dispatch.color = to_ffx_resource(info->color);
    dispatch.depth = to_ffx_resource(info->depth);
    dispatch.motionVectors = to_ffx_resource(info->motion);
    dispatch.exposure = to_ffx_resource(info->exposure);
    dispatch.reactive = to_ffx_resource(info->reactive);
    dispatch.transparencyAndComposition = to_ffx_resource(info->composition);
    dispatch.output = to_ffx_resource(info->output);
    dispatch.jitterOffset = {info->jitter_x, info->jitter_y};
    dispatch.motionVectorScale = {info->motion_scale_x, info->motion_scale_y};
    dispatch.renderSize = {info->render_width, info->render_height};
    dispatch.upscaleSize = {info->output_width, info->output_height};
    dispatch.enableSharpening = info->enable_sharpening != 0;
    dispatch.sharpness = info->sharpness;
    dispatch.frameTimeDelta = info->frame_time_ms;
    dispatch.preExposure = info->pre_exposure;
    dispatch.reset = info->reset != 0 || context->pending_reset;
    dispatch.cameraNear = info->camera_near;
    dispatch.cameraFar = info->camera_far;
    dispatch.cameraFovAngleVertical = info->camera_fov_y;
    dispatch.viewSpaceToMetersFactor = info->view_space_to_meters;
    dispatch.flags = 0;

    const ffxReturnCode_t status = ffxDispatch(
        &context->sdk_context,
        &dispatch.header);
    if (status != FFX_API_RETURN_OK) {
        return TUX_FFX_DISPATCH_FAILED;
    }

    context->pending_reset = false;
    return TUX_FFX_OK;
}

extern "C" TUX_FFX_API int32_t tux_ffx_reset(TuxFfxContext* context)
{
    if (!context || !context->sdk_context) {
        return TUX_FFX_INVALID_ARGUMENT;
    }

    context->pending_reset = true;
    return TUX_FFX_OK;
}

extern "C" TUX_FFX_API void tux_ffx_destroy(TuxFfxContext* context)
{
    if (!context) {
        return;
    }

    if (context->sdk_context) {
        ffxDestroyContext(&context->sdk_context, nullptr);
        context->sdk_context = nullptr;
    }
    unregister_physical_dispatch(to_vk_physical_device(context->physical_device));
    unregister_device_dispatch(to_vk_device(context->device));
    delete context;
}
