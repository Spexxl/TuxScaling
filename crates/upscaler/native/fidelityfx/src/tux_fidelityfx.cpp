#include "tux_fidelityfx.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_types.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <cstdint>
#include <new>

struct TuxFfxContext {
    ffxContext sdk_context = nullptr;
    bool pending_reset = false;
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

extern "C" TUX_FFX_API int32_t tux_ffx_create(
    const TuxFfxCreateInfo* info,
    TuxFfxContext** context)
{
    if (!info || !context || *context || info->physical_device == 0 || info->device == 0 ||
        info->get_device_proc_addr == 0 || info->max_render_width == 0 ||
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
    backend.vkDeviceProcAddr = to_vk_get_device_proc_addr(info->get_device_proc_addr);

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
        info->output_height == 0 || info->pre_exposure <= 0.0f) {
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
    dispatch.enableSharpening = false;
    dispatch.sharpness = 0.0f;
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
    delete context;
}
