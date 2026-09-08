#ifndef TUX_FIDELITYFX_H
#define TUX_FIDELITYFX_H

#include <stdint.h>

#if defined(_WIN32)
#define TUX_FFX_API __declspec(dllexport)
#else
#define TUX_FFX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TuxFfxContext TuxFfxContext;

typedef struct TuxFfxVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} TuxFfxVersion;

typedef struct TuxFfxImage {
    uint64_t image;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t usage;
    uint32_t state;
} TuxFfxImage;

typedef struct TuxFfxCreateInfo {
    uint64_t physical_device;
    uint64_t device;
    uint64_t get_device_proc_addr;
    uint64_t enumerate_device_extension_properties;
    uint64_t get_physical_device_features;
    uint64_t get_physical_device_features2;
    uint64_t get_physical_device_memory_properties;
    uint64_t get_physical_device_properties;
    uint64_t get_physical_device_properties2;
    uint32_t max_render_width;
    uint32_t max_render_height;
    uint32_t max_output_width;
    uint32_t max_output_height;
    uint32_t flags;
    uint32_t vulkan_api_version;
} TuxFfxCreateInfo;

typedef struct TuxFfxDispatchInfo {
    uint64_t command_buffer;
    TuxFfxImage color;
    TuxFfxImage depth;
    TuxFfxImage motion;
    TuxFfxImage exposure;
    TuxFfxImage reactive;
    TuxFfxImage composition;
    TuxFfxImage output;
    float jitter_x;
    float jitter_y;
    float motion_scale_x;
    float motion_scale_y;
    float frame_time_ms;
    float pre_exposure;
    float camera_near;
    float camera_far;
    float camera_fov_y;
    float view_space_to_meters;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t reset;
} TuxFfxDispatchInfo;

enum {
    TUX_FFX_OK = 0,
    TUX_FFX_INVALID_ARGUMENT = 1,
    TUX_FFX_UNSUPPORTED_VERSION = 2,
    TUX_FFX_CREATE_FAILED = 3,
    TUX_FFX_DISPATCH_FAILED = 4
};

enum {
    TUX_FFX_CREATE_HIGH_DYNAMIC_RANGE = 1u << 0,
    TUX_FFX_CREATE_DISPLAY_RESOLUTION_MOTION = 1u << 1,
    TUX_FFX_CREATE_MOTION_VECTORS_JITTER_CANCELLATION = 1u << 2,
    TUX_FFX_CREATE_DEPTH_INVERTED = 1u << 3,
    TUX_FFX_CREATE_DEPTH_INFINITE = 1u << 4,
    TUX_FFX_CREATE_AUTO_EXPOSURE = 1u << 5,
    TUX_FFX_CREATE_DYNAMIC_RESOLUTION = 1u << 6,
    TUX_FFX_CREATE_DEBUG_CHECKING = 1u << 7,
    TUX_FFX_CREATE_NON_LINEAR_COLORSPACE = 1u << 8
};

TUX_FFX_API TuxFfxVersion tux_ffx_version(void);
TUX_FFX_API int32_t tux_ffx_create(const TuxFfxCreateInfo* info, TuxFfxContext** context);
TUX_FFX_API int32_t tux_ffx_dispatch(TuxFfxContext* context, const TuxFfxDispatchInfo* info);
TUX_FFX_API int32_t tux_ffx_reset(TuxFfxContext* context);
TUX_FFX_API void tux_ffx_destroy(TuxFfxContext* context);

#ifdef __cplusplus
}
#endif

#endif
