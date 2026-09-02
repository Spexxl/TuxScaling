#ifndef TUXSCALING_BACKEND_H
#define TUXSCALING_BACKEND_H

#include <stdint.h>

#define TUXSCALING_BACKEND_ABI_VERSION 1u

typedef struct tuxscaling_backend_context tuxscaling_backend_context;
typedef struct tuxscaling_backend_resource tuxscaling_backend_resource;

typedef struct tuxscaling_backend_capabilities {
    uint32_t abi_version;
    uint32_t supports_temporal;
    uint32_t supports_frame_generation;
} tuxscaling_backend_capabilities;

/* The caller owns the returned context and must release it with destroy. */
uint32_t tuxscaling_backend_query(
    uint32_t requested_abi_version,
    tuxscaling_backend_capabilities *out_capabilities);

void tuxscaling_backend_destroy(tuxscaling_backend_context *context);

#endif
