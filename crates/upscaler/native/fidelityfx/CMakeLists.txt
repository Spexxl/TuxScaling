cmake_minimum_required(VERSION 3.23)

project(tuxscaling_fidelityfx_vk LANGUAGES CXX)

set(FFX_SDK_ROOT "" CACHE PATH "External FidelityFX SDK checkout used only for rebuilding the companion library")
set(FFX_GENERATED_SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/generated/vk" CACHE PATH "Generated FidelityFX Vulkan shader headers")

if (NOT FFX_SDK_ROOT OR NOT EXISTS "${FFX_SDK_ROOT}/ffx-api/CMakeLists.txt")
    message(FATAL_ERROR "FFX_SDK_ROOT must point to an external FidelityFX SDK checkout")
endif()

set(FFX_API_BACKEND VK_X64 CACHE STRING "FidelityFX API backend")
set(FFX_FSR2 OFF CACHE BOOL "" FORCE)
set(FFX_FSR3 OFF CACHE BOOL "" FORCE)
set(FFX_FI OFF CACHE BOOL "" FORCE)
set(FFX_OF OFF CACHE BOOL "" FORCE)
set(FFX_FSR3UPSCALER ON CACHE BOOL "" FORCE)
set(FFX_AUTO_COMPILE_SHADERS OFF CACHE BOOL "" FORCE)
set(FFX_BUILD_AS_DLL OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FFX_LINUX_COMPAT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/tux_fidelityfx_linux_compat.h")
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-include${FFX_LINUX_COMPAT_HEADER}>)

add_subdirectory("${FFX_SDK_ROOT}/ffx-api" "${CMAKE_CURRENT_BINARY_DIR}/ffx-api")

target_include_directories(amd_fidelityfx_vk PRIVATE
    "${FFX_SDK_ROOT}/ffx-api/include"
    "${FFX_SDK_ROOT}/sdk/include"
)
target_include_directories(ffx_backend_vk_x64 PUBLIC "${FFX_GENERATED_SHADER_PATH}")

add_library(tuxscaling_fidelityfx_vk SHARED
    src/tux_fidelityfx.cpp
    src/tux_fidelityfx_provider.cpp
)

target_compile_features(tuxscaling_fidelityfx_vk PRIVATE cxx_std_17)
target_compile_definitions(tuxscaling_fidelityfx_vk PRIVATE TUX_FFX_ABI_VERSION=2)
target_include_directories(tuxscaling_fidelityfx_vk PRIVATE
    include
    "${FFX_SDK_ROOT}/ffx-api/src"
    "${FFX_SDK_ROOT}/ffx-api/include"
    "${FFX_SDK_ROOT}/sdk/include"
)
target_link_libraries(tuxscaling_fidelityfx_vk PRIVATE amd_fidelityfx_vk)
target_link_options(tuxscaling_fidelityfx_vk PRIVATE "LINKER:-Bsymbolic-functions")
set_target_properties(tuxscaling_fidelityfx_vk PROPERTIES
    OUTPUT_NAME tuxscaling_fidelityfx_vk
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output"
)
