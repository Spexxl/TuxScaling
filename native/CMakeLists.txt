cmake_minimum_required(VERSION 3.20)
project(tuxscaling_native LANGUAGES C)

add_library(tuxscaling_backend_abi INTERFACE)
target_include_directories(tuxscaling_backend_abi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
