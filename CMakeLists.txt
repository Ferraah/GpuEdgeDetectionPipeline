cmake_minimum_required(VERSION 3.10)

project(EdgeDetectionPipeline LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP for C/C++ only (not CUDA)
find_package(OpenMP REQUIRED COMPONENTS C CXX)

# Default to Debug builds unless the user explicitly sets a build type.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

include_directories(include)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 40 series (compute capability 8.9)

file(GLOB_RECURSE sources      src/*.cpp src/*.c src/*.cu include/*.h include/*.hpp include/*.cuh)

add_executable(EdgeDetectionPipeline ${sources})
target_link_libraries(EdgeDetectionPipeline PRIVATE cuda OpenMP::OpenMP_CXX)