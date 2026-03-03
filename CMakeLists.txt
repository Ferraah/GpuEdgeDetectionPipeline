cmake_minimum_required(VERSION 3.10)

project(EdgeDetectionPipeline LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Debug builds unless the user explicitly sets a build type.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

include_directories(include)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 89)

file(GLOB_RECURSE sources      src/*.cpp src/*.c src/*.cu include/*.h include/*.hpp include/*.cuh)

add_executable(EdgeDetectionPipeline ${sources})
target_link_libraries(EdgeDetectionPipeline PRIVATE cuda)

# Unit tests
add_executable(TestOutputImages tests/test_output_images.cpp)
enable_testing()
add_test(NAME OutputImageComparison COMMAND TestOutputImages)