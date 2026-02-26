cmake_minimum_required(VERSION 3.10)

project(EdgeDetectionPipeline LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)

file(GLOB_RECURSE sources      src/*.cpp src/*.c src/*.cu include/*.h include/*.hpp include/*.cuh)

add_executable(EdgeDetectionPipeline ${sources})
target_link_libraries(EdgeDetectionPipeline PRIVATE cuda)