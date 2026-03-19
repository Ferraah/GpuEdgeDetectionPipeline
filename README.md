# GPU Edge Detection Pipeline

A GPU-accelerated, pipelined image processing system for edge detection and visualization using CUDA.
The project was structured to comply as much as possible with the proposed pipelined architecture, which is NOT the most optimized version (see full_parallel branch).

**Language:** C++17  
**IDE:** Visual Studio 2022  
**Platform:** Windows  

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [What Was Implemented](#what-was-implemented)
4. [What Was NOT Implemented](#what-was-not-implemented)
5. [Design Decisions & Optimizations](#design-decisions--optimizations)
6. [Performance Metrics](#performance-metrics)
7. [Project Structure](#project-structure)
8. [Building & Running](#building--running)
9. [Future Work](#future-work)

---

## Overview

This project implements a **5-stage GPU-accelerated pipeline** for edge detection, designed to process multiple images concurrently using CUDA streams. The system detects edges using Gaussian smoothing followed by Laplacian edge detection, then visualizes results by overlaying colored edges on the original image.

---

## Pipeline Architecture

```
Tick:   0       1         2          3          4         5    ...
Img0: [Load] [Gaussian] [Laplacian] [Binarize] [Visualize]
Img1:        [Load]     [Gaussian]  [Laplacian] [Binarize] [Visualize]
Img2:                   [Load]      [Gaussian]  [Laplacian] [Binarize] ...
```

Each image is assigned its own **CUDA stream**, enabling concurrent execution of different pipeline stages across multiple images. This maximizes GPU utilization by overlapping:
- Memory transfers (HtoD / DtoH)
- Kernel execution (convolutions, binarization)
- CPU work (visualization overlay)

### Pipeline Stages

| Stage | Location | Description |
|-------|----------|-------------|
| **1. (Pre-Load)** | CPU | All images loaded into pinned memory before pipeline starts |
| **2. Load** | GPU | Async HtoD transfer via CUDA stream |
| **3. Gaussian Filter** | GPU | 3x3 Gaussian convolution (σ=1) for noise reduction |
| **4. Laplacian Filter** | GPU | 3x3 Laplacian convolution for edge detection |
| **5. Binarize** | GPU | Threshold edge response + async DtoH transfer |
| **6. Visualize** | CPU | Create colored edge overlay on original image |
| **7. (Save)** | CPU | Batch save all output images at pipeline end |

---

## What Was Implemented

### Core Requirements ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| BMP image loading | ✅ | Supports any resolution via stb_image |
| GPU Gaussian filtering (3x3) | ✅ | Separable kernel, normalized weights |
| GPU Laplacian edge detection | ✅ | 4-connectivity kernel |
| GPU thresholding/binarization | ✅ | Configurable threshold value |
| Colored edge overlay | ✅ | Edges rendered in red on original |
| Pipelined execution | ✅ | Per-image CUDA streams |
| Per-stage timing | ✅ | CUDA events for GPU profiling |
| Output images saved | ✅ | BMP format in output folder |
| Per-image stats file | ✅ | Timing breakdown per stage |
| GPU utilization metrics | ✅ | Nsight analysis of the executable |

### Performance Optimizations ✅

| Optimization | Status | Description |
|--------------|--------|-------------|
| Pinned host memory | ✅ | `cudaHostAlloc` for faster DMA transfers |
| Async memory transfers | ✅ | `cudaMemcpyAsync` with streams |
| Double-buffered device memory | ✅ | Avoids in-place convolution race conditions |
| Pre-loading all images | ✅ | Eliminates disk I/O from pipeline hot path |
| Deferred image saving | ✅ | Batch save at end, not blocking pipeline |
| Stream synchronization | ✅ | Minimal syncs, only where required |

### Code Quality ✅

| Aspect | Status | Description |
|--------|--------|-------------|
| Modular stage functions | ✅ | Each stage is a separate function |
| Clear documentation | ✅ | Doxygen-style comments throughout |
| Error handling | ✅ | CUDA error checking macro `ck()` |
| Configurable parameters | ✅ | Threshold, kernel size as constants |

---

## What Was NOT Implemented

### Missing Requirements ❌

| Requirement | Status | Reason/Notes |
|-------------|--------|--------------|
| Centroid detection | ❌ | Connected component analysis not implemented |
| Object counting | ❌ | Requires centroid/blob detection |
| Balanced stage timing | ⚠️ | Convolutions are faster than transfers with this current dataset, but acceptable|

### Potential Improvements Not Done

| Improvement | Status | Description |
|-------------|--------|-------------|
| Threading for I/O | ❌ | Pre-load is sequential (could parallelize) |
| Shared memory convolution | ❌ | Using global memory (could optimize) |
| Threading image saving | ❌ | Sequential save at end (could parallelize) |
| Unit test framework | ❌ | More granular testing, using gtest or other tools |

---

## Design Decisions & Optimizations

### 1. Per-Image CUDA Streams
Each image gets its **own CUDA stream**, allowing the GPU to interleave operations from different images. This is critical for hiding memory transfer latency behind kernel execution.

### 2. Pre-Loading Phase
All images are loaded into CPU pinned memory **before** the pipeline starts. This eliminates disk I/O as a bottleneck during pipeline execution, at the cost of higher memory usage. 

### 3. Double-Buffered Device Memory
Convolution requires reading neighbor pixels while writing output.

### 4. Deferred Image Saving
The visualization stage prepares the overlay but does **not** write to disk. All images are saved in a batch at the end, keeping the pipeline CPU-bound work minimal. Overlapping and other operations related to this step are ran sequentially in the CPU, while the cuda streams work in the background, since I evaluated that using the GPU would have been less efficient.

### 5. Pinned Memory for Transfers
Using `cudaHostAlloc` for host buffers enables DMA transfers that don't block the CPU and achieve higher bandwidth than pageable memory.

Preloading and deferred image saving was detached from the pipeline logic as the profilation shows how expensive they are in terms of time spent. This way I were able to overlap the execution of the stages better, as also shown in the Nsight report. 
---

## Performance Metrics

Per-stage timing is collected using CUDA events and reported at pipeline completion:

```
========================================
Pipeline Performance Statistics
========================================
  Load+Transfer: X.XX ms avg (N images)
  Gaussian:      X.XX ms avg (N images)
  Laplacian:     X.XX ms avg (N images)
  Binarization:  X.XX ms avg (N images)
  Save:          X.XX ms avg (N images)
----------------------------------------
  Total: X.XX ms avg per image
========================================
```

Individual image stats are written to `output_images/output_image_N_stats.txt`.

---

## Project Structure

```
EdgeDetectionPipeline/
├── src/
│   ├── main.cpp           # Pipeline orchestration and entry point
│   ├── kernel.cu          # CUDA kernels (convolution, binarization)
│   └── bmpread.c          # BMP file parsing (unused, using stb_image)
├── include/
│   ├── kernel.cuh         # CUDA kernel declarations
│   ├── image_descriptor.hpp # Per-image data structure
│   ├── utils.hpp          # I/O and utility functions
│   ├── thread_pool.hpp    # Thread pool (available for future use)
│   ├── stb_image.h        # Image loading library
│   └── stb_image_write.h  # Image writing library
├── tests/
│   └── test_output_images.cpp  # Output validation tests
├── output_images/         # Generated output images and stats
├── CMakeLists.txt         # Build configuration
└── README.md              # This document
```

---

## Building & Running

### Prerequisites

- **Visual Studio 2022** with C++ desktop development
- **CUDA Toolkit 11.0+** (tested with 12.x)
- **CMake 3.18+**

### Build Steps

```powershell
# From project root
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Running

1. Place input images (BMP format) in `input_images/` folder
2. Run the executable:

```powershell
cd build
.\Release\EdgeDetectionPipeline.exe
```

3. Output images appear in `output_images/`

### Configuration

Edit constants in `src/main.cpp`:

```cpp
constexpr uint8_t BINARIZE_THRESHOLD = 5;  // Edge threshold (0-255)
const std::string input_folder = "...";     // Input path
const std::string output_folder = "...";    // Output path
```

---

## Future Work

1. **Centroid Detection**: Implement connected component labeling on GPU to identify objects and compute centroids
2. **GPU Utilization Metrics**: Integrate CUPTI or use Nsight for detailed GPU occupancy reporting
3. **Memory Pooling**: Pre-allocate device buffers for max image size to avoid per-image cudaMalloc
4. **Shared Memory Convolution**: Use shared memory tiling for 2-3x faster convolutions
5. **Parallel I/O**: Use thread pool for parallel image loading and saving
6. **Configurable Kernels**: Support different kernel sizes and custom filter coefficients

---

## Author

Daniele Ferrario

