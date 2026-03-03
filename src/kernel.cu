/**
 * @file kernel.cu
 * @brief CUDA kernel implementations for edge detection image processing.
 * 
 * Implements GPU-accelerated convolution and binarization kernels for the
 * edge detection pipeline. Uses 16x16 thread blocks for efficient execution
 * on modern NVIDIA GPUs.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "kernel.cuh"

// ============================================================================
// Configuration Constants
// ============================================================================

/// Thread block dimensions for 2D kernel launches
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief 2D convolution kernel with border clamping.
 * 
 * Each thread computes one output pixel by applying the convolution kernel
 * to the corresponding input region. Border pixels are handled by clamping
 * coordinates to valid range (replicate edge strategy).
 */
__global__ void applyKernel(uint8_t* inputImage, uint8_t* outputImage, 
                            int width, int height, float* kernel, int kernelSize) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    const int halfKernelSize = kernelSize / 2;

    // Apply convolution
    for (int ky = -halfKernelSize; ky <= halfKernelSize; ky++) {
        for (int kx = -halfKernelSize; kx <= halfKernelSize; kx++) {
            // Clamp coordinates to image bounds (border replication)
            const int ix = min(max(x + kx, 0), width - 1);
            const int iy = min(max(y + ky, 0), height - 1);

            const int kernelIdx = (ky + halfKernelSize) * kernelSize + (kx + halfKernelSize);
            sum += inputImage[iy * width + ix] * kernel[kernelIdx];
        }
    }

    // Clamp result to valid uint8 range
    outputImage[y * width + x] = static_cast<uint8_t>(min(max(sum, 0.0f), 255.0f));
}

/**
 * @brief Threshold-based image binarization kernel.
 * 
 * Each thread processes one pixel. Output is 255 if input > threshold, else 0.
 * Safe for in-place operation since each thread reads/writes only its own pixel.
 */
__global__ void binarizeImage(uint8_t* inputImage, uint8_t* outputImage, 
                              int width, int height, uint8_t threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) {
        return;
    }

    const uint8_t pixelValue = inputImage[y * width + x];
    outputImage[y * width + x] = (pixelValue > threshold) ? 255 : 0;
}

// ============================================================================
// Synchronous Launch Functions
// ============================================================================

/**
 * @brief Compute grid dimensions for a given image size.
 */
inline dim3 computeGridSize(int width, int height) {
    return dim3(
        (width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );
}

void launchBinarizeImage(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                         int width, int height, uint8_t threshold) {
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 gridSize = computeGridSize(width, height);
    
    binarizeImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, 
                                           width, height, threshold);
    cudaDeviceSynchronize();
}

void launchGaussianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                          int width, int height, float* d_kernel, int kernelSize) {
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 gridSize = computeGridSize(width, height);
    
    applyKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, 
                                         width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();
}

void launchLaplacianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                           int width, int height, float* d_kernel, int kernelSize) {
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 gridSize = computeGridSize(width, height);
    
    applyKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, 
                                         width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();
}

// ============================================================================
// Asynchronous Launch Functions (Stream-based)
// ============================================================================

void launchConvolutionAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                            int width, int height, float* d_kernel, 
                            int kernelSize, cudaStream_t stream) {
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 gridSize = computeGridSize(width, height);
    
    applyKernel<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, 
                                                    width, height, d_kernel, kernelSize);
    
    // Check for launch errors (async - doesn't wait for completion)
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Convolution kernel launch failed: " 
                  << cudaGetErrorString(launchErr) << std::endl;
    }
}

void launchBinarizeImageAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                              int width, int height, uint8_t threshold, 
                              cudaStream_t stream) {
    const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const dim3 gridSize = computeGridSize(width, height);
    
    binarizeImage<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, 
                                                      width, height, threshold);
    
    // Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Binarize kernel launch failed: " 
                  << cudaGetErrorString(launchErr) << std::endl;
    }
}