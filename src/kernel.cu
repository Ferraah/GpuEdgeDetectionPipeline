// ============================================================================
// CUDA Kernel Implementation for Edge Detection Pipeline
// Contains kernels for image convolution and binarization operations
// ============================================================================

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "kernel.cuh"

using namespace std;

/**
 * @brief Function that applies a 2D convolution kernel to a single pixel
 * 
 * @param inputImage Input image data (grayscale, 8-bit)
 * @param outputImage Output image data (grayscale, 8-bit)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param kernel Convolution kernel coefficients
 * @param kernelSize Size of square kernel (3x3, 5x5, etc.)
 * 
 * Each thread processes one pixel using the provided kernel.
 * Border pixels are handled with clamping (repeating edge values).
 */
__global__ void applyKernel(uint8_t* inputImage, uint8_t* outputImage, int width, int height, float* kernel, int kernelSize) {
    // Calculate the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within image bounds
    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    int halfKernelSize = kernelSize / 2;

    // Apply convolution: multiply each kernel element by corresponding image pixel
    for (int ky = -halfKernelSize; ky <= halfKernelSize; ky++) {
        for (int kx = -halfKernelSize; kx <= halfKernelSize; kx++) {

            // Clamp coordinates to image boundaries to handle edges
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            // Accumulate weighted sum
            sum += inputImage[iy * width + ix] * kernel[(ky + halfKernelSize) * kernelSize + (kx + halfKernelSize)];
        }
    }

    // Store result, clamp to 8-bit range [0, 255]
    outputImage[y * width + x] = static_cast<uint8_t>(sum);

}

/**
 * @brief Function that binarizes an image using a threshold
 * 
 * @param inputImage Input grayscale image
 * @param outputImage Output binary image (0 or 255)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param threshold Threshold value for binarization
 * 
 * Pixels above threshold become white (255), below become black (0).
 */
__global__ void binarizeImage(uint8_t* inputImage, uint8_t* outputImage, int width, int height, uint8_t threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint8_t pixelValue = inputImage[y * width + x];
    // Apply threshold: pixel > threshold -> 255 (white), else -> 0 (black)
    outputImage[y * width + x] = (pixelValue > threshold) ? 255 : 0;
}

/**
 * @brief Host function that launches the convolution kernel asynchronously on GPU
 * 
 * @param d_inputImage Device memory pointer to input image
 * @param d_outputImage Device memory pointer to output image
 * @param width Image width
 * @param height Image height
 * @param d_kernel Device memory pointer to convolution kernel
 * @param kernelSize Kernel size 
 * @param stream CUDA stream for async execution
 * 
 * Configures grid and block dimensions for efficient GPU execution,
 * launches kernel, and checks for launch errors.
 */
void launchConvolutionAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize, cudaStream_t stream) {
    // Configure block size (16x16 = 256 threads per block)
    dim3 blockSize(16, 16);
    // Calculate grid size to cover all image pixels
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    // Launch kernel asynchronously on specified stream
    applyKernel<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, width, height, d_kernel, kernelSize);
    // Check for kernel launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }
}

/**
 * @brief Host function that launches the binarization kernel asynchronously on GPU
 * 
 * @param d_inputImage Device memory pointer to input grayscale image
 * @param d_outputImage Device memory pointer to output binary image
 * @param width Image width
 * @param height Image height
 * @param threshold Binarization threshold value
 * @param stream CUDA stream for async execution
 * 
 * Configures grid and block dimensions and launches the binarization kernel.
 */
void launchBinarizeImageAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, int width, int height, uint8_t threshold, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    binarizeImage<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, width, height, threshold);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }
}

#endif // KERNEL_CUH