/**
 * @file kernel.cuh
 * @brief CUDA kernel declarations for edge detection image processing.
 * 
 * This header provides the interface for GPU-accelerated image processing kernels
 * including convolution (for Gaussian/Laplacian filtering) and binarization.
 * Both synchronous and asynchronous (stream-based) versions are available.
 */

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// CUDA Kernels (device code)
// ============================================================================

/**
 * @brief Apply a convolution kernel to an image.
 * 
 * Performs 2D convolution with border handling via clamping (replicate edge pixels).
 * Each thread processes one output pixel.
 * 
 * @param inputImage  Input grayscale image (device memory)
 * @param outputImage Output image buffer (device memory, must be separate from input)
 * @param width       Image width in pixels
 * @param height      Image height in pixels
 * @param kernel      Convolution kernel coefficients (device memory)
 * @param kernelSize  Kernel dimensions (e.g., 3 for 3x3 kernel)
 */
__global__ void applyKernel(uint8_t* inputImage, uint8_t* outputImage, 
                            int width, int height, float* kernel, int kernelSize);

/**
 * @brief Binarize an image using a threshold.
 * 
 * Pixels above threshold become 255 (white), others become 0 (black).
 * Safe for in-place operation (input == output).
 * 
 * @param inputImage  Input grayscale image (device memory)
 * @param outputImage Output binary image (device memory)
 * @param width       Image width in pixels
 * @param height      Image height in pixels
 * @param threshold   Intensity threshold [0-255]
 */
__global__ void binarizeImage(uint8_t* inputImage, uint8_t* outputImage, 
                              int width, int height, uint8_t threshold);

// ============================================================================
// Synchronous Host Functions (block until completion)
// ============================================================================

/**
 * @brief Launch Gaussian filter convolution (synchronous).
 * @note Calls cudaDeviceSynchronize() internally - blocks host thread.
 */
void launchGaussianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                          int width, int height, float* d_kernel, int kernelSize);

/**
 * @brief Launch Laplacian filter convolution (synchronous).
 * @note Calls cudaDeviceSynchronize() internally - blocks host thread.
 */
void launchLaplacianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                           int width, int height, float* d_kernel, int kernelSize);

/**
 * @brief Launch binarization kernel (synchronous).
 * @note Calls cudaDeviceSynchronize() internally - blocks host thread.
 */
void launchBinarizeImage(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                         int width, int height, uint8_t threshold);

// ============================================================================
// Asynchronous Host Functions (stream-based, non-blocking)
// ============================================================================

/**
 * @brief Launch convolution kernel asynchronously on a CUDA stream.
 * 
 * Suitable for pipelined processing where multiple images are processed
 * concurrently on different streams.
 * 
 * @param d_inputImage  Input image (device memory)
 * @param d_outputImage Output image (device memory, must differ from input)
 * @param width         Image width
 * @param height        Image height
 * @param d_kernel      Convolution kernel (device memory)
 * @param kernelSize    Kernel size (e.g., 3)
 * @param stream        CUDA stream for asynchronous execution
 */
void launchConvolutionAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                            int width, int height, float* d_kernel, 
                            int kernelSize, cudaStream_t stream);

/**
 * @brief Launch binarization kernel asynchronously on a CUDA stream.
 * 
 * @param d_inputImage  Input image (device memory)
 * @param d_outputImage Output image (device memory, can be same as input)
 * @param width         Image width
 * @param height        Image height
 * @param threshold     Binarization threshold [0-255]
 * @param stream        CUDA stream for asynchronous execution
 */
void launchBinarizeImageAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
                              int width, int height, uint8_t threshold, 
                              cudaStream_t stream);

#endif // KERNEL_CUH