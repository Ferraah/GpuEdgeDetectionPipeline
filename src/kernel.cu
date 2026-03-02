// This file contains CUDA kernel declarations and functions for edge detection operations.

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "kernel.cuh"

using namespace std;

// Gaussian filtering kernel
__global__ void applyKernel(uint8_t* inputImage, uint8_t* outputImage, int width, int height, float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    int halfKernelSize = kernelSize / 2;

    for (int ky = -halfKernelSize; ky <= halfKernelSize; ky++) {
        for (int kx = -halfKernelSize; kx <= halfKernelSize; kx++) {

            // Convolution of the border handled with min & max
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            sum += inputImage[iy * width + ix] * kernel[(ky + halfKernelSize) * kernelSize + (kx + halfKernelSize)];
        }
    }

    outputImage[y * width + x] = static_cast<uint8_t>(sum);

}

__global__ void binarizeImage(uint8_t* inputImage, uint8_t* outputImage, int width, int height, uint8_t threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    uint8_t pixelValue = inputImage[y * width + x];
    outputImage[y * width + x] = (pixelValue > threshold) ? 255 : 0;
}

void launchBinarizeImage(uint8_t* d_inputImage, uint8_t* d_outputImage, int width, int height, uint8_t threshold) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    binarizeImage<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, threshold);
    cudaDeviceSynchronize();
}

void launchGaussianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    applyKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();
}

void launchLaplacianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    applyKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();
}

void launchConvolutionAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    applyKernel<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, width, height, d_kernel, kernelSize);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr) << std::endl;
    }
}

void launchBinarizeImageAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, int width, int height, uint8_t threshold, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    binarizeImage<<<gridSize, blockSize, 0, stream>>>(d_inputImage, d_outputImage, width, height, threshold);

}

#endif // KERNEL_CUH