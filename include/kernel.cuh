// This file contains CUDA kernel declarations and functions for edge detection operations.

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

using namespace std;

// Gaussian filtering kernel
__global__ void applyKernel(uint8_t* inputImage, uint8_t* outputImage, int width, int height, float* kernel, int kernelSize);

void launchGaussianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize); 

void launchLaplacianFilter(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize);

void launchBinarizeImage(uint8_t* d_inputImage, uint8_t* d_outputImage, int width, int height, uint8_t threshold);  


void launchConvolutionAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, 
    int width, int height, float* d_kernel, int kernelSize, cudaStream_t stream);

void launchBinarizeImageAsync(uint8_t* d_inputImage, uint8_t* d_outputImage, int width, int height, uint8_t threshold, cudaStream_t stream);
#endif // KERNEL_CUH