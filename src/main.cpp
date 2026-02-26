#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include "kernel.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"

#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }


int main() {

    std::string img_folder = "/home/dferrario/EdgeDetectionPipeline/UDED/imgs_bmp/";
    std::vector<std::string> img_paths;
    get_images_paths(img_folder, img_paths);
    
    int width, height;
    int channels = 1;

    std::string curr_img_path = img_paths[0]; // Just take the first image for testing
    std::cout << "Loading image: " << curr_img_path << std::endl;

    uint8_t* data, *data_bgr;
    load_image(curr_img_path, data, data_bgr, width, height);
    printf("Loaded image: %d x %d, \n", width, height);
    
    // Call CUDA functions to process images
    uint8_t* d_input;
    uint8_t* d_output;
    float* d_kernel;
    uint8_t* h_output = new uint8_t[width * height];
    float kernel[9] = { 0.0625, 0.125, 0.0625,
                        0.125,  0.25, 0.125,
                        0.0625, 0.125, 0.0625 };
    float laplacian_kernel[9] = { 0,  1, 0,
                             1, -4, 1,
                             0,  1, 0 };
    int kernel_size = 3;

    ck(cudaMalloc(&d_input, width * height * sizeof(uint8_t)));
    ck(cudaMalloc(&d_output, width * height * sizeof(uint8_t)));
    ck(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

    // Gaussian
    ck(cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_input, data, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    launchGaussianFilter(d_input, d_output, width, height, d_kernel, kernel_size);
   
    // Laplacian
    ck(cudaMemcpy(d_kernel, laplacian_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    launchLaplacianFilter(d_output, d_output, width, height, d_kernel, kernel_size);

    // Binarization
    uint8_t threshold = 16; // Example threshold value
    launchBinarizeImage(d_output, d_output, width, height, threshold);
    ck(cudaMemcpy(h_output, d_output, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    uint8_t* bgrData = new uint8_t[width * height * 3];
    mono_to_bgr(h_output, bgrData, width * height);
    overlap_images(data_bgr, bgrData, bgrData, width, height, 3);

    // Save image data to file for verification
    stbi_write_bmp("output_image.bmp", width, height, 1, h_output);
    stbi_write_bmp("output_image_bgr.bmp", width, height, 3, bgrData);

    // Free allocated memory
    delete[] h_output;
    delete[] bgrData;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(data);
    free(data_bgr);

    std::cout << "Edge Detection Pipeline completed." << std::endl;
    return 0;
}