#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <queue>
#include "image_descriptor.hpp"
#include "kernel.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"


void runPipeline(std::vector<std::string>& images_paths, const std::string& output_folder) {

    constexpr int PIPELINE_DEPTH = 4; // one slot per stage

    std::vector<ImageDescriptor> imgs;

    std::cout << "Running pipeline with n. Images = " << images_paths.size() << std::endl;
    
    const float kernel[9] = { 0.0625, 0.125, 0.0625,
                        0.125,  0.25, 0.125,
                        0.0625, 0.125, 0.0625 };
    const float laplacian_kernel[9] = { 0,  1, 0,
                             1, -4, 1,
                             0,  1, 0 };
    const int kernel_size = 3;

    float* d_kernel_gaussian, *d_kernel_laplacian;
    ck(cudaMalloc(&d_kernel_gaussian, kernel_size * kernel_size * sizeof(float)));
    ck(cudaMalloc(&d_kernel_laplacian, kernel_size * kernel_size * sizeof(float)));

    ck(cudaMemcpy(d_kernel_gaussian, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_kernel_laplacian, laplacian_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    size_t N_IMAGES = images_paths.size();
    for(size_t t = 0; t < N_IMAGES + PIPELINE_DEPTH; t++){
        std::cout << "Pipeline tick: " << t << std::endl;
        // Loading
        if(t < N_IMAGES){
            ImageDescriptor img;
            img.path = images_paths[t];
            std::cout<< "Loading image: " << img.path << std::endl;
            utils::load_image(images_paths[t], img.h_data_mono, img.h_data_bgr, img.width, img.height);
            // Create stream for this image
            ck(cudaStreamCreate(&img.stream));
            // Bring on device
            ck(cudaMalloc(&img.d_data_mono, img.width * img.height * sizeof(uint8_t)));
            ck(cudaMemcpyAsync(img.d_data_mono, img.h_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyHostToDevice, img.stream));    
            imgs.push_back(img);  
        }
        // Gaussian
        if(t>=1 && t-1 < N_IMAGES){
            ImageDescriptor img = imgs[t-1];
            ck(cudaStreamSynchronize(img.stream)); // Ensure Gaussian is done before starting Laplacian
            std::cout << "Gaussianing image: " << img.path << std::endl;
            launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_gaussian, kernel_size, img.stream);
        }
        // Laplacian
        if(t>=2 && t-2 < N_IMAGES){
            ImageDescriptor img = imgs[t-2];
            ck(cudaStreamSynchronize(img.stream)); // Ensure Gaussian is done before starting Laplacian
            std::cout << "Laplacing image: " << img.path << std::endl;
            launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_laplacian, kernel_size, img.stream);
        }
        // Binarization
        if(t>=3 && t-3 < N_IMAGES){
            ImageDescriptor img = imgs[t-3];
            ck(cudaStreamSynchronize(img.stream)); // Ensure Gaussian is done before starting Laplacian
            std::cout << "Binarizing image: " << img.path << std::endl;
            launchBinarizeImageAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, 16, img.stream);
            ck(cudaMemcpyAsync(img.h_data_mono, img.d_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyDeviceToHost, img.stream));
        }
        // Saving
        if(t>=4 && t-4 < N_IMAGES){
            ImageDescriptor img = imgs[t-4];
              
            ck(cudaStreamSynchronize(img.stream)); // Ensure all processing is done before saving
            std::cout << "Saving image: " << img.path << std::endl;
             
            uint8_t* h_data_rgb_new = new uint8_t[img.width * img.height*3]; 
            // Convert binary image to RGB with (255,0,0)
            utils::mono_to_bgr(img.h_data_mono, h_data_rgb_new, img.width * img.height);
            // Overlay the binary image on top of the original RGB image
            utils::overlap_images(img.h_data_bgr, h_data_rgb_new, img.h_data_bgr, img.width, img.height, 3);
            
            // Save image data to file for verification
            utils::save_image(output_folder + "/output_image_" + std::to_string(t-4) + ".bmp", img.h_data_bgr, img.width, img.height, 3);
            utils::save_image(output_folder + "/output_mono_image_" + std::to_string(t-4) + ".bmp", img.h_data_mono, img.width, img.height, 1);
            cudaFreeHost(img.h_data_mono);
            cudaFreeHost(img.h_data_bgr);
            cudaFree(img.d_data_mono);
            ck(cudaStreamDestroy(img.stream));
            delete[] h_data_rgb_new;

        }
    }

    cudaFree(d_kernel_gaussian);
    cudaFree(d_kernel_laplacian);
}

int main() {
   
    std::cout << "Starting Edge Detection Pipeline..." << std::endl;
    std::string img_folder = "C:\\Users\\dferrario\\EdgeDetectionPipeline\\UDED\\imgs_bmp";
    std::string output_folder = "output_images";
    std::vector<std::string> images_paths;
    utils::get_images_paths(img_folder, images_paths);
    
    runPipeline(images_paths, output_folder);
    return 0;
}