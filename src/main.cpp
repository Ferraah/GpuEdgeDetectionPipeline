#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <array>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include "image_descriptor.hpp"
#include "kernel.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"


void runPipeline(std::vector<std::string>& images_paths, const std::string& output_folder) {

    constexpr int PIPELINE_DEPTH = 5; // one slot per stage


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

    // Timing storage for each stage
    std::array<std::vector<float>, PIPELINE_DEPTH> timings; 

    size_t N_IMAGES = images_paths.size();

    for(size_t t = 0; t < N_IMAGES + PIPELINE_DEPTH; t++){
        std::cout << "Pipeline tick: " << t << std::endl;
        // Loading
        if(t < N_IMAGES){
            ImageDescriptor img;
            img.path = images_paths[t];

            std::cout<< "Loading image: " << img.path << std::endl;

            // Create stream and timing events for this image
            ck(cudaStreamCreate(&img.stream));
            for (int s = 0; s < 5; s++) {
                ck(cudaEventCreate(&img.stage_start[s]));
                ck(cudaEventCreate(&img.stage_end[s]));
            }

            // Record start of Loading stage (CPU-bound, use default stream for timing reference)
            utils::load_image(images_paths[t], img.h_data_mono, img.h_data_bgr, img.width, img.height);
            ck(cudaEventRecord(img.stage_start[0], img.stream)); // NOTE: Timing starts after CPU load, focuses on GPU processing time 

            // Allocate device memory (double-buffer to avoid in-place convolution race)
            ck(cudaMalloc(&img.d_data_mono, img.width * img.height * sizeof(uint8_t)));
            ck(cudaMalloc(&img.d_data_mono_out, img.width * img.height * sizeof(uint8_t)));

            // Bring on device
            ck(cudaMemcpyAsync(img.d_data_mono, img.h_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyHostToDevice, img.stream));
            ck(cudaEventRecord(img.stage_end[0], img.stream));
            imgs.push_back(img);  
        }
        // Gaussian
        if(t>=1 && t-1 < N_IMAGES){
            ImageDescriptor& img = imgs[t-1];  // Use reference to avoid copy

            ck(cudaEventRecord(img.stage_start[1], img.stream));
            std::cout << "Gaussianing image: " << img.path << std::endl;
            launchConvolutionAsync(img.d_data_mono, img.d_data_mono_out, img.width, img.height, d_kernel_gaussian, kernel_size, img.stream);
            std::swap(img.d_data_mono, img.d_data_mono_out);
            ck(cudaEventRecord(img.stage_end[1], img.stream));
        }
        // Laplacian
        if(t>=2 && t-2 < N_IMAGES){
            ImageDescriptor& img = imgs[t-2];

            ck(cudaEventRecord(img.stage_start[2], img.stream));
            std::cout << "Laplacing image: " << img.path << std::endl;
            launchConvolutionAsync(img.d_data_mono, img.d_data_mono_out, img.width, img.height, d_kernel_laplacian, kernel_size, img.stream);
            std::swap(img.d_data_mono, img.d_data_mono_out);
            ck(cudaEventRecord(img.stage_end[2], img.stream));
        }
        // Binarization
        if(t>=3 && t-3 < N_IMAGES){
            ImageDescriptor& img = imgs[t-3];

            ck(cudaEventRecord(img.stage_start[3], img.stream));
            std::cout << "Binarizing image: " << img.path << std::endl;
            launchBinarizeImageAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, 16, img.stream);
            ck(cudaMemcpyAsync(img.h_data_mono, img.d_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyDeviceToHost, img.stream));
            ck(cudaEventRecord(img.stage_end[3], img.stream));
        }
        // Saving
        if(t>=4 && t-4 < N_IMAGES){
            ImageDescriptor& img = imgs[t-4];  // Use reference
            ck(cudaStreamSynchronize(img.stream));  // Ensure GPU work is done before CPU save

            ck(cudaEventRecord(img.stage_start[4], img.stream));
            std::cout << "Saving image: " << img.path << std::endl;

            uint8_t* h_data_rgb_new = new uint8_t[img.width * img.height*3]; 
            // Convert binary image to RGB with (255,0,0)
            utils::mono_to_bgr(img.h_data_mono, h_data_rgb_new, img.width * img.height);
            // Overlay the binary image on top of the original RGB image
            utils::overlap_images(img.h_data_bgr, h_data_rgb_new, img.h_data_bgr, img.width, img.height, 3);
            
            ck(cudaEventRecord(img.stage_end[4], img.stream)); // NOTE: Timing excludes CPU save time, focuses on GPU processing time up to this point
            // Save image data to file for verification
            utils::save_image(output_folder + "/output_image_" + std::to_string(t-4) + ".bmp", img.h_data_bgr, img.width, img.height, 3);

            // Collect timing for this image (all events are on the same stream)
            float stage_times[5];
            for (int s = 0; s < 5; s++) {
                float elapsed;
                ck(cudaEventSynchronize(img.stage_end[s]));
                ck(cudaEventElapsedTime(&elapsed, img.stage_start[s], img.stage_end[s]));
                stage_times[s] = elapsed;
                timings[s].push_back(elapsed);
                ck(cudaEventDestroy(img.stage_start[s]));
                ck(cudaEventDestroy(img.stage_end[s]));
            }

            // Save per-image stats to file
            std::string stats_path = output_folder + "/output_image_" + std::to_string(t-4) + "_stats.txt";
            utils::save_image_stats(stats_path, img.path, img.width, img.height, stage_times);

            cudaFreeHost(img.h_data_mono);
            cudaFreeHost(img.h_data_bgr);
            cudaFree(img.d_data_mono);
            cudaFree(img.d_data_mono_out);  // Free double-buffer
            ck(cudaStreamDestroy(img.stream));
            delete[] h_data_rgb_new;
        }
    }

    cudaFree(d_kernel_gaussian);
    cudaFree(d_kernel_laplacian);



    // Print statistics for each stage
    std::cout << "\nPipeline Performance Statistics:\n";
    for(int i = 0; i < 5; i++) {
    std::cout << "Stage " << (i) << ": ";
        float sum = 0;
        for(float &t : timings[i]) {
            sum += t;
        }
        if(timings[i].size() > 0) {
            std::cout << sum / timings[i].size() << " ms average over " << timings[i].size() << " images" << std::endl;
        } else {
            std::cout << "No data available" << std::endl;
        }
    }

}

int main() {
   
    std::cout << "checkpointing Edge Detection Pipeline..." << std::endl;
    std::string img_folder = "C:\\Users\\dferrario\\EdgeDetectionPipeline\\UDED\\imgs_bmp";
    std::string output_folder = "..\\output_images";
    std::vector<std::string> images_paths;
    utils::get_images_paths(img_folder, images_paths);
    
    runPipeline(images_paths, output_folder);
    return 0;
}