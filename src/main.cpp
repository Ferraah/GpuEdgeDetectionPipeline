#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <array>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include "image_descriptor.hpp"
#include "kernel.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"


// Thread-safe timing storage for profiling
struct PipelineStats {
    std::vector<std::vector<float>> stage_timings; // [stage][measurements]
    omp_lock_t lock;
    
    PipelineStats() {
        stage_timings.resize(5);
        omp_init_lock(&lock);
    }
    
    ~PipelineStats() {
        omp_destroy_lock(&lock);
    }
    
    void add_timing(int stage, float ms) {
        omp_set_lock(&lock);
        stage_timings[stage].push_back(ms);
        omp_unset_lock(&lock);
    }
    
    void print_stats() {
        const char* stage_names[] = {"Load", "Gaussian", "Laplacian", "Binarize", "Save"};
        std::cout << "\n=============== Pipeline Performance Statistics ===============\n";
        
        for(int i = 0; i < 5; i++) {
            if(stage_timings[i].empty()) {
                std::cout << stage_names[i] << ": No data available\n";
                continue;
            }
            
            float sum = 0, min_val = stage_timings[i][0], max_val = stage_timings[i][0];
            for(float t : stage_timings[i]) {
                sum += t;
                min_val = std::min(min_val, t);
                max_val = std::max(max_val, t);
            }
            
            float avg = sum / stage_timings[i].size();
            std::cout << stage_names[i] << ": " << avg << " ms avg"
                      << " (min: " << min_val << " ms, max: " << max_val << " ms)"
                      << " over " << stage_timings[i].size() << " images\n";
        }
        std::cout << "=============================================================\n";
    }
};

void runPipeline(std::vector<std::string>& images_paths, const std::string& output_folder) {
    
    size_t N_IMAGES = images_paths.size();
    std::cout << "Running OpenMP parallel pipeline with " << N_IMAGES << " images\n";
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";
    
    // Kernels (shared across all threads)
    const float kernel[9] = { 0.0625, 0.125, 0.0625,
                              0.125,  0.25,  0.125,
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

    PipelineStats stats;
    
    auto global_start = std::chrono::high_resolution_clock::now();

    // Process all images in parallel - each thread handles full pipeline for one image
    #pragma omp parallel for schedule(dynamic)
    for(int idx = 0; idx < (int)N_IMAGES; idx++) {
        ImageDescriptor img;
        img.path = images_paths[idx];
        
        auto t_start = std::chrono::high_resolution_clock::now();
        auto t_end = t_start;
        
        // Stage 0: Load
        t_start = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::cout << "[Thread " << omp_get_thread_num() << "] Loading image " << idx << ": " << img.path << "\n";
        }
        utils::load_image(images_paths[idx], img.h_data_mono, img.h_data_bgr, img.width, img.height);

        ck(cudaStreamCreate(&img.stream));
        ck(cudaMalloc(&img.d_data_mono, img.width * img.height * sizeof(uint8_t)));
        ck(cudaMemcpyAsync(img.d_data_mono, img.h_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyHostToDevice, img.stream));
        ck(cudaStreamSynchronize(img.stream));
        
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(0, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // Stage 1: Gaussian
        t_start = std::chrono::high_resolution_clock::now();
        launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_gaussian, kernel_size, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(1, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // Stage 2: Laplacian
        t_start = std::chrono::high_resolution_clock::now();
        launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_laplacian, kernel_size, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(2, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // Stage 3: Binarization
        t_start = std::chrono::high_resolution_clock::now();
        launchBinarizeImageAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, 16, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(3, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // Stage 4: Save
        t_start = std::chrono::high_resolution_clock::now();
        
        ck(cudaMemcpyAsync(img.h_data_mono, img.d_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyDeviceToHost, img.stream));
        ck(cudaStreamSynchronize(img.stream));


        uint8_t* h_data_rgb_new = new uint8_t[img.width * img.height * 3];
        utils::mono_to_bgr(img.h_data_mono, h_data_rgb_new, img.width * img.height);
        utils::overlap_images(img.h_data_bgr, h_data_rgb_new, h_data_rgb_new, img.width, img.height, 3);
        utils::save_image(output_folder + "/output_image_" + std::to_string(idx) + ".bmp", h_data_rgb_new, img.width, img.height, 3);
        // utils::save_image(output_folder + "/output_image_" + std::to_string(idx) + ".bmp", img.h_data_mono, img.width, img.height, 1);
        
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(4, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // Cleanup
        cudaFreeHost(img.h_data_mono);
        cudaFreeHost(img.h_data_bgr);
        cudaFree(img.d_data_mono);
        ck(cudaStreamDestroy(img.stream));
        delete[] h_data_rgb_new;
        
        #pragma omp critical
        {
            std::cout << "[Thread " << omp_get_thread_num() << "] Completed image " << idx << "\n";
        }
    }

    cudaFree(d_kernel_gaussian);
    cudaFree(d_kernel_laplacian);
    
    auto global_end = std::chrono::high_resolution_clock::now();
    float global_time_ms = std::chrono::duration<float, std::milli>(global_end - global_start).count();
    
    stats.print_stats();
    std::cout << "Total pipeline time: " << global_time_ms << " ms (" << global_time_ms / 1000.0f << " s)\n";
    std::cout << "Throughput: " << (N_IMAGES / (global_time_ms / 1000.0f)) << " images/sec\n";
}

int main(int argc, char* argv[]) {
    
    // Get number of threads from command line (default: max available)
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
        if (num_threads <= 0) {
            std::cerr << "Invalid number of threads. Using default.\n";
            num_threads = omp_get_max_threads();
        }
    }
    omp_set_num_threads(num_threads);
    
    std::cout << "Edge Detection Pipeline\n";
    std::cout << "Usage: " << argv[0] << " [num_threads]\n\n";
    std::string img_folder = "C:\\Users\\dferrario\\EdgeDetectionPipeline\\UDED\\imgs_bmp";
    std::string output_folder = "..\\output_images";
    std::vector<std::string> images_paths;
    utils::get_images_paths(img_folder, images_paths);
    
    runPipeline(images_paths, output_folder);
    return 0;
}