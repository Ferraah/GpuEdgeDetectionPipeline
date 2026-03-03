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


// ============================================================================
// Pipeline Statistics Structure
// Thread-safe storage and reporting of execution times for each pipeline stage
// ============================================================================
struct PipelineStats {
    std::vector<std::vector<float>> stage_timings; // [stage][measurements]
    omp_lock_t lock;                               // OpenMP lock for thread-safe access
    
    /**
     * Constructor: Initialize stages and OpenMP lock
     */
    PipelineStats() {
        stage_timings.resize(5); // 5 stages: Load, Gaussian, Laplacian, Binarize, Save
        omp_init_lock(&lock);
    }
    
    /**
     * Destructor: Clean up OpenMP lock
     */
    ~PipelineStats() {
        omp_destroy_lock(&lock);
    }
    
    /**
     * @brief Add timing measurement for a pipeline stage (thread-safe)
     * @param stage Stage index (0-4)
     * @param ms Execution time in milliseconds
     */
    void add_timing(int stage, float ms) {
        omp_set_lock(&lock);
        stage_timings[stage].push_back(ms);
        omp_unset_lock(&lock);
    }
    
    /**
     * @brief Print performance statistics for all stages
     * Displays average, min, and max times for each stage across all images
     */
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

// ============================================================================
// Main Pipeline Function
// Processes images in parallel using OpenMP, with CUDA GPU acceleration
// ============================================================================
/**
 * @brief Execute the edge detection pipeline on all images
 * 
 * Pipeline stages:
 *   0. Load: Read image from disk and transfer to GPU
 *   1. Gaussian blur: Smooth image to reduce noise
 *   2. Laplacian: Apply edge detection filter
 *   3. Binarization: Convert to binary (threshold-based)
 *   4. Save: Overlay edges on original and write to disk
 * 
 * @param images_paths Vector of paths to input images
 * @param output_folder Directory where output images will be saved
 * 
 * Execution is parallelized using OpenMP: each thread processes one complete
 * image through all 5 stages. GPU operations use CUDA streams for efficiency.
 */
void runPipeline(std::vector<std::string>& images_paths, const std::string& output_folder) {
    
    size_t N_IMAGES = images_paths.size();
    std::cout << "Running OpenMP parallel pipeline with " << N_IMAGES << " images\n";
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";
    
    // Define and allocate convolution kernels
    // Gaussian kernel: 3x3 Gaussian blur filter
    const float kernel[9] = { 0.0625, 0.125, 0.0625,
                              0.125,  0.25,  0.125,
                              0.0625, 0.125, 0.0625 };
    // Laplacian kernel: 3x3 edge detection filter
    const float laplacian_kernel[9] = { 0,  1, 0,
                                        1, -4, 1,
                                        0,  1, 0 };
    const int kernel_size = 3;

    // Allocate and copy kernels to GPU device memory
    float* d_kernel_gaussian, *d_kernel_laplacian;
    ck(cudaMalloc(&d_kernel_gaussian, kernel_size * kernel_size * sizeof(float)));
    ck(cudaMalloc(&d_kernel_laplacian, kernel_size * kernel_size * sizeof(float)));
    ck(cudaMemcpy(d_kernel_gaussian, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_kernel_laplacian, laplacian_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize profiling statistics
    PipelineStats stats;
    
    // Record global start time for total pipeline execution
    auto global_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Main processing loop - Parallel over all images
    // ========================================================================
    // OpenMP parallel loop: schedule(dynamic) allows dynamic work distribution
    #pragma omp parallel for schedule(dynamic)
    for(int idx = 0; idx < (int)N_IMAGES; idx++) {
        ImageDescriptor img;
        img.path = images_paths[idx];
        
        auto t_start = std::chrono::high_resolution_clock::now();
        auto t_end = t_start;
        
        // ====================================================================
        // Stage 0: Load - Read image from disk and transfer to GPU
        // ====================================================================
        t_start = std::chrono::high_resolution_clock::now();
        #pragma omp critical
        {
            std::cout << "[Thread " << omp_get_thread_num() << "] Loading image " << idx << ": " << img.path << "\n";
        }
        // Load image in grayscale and BGR formats
        utils::load_image(images_paths[idx], img.h_data_mono, img.h_data_bgr, img.width, img.height);

        // Create CUDA stream for asynchronous GPU operations
        ck(cudaStreamCreate(&img.stream));
        // Allocate GPU memory for grayscale image data
        ck(cudaMalloc(&img.d_data_mono, img.width * img.height * sizeof(uint8_t)));
        // Asynchronously copy image to GPU
        ck(cudaMemcpyAsync(img.d_data_mono, img.h_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyHostToDevice, img.stream));
        ck(cudaStreamSynchronize(img.stream));
        
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(0, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // ====================================================================
        // Stage 1: Gaussian Blur - Smooth image to reduce noise
        // ====================================================================
        t_start = std::chrono::high_resolution_clock::now();
        launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_gaussian, kernel_size, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(1, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // ====================================================================
        // Stage 2: Laplacian Filter - Detect edges
        // ====================================================================
        t_start = std::chrono::high_resolution_clock::now();
        launchConvolutionAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, d_kernel_laplacian, kernel_size, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(2, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // ====================================================================
        // Stage 3: Binarization - Convert to binary image (0 or 255)
        // ====================================================================
        int threshold = 16; // Threshold value for binarization (tuned for edge detection)
        t_start = std::chrono::high_resolution_clock::now();
        launchBinarizeImageAsync(img.d_data_mono, img.d_data_mono, img.width, img.height, threshold, img.stream);
        ck(cudaStreamSynchronize(img.stream));
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(3, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // ====================================================================
        // Stage 4: Save - Overlay edges on original and save to disk
        // ====================================================================
        t_start = std::chrono::high_resolution_clock::now();
        
        // Copy processed image back from GPU to host memory
        ck(cudaMemcpyAsync(img.h_data_mono, img.d_data_mono, img.width * img.height * sizeof(uint8_t), cudaMemcpyDeviceToHost, img.stream));
        ck(cudaStreamSynchronize(img.stream));

        // Allocate temporary buffer for RGB conversion
        uint8_t* h_data_rgb_new = new uint8_t[img.width * img.height * 3];
        // Convert binary edge image to BGR format (red edges)
        utils::mono_to_bgr(img.h_data_mono, h_data_rgb_new, img.width * img.height);
        // Overlay edges on original image
        utils::overlap_images(img.h_data_bgr, h_data_rgb_new, h_data_rgb_new, img.width, img.height, 3);
        // Save output image to file
        utils::save_image(output_folder + "/output_image_" + std::to_string(idx) + ".bmp", h_data_rgb_new, img.width, img.height, 3);
        
        t_end = std::chrono::high_resolution_clock::now();
        stats.add_timing(4, std::chrono::duration<float, std::milli>(t_end - t_start).count());
        
        // ====================================================================
        // Cleanup - Free GPU and CPU memory
        // ====================================================================
        ck(cudaFreeHost(img.h_data_mono));
        ck(cudaFreeHost(img.h_data_bgr));
        ck(cudaFree(img.d_data_mono));
        ck(cudaStreamDestroy(img.stream));
        delete[] h_data_rgb_new;
        
        #pragma omp critical
        {
            std::cout << "[Thread " << omp_get_thread_num() << "] Completed image " << idx << "\n";
        }
    }

    // Free GPU memory used by kernels
    ck(cudaFree(d_kernel_gaussian));
    ck(cudaFree(d_kernel_laplacian));
    
    // Calculate and report total execution time
    auto global_end = std::chrono::high_resolution_clock::now();
    float global_time_ms = std::chrono::duration<float, std::milli>(global_end - global_start).count();
    
    stats.print_stats();
    std::cout << "Total pipeline time: " << global_time_ms << " ms (" << global_time_ms / 1000.0f << " s)\n";
    std::cout << "Throughput: " << (N_IMAGES / (global_time_ms / 1000.0f)) << " images/sec\n";
}

// ============================================================================
// Main Function - Entry Point
// ============================================================================
/**
 * @brief Application entry point
 * 
 * @param argc Number of command-line arguments
 * @param argv Command-line argument vector
 *   argv[1] (optional): Number of OpenMP threads to use
 *           If not provided, uses hardware concurrency
 * 
 * Example usage:
 *   EdgeDetectionPipeline.exe          # Use all available CPU cores
 *   EdgeDetectionPipeline.exe 4        # Use 4 threads
 */
int main(int argc, char* argv[]) {
    
    // ========================================================================
    // Parse command-line arguments for thread configuration
    // ========================================================================
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
    
    // Display usage information
    std::cout << "Edge Detection Pipeline\n";
    std::cout << "Usage: " << argv[0] << " [num_threads]\n\n";
    std::string img_folder = "C:\\Users\\dferrario\\EdgeDetectionPipeline\\UDED\\imgs_bmp";
    std::string output_folder = "..\\output_images";
    std::vector<std::string> images_paths;
    utils::get_images_paths(img_folder, images_paths);
    
    runPipeline(images_paths, output_folder);
    return 0;
}