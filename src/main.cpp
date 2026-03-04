/**
 * @file main.cpp
 * @brief GPU-accelerated Edge Detection Pipeline using CUDA.
 * 
 * This application implements a pipelined edge detection system that processes
 * multiple images concurrently using CUDA streams. The pipeline consists of:
 * 
 * 1. **Load/Transfer**: Load image from disk and transfer to GPU
 * 2. **Gaussian Filter**: Apply smoothing to reduce noise
 * 3. **Laplacian Filter**: Detect edges using second-order derivative
 * 4. **Binarization**: Threshold the edge response
 * 5. **Save**: Overlay edges on original image and save result
 * 
 * Each image is assigned its own CUDA stream, enabling concurrent execution
 * of different stages across multiple images for improved throughput.
 * 
 * @author Edge Detection Pipeline Team
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <array>
#include <string>

#include "image_descriptor.hpp"
#include "kernel.cuh"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Binarization threshold for edge detection (0-255)
constexpr uint8_t BINARIZE_THRESHOLD = 5;

/// Convolution kernel size (3x3)
constexpr int KERNEL_SIZE = 3;

/// Gaussian smoothing kernel (normalized 3x3)
const float GAUSSIAN_KERNEL[9] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
};

/// Laplacian edge detection kernel (4-connectivity)
const float LAPLACIAN_KERNEL[9] = {
    0.0f,  1.0f, 0.0f,
    1.0f, -4.0f, 1.0f,
    0.0f,  1.0f, 0.0f
};

// ============================================================================
// Pipeline Stage Functions
// ============================================================================

/**
 * @brief Execute the Load stage - transfer pre-loaded image to GPU.
 * 
 * @param img   Image descriptor with data already loaded in h_data_mono/h_data_bgr
 */
void executeLoadStage(ImageDescriptor& img) {
    std::cout << "  [Transfer] " << img.path << std::endl;
    
    // Initialize CUDA resources for this image
    ck(img.initCudaResources());
    
    // Start timing for GPU transfer
    ck(cudaEventRecord(img.stage_start[STAGE_LOAD], img.stream));
    
    // Allocate device memory with double-buffering
    const size_t mono_size = static_cast<size_t>(img.width) * img.height * sizeof(uint8_t);
    ck(cudaMalloc(&img.d_data_mono, mono_size));
    ck(cudaMalloc(&img.d_data_mono_out, mono_size));
    
    // Async transfer to device
    ck(cudaMemcpyAsync(img.d_data_mono, img.h_data_mono, mono_size, 
                       cudaMemcpyHostToDevice, img.stream));
    
    ck(cudaEventRecord(img.stage_end[STAGE_LOAD], img.stream));
}

/**
 * @brief Execute the Gaussian filter stage.
 * 
 * @param img        Image descriptor
 * @param d_kernel   Device pointer to Gaussian kernel
 */
void executeGaussianStage(ImageDescriptor& img, float* d_kernel) {
    std::cout << "  [Gaussian] " << img.path << std::endl;
    
    ck(cudaEventRecord(img.stage_start[STAGE_GAUSSIAN], img.stream));
    
    launchConvolutionAsync(img.d_data_mono, img.d_data_mono_out, 
                           img.width, img.height, d_kernel, KERNEL_SIZE, img.stream);
    std::swap(img.d_data_mono, img.d_data_mono_out);  // Result now in d_data_mono
    
    ck(cudaEventRecord(img.stage_end[STAGE_GAUSSIAN], img.stream));
}

/**
 * @brief Execute the Laplacian filter stage.
 * 
 * @param img        Image descriptor
 * @param d_kernel   Device pointer to Laplacian kernel
 */
void executeLaplacianStage(ImageDescriptor& img, float* d_kernel) {
    std::cout << "  [Laplacian] " << img.path << std::endl;
    
    ck(cudaEventRecord(img.stage_start[STAGE_LAPLACIAN], img.stream));
    
    launchConvolutionAsync(img.d_data_mono, img.d_data_mono_out, 
                           img.width, img.height, d_kernel, KERNEL_SIZE, img.stream);
    std::swap(img.d_data_mono, img.d_data_mono_out);
    
    ck(cudaEventRecord(img.stage_end[STAGE_LAPLACIAN], img.stream));
}

/**
 * @brief Execute the Binarization stage.
 * 
 * @param img   Image descriptor
 */
void executeBinarizeStage(ImageDescriptor& img) {
    std::cout << "  [Binarize] " << img.path << std::endl;
    
    ck(cudaEventRecord(img.stage_start[STAGE_BINARIZE], img.stream));
    
    // Binarization is safe in-place (each thread accesses only its own pixel)
    launchBinarizeImageAsync(img.d_data_mono, img.d_data_mono, 
                             img.width, img.height, BINARIZE_THRESHOLD, img.stream);
    
    // Transfer result back to host
    const size_t mono_size = static_cast<size_t>(img.width) * img.height * sizeof(uint8_t);
    ck(cudaMemcpyAsync(img.h_data_mono, img.d_data_mono, mono_size, 
                       cudaMemcpyDeviceToHost, img.stream));
    
    ck(cudaEventRecord(img.stage_end[STAGE_BINARIZE], img.stream));
}

/**
 * @brief Execute the Save stage - create overlay and write to disk.
 * 
 * @param img            Image descriptor
 * @param output_path    Path for the output image file
 * @param stats_path     Path for the statistics file
 * @param timings        Reference to timing storage arrays
 */
void executeVisualizationStage(ImageDescriptor& img, 
                      const std::string& output_path,
                      const std::string& stats_path,
                      std::array<std::vector<float>, NUM_PIPELINE_STAGES>& timings) {
    // Wait for GPU work to complete before reading h_data_mono
    ck(cudaStreamSynchronize(img.stream));
    
    ck(cudaEventRecord(img.stage_start[STAGE_SAVE], img.stream));
    std::cout << "  [Visualize] " << img.path << std::endl;
    
    const size_t num_pixels = static_cast<size_t>(img.width) * img.height;
    
    // Create RGB overlay buffer
    uint8_t* h_data_overlay = new uint8_t[num_pixels * 3];
    
    // Convert binary edges to BGR (edges appear as blue)
    utils::mono_to_bgr(img.h_data_mono, h_data_overlay, num_pixels);
    
    // Overlay edges on original image using max blending
    utils::overlap_images(img.h_data_bgr, h_data_overlay, img.h_data_bgr, 
                          img.width, img.height, 3);
    
    ck(cudaEventRecord(img.stage_end[STAGE_SAVE], img.stream));
    
    // Store output path for later saving
    img.output_path = output_path;
    
    // Collect timing data for all stages
    float stage_times[NUM_PIPELINE_STAGES];
    for (int s = 0; s < NUM_PIPELINE_STAGES; s++) {
        ck(cudaEventSynchronize(img.stage_end[s]));
        ck(cudaEventElapsedTime(&stage_times[s], img.stage_start[s], img.stage_end[s]));
        timings[s].push_back(stage_times[s]);
    }
    
    // Write statistics file
    utils::save_image_stats(stats_path, img.path, img.width, img.height, stage_times);
    
    // Cleanup overlay buffer
    delete[] h_data_overlay;
}

/**
 * @brief Print aggregate pipeline performance statistics.
 * 
 * @param timings   Array of timing vectors for each stage
 */
void printPipelineStatistics(const std::array<std::vector<float>, NUM_PIPELINE_STAGES>& timings) {
    std::cout << "\n========================================\n";
    std::cout << "Pipeline Performance Statistics\n";
    std::cout << "========================================\n";
    
    float total_avg = 0.0f;
    
    for (int i = 0; i < NUM_PIPELINE_STAGES; i++) {
        std::cout << "  " << STAGE_NAMES[i] << ": ";
        
        if (!timings[i].empty()) {
            float sum = 0.0f;
            for (const float& t : timings[i]) {
                sum += t;
            }
            const float avg = sum / timings[i].size();
            total_avg += avg;
            std::cout << avg << " ms avg (" << timings[i].size() << " images)\n";
        } else {
            std::cout << "No data\n";
        }
    }
    
    std::cout << "----------------------------------------\n";
    std::cout << "  Total: " << total_avg << " ms avg per image\n";
    std::cout << "========================================\n";
}

// ============================================================================
// Main Pipeline Function
// ============================================================================

/**
 * @brief Run the edge detection pipeline on a batch of images.
 * 
 * Implements a software pipeline where each image progresses through stages
 * independently, with different images in different stages at any given time.
 * This maximizes GPU utilization through concurrent kernel execution.
 * 
 * Pipeline timing diagram (for 3 images):
 * ```
 * Tick:   0    1    2    3    4    5    6    7
 * Img0: [Load][Gaus][Lapl][Bina][Save]
 * Img1:      [Load][Gaus][Lapl][Bina][Save]
 * Img2:           [Load][Gaus][Lapl][Bina][Save]
 * ```
 * 
 * @param images_paths    Vector of input image file paths
 * @param output_folder   Directory for output files
 */
void runPipeline(std::vector<std::string>& images_paths, const std::string& output_folder) {
    const size_t num_images = images_paths.size();
    
    std::cout << "\n========================================\n";
    std::cout << "Edge Detection Pipeline\n";
    std::cout << "========================================\n";
    std::cout << "Input images: " << num_images << "\n";
    std::cout << "Output folder: " << output_folder << "\n";
    std::cout << "========================================\n\n";
    
    // Pre-load all images into CPU memory
    std::cout << "--- Pre-loading all images ---\n";
    std::vector<ImageDescriptor> images(num_images);
    for (size_t i = 0; i < num_images; i++) {
        images[i].path = images_paths[i];
        utils::load_image(images_paths[i], images[i].h_data_mono, images[i].h_data_bgr, 
                          images[i].width, images[i].height);
        std::cout << "  [" << (i + 1) << "/" << num_images << "] " << images_paths[i] << std::endl;
    }
    std::cout << std::endl;
    
    // Timing storage for each stage
    std::array<std::vector<float>, NUM_PIPELINE_STAGES> timings;
    
    // Allocate and upload convolution kernels to device
    float* d_kernel_gaussian = nullptr;
    float* d_kernel_laplacian = nullptr;
    const size_t kernel_bytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    
    ck(cudaMalloc(&d_kernel_gaussian, kernel_bytes));
    ck(cudaMalloc(&d_kernel_laplacian, kernel_bytes));
    ck(cudaMemcpy(d_kernel_gaussian, GAUSSIAN_KERNEL, kernel_bytes, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_kernel_laplacian, LAPLACIAN_KERNEL, kernel_bytes, cudaMemcpyHostToDevice));
    
    // Pipeline execution loop
    // Total ticks = num_images + (NUM_PIPELINE_STAGES - 1) to drain pipeline
    const size_t total_ticks = num_images + NUM_PIPELINE_STAGES - 1;
    
    for (size_t tick = 0; tick < total_ticks; tick++) {
        std::cout << "--- Pipeline Tick " << tick << " ---\n";
        
        // Stage 0: Transfer pre-loaded image to GPU
        if (tick < num_images) {
            executeLoadStage(images[tick]);
        }
        
        // Stage 1: Gaussian filter (image loaded 1 tick ago)
        if (tick >= 1 && tick - 1 < num_images) {
            executeGaussianStage(images[tick - 1], d_kernel_gaussian);
        }
        
        // Stage 2: Laplacian filter (image loaded 2 ticks ago)
        if (tick >= 2 && tick - 2 < num_images) {
            executeLaplacianStage(images[tick - 2], d_kernel_laplacian);
        }
        
        // Stage 3: Binarization (image loaded 3 ticks ago)
        if (tick >= 3 && tick - 3 < num_images) {
            executeBinarizeStage(images[tick - 3]);
        }
        
        // Stage 4: Save result (image loaded 4 ticks ago)
        if (tick >= 4 && tick - 4 < num_images) {
            const size_t img_idx = tick - 4;
            const std::string output_path = output_folder + "/output_image_" + 
                                            std::to_string(img_idx) + ".bmp";
            const std::string stats_path = output_folder + "/output_image_" + 
                                           std::to_string(img_idx) + "_stats.txt";
            executeVisualizationStage(images[img_idx], output_path, stats_path, timings);
        }
        
        std::cout << std::endl;
    }
    
    // Cleanup device kernels
    cudaFree(d_kernel_gaussian);
    cudaFree(d_kernel_laplacian);
    
    // Save all images at the end of the pipeline
    std::cout << "--- Saving All Images ---\n";
    for (size_t i = 0; i < images.size(); i++) {
        std::cout << "  [Save] " << images[i].output_path << std::endl;
        utils::save_image(images[i].output_path, images[i].h_data_bgr, 
                          images[i].width, images[i].height, 3);
        images[i].cleanup();
    }
    std::cout << std::endl;
    
    // Print aggregate statistics
    printPipelineStatistics(timings);
}

// ============================================================================
// Entry Point
// ============================================================================

/**
 * @brief Application entry point.
 * 
 * Configures input/output paths and launches the edge detection pipeline.
 */
int main() {
    std::cout << "GPU Edge Detection Pipeline\n";
    std::cout << "===========================\n\n";
    
    // Configuration
    const std::string input_folder = "C:\\Users\\dferrario\\EdgeDetectionPipeline\\input_images";
    const std::string output_folder = "..\\output_images";
    
    // Gather input images
    std::vector<std::string> image_paths;
    utils::get_images_paths(input_folder, image_paths);
    
    if (image_paths.empty()) {
        std::cerr << "Error: No images found in " << input_folder << std::endl;
        return 1;
    }
    
    // Run the pipeline
    runPipeline(image_paths, output_folder);
    
    std::cout << "\nPipeline completed successfully.\n";
    return 0;
}