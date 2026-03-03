/**
 * @file utils.hpp
 * @brief Utility functions for the edge detection pipeline.
 * 
 * Provides helper functions for:
 * - CUDA error checking
 * - Image I/O operations
 * - Image processing utilities (format conversion, overlay)
 * - Performance statistics output
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <algorithm>

// ============================================================================
// CUDA Error Checking
// ============================================================================

/**
 * @brief CUDA error checking macro.
 * 
 * Wraps CUDA API calls and aborts on failure with file/line information.
 * Usage: ck(cudaMalloc(...));
 */
#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * @brief CUDA error assertion helper.
 * @param code   CUDA error code to check
 * @param file   Source file where the call was made
 * @param line   Line number of the call
 * @param abort  If true, exit program on error
 */
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) 
                  << " at " << file << ":" << line << std::endl;
        if (abort) {
            exit(code);
        }
    }
}

// ============================================================================
// Utility Namespace
// ============================================================================

namespace utils {

// ----------------------------------------------------------------------------
// Image I/O Functions
// ----------------------------------------------------------------------------

/**
 * @brief Load an image from disk into pinned host memory.
 * 
 * Loads the image using stb_image (must be included before this header),
 * extracts grayscale (from red channel) and BGR data, using CUDA pinned
 * memory for efficient host-to-device transfers.
 * 
 * @param[in]  path       Path to the image file
 * @param[out] data_mono  Pinned memory for grayscale data
 * @param[out] data_bgr   Pinned memory for BGR color data  
 * @param[out] width      Image width in pixels
 * @param[out] height     Image height in pixels
 */
inline void load_image(const std::string& path, uint8_t*& data_mono, 
                       uint8_t*& data_bgr, int& width, int& height) {
    int channels;
    unsigned char* raw_data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    
    if (!raw_data) {
        std::cerr << "Error: Failed to load image: " << path << std::endl;
        return;
    }

    const size_t num_pixels = static_cast<size_t>(width) * height;
    
    // Allocate pinned memory for efficient DMA transfers
    ck(cudaHostAlloc(&data_mono, num_pixels * sizeof(uint8_t), cudaHostAllocDefault));
    ck(cudaHostAlloc(&data_bgr, num_pixels * 3 * sizeof(uint8_t), cudaHostAllocDefault));

    // Extract grayscale (red channel) and copy BGR data
    for (size_t i = 0; i < num_pixels; i++) {
        data_mono[i] = raw_data[i * channels];     // Grayscale from red channel
        data_bgr[i * 3 + 0] = raw_data[i * channels + 0];  // Blue
        data_bgr[i * 3 + 1] = raw_data[i * channels + 1];  // Green
        data_bgr[i * 3 + 2] = raw_data[i * channels + 2];  // Red
    }

    stbi_image_free(raw_data);
}

/**
 * @brief Save an image to disk as BMP format.
 * 
 * @param path     Output file path
 * @param data     Image data buffer
 * @param width    Image width
 * @param height   Image height
 * @param channels Number of color channels (1 or 3)
 */
inline void save_image(const std::string& path, uint8_t* data, 
                       int width, int height, int channels) {
    if (!stbi_write_bmp(path.c_str(), width, height, channels, data)) {
        std::cerr << "Error: Failed to save image: " << path << std::endl;
    }
}

/**
 * @brief Get all image file paths from a directory.
 * 
 * @param[in]  folder_path  Directory to scan
 * @param[out] img_paths    Vector to populate with file paths
 */
namespace fs = std::filesystem;
inline void get_images_paths(const std::string& folder_path, 
                             std::vector<std::string>& img_paths) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        img_paths.push_back(entry.path().string());
    }
}

// ----------------------------------------------------------------------------
// Image Processing Functions
// ----------------------------------------------------------------------------

/**
 * @brief Convert grayscale/binary image to BGR format (red channel only).
 * 
 * Maps grayscale values to the blue channel, leaving green and red as 0.
 * Used for creating colored overlays of edge detection results.
 * 
 * @param binaryData   Input grayscale/binary data
 * @param bgrData      Output BGR buffer (must be pre-allocated, 3x size of input)
 * @param totalPixels  Number of pixels to process
 */
inline void mono_to_bgr(uint8_t* binaryData, uint8_t* bgrData, size_t totalPixels) {
    for (size_t i = 0; i < totalPixels; ++i) {
        const uint8_t pixelValue = binaryData[i];
        bgrData[i * 3 + 0] = pixelValue;  // Blue channel gets the edge value
        bgrData[i * 3 + 1] = 0;           // Green = 0
        bgrData[i * 3 + 2] = 0;           // Red = 0
    }
}

/**
 * @brief Overlay two images using maximum blending.
 * 
 * For each pixel, takes the maximum value from img1 and img2.
 * Used to superimpose edge detection results on original image.
 * 
 * @param img1      First image (typically original)
 * @param img2      Second image (typically edges)
 * @param output    Output buffer (can be same as img1 for in-place)
 * @param width     Image width
 * @param height    Image height
 * @param channels  Number of channels (typically 3 for BGR)
 */
inline void overlap_images(uint8_t* img1, uint8_t* img2, uint8_t* output, 
                           int width, int height, int channels) {
    const size_t totalBytes = static_cast<size_t>(width) * height * channels;
    for (size_t i = 0; i < totalBytes; ++i) {
        output[i] = std::max(img1[i], img2[i]);
    }
}

// ----------------------------------------------------------------------------
// Statistics and Reporting
// ----------------------------------------------------------------------------

/**
 * @brief Save per-image processing statistics to a text file.
 * 
 * Outputs image metadata and per-stage timing information.
 * 
 * @param stats_path    Output file path
 * @param source_path   Original image path
 * @param width         Image width
 * @param height        Image height
 * @param stage_times   Array of 5 timing values (one per stage, in ms)
 */
inline void save_image_stats(const std::string& stats_path, 
                             const std::string& source_path, 
                             int width, int height, 
                             const float stage_times[5]) {
    const char* STAGE_NAMES[] = {"Load+Transfer", "Gaussian", "Laplacian", 
                                  "Binarization", "Save"};
    
    float total_time = 0.0f;
    for (int s = 0; s < 5; s++) {
        total_time += stage_times[s];
    }

    std::ofstream file(stats_path);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to write stats file: " << stats_path << std::endl;
        return;
    }

    // Header
    file << "Image Statistics\n";
    file << "================\n";
    file << "Source: " << source_path << "\n";
    file << "Dimensions: " << width << "x" << height 
         << " (" << (width * height) << " pixels)\n\n";

    // Timing information
    file << "Stage Timings (ms):\n";
    file << "-------------------\n";
    for (int s = 0; s < 5; s++) {
        file << "  " << std::setw(14) << std::left << STAGE_NAMES[s] 
             << ": " << std::fixed << std::setprecision(3) << stage_times[s] << " ms\n";
    }
    file << "-------------------\n";
    file << "  " << std::setw(14) << std::left << "Total" 
         << ": " << std::fixed << std::setprecision(3) << total_time << " ms\n";
    
    file.close();
}

// ----------------------------------------------------------------------------
// Legacy/Deprecated Functions
// ----------------------------------------------------------------------------

/**
 * @brief Profile elapsed time between two CUDA events.
 * @deprecated Use per-image event pairs instead for accurate stream-based timing.
 */
[[deprecated("Use per-image events for accurate timing")]]
inline void profile_previous_step(cudaEvent_t start, std::vector<float>& timings) {
    cudaEvent_t end;
    cudaEventCreate(&end);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);
    timings.push_back(elapsed);
    cudaEventDestroy(end);
}

} // namespace utils

#endif // UTILS_HPP