/**
 * @file image_descriptor.hpp
 * @brief Data structure for managing image data throughout the edge detection pipeline.
 * 
 * This header defines the ImageDescriptor struct which holds all data associated with
 * a single image as it flows through the GPU-accelerated pipeline stages.
 */

#ifndef IMAGE_DESCRIPTOR_HPP
#define IMAGE_DESCRIPTOR_HPP

#include <cuda_runtime.h>
#include <string>
#include <cstdint>

/**
 * @brief Number of processing stages in the pipeline.
 * 
 * Stages: Load/Transfer, Gaussian, Laplacian, Binarization, Save
 */
constexpr int NUM_PIPELINE_STAGES = 5;

/**
 * @brief Stage indices for the pipeline.
 */
enum PipelineStage {
    STAGE_LOAD = 0,        ///< Image loading and host-to-device transfer
    STAGE_GAUSSIAN = 1,    ///< Gaussian blur filtering
    STAGE_LAPLACIAN = 2,   ///< Laplacian edge detection
    STAGE_BINARIZE = 3,    ///< Threshold binarization
    STAGE_SAVE = 4         ///< Result overlay and file output
};

/**
 * @brief Human-readable names for each pipeline stage.
 */
inline const char* STAGE_NAMES[NUM_PIPELINE_STAGES] = {
    "Load (to GPU)",
    "Gaussian",
    "Laplacian", 
    "Binarization",
    "Visualization"
};

/**
 * @brief Descriptor holding all data for a single image in the pipeline.
 * 
 * Each image gets its own CUDA stream for asynchronous processing,
 * double-buffered device memory to avoid race conditions during convolution,
 * and timing events to measure per-stage performance.
 */
struct ImageDescriptor {
    // ===== Host Memory =====
    uint8_t* h_data_mono;     ///< Pinned host memory for grayscale data
    uint8_t* h_data_bgr;      ///< Pinned host memory for BGR color data (overlay output)

    // ===== Device Memory =====
    uint8_t* d_data_mono;     ///< Primary device buffer for processing
    uint8_t* d_data_mono_out; ///< Secondary buffer for double-buffering (avoids in-place convolution race)

    // ===== CUDA Resources =====
    cudaStream_t stream;      ///< Dedicated CUDA stream for this image's operations
    cudaEvent_t loaded;       ///< Event marking when image data is on device (legacy, kept for compatibility)

    // ===== Timing Events =====
    cudaEvent_t stage_start[NUM_PIPELINE_STAGES]; ///< Start timestamp for each pipeline stage
    cudaEvent_t stage_end[NUM_PIPELINE_STAGES];   ///< End timestamp for each pipeline stage

    // ===== Metadata =====
    std::string path;         ///< Source file path
    std::string output_path;  ///< Destination file path for saving
    int width;                ///< Image width in pixels
    int height;               ///< Image height in pixels

    /**
     * @brief Initialize CUDA resources (stream and timing events).
     * @return cudaSuccess on success, error code otherwise.
     */
    cudaError_t initCudaResources() {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) return err;

        for (int s = 0; s < NUM_PIPELINE_STAGES; s++) {
            err = cudaEventCreate(&stage_start[s]);
            if (err != cudaSuccess) return err;
            err = cudaEventCreate(&stage_end[s]);
            if (err != cudaSuccess) return err;
        }
        return cudaSuccess;
    }

    /**
     * @brief Release all CUDA and host resources.
     */
    void cleanup() {
        // Destroy timing events
        for (int s = 0; s < NUM_PIPELINE_STAGES; s++) {
            cudaEventDestroy(stage_start[s]);
            cudaEventDestroy(stage_end[s]);
        }

        // Free host pinned memory
        if (h_data_mono) cudaFreeHost(h_data_mono);
        if (h_data_bgr) cudaFreeHost(h_data_bgr);

        // Free device memory
        if (d_data_mono) cudaFree(d_data_mono);
        if (d_data_mono_out) cudaFree(d_data_mono_out);

        // Destroy stream
        cudaStreamDestroy(stream);
    }
};

#endif // IMAGE_DESCRIPTOR_HPP