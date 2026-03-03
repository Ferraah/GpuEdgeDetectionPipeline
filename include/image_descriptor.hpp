#ifndef PIPELINE_HPP
#define PIPELINE_HPP
#include <cuda_runtime.h>

struct ImageDescriptor {
    uint8_t* h_data_mono; // Grayscale data
    uint8_t* h_data_bgr;  // RGB data for overlay
    uint8_t* d_data_mono; // Device pointer to grayscale data
    uint8_t* d_data_mono_out; // Second buffer for double-buffering (avoids in-place convolution race)

    cudaStream_t stream;   // Stream for this image's operations
    cudaEvent_t loaded;    // Event for when the image is loaded to device

    // Per-stage timing events (recorded on this image's stream)
    cudaEvent_t stage_start[5];  // Start events for: Load, Gaussian, Laplacian, Binarize, Save
    cudaEvent_t stage_end[5];    // End events for each stage

    std::string path;
    int width;
    int height;
};


#endif // PIPELINE_HPP