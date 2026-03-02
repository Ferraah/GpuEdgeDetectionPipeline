#ifndef PIPELINE_HPP
#define PIPELINE_HPP
#include <cstdint>
#include <cuda_runtime.h>

struct ImageDescriptor {
    uint8_t* h_data_mono; // Grayscale data
    uint8_t* h_data_bgr;  // RGB data for overlay
    uint8_t* d_data_mono; // Device pointer to grayscale data

    cudaStream_t stream;   // Stream for this image's operations
    cudaEvent_t loaded;    // Event for when the image is loaded to device

    std::string path;
    int width;
    int height;
};


#endif // PIPELINE_HPP