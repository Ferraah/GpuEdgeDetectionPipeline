#ifndef PIPELINE_HPP
#define PIPELINE_HPP

constexpr int PIPELINE_DEPTH = 4; // one slot per stage

struct PipelineSlot {
    // Host pinned memory (required for async memcpy)
    uint8_t* h_input;      // pinned, grayscale
    uint8_t* h_input_rgb;  // pinned, RGB for overlay
    uint8_t* h_output;     // pinned, final result
    size_t width, height;
    
    // Device buffers
    uint8_t* d_input;
    uint8_t* d_gaussian;   // output of stage 1
    uint8_t* d_laplacian;  // output of stage 2
    uint8_t* d_output;     // output of stage 3

    // One stream per slot
    cudaStream_t stream;

    // Events to chain stages
    cudaEvent_t loaded;    // H2D transfer done
    cudaEvent_t gaussian_done;
    cudaEvent_t laplacian_done;
    cudaEvent_t done;      // D2H transfer done
};


void 

#endif // PIPELINE_HPP