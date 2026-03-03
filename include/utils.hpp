#ifndef UTILS_HPP
#define UTILS_HPP

#define ck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) 
                    << " " << file << " " << line << std::endl;
        if (abort) exit(code);
}
}


namespace utils {
    
    void profile_previous_step(cudaEvent_t start, std::vector<float> &timings){
        cudaEvent_t end;
        ck(cudaEventCreate(&end));
        ck(cudaEventRecord(end));
        ck(cudaEventSynchronize(end));
        float tmp;
        ck(cudaEventElapsedTime(&tmp, start, end));
        timings.push_back(tmp);
        ck(cudaEventDestroy(end));
    }

    void load_image(std::string path, uint8_t*& data_mono, uint8_t*& data_bgr, int& width, int& height) {
        int channels;
        unsigned char* _full_data = stbi_load(path.c_str(), &width, &height, &channels, 3);
        // data_mono = new uint8_t[width * height];
        // data_bgr = new uint8_t[width * height * 3];
        ck(cudaHostAlloc(&data_mono, width * height * sizeof(uint8_t), cudaHostAllocDefault));
        ck(cudaHostAlloc(&data_bgr, width * height * 3 * sizeof(uint8_t), cudaHostAllocDefault));

        for (size_t i = 0; i < width * height; i++) {
            data_mono[i] = static_cast<uint8_t>(_full_data[i * channels]); // Take only the first channel (R)
            data_bgr[i * 3 + 0] = _full_data[i * channels + 0]; // Blue
            data_bgr[i * 3 + 1] = _full_data[i * channels + 1]; // Green
            data_bgr[i * 3 + 2] = _full_data[i * channels + 2]; // Red
        }
        stbi_image_free(_full_data);

    }

    void save_image(std::string path, uint8_t* data, int width, int height, int channels) {
        stbi_write_bmp(path.c_str(), width, height, channels, data);
    }

    void mono_to_bgr(uint8_t*& binaryData, uint8_t*& bgrData, size_t totalPixels) {

        for (size_t i = 0; i < totalPixels; ++i) {
            // Binary images usually use 255 for white, 0 for black
            uint8_t pixelValue = binaryData[i];

            // BMP format is typically Blue-Green-Red (BGR)
            bgrData[i * 3 + 0] = pixelValue;          
            bgrData[i * 3 + 1] = 0;          
            bgrData[i * 3 + 2] = 0; 
        }

    }

    void overlap_images(uint8_t*& img1, uint8_t*& img2, uint8_t*& output, int width, int height, int channels) {
        for (int i = 0; i < width * height * channels; ++i) {
            output[i] = std::max(img1[i], img2[i]);
        }
    }

    namespace fs = std::filesystem;
    void get_images_paths(const std::string& folder_path, std::vector<std::string>& img_paths) {
        for (const auto & entry : fs::directory_iterator(folder_path))
            img_paths.push_back(entry.path().string());
    }

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
}
#endif // UTILS_HPP