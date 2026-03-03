#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>

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
        cudaEventCreate(&end);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float tmp;
        cudaEventElapsedTime(&tmp, start, end);
        timings.push_back(tmp);
        cudaEventDestroy(end);
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

    void save_image_stats(const std::string& stats_path, const std::string& source_path, 
                          int width, int height, const float stage_times[5]) {
        const char* stage_names[5] = {"Load+Transfer", "Gaussian", "Laplacian", "Binarization", "Save"};
        float total_time = 0.0f;
        for (int s = 0; s < 5; s++) {
            total_time += stage_times[s];
        }

        std::ofstream stats_file(stats_path);
        if (stats_file.is_open()) {
            stats_file << "Image Statistics\n";
            stats_file << "================\n";
            stats_file << "Source: " << source_path << "\n";
            stats_file << "Dimensions: " << width << "x" << height << " (" << (width * height) << " pixels)\n\n";
            stats_file << "Stage Timings (ms):\n";
            stats_file << "-------------------\n";
            for (int s = 0; s < 5; s++) {
                stats_file << "  " << stage_names[s] << ": " << stage_times[s] << " ms\n";
            }
            stats_file << "-------------------\n";
            stats_file << "  Total: " << total_time << " ms\n";
            stats_file.close();
        }
    }

}
#endif // UTILS_HPP