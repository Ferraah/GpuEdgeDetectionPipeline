#ifndef UTILS_HPP
#define UTILS_HPP

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) 
                << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

void load_image(std::string path, uint8_t*& data_mono, uint8_t*& data_bgr, int& width, int& height) {
    int channels;
    unsigned char* _full_data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    data_mono = new uint8_t[width * height];
    data_bgr = new uint8_t[width * height * 3];
    for (size_t i = 0; i < width * height; i++) {
        data_mono[i] = static_cast<uint8_t>(_full_data[i * channels]); // Take only the first channel (R)
        data_bgr[i * 3 + 0] = _full_data[i * channels + 0]; // Blue
        data_bgr[i * 3 + 1] = _full_data[i * channels + 1]; // Green
        data_bgr[i * 3 + 2] = _full_data[i * channels + 2]; // Red
    }
    stbi_image_free(_full_data);

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
        img_paths.push_back(entry.path());
}
#endif // UTILS_HPP