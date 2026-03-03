#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>

namespace fs = std::filesystem;

bool compareFiles(const std::string& file1, const std::string& file2) {
    std::ifstream f1(file1, std::ios::binary | std::ios::ate);
    std::ifstream f2(file2, std::ios::binary | std::ios::ate);

    if (!f1.is_open()) {
        std::cerr << "  Cannot open: " << file1 << std::endl;
        return false;
    }
    if (!f2.is_open()) {
        std::cerr << "  Cannot open: " << file2 << std::endl;
        return false;
    }

    // Compare sizes first
    auto size1 = f1.tellg();
    auto size2 = f2.tellg();
    if (size1 != size2) {
        std::cerr << "  Size mismatch: " << size1 << " vs " << size2 << std::endl;
        return false;
    }

    f1.seekg(0);
    f2.seekg(0);

    // Compare content
    std::vector<char> buf1(size1), buf2(size2);
    f1.read(buf1.data(), size1);
    f2.read(buf2.data(), size2);

    if (std::memcmp(buf1.data(), buf2.data(), size1) != 0) {
        std::cerr << "  Content mismatch" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string outputDir = "..\\output_images";
    std::string refDir = "..\\output_images_ref";

    // Allow overriding paths via command line
    if (argc >= 3) {
        outputDir = argv[1];
        refDir = argv[2];
    }

    int passed = 0;
    int failed = 0;
    int total = 0;

    std::cout << "Comparing images in:\n  Output: " << outputDir << "\n  Reference: " << refDir << "\n\n";

    for (const auto& entry : fs::directory_iterator(refDir)) {
        if (entry.path().extension() == ".bmp") {
            total++;
            std::string filename = entry.path().filename().string();
            std::string outputPath = outputDir + "\\" + filename;
            std::string refPath = entry.path().string();

            std::cout << "[TEST] " << filename << " ... ";

            if (compareFiles(outputPath, refPath)) {
                std::cout << "PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "FAILED" << std::endl;
                failed++;
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << "/" << total << " tests passed";
    if (failed > 0) {
        std::cout << " (" << failed << " failed)";
    }
    std::cout << std::endl;

    return failed > 0 ? 1 : 0;
}
