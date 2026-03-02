#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

class PipelineProfiler {
private:
    struct StageTime {
        std::string name;
        std::vector<double> times_ms;  // Time in milliseconds for each image
        
        double getTotalTime() const {
            return std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        }
        
        double getAverageTime() const {
            if (times_ms.empty()) return 0.0;
            return getTotalTime() / times_ms.size();
        }
        
        double getMinTime() const {
            if (times_ms.empty()) return 0.0;
            return *std::min_element(times_ms.begin(), times_ms.end());
        }
        
        double getMaxTime() const {
            if (times_ms.empty()) return 0.0;
            return *std::max_element(times_ms.begin(), times_ms.end());
        }
    };
    
    std::vector<StageTime> stages;
    std::chrono::high_resolution_clock::time_point stage_start;
    std::chrono::high_resolution_clock::time_point pipeline_start;
    int current_image_idx = -1;
    bool enabled;
    
public:
    enum Stage {
        LOAD = 0,
        GAUSSIAN = 1,
        LAPLACIAN = 2,
        BINARIZE = 3,
        SAVE = 4
    };
    
    PipelineProfiler(bool enable_profiling = false) : enabled(enable_profiling) {
        stages.resize(5);
        stages[LOAD] = {"Load", {}};
        stages[GAUSSIAN] = {"Gaussian", {}};
        stages[LAPLACIAN] = {"Laplacian", {}};
        stages[BINARIZE] = {"Binarize", {}};
        stages[SAVE] = {"Save", {}};
        
        if (enabled) {
            pipeline_start = std::chrono::high_resolution_clock::now();
        }
    }
    
    void startStage(Stage stage, int image_idx) {
        if (!enabled) return;
        
        current_image_idx = image_idx;
        stage_start = std::chrono::high_resolution_clock::now();
    }
    
    void endStage(Stage stage, int image_idx) {
        if (!enabled) return;
        
        auto stage_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
        double duration_ms = duration / 1000.0;
        
        // Ensure we have enough space for this image index
        if (image_idx >= stages[stage].times_ms.size()) {
            stages[stage].times_ms.resize(image_idx + 1, 0.0);
        }
        
        stages[stage].times_ms[image_idx] += duration_ms;
    }
    
    void printStatistics() const {
        if (!enabled) {
            std::cout << "Profiling was not enabled." << std::endl;
            return;
        }
        
        auto pipeline_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start).count();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "PIPELINE PROFILING STATISTICS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << std::left << std::setw(15) << "Stage"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Images"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        double total_avg = 0.0;
        for (const auto& stage : stages) {
            if (!stage.times_ms.empty()) {
                std::cout << std::left
                          << std::setw(15) << stage.name
                          << std::setw(15) << std::fixed << std::setprecision(3) << stage.getAverageTime()
                          << std::setw(15) << std::fixed << std::setprecision(3) << stage.getMinTime()
                          << std::setw(15) << std::fixed << std::setprecision(3) << stage.getMaxTime()
                          << std::setw(15) << std::fixed << std::setprecision(1) << stage.getTotalTime()
                          << std::setw(15) << stage.times_ms.size()
                          << std::endl;
                total_avg += stage.getAverageTime();
            }
        }
        
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Total Pipeline Time: " << total_duration << " ms" << std::endl;
        std::cout << "Average Time per Image (sum of stages): " << std::fixed << std::setprecision(3) << total_avg << " ms" << std::endl;
        
        // Count how many images were fully processed
        int num_images = 0;
        if (!stages[LOAD].times_ms.empty()) {
            num_images = stages[LOAD].times_ms.size();
        }
        if (num_images > 0) {
            std::cout << "Total Images Processed: " << num_images << std::endl;
            std::cout << "Average Pipeline Time per Image: " << std::fixed << std::setprecision(3) 
                      << (total_duration / static_cast<double>(num_images)) << " ms" << std::endl;
        }
        
        std::cout << std::string(80, '=') << std::endl << std::endl;
    }
    
    bool isEnabled() const {
        return enabled;
    }
};

#endif // PROFILER_HPP
