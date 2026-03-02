#pragma once

#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : shutdown_(false) {
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_thread(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F>
    void enqueue(F&& task) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (shutdown_) throw std::runtime_error("ThreadPool is shutdown");
            tasks_.emplace(std::forward<F>(task));
        }
        cv_.notify_one();
    }

    size_t get_num_threads() const { return workers_.size(); }

private:
    void worker_thread() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !tasks_.empty() || shutdown_; });
            
            if (shutdown_ && tasks_.empty()) break;
            
            if (!tasks_.empty()) {
                auto task = std::move(tasks_.front());
                tasks_.pop();
                lock.unlock();
                task();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_;
};
