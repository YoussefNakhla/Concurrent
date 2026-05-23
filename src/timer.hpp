#pragma once
#include <chrono>

//  helper file for benchmarks later
class Timer {
    // start point of the timer
    std::chrono::high_resolution_clock::time_point start_;

public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    // return the elapsed time in milliseconds
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};
