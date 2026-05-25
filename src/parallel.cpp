#include "parallel.hpp"
#include <thread>
#include <mutex>
#include <vector>
#include <algorithm>

// per thread work function
void work(const Dataset& d, const Params& beta, int repeat, bool with_grad, int num_threads, Result& res, std::mutex& m, int thread_id) {
    std::size_t chunk = (d.N + static_cast<std::size_t>(num_threads) - 1)
                        / static_cast<std::size_t>(num_threads);
    std::size_t begin = static_cast<std::size_t>(thread_id) * chunk;
    std::size_t end = std::min(begin + chunk, d.N);
    
    for (std::size_t i = begin; i < end; ++i) {
        double ll = loglik_one(d, beta, i, repeat);
        std::lock_guard<std::mutex> lock(m);
        res.loglik += ll;
        if (with_grad){
            grad_one(d, beta, i, repeat, res.grad);
        }
    }
}

// version number of 1 of our implementation of the parallel version: all threads update same shared variable, with a lock (next step is to use partial sums)
Result loglik_parallel(const Dataset& d, const Params& beta,
                       int repeat, bool with_grad, int num_threads) {
    if (num_threads < 1){
        num_threads = 1;
    }

    Result res{0.0, std::vector<double>(NPARAM, 0.0)};
    std::mutex m;

    std::vector<std::thread> threads(num_threads);
    for (int thread_id = 0; thread_id < num_threads; ++thread_id)
        threads[thread_id] = std::thread(work, std::cref(d), std::cref(beta), repeat, with_grad, num_threads, std::ref(res), std::ref(m), thread_id);
    for (std::thread& thread : threads)
        thread.join();

    return res;
}
