#include "parallel_v2.hpp"
#include <thread>
#include <vector>
#include <algorithm>

// per thread work function
void work_v2(const Dataset& d, const Params& beta, int repeat, bool with_grad,
             size_t num_threads, std::vector<double>& partial_ll,
             std::vector<std::vector<double>>& partial_grad, int thread_id) {
    std::size_t chunk = (d.N + num_threads - 1) / num_threads;
    std::size_t begin = thread_id * chunk;
    std::size_t end = std::min(begin + chunk, d.N);

    double local_ll = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        local_ll += loglik_one(d, beta, i, repeat);
        if (with_grad) {
            grad_one(d, beta, i, repeat, partial_grad[thread_id]);
        }
    }
    partial_ll[thread_id] = local_ll;
}

// version 2 of our implementation: all threads update their own partial sums, and then we combune them after the join (no lock)
Result loglik_parallel_v2(const Dataset& d, const Params& beta,
                          int repeat, bool with_grad, size_t num_threads) {
    if (num_threads < 1) {
        num_threads = 1;
    }
    
    Result res{0.0, std::vector<double>(NPARAM, 0.0)};
    std::vector<double> partial_ll(num_threads, 0.0);
    std::vector<std::vector<double>> partial_grad(
        num_threads, std::vector<double>(NPARAM, 0.0));

    std::vector<std::thread> threads(num_threads);
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        threads[thread_id] = std::thread(
            work_v2, std::cref(d), std::cref(beta), repeat, with_grad, num_threads,
            std::ref(partial_ll), std::ref(partial_grad), thread_id);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    for (size_t t = 0; t < num_threads; ++t) {
        res.loglik += partial_ll[t];
        if (with_grad) {
            const std::vector<double>& pg = partial_grad[t];
            for (int k = 0; k < NPARAM; ++k) {
                res.grad[k] += pg[k];
            }
        }
    }
    return res;
}
