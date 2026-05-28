#pragma once
#include "serial.hpp"

// same signature as loglik_parallel in parallel.hpp
Result loglik_parallel_v2(const Dataset& d, const Params& beta,
                          int repeat, bool with_grad, std::size_t num_threads);
