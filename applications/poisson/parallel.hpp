#pragma once
#include "serial.hpp"  


Result loglik_parallel(const Dataset& d, const Params& beta,
                       int repeat, bool with_grad, std::size_t num_threads);
