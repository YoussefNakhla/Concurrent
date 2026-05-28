#pragma once
#include "serial.hpp"   // reuses Result { loglik, grad }

// basically same as loglik_serial, but with num_threads threads instead
Result loglik_parallel(const Dataset& d, const Params& beta,
                       int repeat, bool with_grad, std::size_t num_threads);
