#pragma once
#include "model.hpp"
#include <vector>

// Result of one likelihood evaluation: the total log-likelihood over all observations and the gradient accumulated over all observations.
struct Result {
    double loglik;
    std::vector<double> grad;
};

// Serial driver: sum loglik_one over all N observations.
// This is the baseline the parallel version must agree with and beat.
Result loglik_serial(const Dataset& d, const Params& beta, int repeat, bool with_grad);