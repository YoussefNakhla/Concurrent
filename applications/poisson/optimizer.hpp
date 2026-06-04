#pragma once
#include "serial.hpp"
#include <functional>

struct OptResult {
    Params beta; 
    double loglik;
    int iterations;
    bool converged; 
};


using Evaluator = std::function<Result(const Params& beta, bool with_grad)>;

OptResult optimize(const Evaluator& eval, int nparam,
                   double tol = 1e-4, int max_iter = 500);
