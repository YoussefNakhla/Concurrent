#pragma once
#include "serial.hpp"
#include <functional>

// Outcome of a full maximum-likelihood solve.
struct OptResult {
    Params beta; // the parameter vector we converged to
    double loglik; // log-likelihood at that beta
    int iterations; // how many ascent steps were taken
    bool converged; // true if grad dropped below tol before max_iter
};

// The optimizer calls this instead of a fixed backend so the we can use the same optimize() runs on top of the serial, mutex or partial-sum likelihood.
using Evaluator = std::function<Result(const Params& beta, bool with_grad)>;

// Gradient ascent with backtracking line search. Maximizes the log-likelihood.
// nparam is the number of parameters
OptResult optimize(const Evaluator& eval, int nparam,
                   double tol = 1e-4, int max_iter = 500);