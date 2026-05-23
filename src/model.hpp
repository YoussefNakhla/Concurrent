#pragma once
#include <vector>
#include <cstddef>


constexpr int K = 5;   // number of choices 
constexpr int P = 5;   // regressors: constant term / bias, age, education, black, children

struct Dataset {
    std::size_t N = 0;            // number of observations
    std::vector<double> X;        // full design matrix 
    std::vector<int>    choice;   // vector of chosen categories in [0, K)
};

// Parameters: beta for choices 1..K-1 (choice 0 is base, all zeros).
using Params = std::vector<double>;
constexpr int NPARAM = (K - 1) * P;

// Log-likelihood contribution of 1 observation i.
double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat);

// Gradient contribution of 1 observation accumulated into grad
void grad_one(const Dataset& d, const Params& beta, std::size_t i,
              int repeat, std::vector<double>& grad);
