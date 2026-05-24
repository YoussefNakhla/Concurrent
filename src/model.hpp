#pragma once
#include <vector>
#include <cstddef>

constexpr int K = 5; // number of choices
constexpr int P = 5; // number of regressors

struct Dataset {
    std::size_t N = 0;
    std::vector<double> X;
    std::vector<int> choice;
};

using Params = std::vector<double>;
constexpr int NPARAM = (K - 1) * P;

double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat);

void grad_one(const Dataset& d, const Params& beta, std::size_t i, int repeat, std::vector<double>& grad);