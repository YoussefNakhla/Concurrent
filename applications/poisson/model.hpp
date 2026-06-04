#pragma once
#include <vector>
#include <cstddef>

// poisson regression: predict a count response y_i >= 0 from covariates x_i

constexpr int P = 5;       
constexpr int NPARAM = P; 

struct Dataset {
    std::size_t N = 0;
    std::vector<double> X;    
    std::vector<double> y;   
};

using Params = std::vector<double>;

double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat);
void   grad_one  (const Dataset& d, const Params& beta, std::size_t i, int repeat,
                  std::vector<double>& grad);
