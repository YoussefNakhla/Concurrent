#pragma once
#include "model.hpp"
#include <vector>


struct Result {
    double loglik;
    std::vector<double> grad;
};

Result loglik_serial(const Dataset& d, const Params& beta, int repeat, bool with_grad);
