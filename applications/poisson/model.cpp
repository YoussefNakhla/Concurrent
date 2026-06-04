#include "model.hpp"
#include <cmath>

// dot product x_i' beta
static double linear_index(const Dataset& d, const Params& beta, std::size_t i) {
    const double* xi = &d.X[i * P];
    double s = 0.0;
    for (int p = 0; p < P; ++p){
        s += xi[p] * beta[p];
    }
    return s;
}

// contribution of observation i to the poisson log-likelihood
double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat) {
    double acc = 0.0;
    const double yi = d.y[i];
    for (int r = 0; r < repeat; ++r) {
        double eta = linear_index(d, beta, i);
        acc += yi * eta - std::exp(eta);
    }
    return acc / repeat;
}


// grad += (y_i - mu_i) * x_i
void grad_one(const Dataset& d, const Params& beta, std::size_t i,
              int repeat, std::vector<double>& grad) {
    const double* xi = &d.X[i * P];
    const double yi  = d.y[i];

  
    double mu_acc = 0.0;
    for (int r = 0; r < repeat; ++r) {
        mu_acc += std::exp(linear_index(d, beta, i));
    }
    double mu = mu_acc / repeat;

    double factor = yi - mu;
    for (int p = 0; p < P; ++p){
        grad[p] += factor * xi[p];
    }
}
