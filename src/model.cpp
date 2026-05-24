#include "model.hpp"
#include <cmath>

// Computes the utility index for each possible choice for observation i.
// The first choice is kept fixed at zero for identification.
static void linear_indices(const Dataset& d, const Params& beta,
                           std::size_t i, double* out) {
    const double* xi = &d.X[i * P];
    out[0] = 0.0;
    for (int j = 1; j < K; ++j) {
        double s = 0.0;
        const double* bj = &beta[(j - 1) * P];
        for (int p = 0; p < P; ++p) {
            s += xi[p] * bj[p];
        }
        out[j] = s;
    }
}

// Turns the utility indices into probabilities, using the usual max subtraction
// to avoid overflow in the exponentials.
static void choice_probs(const double* idx, double* prob) {
    double mx = idx[0];
    for (int j = 1; j < K; ++j) {
        if (idx[j] > mx) {
            mx = idx[j];
        }
    }
    double denom = 0.0;
    for (int j = 0; j < K; ++j) {
        prob[j] = std::exp(idx[j] - mx);
        denom += prob[j];
    }
    for (int j = 0; j < K; ++j) {
        prob[j] /= denom;
    }
}

// Contribution of a single observation to the log-likelihood.
// repeat is only there to make each observation more expensive during benchmarks.
double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat) {
    double result = 0.0;
    double idx[K], prob[K];
    for (int r = 0; r < repeat; ++r) {
        linear_indices(d, beta, i, idx);
        choice_probs(idx, prob);
        result = std::log(prob[d.choice[i]]);
    }
    return result;
}

// Adds the gradient contribution of one observation into the running total.
// The formula is the standard multinomial logit score for each estimated choice.
void grad_one(const Dataset& d, const Params& beta, std::size_t i,
              int repeat, std::vector<double>& grad) {
    double idx[K], prob[K];
    for (int r = 0; r < repeat; ++r) {
        linear_indices(d, beta, i, idx);
        choice_probs(idx, prob);
    }
    const double* xi = &d.X[i * P];
    int chosen = d.choice[i];
    for (int j = 1; j < K; ++j) {
        double factor = (chosen == j ? 1.0 : 0.0) - prob[j];
        for (int p = 0; p < P; ++p) {
            grad[(j - 1) * P + p] += factor * xi[p];
        }
    }
}