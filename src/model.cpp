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
        for (int p = 0; p < P; ++p){
            s += xi[p] * bj[p];
        }
        out[j] = s;
    }
}

// Turns the utility indices into probabilities, using the usual max subtraction
// to avoid overflow in the exponentials.
static void choice_probs(const double* idx, double* prob) {
    double mx = idx[0];
    for (int j = 1; j < K; ++j){
        if (idx[j] > mx){
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
// Repeat inflates per observation cost for benchmarking. To stop the optimizer from deleting the redundant passes (called dead-code elimination) each pass must contribute to the result. 
// We accumulate then divide back out, so the returned value is unchanged but every pass is observably necessary.
double loglik_one(const Dataset& d, const Params& beta, std::size_t i, int repeat) {
    double acc = 0.0;
    double idx[K], prob[K];
    for (int r = 0; r < repeat; ++r) {
        linear_indices(d, beta, i, idx);
        choice_probs(idx, prob);
        acc += std::log(prob[d.choice[i]]);   // accumulate so each pass matters
    }
    return acc / repeat;                       // average = the single pass value from before
}

// Adds the gradient contribution of one observation into the running total.
// The formula is the standard multinomial logit score for each estimated choice.
void grad_one(const Dataset& d, const Params& beta, std::size_t i,
              int repeat, std::vector<double>& grad) {
    double idx[K], prob[K];
    // Accumulate probabilities across passes so they cannot be optimized away then average before forming the gradient (math result is unchanged).
    double prob_acc[K] = {0,0,0,0,0};
    for (int r = 0; r < repeat; ++r) {
        linear_indices(d, beta, i, idx);
        choice_probs(idx, prob);
        for (int j = 0; j < K; ++j) {
            prob_acc[j] += prob[j];
        }
    }
    for (int j = 0; j < K; ++j) {
        prob[j] = prob_acc[j] / repeat;
    }

    const double* xi = &d.X[i * P];
    int chosen = d.choice[i];
    for (int j = 1; j < K; ++j) {
        double factor;
        if (chosen == j){
            factor = 1.0 - prob[j];
        } else {
            factor = -prob[j];
        }
        for (int p = 0; p < P; ++p) {
            grad[(j - 1) * P + p] += factor * xi[p];
        }
    }
}
