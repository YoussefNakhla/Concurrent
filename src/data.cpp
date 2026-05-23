#include "data.hpp"
#include <random>
#include <cmath>

Dataset generate_data(std::size_t N, std::uint64_t seed) {
    Dataset d;
    d.N = N;
    d.X.resize(N * P);
    d.choice.resize(N);

    std::mt19937_64 rng(seed);
    // chose distributions described in the paper to generate fake data that follows the real data from PSID
    std::normal_distribution<double> age(26.0, 5.0);     
    std::normal_distribution<double> educ(11.9, 1.7);
    std::bernoulli_distribution     black(0.49);
    std::poisson_distribution<int>  kids(1.37);

    // Initialisation of the true beta parameters
    Params truth(NPARAM, 0.0);
    std::mt19937_64 brng(seed ^ 0xABCDEF); // random number generator where we mix the original seed with a fixed value to get a different random number sequence
    std::uniform_real_distribution<double> bdist(-0.3, 0.3);
    for (double& b : truth){
        b = bdist(brng);
    }
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (std::size_t i = 0; i < N; ++i) {
        double* xi = &d.X[i * P];
        xi[0] = 1.0; // constant term / bias / intercept
        xi[1] = age(rng);
        xi[2] = educ(rng);
        if (black(rng)) {
            xi[3] = 1.0;
        } else {
            xi[3] = 0.0;
        }
        xi[4] = (double)(kids(rng));

        // softmax from truth then sample one choice 
        double idx[K], prob[K], denom = 0.0;
        idx[0] = 0.0;
        for (int j = 1; j < K; ++j) {
            double s = 0.0;
            for (int p = 0; p < P; ++p){
                s += xi[p] * truth[(j - 1) * P + p];
            }
            idx[j] = s;
        }
        double mx = idx[0];
        for (int j = 1; j < K; ++j){
            if (idx[j] > mx){
                mx = idx[j];
            }
        }
        for (int j = 0; j < K; ++j) {
            prob[j] = std::exp(idx[j] - mx);
            denom += prob[j];
        }
        double u = uni(rng) * denom, acc = 0.0;
        int c = K - 1;
        for (int j = 0; j < K; ++j) {
            acc += prob[j];
            if (u <= acc) { 
                c = j; 
                break; 
            }
        }
        d.choice[i] = c;
    }
    return d;
}
