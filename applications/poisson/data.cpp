#include "data.hpp"
#include <random>
#include <cmath>

Dataset generate_data(std::size_t N, std::uint64_t seed) {
    Dataset d;
    d.N = N;
    d.X.resize(N * P);
    d.y.resize(N);

    std::mt19937_64 rng(seed);

    // covariate distributions from RAND Health Insurance Experiment
    std::normal_distribution<double>  age(45.0, 12.0);      // age in years
    std::normal_distribution<double>  educ(12.5, 3.0);      // years of schooling
    std::normal_distribution<double>  logincome(10.5, 0.8); // log of annual income (USD)
    std::bernoulli_distribution       chronic(0.30);        // 1 = has a chronic illness

    Params truth(NPARAM);
    std::mt19937_64 brng(seed ^ 0xDEADBEEF);
    std::uniform_real_distribution<double> bdist(-0.15, 0.15);
    for (double& b : truth) b = bdist(brng);
    truth[0] = 0.5;
    for (std::size_t i = 0; i < N; ++i) {
        double* xi = &d.X[i * P];
        xi[0] = 1.0;            
        xi[1] = age(rng);
        xi[2] = educ(rng);
        xi[3] = logincome(rng);
        xi[4] = chronic(rng) ? 1.0 : 0.0;

        // compute true linear predictor and mean count
        double eta = 0.0;
        for (int p = 0; p < P; ++p){
            eta += xi[p] * truth[p];
        }
        double mu = std::exp(eta);

        // draw y_i ~ Poisson(mu)
        std::poisson_distribution<int> poisson(mu);
        d.y[i] = static_cast<double>(poisson(rng));
    }
    return d;
}
