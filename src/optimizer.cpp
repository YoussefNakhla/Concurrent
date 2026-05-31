#include "optimizer.hpp"
#include <cmath>

static double norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// Gradient ascent with backtracking line search. Maximizes the log-likelihood.
// The multinomial-logit log-likelihood is globally concave, so there are no local maxima
// Only difficulty is choosing a step size, which the line search handles (we start with a step and then halve it until the likelihood improves)
// Stop when the relative gain in log-likelihood becomes very small which avoids using an absolute gradient threshold that depends on dataset size.
OptResult optimize(const Evaluator& eval, int nparam, double tol, int max_iter) {
    Params beta(nparam, 0.0);
    Result cur = eval(beta, true);
    int it = 0;
    bool converged = false;
    for (; it < max_iter; ++it) {
        double gnorm = norm(cur.grad);
        if (gnorm == 0.0) { converged = true; break; }

        // Normalized search direction so the initial step means the same thing regardless of the gradient's magnitude.
        std::vector<double> dir(nparam);
        for (int k = 0; k < nparam; ++k) dir[k] = cur.grad[k] / gnorm;

        double step = 1.0;
        Params trial(nparam);
        Result tr{cur.loglik, std::vector<double>(nparam, 0.0)};
        bool improved = false;

        for (int b = 0; b < 60; ++b) {
            for (int k = 0; k < nparam; ++k) trial[k] = beta[k] + step * dir[k];
            tr = eval(trial, false);
            if (tr.loglik > cur.loglik) { improved = true; break; }
            step *= 0.5;
        }

        if (!improved) {
            converged = true;
            break;
        }  // can't climb further

        // Relative improvement in log-likelihood
        double rel_gain = (tr.loglik - cur.loglik) / (1.0 + std::fabs(cur.loglik));

        beta = trial;
        cur = eval(beta, true);

        if (rel_gain < tol) {
            converged = true;
            ++it;
            break;
        }
    }

    return OptResult{beta, cur.loglik, it, converged};
}