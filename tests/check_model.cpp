#include "model.hpp"
#include "serial.hpp"
#include "data.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

static int failures = 0;
static void check(bool cond, const std::string& name) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!cond) ++failures;
}

int main() {
    // Basic one-observation checks for the likelihood and gradient.
    Dataset d;
    d.N = 1;
    d.X = {1.0, 2.0, 3.0, 4.0, 5.0};
    d.choice = {3};

    Params beta(NPARAM, 0.0);

    double ll = loglik_one(d, beta, 0, 1);
    double expected_ll = std::log(1.0 / K);
    check(std::fabs(ll - expected_ll) < 1e-12, "loglik at beta=0 equals log(1/K)");

    std::vector<double> grad(NPARAM, 0.0);
    grad_one(d, beta, 0, 1, grad);
    bool grad_ok = true;
    for (int j = 1; j < K; ++j) {
        double indicator = (d.choice[0] == j) ? 1.0 : 0.0;
        for (int p = 0; p < P; ++p) {
            double want = (indicator - 1.0 / K) * d.X[p];
            if (std::fabs(grad[(j - 1) * P + p] - want) > 1e-12) grad_ok = false;
        }
    }
    check(grad_ok, "gradient at beta=0 matches closed form");

    // A finite-difference check catches mistakes that the zero-beta case can miss.
    Params b(NPARAM, 0.0);
    for (int k = 0; k < NPARAM; ++k) b[k] = 0.05 * (k + 1) - 0.1;

    std::vector<double> ganalytic(NPARAM, 0.0);
    grad_one(d, b, 0, 1, ganalytic);

    double eps = 1e-6;
    bool fd_ok = true;
    for (int k = 0; k < NPARAM; ++k) {
        Params bp = b, bm = b;
        bp[k] += eps; bm[k] -= eps;
        double fd = (loglik_one(d, bp, 0, 1) - loglik_one(d, bm, 0, 1)) / (2 * eps);
        if (std::fabs(fd - ganalytic[k]) > 1e-5) {
            fd_ok = false;
            std::cout << "   k=" << k << " analytic=" << ganalytic[k]
                      << " finite_diff=" << fd << "\n";
        }
    }
    check(fd_ok, "analytic gradient matches finite differences (nonzero beta)");

    check(loglik_one(d, b, 0, 1) <= 1e-12, "loglik is non-positive (log of a probability)");

    check(std::fabs(loglik_one(d, b, 0, 1) - loglik_one(d, b, 0, 100)) < 1e-12,
          "loglik invariant under repeat");

    // Check the serial driver against the same calculation written out directly.
    {
        Dataset ds = generate_data(5000);
        Params bb(NPARAM, 0.1);

        Result r = loglik_serial(ds, bb, 1, true);

        double ref_ll = 0.0;
        std::vector<double> ref_g(NPARAM, 0.0);
        for (std::size_t i = 0; i < ds.N; ++i) {
            ref_ll += loglik_one(ds, bb, i, 1);
            grad_one(ds, bb, i, 1, ref_g);
        }
        bool g_ok = true;
        for (int k = 0; k < NPARAM; ++k)
            if (std::fabs(r.grad[k] - ref_g[k]) > 1e-9) g_ok = false;
        check(std::fabs(r.loglik - ref_ll) < 1e-9, "serial loglik matches reference loop");
        check(g_ok, "serial gradient matches reference loop");

        Result r2 = loglik_serial(ds, bb, 1, false);
        double gsum = 0.0; for (double x : r2.grad) gsum += std::fabs(x);
        check(gsum == 0.0, "serial with_grad=false leaves gradient zero");
        check(std::fabs(r.loglik - r2.loglik) < 1e-9,
              "serial loglik independent of with_grad flag");
    }

    std::cout << (failures == 0 ? "\nALL TESTS PASSED\n" : "\nSOME TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
