#include "data.hpp"
#include "serial.hpp"
#include "parallel.hpp"
#include "parallel_v2.hpp"
#include <iostream>
#include <cmath>
#include <string>

static int failures = 0;

static void check(bool cond, const std::string& name) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!cond) ++failures;
}

// Serial and parallel sum in different orders, so use a small relative tolerance.
static bool close(double a, double b, double rel_tol) {
    return std::fabs(a - b) <= rel_tol * (1.0 + std::fabs(a));
}

static bool same_result(const Result& ref, const Result& got) {
    if (!close(ref.loglik, got.loglik, 1e-6)) return false;
    for (int k = 0; k < NPARAM; ++k) {
        if (!close(ref.grad[k], got.grad[k], 1e-5)) return false;
    }
    return true;
}

int main() {
    Dataset d = generate_data(50000);
    Params beta(NPARAM, 0.1);
    int repeat = 10;

    Result ref = loglik_serial(d, beta, repeat, true);

    for (int th : {1, 2, 3, 4, 8}) {
        std::size_t t = static_cast<std::size_t>(th);

        Result p1 = loglik_parallel(d, beta, repeat, true, t);
        check(same_result(std::cref(ref), std::cref(p1)),
              "parallel v1 (mutex) matches serial, threads=" + std::to_string(th));

        Result p2 = loglik_parallel_v2(d, beta, repeat, true, t);
        check(same_result(std::cref(ref), std::cref(p2)),
              "parallel v2 (partial sums) matches serial, threads=" + std::to_string(th));
    }

    if (failures == 0) {
        std::cout << "\nALL TESTS PASSED\n";
        return 0;
    }
    std::cout << failures << " tests failed\n";
    return 1;
}
