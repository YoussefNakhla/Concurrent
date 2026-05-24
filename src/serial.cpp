#include "serial.hpp"

Result loglik_serial(const Dataset& d, const Params& beta, int repeat, bool with_grad) {
    Result res{0.0, std::vector<double>(NPARAM, 0.0)};
    for (std::size_t i = 0; i < d.N; ++i) {
        res.loglik += loglik_one(d, beta, i, repeat);
        if (with_grad) {
            grad_one(d, beta, i, repeat, res.grad);
        }
    }
    return res;
}