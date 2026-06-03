#include "data.hpp"
#include "serial.hpp"
#include "parallel_v2.hpp"
#include "optimizer.hpp"
#include <cstdio>
#include <cmath>

int main() {
    Dataset d = generate_data(20000);
    int repeat = 1;

    // Inject serial, then parallel using the same optimixe()
    Evaluator es = [&](const Params& b, bool g){ return loglik_serial(d, b, repeat, g); };
    Evaluator ep = [&](const Params& b, bool g){ return loglik_parallel_v2(d, b, repeat, g, 4); };

    OptResult rs = optimize(es, NPARAM);
    OptResult rp = optimize(ep, NPARAM);

    std::printf("serial:   converged=%d iters=%d loglik=%.6f\n", rs.converged, rs.iterations, rs.loglik);
    std::printf("parallel: converged=%d iters=%d loglik=%.6f\n", rp.converged, rp.iterations, rp.loglik);

    // loglik at the starting point beta=0, for comparison
    Result at_zero = loglik_serial(d, Params(NPARAM, 0.0), repeat, false);
    std::printf("beta=0 loglik=%.2f  -->  optimized loglik=%.2f\n", at_zero.loglik, rs.loglik);
    return 0;
}