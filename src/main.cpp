#include "data.hpp"
#include "serial.hpp"
#include "parallel.hpp"
#include "parallel_v2.hpp"
#include "optimizer.hpp"
#include "timer.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

// Usage: mle <mode> N threads repeat
//
//   solve    : run a full optimize() under serial, v1 (mutex), and v2 (partial
//              sums); report each solve's wall-clock time and final loglik.
//              This is the main benchmark mode.
//   serial / parallel / parallel_v2 : time a single likelihood+gradient
//              evaluation under one backend (kept for quick checks).
//   compare  : single-evaluation comparison of all three backends.
//
// solve CSV (10 columns):
//   solve,N,threads,repeat,iters_serial,loglik_serial,
//   ms_serial,ms_v1,ms_v2,loglik_match
//   (loglik_match = 1 if all three solves agree to a relative 1e-6, else 0)

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "solve";
    std::size_t N    = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 100000;
    int threads      = (argc > 3) ? std::atoi(argv[3]) : 4;
    int repeat       = (argc > 4) ? std::atoi(argv[4]) : 50;

    Dataset d = generate_data(N);
    const std::size_t nt = static_cast<std::size_t>(threads);

    if (mode == "solve") {
        Evaluator e_serial = [&](const Params& b, bool g){ return loglik_serial(d, b, repeat, g); };
        Evaluator e_v1     = [&](const Params& b, bool g){ return loglik_parallel(d, b, repeat, g, nt); };
        Evaluator e_v2     = [&](const Params& b, bool g){ return loglik_parallel_v2(d, b, repeat, g, nt); };

        Timer ts; OptResult rs = optimize(e_serial, NPARAM); double ms_serial = ts.elapsed_ms();
        Timer t1; OptResult r1 = optimize(e_v1,     NPARAM); double ms_v1     = t1.elapsed_ms();
        Timer t2; OptResult r2 = optimize(e_v2,     NPARAM); double ms_v2     = t2.elapsed_ms();

        double tol = 1e-6 * (1.0 + std::fabs(rs.loglik));
        int match = (std::fabs(rs.loglik - r1.loglik) < tol &&
                     std::fabs(rs.loglik - r2.loglik) < tol) ? 1 : 0;

        std::cout << "solve," << N << "," << threads << "," << repeat
                  << "," << rs.iterations << "," << rs.loglik
                  << "," << ms_serial << "," << ms_v1 << "," << ms_v2
                  << "," << match << "\n";
        return 0;
    }

    Params beta(NPARAM, 0.1);

    if (mode == "serial") {
        Timer t; Result r = loglik_serial(d, beta, repeat, true);
        std::cout << "serial," << N << ",1," << repeat << "," << r.loglik
                  << "," << t.elapsed_ms() << "\n";
        return 0;
    }
    if (mode == "parallel") {
        Timer t; Result r = loglik_parallel(d, beta, repeat, true, nt);
        std::cout << "parallel," << N << "," << threads << "," << repeat << "," << r.loglik
                  << "," << t.elapsed_ms() << "\n";
        return 0;
    }
    if (mode == "parallel_v2") {
        Timer t; Result r = loglik_parallel_v2(d, beta, repeat, true, nt);
        std::cout << "parallel_v2," << N << "," << threads << "," << repeat << "," << r.loglik
                  << "," << t.elapsed_ms() << "\n";
        return 0;
    }
    if (mode == "compare") {
        Timer ts; Result s  = loglik_serial(d, beta, repeat, true);          double ms_s = ts.elapsed_ms();
        Timer t1; Result p1 = loglik_parallel(d, beta, repeat, true, nt);    double ms1  = t1.elapsed_ms();
        Timer t2; Result p2 = loglik_parallel_v2(d, beta, repeat, true, nt); double ms2  = t2.elapsed_ms();
        std::cout << "compare," << N << "," << threads << "," << repeat
                  << "," << s.loglik << "," << p1.loglik << "," << p2.loglik
                  << "," << std::fabs(s.loglik - p1.loglik)
                  << "," << std::fabs(s.loglik - p2.loglik)
                  << "," << ms_s << "," << ms1 << "," << ms2 << "\n";
        return 0;
    }

    std::cerr << "Usage: mle solve|serial|parallel|parallel_v2|compare N threads repeat\n";
    return 1;
}
