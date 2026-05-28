#include "data.hpp"
#include "serial.hpp"
#include "parallel.hpp"
#include "parallel_v2.hpp"
#include "timer.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

// Usage: mle serial|parallel|parallel_v2|compare N threads repeat
//
// CSV (12 columns):
//   mode,N,threads,repeat,loglik_serial,loglik_parallel,loglik_parallel_v2,
//   loglik_diff_v1,loglik_diff_v2,ms_serial,ms_parallel,ms_parallel_v2

static void print_csv(const std::string& mode, std::size_t N, int threads, int repeat,
                      double ll_s, double ll_p1, double ll_p2,
                      double diff_v1, double diff_v2,
                      double ms_s, double ms_p1, double ms_p2) {
    std::cout << mode << "," << N << "," << threads << "," << repeat
              << "," << ll_s << "," << ll_p1 << "," << ll_p2
              << "," << diff_v1 << "," << diff_v2
              << "," << ms_s << "," << ms_p1 << "," << ms_p2 << "\n";
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "serial";
    std::size_t N    = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 100000;
    int threads      = (argc > 3) ? std::atoi(argv[3]) : 4;
    int repeat       = (argc > 4) ? std::atoi(argv[4]) : 100;

    Dataset d = generate_data(N);
    Params beta(NPARAM, 0.1);

    if (mode == "compare") {
        Timer t_serial;
        Result s = loglik_serial(d, beta, repeat, true);
        double ms_serial = t_serial.elapsed_ms();

        Timer t_p1;
        Result p1 = loglik_parallel(d, beta, repeat, true,
                                    static_cast<std::size_t>(threads));
        double ms_p1 = t_p1.elapsed_ms();

        Timer t_p2;
        Result p2 = loglik_parallel_v2(d, beta, repeat, true,
                                       static_cast<std::size_t>(threads));
        double ms_p2 = t_p2.elapsed_ms();

        print_csv("compare", N, threads, repeat,
                  s.loglik, p1.loglik, p2.loglik,
                  std::fabs(s.loglik - p1.loglik), std::fabs(s.loglik - p2.loglik),
                  ms_serial, ms_p1, ms_p2);
        return 0;
    }

    if (mode == "serial") {
        Timer t;
        Result r = loglik_serial(d, beta, repeat, true);
        print_csv("serial", N, 1, repeat,
                  r.loglik, 0, 0, 0, 0, t.elapsed_ms(), 0, 0);
        return 0;
    }

    if (mode == "parallel") {
        Timer t;
        Result r = loglik_parallel(d, beta, repeat, true,
                                   static_cast<std::size_t>(threads));
        print_csv("parallel", N, threads, repeat,
                  0, r.loglik, 0, 0, 0, 0, t.elapsed_ms(), 0);
        return 0;
    }

    if (mode == "parallel_v2") {
        Timer t;
        Result r = loglik_parallel_v2(d, beta, repeat, true,
                                      static_cast<std::size_t>(threads));
        print_csv("parallel_v2", N, threads, repeat,
                  0, 0, r.loglik, 0, 0, 0, 0, t.elapsed_ms());
        return 0;
    }

    std::cerr << "Usage: mle serial|parallel|parallel_v2|compare N threads repeat\n";
    return 1;
}
