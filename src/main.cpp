#include "data.hpp"
#include "serial.hpp"
#include "parallel.hpp"
#include "timer.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

// Small driver for timing the serial version.
// Usage: mle serial|parallel|compare N threads repeat
// Output format is kept CSV-like so later parallel runs can use the same file.
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

        Timer t_parallel;
        Result p = loglik_parallel(d, beta, repeat, true, threads);
        double ms_parallel = t_parallel.elapsed_ms();

        double diff = std::fabs(s.loglik - p.loglik);
        std::cout << "compare," << N << "," << threads << "," << repeat
                  << "," << s.loglik << "," << p.loglik << "," << diff
                  << "," << ms_serial << "," << ms_parallel << "\n";
        return 0;
    }

    if (mode == "serial") {
        Timer t;
        Result r = loglik_serial(d, beta, repeat, true);
        std::cout << "serial," << N << ",1," << repeat
                  << "," << r.loglik << ",,," << t.elapsed_ms() << ",\n";
        return 0;
    }

    if (mode == "parallel") {
        Timer t;
        Result r = loglik_parallel(d, beta, repeat, true, threads);
        std::cout << "parallel," << N << "," << threads << "," << repeat
                  << ",," << r.loglik << ",," << t.elapsed_ms() << "\n";
        return 0;
    }

    std::cerr << "Usage: mle serial|parallel|compare N threads repeat\n";
    return 1;
}
