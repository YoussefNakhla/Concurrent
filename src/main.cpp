#include "data.hpp"
#include "serial.hpp"
#include "timer.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

// Small driver for timing the serial version.
// Usage: mle [N] [repeat]
// Output format is kept CSV-like so later parallel runs can use the same file.
int main(int argc, char** argv) {
    std::size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 100000;
    int repeat    = (argc > 2) ? std::atoi(argv[2]) : 100;

    Dataset d = generate_data(N);
    Params beta(NPARAM, 0.1);   // fixed test point for the baseline run

    Timer timer;
    Result r = loglik_serial(d, beta, repeat, true);
    double ms = timer.elapsed_ms();

    std::cout << "serial," << N << ",1," << repeat
              << "," << r.loglik << "," << ms << "\n";
    return 0;
}