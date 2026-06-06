# Concurrent — Parallel MLE

Reformulation of Christopher Swann's MPI-based maximum likelihood estimation for a
multinomial logit model as a shared-memory `std::thread` computation. The same code
is reused for a second model, a Poisson regression for count data.

**Team:** Youssef Nakhla · Nicolas Asseo

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

Arguments are `N threads repeat`.

```bash
./build/mle solve 100000 4 50           # logit
./build/poisson_mle solve 100000 4 50   # poisson
```

## Tests

```bash
./build/check_model        # model math and serial driver
./build/check_optimizer    # optimizer convergence
./build/test_correctness   # serial vs parallel agreement
```

## Benchmarks

Runs the full sweep for both models and writes CSVs, plots and tables to
`results/<machine>/`:

```bash
./bench/run_all.sh
```
