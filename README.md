# Concurrent — Parallel MLE (Multinomial Logit)

Reformulation of Christopher Swann's MPI-based maximum likelihood estimation for a
multinomial logit model as a shared-memory `std::thread` computation.

**Team:** Youssef Nakhla (serial track) · Nicolas Asseo (parallel track)

## Layout

```
src/       Model, data, serial/parallel drivers, optimizer, main
tests/     Correctness (serial vs parallel)
bench/     Benchmark scripts and plotting
results/   CSV and plot output (gitignored)
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

(Full targets and sources are added incrementally as implementation proceeds.)

## Run

```bash
./build/mle serial <N> <threads> <repeat>
./build/mle parallel <N> <threads> <repeat>
```
