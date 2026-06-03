#!/usr/bin/env bash
# Sweep full-solve timings over dataset sizes and thread counts.
# Run from the repo root after building cmake --build build
# Output: results/bench.csv
# I used ai here to help create a script which would run our code to benchmark

set -e
MLE=./build/mle
OUT=results/bench.csv
mkdir -p results

REPEAT=50
SIZES=(50000 100000 200000 500000)
THREADS=(1 2 4 8)
TRIALS=3

echo "solve,N,threads,repeat,iters,loglik,ms_serial,ms_v1,ms_v2,match" > "$OUT"

for N in "${SIZES[@]}"; do
  for T in "${THREADS[@]}"; do
    best_line=""
    best_v2=1e30
    for trial in $(seq 1 "$TRIALS"); do
      line=$("$MLE" solve "$N" "$T" "$REPEAT")
      v2=$(echo "$line" | cut -d, -f9)
      if awk "BEGIN{exit !($v2 < $best_v2)}"; then
        best_v2=$v2
        best_line=$line
      fi
    done
    echo "$best_line" >> "$OUT"
    echo "  N=$N threads=$T -> $best_line"
  done
done

echo "Wrote $OUT"
