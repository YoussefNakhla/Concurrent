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
THREADS=(1 2 3 4 8)
TRIALS=3

echo "solve,N,threads,repeat,iters,loglik,ms_serial,ms_v1,ms_v2,match" > "$OUT"

for N in "${SIZES[@]}"; do
  for T in "${THREADS[@]}"; do
    best_serial=1e30
    best_v1=1e30
    best_v2=1e30
    meta_line=""
    for trial in $(seq 1 "$TRIALS"); do
      line=$("$MLE" solve "$N" "$T" "$REPEAT")
      ms_serial=$(echo "$line" | cut -d, -f7)
      ms_v1=$(echo "$line"     | cut -d, -f8)
      ms_v2=$(echo "$line"     | cut -d, -f9)
      # Save first trial for metadata (iters, loglik, match are deterministic).
      [ -z "$meta_line" ] && meta_line="$line"
      awk "BEGIN{exit !($ms_serial < $best_serial)}" && best_serial=$ms_serial
      awk "BEGIN{exit !($ms_v1    < $best_v1)}"    && best_v1=$ms_v1
      awk "BEGIN{exit !($ms_v2    < $best_v2)}"    && best_v2=$ms_v2
    done
    # Reconstruct row with per-backend best times; metadata fields are invariant.
    prefix=$(echo "$meta_line" | cut -d, -f1-6)
    match=$(echo  "$meta_line" | cut -d, -f10)
    row="$prefix,$best_serial,$best_v1,$best_v2,$match"
    echo "$row" >> "$OUT"
    echo "  N=$N T=$T -> $row"
  done
done

echo "Wrote $OUT"
