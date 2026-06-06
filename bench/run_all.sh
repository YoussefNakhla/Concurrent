#!/usr/bin/env bash
#I used ai to help automate the tests I wanted to do
set -e

# Benchmark settings
SIZES=(50000 100000 200000 500000)
THREADS=(1 2 3 4 8)

REPEAT=50
REPEATS=(1 10 50 100 200 500)

REP_N=200000
REP_T=4
TRIALS=3

# Machine/output setup
if [ -z "$MACHINE" ]; then
  if [ "$(uname)" = "Darwin" ]; then
    MACHINE="mac"
  elif grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    MACHINE="xps"
  else
    MACHINE="$(uname -n | tr -cd '[:alnum:]')"
  fi
fi

OUTDIR="results/$MACHINE"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "Machine: $MACHINE"
echo "Output : $OUTDIR"
echo "Cores  : $( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo '?' )"
echo "Sizes  : ${SIZES[*]}"
echo "Threads: ${THREADS[*]}"
echo "Repeat : $REPEAT"
echo "Trials : $TRIALS"
echo "============================================================"

# Check build files
for exe in build/mle build/poisson_mle; do
  if [ ! -x "$exe" ]; then
    echo "ERROR: $exe not found."
    echo "Build first with:"
    echo "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build"
    exit 1
  fi
done

# Sweep dataset sizes and thread counts
sweep_threads() {
  local exe="$1"
  local out="$2"

  echo "solve,N,threads,repeat,iters,loglik,ms_serial,ms_v1,ms_v2,match" > "$out"

  for N in "${SIZES[@]}"; do
    for T in "${THREADS[@]}"; do
      local best_s=1e30
      local best_v1=1e30
      local best_v2=1e30
      local meta=""

      for trial in $(seq 1 "$TRIALS"); do
        local line
        line=$("$exe" solve "$N" "$T" "$REPEAT")

        local s
        local v1
        local v2

        s=$(echo "$line" | cut -d, -f7)
        v1=$(echo "$line" | cut -d, -f8)
        v2=$(echo "$line" | cut -d, -f9)

        [ -z "$meta" ] && meta="$line"

        awk "BEGIN{exit !($s < $best_s)}" && best_s=$s
        awk "BEGIN{exit !($v1 < $best_v1)}" && best_v1=$v1
        awk "BEGIN{exit !($v2 < $best_v2)}" && best_v2=$v2
      done

      local iters
      local loglik
      local match

      iters=$(echo "$meta" | cut -d, -f5)
      loglik=$(echo "$meta" | cut -d, -f6)
      match=$(echo "$meta" | cut -d, -f10)

      echo "solve,$N,$T,$REPEAT,$iters,$loglik,$best_s,$best_v1,$best_v2,$match" >> "$out"

      printf "N=%-7s T=%s serial=%-9s v1=%-9s v2=%-9s match=%s\n" \
        "$N" "$T" "$best_s" "$best_v1" "$best_v2" "$match"
    done
  done

  echo "Wrote $out"
}

# Sweep repeat factor
sweep_repeat() {
  local exe="$1"
  local out="$2"

  echo "repeat,N,threads,ms_serial,ms_v2,speedup" > "$out"

  for R in "${REPEATS[@]}"; do
    local best_s=1e30
    local best_v2=1e30

    for trial in $(seq 1 "$TRIALS"); do
      local line
      line=$("$exe" solve "$REP_N" "$REP_T" "$R")

      local s
      local v2

      s=$(echo "$line" | cut -d, -f7)
      v2=$(echo "$line" | cut -d, -f9)

      awk "BEGIN{exit !($s < $best_s)}" && best_s=$s
      awk "BEGIN{exit !($v2 < $best_v2)}" && best_v2=$v2
    done

    local speedup
    speedup=$(awk "BEGIN{printf \"%.3f\", $best_s / $best_v2}")

    echo "$R,$REP_N,$REP_T,$best_s,$best_v2,$speedup" >> "$out"

    printf "repeat=%-4s serial=%-9s v2=%-9s speedup=%s\n" \
      "$R" "$best_s" "$best_v2" "$speedup"
  done

  echo "Wrote $out"
}

# Run benchmarks
echo
echo "[1/5] Logit benchmark"
sweep_threads build/mle "$OUTDIR/logit_bench.csv"

echo
echo "[2/5] Poisson benchmark"
sweep_threads build/poisson_mle "$OUTDIR/poisson_bench.csv"

echo
echo "[3/5] Logit repeat sweep"
sweep_repeat build/mle "$OUTDIR/logit_repeat.csv"

echo
echo "[4/5] Poisson repeat sweep"
sweep_repeat build/poisson_mle "$OUTDIR/poisson_repeat.csv"

# Create plots and tables
echo
echo "[5/5] Plots and tables"
python3 bench/plot2.py "$OUTDIR/logit_bench.csv" --label "Logit ($MACHINE)" --prefix "$OUTDIR/logit"
python3 bench/plot2.py "$OUTDIR/poisson_bench.csv" --label "Poisson ($MACHINE)" --prefix "$OUTDIR/poisson"

python3 bench/plot2.py "$OUTDIR/logit_bench.csv" --repeat "$OUTDIR/logit_repeat.csv" --label "Logit ($MACHINE)" --prefix "$OUTDIR/logit" --quiet
python3 bench/plot2.py "$OUTDIR/poisson_bench.csv" --repeat "$OUTDIR/poisson_repeat.csv" --label "Poisson ($MACHINE)" --prefix "$OUTDIR/poisson" --quiet

# Final summary
echo
echo "============================================================"
echo "Done. Results are in $OUTDIR"
echo
echo "CSV:"
echo "  logit_bench.csv"
echo "  poisson_bench.csv"
echo "  logit_repeat.csv"
echo "  poisson_repeat.csv"
echo
echo "Plots:"
echo "  logit_runtime_vs_threads.png"
echo "  logit_speedup_vs_threads.png"
echo "  logit_efficiency_vs_threads.png"
echo "  logit_v1_vs_v2_speedup.png"
echo "  logit_repeat.png"
echo "  poisson_runtime_vs_threads.png"
echo "  poisson_speedup_vs_threads.png"
echo "  poisson_efficiency_vs_threads.png"
echo "  poisson_v1_vs_v2_speedup.png"
echo "  poisson_repeat.png"
echo
echo "Tables:"
echo "  logit_efficiency_table.txt"
echo "  poisson_efficiency_table.txt"
echo "============================================================"