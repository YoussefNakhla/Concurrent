#!/usr/bin/env python3
"""Plot benchmark results produced by bench/run_bench.sh.

Usage:  python3 bench/plot.py [results/bench.csv]
Outputs PNGs into results/.

CSV columns:
  solve,N,threads,repeat,iters,loglik,ms_serial,ms_v1,ms_v2,match

Again use of ai here to help generate graphs
"""
import sys
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "N": int(r["N"]),
                "threads": int(r["threads"]),
                "ms_serial": float(r["ms_serial"]),
                "ms_v1": float(r["ms_v1"]),
                "ms_v2": float(r["ms_v2"]),
            })
    return rows

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/bench.csv"
    rows = load(path)
    if not rows:
        print("No rows in", path); return

    sizes = sorted({r["N"] for r in rows})
    # serial time is thread-independent; take it from the threads==1 row per N
    serial_ms = {}
    for N in sizes:
        cand = [r["ms_serial"] for r in rows if r["N"] == N]
        serial_ms[N] = min(cand)  # all roughly equal; min is safe

    # ---- Plot 1: runtime (v2) vs threads, one line per N ----
    plt.figure()
    for N in sizes:
        pts = sorted((r["threads"], r["ms_v2"]) for r in rows if r["N"] == N)
        xs = [t for t, _ in pts]; ys = [m for _, m in pts]
        plt.plot(xs, ys, marker="o", label=f"N={N}")
    plt.xlabel("threads"); plt.ylabel("runtime (ms)")
    plt.title("Full-solve runtime (v2 partial sums) vs threads")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/runtime_vs_threads.png", dpi=120, bbox_inches="tight")

    # ---- Plot 2: speedup (v2) vs threads, with ideal line ----
    plt.figure()
    max_t = max(r["threads"] for r in rows)
    plt.plot([1, max_t], [1, max_t], "k--", alpha=0.5, label="ideal")
    for N in sizes:
        pts = sorted((r["threads"], serial_ms[N] / r["ms_v2"])
                     for r in rows if r["N"] == N)
        xs = [t for t, _ in pts]; ys = [s for _, s in pts]
        plt.plot(xs, ys, marker="o", label=f"N={N}")
    plt.xlabel("threads"); plt.ylabel("speedup  (serial / v2)")
    plt.title("Parallel speedup (v2 partial sums)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/speedup_vs_threads.png", dpi=120, bbox_inches="tight")

    # ---- Plot 3: v1 (mutex) vs v2 (partial sums) speedup, largest N ----
    Nbig = sizes[-1]
    plt.figure()
    plt.plot([1, max_t], [1, max_t], "k--", alpha=0.5, label="ideal")
    for col, lab in (("ms_v1", "v1 (mutex)"), ("ms_v2", "v2 (partial sums)")):
        pts = sorted((r["threads"], serial_ms[Nbig] / r[col])
                     for r in rows if r["N"] == Nbig)
        xs = [t for t, _ in pts]; ys = [s for _, s in pts]
        plt.plot(xs, ys, marker="o", label=lab)
    plt.xlabel("threads"); plt.ylabel("speedup  (serial / parallel)")
    plt.title(f"Lock contention: v1 vs v2  (N={Nbig})")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/v1_vs_v2_speedup.png", dpi=120, bbox_inches="tight")

    print("Wrote results/runtime_vs_threads.png, "
          "results/speedup_vs_threads.png, results/v1_vs_v2_speedup.png")

if __name__ == "__main__":
    main()