#!/usr/bin/env python3
#Help of ai to generate the graphs
import argparse
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Load benchmark CSV
def load_bench(path):
    rows = []

    with open(path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "N": int(row["N"]),
                "threads": int(row["threads"]),
                "ms_serial": float(row["ms_serial"]),
                "ms_v1": float(row["ms_v1"]),
                "ms_v2": float(row["ms_v2"]),
            })

    return rows


# Load repeat sweep CSV
def load_repeat(path):
    rows = []

    with open(path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "repeat": int(row["repeat"]),
                "ms_serial": float(row["ms_serial"]),
                "ms_v2": float(row["ms_v2"]),
                "speedup": float(row["speedup"]),
            })

    return rows


# Pick serial baseline for one N
def serial_baseline(rows, N):
    return min(row["ms_serial"] for row in rows if row["N"] == N)


# Plot repeat sweep
def save_repeat_plot(rows, args, tag):
    rows.sort(key=lambda row: row["repeat"])

    repeats = [row["repeat"] for row in rows]
    speedups = [row["speedup"] for row in rows]

    plt.figure()
    plt.plot(repeats, speedups, marker="o")
    plt.xscale("log")
    plt.xlabel("repeat factor")
    plt.ylabel("speedup (serial / v2)")
    plt.title(f"Speedup vs repeated work{tag}")
    plt.grid(True, alpha=0.3)
    plt.savefig(args.prefix + "_repeat.png", dpi=120, bbox_inches="tight")

    if not args.quiet:
        print("Wrote", args.prefix + "_repeat.png")


# Plot runtime
def save_runtime_plot(rows, sizes, args, tag):
    plt.figure()

    for N in sizes:
        points = sorted(
            (row["threads"], row["ms_v2"])
            for row in rows
            if row["N"] == N
        )

        plt.plot(
            [threads for threads, _ in points],
            [ms for _, ms in points],
            marker="o",
            label=f"N={N}",
        )

    plt.xlabel("threads")
    plt.ylabel("runtime (ms)")
    plt.title(f"Runtime using v2{tag}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.prefix + "_runtime_vs_threads.png", dpi=120, bbox_inches="tight")


# Plot speedup
def save_speedup_plot(rows, sizes, max_threads, args, tag):
    plt.figure()
    plt.plot([1, max_threads], [1, max_threads], "k--", alpha=0.5, label="ideal")

    for N in sizes:
        base = serial_baseline(rows, N)

        points = sorted(
            (row["threads"], base / row["ms_v2"])
            for row in rows
            if row["N"] == N
        )

        plt.plot(
            [threads for threads, _ in points],
            [speedup for _, speedup in points],
            marker="o",
            label=f"N={N}",
        )

    plt.xlabel("threads")
    plt.ylabel("speedup (serial / v2)")
    plt.title(f"Parallel speedup using v2{tag}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.prefix + "_speedup_vs_threads.png", dpi=120, bbox_inches="tight")


# Plot efficiency
def save_efficiency_plot(rows, sizes, args, tag):
    plt.figure()
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="ideal")

    for N in sizes:
        base = serial_baseline(rows, N)

        points = sorted(
            (row["threads"], (base / row["ms_v2"]) / row["threads"])
            for row in rows
            if row["N"] == N
        )

        plt.plot(
            [threads for threads, _ in points],
            [eff for _, eff in points],
            marker="o",
            label=f"N={N}",
        )

    plt.xlabel("threads")
    plt.ylabel("efficiency")
    plt.title(f"Parallel efficiency using v2{tag}")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.prefix + "_efficiency_vs_threads.png", dpi=120, bbox_inches="tight")


# Compare v1 and v2
def save_v1_v2_plot(rows, sizes, max_threads, args, tag):
    N = sizes[-1]
    base = serial_baseline(rows, N)

    plt.figure()
    plt.plot([1, max_threads], [1, max_threads], "k--", alpha=0.5, label="ideal")

    for column, label in (("ms_v1", "v1 mutex"), ("ms_v2", "v2 partial sums")):
        points = sorted(
            (row["threads"], base / row[column])
            for row in rows
            if row["N"] == N
        )

        plt.plot(
            [threads for threads, _ in points],
            [speedup for _, speedup in points],
            marker="o",
            label=label,
        )

    plt.xlabel("threads")
    plt.ylabel("speedup")
    plt.title(f"v1 vs v2 speedup, N={N}{tag}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(args.prefix + "_v1_vs_v2_speedup.png", dpi=120, bbox_inches="tight")


# Write efficiency table
def write_efficiency_table(rows, sizes, args, tag):
    table_path = args.prefix + "_efficiency_table.txt"

    with open(table_path, "w") as f:
        f.write(f"Efficiency table{tag}\n")
        f.write(f"{'N':>8} {'thr':>4} {'serial':>10} {'v2(ms)':>10} ")
        f.write(f"{'speedup':>8} {'eff%':>6}\n")

        for N in sizes:
            base = serial_baseline(rows, N)

            subset = sorted(
                (row for row in rows if row["N"] == N),
                key=lambda row: row["threads"],
            )

            for row in subset:
                speedup = base / row["ms_v2"]
                efficiency = 100.0 * speedup / row["threads"]

                f.write(
                    f"{N:>8} "
                    f"{row['threads']:>4} "
                    f"{row['ms_serial']:>10.1f} "
                    f"{row['ms_v2']:>10.1f} "
                    f"{speedup:>8.2f} "
                    f"{efficiency:>6.1f}\n"
                )

    return table_path


# Parse inputs and run
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_csv")
    parser.add_argument("--prefix", default="results/out")
    parser.add_argument("--label", default="")
    parser.add_argument("--repeat", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    rows = load_bench(args.bench_csv)

    if not rows:
        print("No rows in", args.bench_csv)
        return

    sizes = sorted({row["N"] for row in rows})
    max_threads = max(row["threads"] for row in rows)
    tag = f" -- {args.label}" if args.label else ""

    if args.repeat:
        repeat_rows = load_repeat(args.repeat)
        save_repeat_plot(repeat_rows, args, tag)
        return

    save_runtime_plot(rows, sizes, args, tag)
    save_speedup_plot(rows, sizes, max_threads, args, tag)
    save_efficiency_plot(rows, sizes, args, tag)
    save_v1_v2_plot(rows, sizes, max_threads, args, tag)

    table_path = write_efficiency_table(rows, sizes, args, tag)

    if not args.quiet:
        print("Wrote:")
        print(" ", args.prefix + "_runtime_vs_threads.png")
        print(" ", args.prefix + "_speedup_vs_threads.png")
        print(" ", args.prefix + "_efficiency_vs_threads.png")
        print(" ", args.prefix + "_v1_vs_v2_speedup.png")
        print(" ", table_path)


if __name__ == "__main__":
    main()