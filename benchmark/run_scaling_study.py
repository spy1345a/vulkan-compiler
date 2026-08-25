# benchmark/run_scaling_study.py

import argparse
import contextlib
import io
import os
import sys
import time
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.harness import (
    EXPRESSIONS,
    compile_expression,
    flatten_expression,
    ordered_var_names,
    gen_inputs,
    describe,
    plan_reps,
    cpu_eval_batch,
    gpu_eval_batch,
)
from compiler.gpu.dispatch import GPUExecutor, compile_shader

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIZES = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def get_device_name():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gpu = GPUExecutor()
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            return gpu, line.removeprefix("GPU:").strip()
    return gpu, "unknown"


def get_cpu_name():
    name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return name


def methodology(args):
    return [
        "-" * 110,
        " WHAT THIS TEST DOES",
        "-" * 110,
        f"1. Workload   : {len(EXPRESSIONS)} arithmetic expressions, each"
        f" evaluated for N independent variable assignments",
        f"               (N = batch size, swept from {args.sizes[0]} to"
        f" {args.sizes[-1]:,}), on both backends.",
        "2. CPU mode   : sequential tree-walking evaluation per instance"
        " (includes Python dict construction of the",
        "                variable environment - that is part of interpreter"
        " throughput).",
        "3. GPU mode   : ONE data-parallel dispatch per measurement"
        f" (local_size_x = {GPUExecutor.LOCAL_SIZE}, one invocation per"
        " instance);",
        "                glslc compilation cached, excluded from timings.",
        "4. Statistics : each cell repeats up to"
        f" {args.max_runs} times; repetitions adapt to a {args.budget:.0f} s"
        " per-cell budget",
        f"               (never below {args.min_reps}). Reported: median,"
        " mean +/- std, 95% CI half-width (normal approx), min/max.",
        "5. Device time: Vulkan timestamp queries bracketing the dispatch",
        "6. Correctness: every GPU cell is verified against the CPU result of"
        " the first instance.",
        "",
    ]


def fmt_row(r):
    return (
        f"{r['n']:>9} {r['reps']:>5} "
        f"{r['cpu']['median']:>12.4f} {r['cpu']['ci95']:>10.4f} "
        f"{r['cpu_rate']:>13,.0f} "
        f"{r['gpu']['median']:>12.4f} {r['gpu']['ci95']:>10.4f} "
        f"{r['gpu_rate']:>13,.0f} "
        f"{r['speedup']:>8.2f}x {r['dev_exec']:>12.4f}"
    )


TABLE_HEADER = (
    f"{'N':>9} {'Reps':>5} {'CPU med (ms)':>12} {'CI95':>10} "
    f"{'CPU ev/s':>13} {'GPU med (ms)':>12} {'CI95':>10} "
    f"{'GPU ev/s':>13} {'Speedup':>9} {'DevExec(ms)':>12}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--max-runs", type=int, default=300)
    ap.add_argument("--min-reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=8.0,
                    help="seconds budgeted per benchmark cell")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.perf_counter()

    print("Pre-compiling shader once (cached)...")
    t0 = time.perf_counter()
    compile_shader()
    print(f"  glslc took {(time.perf_counter() - t0) * 1000:.3f} ms")

    cpu_name = get_cpu_name()
    gpu, device_name = get_device_name()
    print(f"CPU: {cpu_name}")
    print(f"GPU: {device_name}")

    report = [
        "=" * 110,
        " CPU vs GPU BATCH SCALING STUDY - STATISTICAL RESULTS",
        "=" * 110,
        f"Started   : {started}",
        f"Platform  : {platform.platform()}",
        f"Python    : {platform.python_version()}",
        f"CPU       : {cpu_name}",
        f"GPU       : {device_name}",
        f"Sizes     : {', '.join(f'{s:,}' for s in args.sizes)}",
        f"Runs      : up to {args.max_runs} per cell"
        f" ({args.budget:.0f}s/cell budget, floor {args.min_reps})",
        "",
    ]
    report += methodology(args)

    all_crossovers = []

    for expr_idx, code in enumerate(EXPRESSIONS, 1):
        ast = compile_expression(code)
        flat, consts, num_vars, names = flatten_expression(ast)

        print(f"\n[{expr_idx}/{len(EXPRESSIONS)}] {code}"
              f"   ({num_vars} vars)")
        section = [
            "-" * 110,
            f" [{expr_idx}] {code}   ({num_vars} variables)",
            "-" * 110,
            TABLE_HEADER,
        ]

        expr_crossover = None
        for n in args.sizes:
            rows = gen_inputs(n, num_vars, seed=args.seed)

            reps = plan_reps(
                lambda: gpu_eval_batch(gpu, flat, consts, num_vars, rows),
                max_reps=args.max_runs, min_reps=args.min_reps,
                budget_s=args.budget)

            cpu_samples, gpu_samples, dev_samples = [], [], []
            ref_cpu = None
            for _ in range(reps):
                cpu_res, cpu_ms = cpu_eval_batch(ast, rows, names)
                if ref_cpu is None:
                    ref_cpu = cpu_res[0]
                cpu_samples.append(cpu_ms)

                g_res, g_ms, t = gpu_eval_batch(gpu, flat, consts, num_vars,
                                                rows)
                gpu_samples.append(g_ms)
                dev_samples.append(t["gpu_exec"])

            c_st, g_st = describe(cpu_samples), describe(gpu_samples)
            d_st = describe(dev_samples)
            speedup = c_st["median"] / g_st["median"]
            if expr_crossover is None and speedup > 1.0:
                expr_crossover = n

            err = abs(g_res[0] - ref_cpu) / max(abs(ref_cpu), 1e-9)
            assert err < 1e-3, f"verification failed at N={n}: {err}"

            row = {
                "n": n, "reps": reps, "cpu": c_st, "gpu": g_st,
                "cpu_rate": n / (c_st["median"] / 1000.0),
                "gpu_rate": n / (g_st["median"] / 1000.0),
                "speedup": speedup,
                "dev_exec": d_st["median"],
            }
            section.append(fmt_row(row))
            print(f"  N={n:>9,} reps={reps:>3}  "
                  f"cpu {c_st['median']:>10.3f} ms  "
                  f"gpu {g_st['median']:>10.3f} ms  "
                  f"({speedup:>6.2f}x)")

        section.append("-" * 110)
        cross_txt = (f"N = {expr_crossover:,}" if expr_crossover
                     else "not reached within tested sizes")
        section.append(f" Crossover (GPU faster than CPU): {cross_txt}")
        section.append("")
        all_crossovers.append((code, expr_crossover))
        report += section

    total_min = (time.perf_counter() - t_start) / 60.0
    report += [
        "=" * 110,
        " OVERALL SUMMARY",
        "=" * 110,
        f"{'Expression':<36} {'Crossover N':>14} {'Max GPU evals/s':>17}",
        "-" * 110,
    ]
    for code, xo in all_crossovers:
        report.append(f"{code:<36} "
                      f"{(f'{xo:,}' if xo else 'not reached'):>14} "
                      f"{'see sections':>17}")
    report += [
        "-" * 110,
        f"Total run time: {total_min:.1f} min",
        "Note: repetition counts are adaptive (time-budgeted); each row shows"
        " its own count. Cells with large",
        "      batches use fewer repetitions to keep total runtime sane;"
        " medians remain robust because",
        "      every repetition aggregates the full batch of N evaluations.",
        "=" * 110,
    ]

    path = os.path.join(OUT_DIR, "scaling_stats.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
