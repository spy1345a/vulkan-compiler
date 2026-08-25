# benchmark/benchmark.py

import io
import os
import sys
import time
import platform
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler import Lexer, Parser, Evaluator
from compiler.gpu import Flattener, GPUExecutor

RUNS = 10
ENV = {"a": 10.0, "b": 5.0, "c": 2.5, "d": 4.0}

EXPRESSIONS = [
    "a + b",
    "a - b",
    "a * b",
    "a / b",
    "a + b * 2",
    "(a + b) * c",
    "a * b - c / d",
    "(a - b) / c + d",
    "a + b - c * d / 2",
    "((a * b) + (c * d)) / (a - b)",
]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_ast(code):
    return Parser(Lexer(code).tokenize()).parse()


def run_cpu(code):
    return Evaluator(ENV).eval(compile_ast(code))


def run_gpu(code, gpu):
    f = Flattener()
    f.flatten(compile_ast(code))
    variables = [None] * len(f.var_map)
    for name, (_, idx) in f.var_map.items():
        variables[idx] = ENV[name]
    return gpu.run(f.get_flat(), f.const_values, variables)


def bench(fn, code, runs=RUNS):
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(code)
        times.append((time.perf_counter() - t0) * 1000.0)
    return result, times


def header(title, device):
    return [
        "=" * 78,
        f" {title}",
        "=" * 78,
        f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform  : {platform.platform()}",
        f"Python    : {platform.python_version()}",
        f"Device    : {device}",
        f"Runs      : {RUNS} per expression",
        "-" * 78,
        "",
    ]


def detail_block(results):
    lines = []
    for i, (code, result, times) in enumerate(results, 1):
        avg = sum(times) / len(times)
        lines.append(f"[{i:02d}] {code}")
        lines.append(f"     result : {result:.6f}")
        for r, t in enumerate(times, 1):
            lines.append(f"     run {r:02d}  : {t:.4f} ms")
        lines.append(f"     avg    : {avg:.4f} ms   "
                     f"min: {min(times):.4f} ms   max: {max(times):.4f} ms")
        lines.append("")
    return lines


def summary_table(results):
    total_avg = 0.0
    lines = [
        "-" * 78,
        f"{'#':<3} {'Expression':<32} {'Avg (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}",
        "-" * 78,
    ]
    for i, (code, _, times) in enumerate(results, 1):
        avg = sum(times) / len(times)
        total_avg += avg
        lines.append(f"{i:<3} {code:<32} {avg:>12.4f} {min(times):>12.4f} {max(times):>12.4f}")
    lines.append("-" * 78)
    lines.append(f"{'':<3} {'TOTAL AVG':<32} {total_avg / len(results):>12.4f}")
    lines.append("=" * 78)
    return lines


def export(path, title, device, results):
    lines = header(title, device)
    lines += detail_block(results)
    lines += summary_table(results)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def get_gpu_device_name():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gpu = GPUExecutor()
    name = ""
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            name = line.removeprefix("GPU:").strip()
    return gpu, name


def main():
    print("Benchmarking CPU...")
    cpu_results = []
    for code in EXPRESSIONS:
        result, times = bench(run_cpu, code)
        cpu_results.append((code, result, times))
        print(f"  [CPU] {code:<35} avg {sum(times)/len(times):.4f} ms -> {result}")

    print("\nBenchmarking GPU (each run includes glslc shader compilation + dispatch)...")
    gpu_exec, gpu_name = get_gpu_device_name()
    gpu_results = []
    for code in EXPRESSIONS:
        result, times = bench(lambda c: run_gpu(c, gpu_exec), code)
        gpu_results.append((code, result, times))
        print(f"  [GPU] {code:<35} avg {sum(times)/len(times):.4f} ms -> {result}")

    cpu_path = os.path.join(OUT_DIR, "cpu_benchmark.txt")
    gpu_path = os.path.join(OUT_DIR, "gpu_benchmark.txt")

    export(cpu_path, "CPU BENCHMARK RESULTS", platform.processor() or "CPU", cpu_results)
    export(gpu_path, "GPU BENCHMARK RESULTS", gpu_name, gpu_results)

    print(f"\nWrote {cpu_path}")
    print(f"Wrote {gpu_path}")


if __name__ == "__main__":
    main()
