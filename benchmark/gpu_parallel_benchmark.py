# benchmark/gpu_parallel_benchmark.py

import argparse
import contextlib
import io
import os
import sys
import time
import platform
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler import Lexer, Parser
from compiler.gpu import Flattener
from compiler.gpu.dispatch import GPUExecutor, compile_shader

STAGES = ["compile", "buffers", "setup", "dispatch", "readback"]
ENV = {"a": 10.0, "b": 5.0, "c": 2.5, "d": 4.0}
EXPRESSION = "((a * b) + (c * d)) / (a - b)"
DEFAULT_SIZES = [10, 25, 50, 100, 250, 500, 750, 1000]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_ast(code):
    return Parser(Lexer(code).tokenize()).parse()


def build_executor_pool(count):
    buf = io.StringIO()
    name = ""
    pool = []
    with contextlib.redirect_stdout(buf):
        for _ in range(count):
            pool.append(GPUExecutor())
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            name = line.removeprefix("GPU:").strip()
    return name, pool


def gpu_task(gpu):
    f = Flattener()
    f.flatten(compile_ast(EXPRESSION))
    variables = [None] * len(f.var_map)
    for var_name, (_, idx) in f.var_map.items():
        variables[idx] = ENV[var_name]
    t0 = time.perf_counter()
    result = gpu.run(f.get_flat(), f.const_values, variables)
    wall = (time.perf_counter() - t0) * 1000.0
    return result, wall, dict(gpu.last_timings)


def run_size(n, executors):
    workers = len(executors)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        futures = [pool.submit(gpu_task, executors[i % workers]) for i in range(n)]
        results = [f.result() for f in futures]
        wall = (time.perf_counter() - t0) * 1000.0

    task_times = [r[1] for r in results]
    stages = {
        s: sum(r[2].get(s, 0.0) for r in results) / n for s in STAGES
    }
    return {
        "n": n,
        "workers": workers,
        "wall": wall,
        "avg": sum(task_times) / len(task_times),
        "min": min(task_times),
        "max": max(task_times),
        "throughput": n / (wall / 1000.0),
        "parallelism": sum(task_times) / wall,
        "stages": stages,
    }


def header(title, device):
    return [
        "=" * 100,
        f" {title}",
        "=" * 100,
        f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform  : {platform.platform()}",
        f"Python    : {platform.python_version()}",
        f"Device    : {device}",
        f"Expression: {EXPRESSION}",
        f"Mode      : ThreadPoolExecutor, pre-built pool of GPUExecutor"
        f" instances (one Vulkan instance each)",
        f"Task      : lex -> parse -> flatten -> dispatch (glslc compile cached,"
        f" timed separately)",
        "-" * 100,
        "",
    ]


def summary_table(rows):
    lines = [
        "-" * 100,
        f"{'N':>6} {'Workers':>8} {'Wall (ms)':>13} {'Tasks/s':>9}"
        f" {'Parallel':>10} {'AvgTask':>10} {'Compile':>10} {'Setup':>10}"
        f" {'Dispatch':>11} {'Readback':>10}",
        "-" * 100,
    ]
    for r in rows:
        st = r["stages"]
        lines.append(
            f"{r['n']:>6} {r['workers']:>8} {r['wall']:>13.4f}"
            f" {r['throughput']:>9.2f} {r['parallelism']:>9.2f}x"
            f" {r['avg']:>10.4f} {st['compile']:>10.4f} {st['buffers'] + st['setup']:>10.4f}"
            f" {st['dispatch']:>11.4f} {st['readback']:>10.4f}"
        )
    lines.append("-" * 100)
    lines.append("=" * 100)
    return lines


def export(rows, device):
    lines = header("GPU PARALLEL BENCHMARK RESULTS", device)
    lines += summary_table(rows)
    path = os.path.join(OUT_DIR, "gpu_parallel_benchmark.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


def get_device_name():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        GPUExecutor()
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            return line.removeprefix("GPU:").strip()
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--workers-cap", type=int, default=64)
    args = ap.parse_args()

    max_workers = max(min(n, args.workers_cap) for n in args.sizes)

    print("Pre-compiling shader once (cached from here on)...")
    t0 = time.perf_counter()
    compile_shader()
    print(f"  glslc took {(time.perf_counter() - t0) * 1000:.4f} ms "
          f"(cached on disk + memory)")

    device_name = get_device_name()
    print(f"Building executor pool ({max_workers} Vulkan instances)...")
    _, executors = build_executor_pool(max_workers)
    print(f"Device: {device_name}")

    rows = []
    for n in args.sizes:
        workers = max(1, min(n, args.workers_cap))
        row = run_size(n, executors[:workers])
        rows.append(row)
        st = row["stages"]
        print(f"  N={n:<5} workers={workers:<4} "
              f"wall {row['wall']:.4f} ms   "
              f"{row['throughput']:.2f} tasks/s   "
              f"{row['parallelism']:.2f}x parallel   "
              f"(compile {st['compile']:.4f}, dispatch {st['dispatch']:.4f} ms/task)")

    export(rows, device_name)


if __name__ == "__main__":
    main()
