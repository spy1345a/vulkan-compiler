# benchmark/gpu_parallel_benchmark.py

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import platform
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler import Lexer, Parser
from compiler.gpu import Flattener
from compiler.gpu import dispatch as gpu_dispatch
from compiler.gpu.dispatch import GPUExecutor, SHADER_PATH

ENV = {"a": 10.0, "b": 5.0, "c": 2.5, "d": 4.0}
EXPRESSION = "((a * b) + (c * d)) / (a - b)"
DEFAULT_SIZES = [10, 25, 50, 100, 250, 500, 750, 1000]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

_local = threading.local()


def compile_ast(code):
    return Parser(Lexer(code).tokenize()).parse()


def safe_compile_shader():
    fd, out_path = tempfile.mkstemp(suffix=".spv")
    os.close(fd)
    try:
        r = subprocess.run(
            ["glslc", SHADER_PATH, "-o", out_path],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"glslc failed:\n{r.stderr}")
        with open(out_path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(out_path)


gpu_dispatch.compile_shader = safe_compile_shader


def get_executor():
    if getattr(_local, "gpu", None) is None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _local.gpu = GPUExecutor()
    return _local.gpu


def get_device_name():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        GPUExecutor()
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            return line.removeprefix("GPU:").strip()
    return "unknown"


def gpu_task():
    gpu = get_executor()
    f = Flattener()
    f.flatten(compile_ast(EXPRESSION))
    variables = [None] * len(f.var_map)
    for name, (_, idx) in f.var_map.items():
        variables[idx] = ENV[name]
    t0 = time.perf_counter()
    result = gpu.run(f.get_flat(), f.const_values, variables)
    return result, (time.perf_counter() - t0) * 1000.0


def run_size(n, workers_cap):
    workers = max(1, min(n, workers_cap))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        warmups = [pool.submit(get_executor) for _ in range(workers)]
        for w in warmups:
            w.result()

        t0 = time.perf_counter()
        futures = [pool.submit(gpu_task) for _ in range(n)]
        task_times = [f.result()[1] for f in futures]
        wall = (time.perf_counter() - t0) * 1000.0

    return {
        "n": n,
        "workers": workers,
        "wall": wall,
        "avg": sum(task_times) / len(task_times),
        "min": min(task_times),
        "max": max(task_times),
        "throughput": n / (wall / 1000.0),
        "parallelism": sum(task_times) / wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--workers-cap", type=int, default=64)
    args = ap.parse_args()

    device_name = get_device_name()
    print(f"Device: {device_name}")

    rows = []
    for n in args.sizes:
        workers = max(1, min(n, args.workers_cap))
        row = run_size(n, args.workers_cap)
        rows.append(row)
        print(f"  N={n:<5} workers={workers:<4} "
              f"wall {row['wall']:.4f} ms   "
              f"task avg {row['avg']:.4f} ms   "
              f"{row['throughput']:.2f} tasks/s   "
              f"{row['parallelism']:.2f}x parallel")

    export(args.sizes, device_name, rows)


def header(title, device):
    return [
        "=" * 78,
        f" {title}",
        "=" * 78,
        f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform  : {platform.platform()}",
        f"Python    : {platform.python_version()}",
        f"Device    : {device}",
        f"Expression: {EXPRESSION}",
        f"Mode      : ThreadPoolExecutor, one GPUExecutor (own Vulkan"
        f" instance) per worker thread",
        f"Task      : lex -> parse -> flatten -> glslc compile -> Vulkan dispatch",
        "-" * 78,
        "",
    ]


def summary_table(rows):
    lines = [
        "-" * 78,
        f"{'N':>6} {'Workers':>8} {'Wall (ms)':>14} {'Avg task (ms)':>15}"
        f" {'Min task (ms)':>15} {'Max task (ms)':>15} {'Tasks/s':>12} {'Parallel':>10}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['n']:>6} {r['workers']:>8} {r['wall']:>14.4f}"
            f" {r['avg']:>15.4f} {r['min']:>15.4f} {r['max']:>15.4f}"
            f" {r['throughput']:>12.2f} {r['parallelism']:>9.2f}x"
        )
    lines.append("-" * 78)
    lines.append("=" * 78)
    return lines


def export(sizes, device, rows):
    lines = header("GPU PARALLEL BENCHMARK RESULTS", device)
    lines += summary_table(rows)
    path = os.path.join(OUT_DIR, "gpu_parallel_benchmark.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
