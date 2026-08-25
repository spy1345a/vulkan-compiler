# benchmark/head_to_head_benchmark.py

import argparse
import contextlib
import io
import os
import sys
import time
import platform
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler import Lexer, Parser, Evaluator
from compiler.gpu import Flattener
from compiler.gpu.dispatch import GPUExecutor, compile_shader

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
STAGES = ["compile", "buffers", "setup", "dispatch", "readback", "gpu_exec"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_ast(code):
    return Parser(Lexer(code).tokenize()).parse()


def cpu_task(idx_code):
    _, code = idx_code
    t0 = time.perf_counter()
    result = Evaluator(ENV).eval(compile_ast(code))
    return result, (time.perf_counter() - t0) * 1000.0


def make_gpu_task(executors):
    def gpu_task(idx_code):
        idx, code = idx_code
        gpu = executors[idx % len(executors)]
        f = Flattener()
        f.flatten(compile_ast(code))
        variables = [None] * len(f.var_map)
        for var_name, (_, vidx) in f.var_map.items():
            variables[vidx] = ENV[var_name]
        t0 = time.perf_counter()
        result = gpu.run(f.get_flat(), f.const_values, variables)
        wall = (time.perf_counter() - t0) * 1000.0
        return result, wall, dict(gpu.last_timings)
    return gpu_task


def run_pass(task_fn, tasks, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        futures = [pool.submit(task_fn, t) for t in tasks]
        results = [f.result() for f in futures]
        wall = (time.perf_counter() - t0) * 1000.0
    return wall, results


def pct(sorted_times, p):
    k = (len(sorted_times) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_times) - 1)
    return sorted_times[f] + (sorted_times[c] - sorted_times[f]) * (k - f)


def stats(times):
    s = sorted(times)
    return {
        "avg": sum(s) / len(s),
        "min": s[0],
        "max": s[-1],
        "p50": pct(s, 50),
        "p95": pct(s, 95),
    }


def fmt_stats(st):
    return (f"avg {st['avg']:.4f}   p50 {st['p50']:.4f}   p95 {st['p95']:.4f}   "
            f"min {st['min']:.4f}   max {st['max']:.4f}")


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


def get_device_name():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        GPUExecutor()
    for line in buf.getvalue().splitlines():
        if line.startswith("GPU:"):
            return line.removeprefix("GPU:").strip()
    return "unknown"


def build_executor_pool(count):
    buf = io.StringIO()
    pool = []
    with contextlib.redirect_stdout(buf):
        for _ in range(count):
            pool.append(GPUExecutor())
    return pool


def methodology(repeats, cpu_workers, gpu_workers):
    return [
        "-" * 100,
        " WHAT THIS TEST DOES",
        "-" * 100,
        f"1. Workload      : evaluate {len(EXPRESSIONS)} arithmetic expressions"
        f" x {repeats} repetitions = {len(EXPRESSIONS) * repeats} tasks",
        "2. Each task     : full pipeline (lex -> parse -> eval on CPU,"
        " lex -> parse -> flatten -> dispatch on GPU)",
        "3. CPU mode      : ThreadPoolExecutor with multiple worker threads."
        " NOTE: CPython's GIL serializes pure-Python bytecode, so CPU scaling",
        "                   reflects concurrent scheduling, not true multicore"
        " execution.",
        "4. GPU mode      : ThreadPoolExecutor with a pre-built pool of"
        " independent GPUExecutor instances (one Vulkan instance/device each),",
        "                   dispatching concurrently at maximum worker count."
        " glslc shader compilation is cached (disk + memory) and excluded",
        f"                   from steady-state timing.",
        "5. Timing        : host wall-clock via time.perf_counter(); device-side"
        " GPU execution measured with Vulkan timestamp queries (ns resolution,",
        "                   TOP_OF_PIPE -> BOTTOM_OF_PIPE around the dispatch),"
        " converted with the device timestampPeriod.",
        "6. Protocol      : 1 untimed warm-up task, then a single-worker"
        " sequential baseline pass, then the multi-worker parallel pass.",
        "",
    ]


def cpu_section(name, repeats, cpu_workers, tasks_count, seq_wall, par_wall, times):
    st = stats(times)
    lines = [
        "=" * 100,
        f" CPU RESULTS ({name}, {cpu_workers} threads)",
        "=" * 100,
        f"Tasks                 : {tasks_count}",
        f"Sequential wall       : {seq_wall:.4f} ms",
        f"Multithreaded wall    : {par_wall:.4f} ms",
        f"Speedup               : {seq_wall / par_wall:.2f}x",
        f"Throughput            : {tasks_count / (par_wall / 1000.0):.2f} tasks/s"
        f" (sequential: {tasks_count / (seq_wall / 1000.0):.2f} tasks/s)",
        f"Task time (ms)        : {fmt_stats(st)}",
        "",
    ]
    return lines


def gpu_section(device, repeats, gpu_workers, tasks_count,
                seq_wall, par_wall, walls, stage_sums):
    st = stats(walls)
    lines = [
        "=" * 100,
        f" GPU RESULTS ({device}, {gpu_workers} workers)",
        "=" * 100,
        f"Tasks                 : {tasks_count}",
        f"Sequential wall       : {seq_wall:.4f} ms",
        f"Parallel wall         : {par_wall:.4f} ms",
        f"Speedup               : {seq_wall / par_wall:.2f}x",
        f"Throughput            : {tasks_count / (par_wall / 1000.0):.2f} tasks/s"
        f" (sequential: {tasks_count / (seq_wall / 1000.0):.2f} tasks/s)",
        f"Effective parallelism : {sum(walls) / par_wall:.2f}x"
        f" (sum of task durations / wall time)",
        f"Host task time (ms)   : {fmt_stats(st)}",
        "",
        " Device-side stage totals across all tasks (Vulkan timestamp queries):",
        f"   gpu_exec (device)  : avg {stage_sums['gpu_exec']['avg']:.4f} ms"
        f"   p95 {stage_sums['gpu_exec']['p95']:.4f}   max {stage_sums['gpu_exec']['max']:.4f}",
        f"   host dispatch      : avg {stage_sums['dispatch']['avg']:.4f} ms",
        f"   buffers upload     : avg {stage_sums['buffers']['avg']:.4f} ms",
        f"   pipeline setup     : avg {stage_sums['setup']['avg']:.4f} ms",
        f"   readback           : avg {stage_sums['readback']['avg']:.4f} ms",
        f"   shader compile     : avg {stage_sums['compile']['avg']:.4f} ms (cached)",
        "",
    ]
    return lines


def comparison(cpu_par, gpu_par, tasks_count):
    lines = [
        "=" * 100,
        " HEAD TO HEAD",
        "=" * 100,
        f"{'Backend':<10} {'Wall (ms)':>14} {'Tasks/s':>12} {'ms / task':>14}",
        "-" * 100,
        f"{'CPU':<10} {cpu_par:>14.4f} {tasks_count / (cpu_par / 1000.0):>12.2f}"
        f" {cpu_par / tasks_count:>14.4f}",
        f"{'GPU':<10} {gpu_par:>14.4f} {tasks_count / (gpu_par / 1000.0):>12.2f}"
        f" {gpu_par / tasks_count:>14.4f}",
        "-" * 100,
    ]
    winner = "CPU" if cpu_par < gpu_par else "GPU"
    factor = max(cpu_par, gpu_par) / min(cpu_par, gpu_par)
    lines.append(f"Winner: {winner} ({factor:.2f}x faster wall-clock)")
    lines.append("=" * 100)
    return lines


def export(path, text_lines):
    with open(path, "w") as f:
        f.write("\n".join(text_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--repeats", type=int, default=5,
                    help="repetitions of the expression set (default: 5)")
    ap.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--gpu-workers", type=int, default=64)
    args = ap.parse_args()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.perf_counter()

    print(f"Pre-compiling shader once (cached from here on)...")
    t0 = time.perf_counter()
    compile_shader()
    print(f"  glslc took {(time.perf_counter() - t0) * 1000:.4f} ms")

    cpu_name = get_cpu_name()
    gpu_name = get_device_name()
    print(f"CPU: {cpu_name} ({args.cpu_workers} threads)")
    print(f"GPU: {gpu_name} ({args.gpu_workers} workers)")

    tasks = [(i, code) for i in range(args.repeats) for code in EXPRESSIONS]
    tasks_count = len(tasks)

    print("\n[CPU] warm-up...")
    cpu_task((0, EXPRESSIONS[0]))
    print(f"[CPU] sequential baseline ({tasks_count} tasks, 1 thread)...")
    cpu_seq_wall, cpu_results = run_pass(cpu_task, tasks, 1)
    cpu_times = [r[1] for r in cpu_results]
    print(f"[CPU] multithreaded pass ({args.cpu_workers} threads)...")
    cpu_par_wall, cpu_results = run_pass(cpu_task, tasks, args.cpu_workers)
    cpu_times_par = [r[1] for r in cpu_results]

    print(f"\n[GPU] building executor pool ({args.gpu_workers} Vulkan instances)...")
    executors = build_executor_pool(args.gpu_workers)
    gpu_task = make_gpu_task(executors)

    print("[GPU] warm-up...")
    gpu_task((0, EXPRESSIONS[0]))
    print(f"[GPU] sequential baseline ({tasks_count} tasks, 1 worker)...")
    gpu_seq_wall, gpu_results = run_pass(gpu_task, tasks, 1)
    print(f"[GPU] max-parallel pass ({args.gpu_workers} workers)...")
    gpu_par_wall, gpu_results = run_pass(gpu_task, tasks, args.gpu_workers)

    gpu_walls = [r[1] for r in gpu_results]
    stage_sums = {}
    for s in STAGES:
        vals = [r[2].get(s, 0.0) for r in gpu_results]
        stage_sums[s] = stats(vals)

    ended = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.perf_counter() - t_start

    report = [
        "=" * 100,
        " MULTITHREADED CPU vs MAX-PARALLEL GPU BENCHMARK",
        "=" * 100,
        f"Started   : {started}",
        f"Finished  : {ended}   (total {elapsed:.1f} s)",
        f"Platform  : {platform.platform()}",
        f"Python    : {platform.python_version()}",
        f"CPU       : {cpu_name}",
        f"GPU       : {gpu_name}",
        "",
    ]
    report += methodology(args.repeats, args.cpu_workers, args.gpu_workers)
    report += [
        "-" * 100,
        " DETAILED RESULTS",
        "-" * 100,
        "",
    ]
    report += cpu_section(cpu_name, args.repeats, args.cpu_workers,
                          tasks_count, cpu_seq_wall, cpu_par_wall, cpu_times_par)
    report += gpu_section(gpu_name, args.repeats, args.gpu_workers,
                          tasks_count, gpu_seq_wall, gpu_par_wall,
                          gpu_walls, stage_sums)
    report += comparison(cpu_par_wall, gpu_par_wall, tasks_count)

    path = os.path.join(OUT_DIR, "head_to_head.txt")
    export(path, report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
