# benchmark/harness.py
"""Reusable benchmark building blocks: statistics, batch timing, input
generation. Import this from your own scripts to benchmark other code, e.g.

    from benchmark.harness import describe, plan_reps, cpu_eval_batch, gpu_eval_batch
"""

import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler import Lexer, Parser, Evaluator
from compiler.gpu import Flattener

CI_Z = 1.959963984540054


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


def compile_expression(code):
    return Parser(Lexer(code).tokenize()).parse()


def flatten_expression(ast):
    """Compile an AST to bytecode. Returns (flat, consts, num_vars, names)
    where names are variable names ordered by their register index."""
    f = Flattener()
    f.flatten(ast)
    names = [None] * len(f.var_map)
    for name, (_, idx) in f.var_map.items():
        names[idx] = name
    return f.get_flat(), f.const_values, len(f.var_map), names


def ordered_var_names(var_map):
    names = [None] * len(var_map)
    for name, (_, idx) in var_map.items():
        names[idx] = name
    return names


def gen_inputs(n, num_vars, seed=42):
    rng = random.Random(seed + num_vars)
    return [tuple(rng.uniform(0.5, 9.5) for _ in range(num_vars))
            for _ in range(n)]


def describe(samples):
    s = sorted(samples)
    n = len(s)
    mean = statistics.fmean(s)
    std = statistics.stdev(s) if n > 1 else 0.0
    ci95 = CI_Z * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "median": statistics.median(s),
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": s[0],
        "max": s[-1],
        "p95": _pct(s, 95),
    }


def _pct(s, p):
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def plan_reps(one_iter_fn, max_reps=300, min_reps=5, budget_s=8.0):
    """Run one untimed pilot iteration, then choose how many timed
    repetitions fit in budget_s, clamped to [min_reps, max_reps]."""
    t0 = time.perf_counter()
    one_iter_fn()
    dt = time.perf_counter() - t0
    if dt <= 0:
        return max_reps
    return int(max(min_reps, min(max_reps, budget_s / dt)))


def cpu_eval_batch(ast, rows, var_names):
    """Evaluate ast for every row of variable values. Returns (results, ms)."""
    t0 = time.perf_counter()
    results = [Evaluator(dict(zip(var_names, row))).eval(ast) for row in rows]
    return results, (time.perf_counter() - t0) * 1000.0


def gpu_eval_batch(gpu, flat, consts, num_vars, rows, count=None):
    """One data-parallel dispatch over all rows. Returns (results, ms, timings)."""
    vars_flat = [v for row in rows for v in row]
    count = count or len(rows)
    t0 = time.perf_counter()
    results = gpu.run_batch(flat, consts, vars_flat, count, num_vars)
    wall = (time.perf_counter() - t0) * 1000.0
    return results, wall, dict(gpu.last_timings)


if __name__ == "__main__":
    ast = compile_expression("a + b * 2")
    flat, consts, nv = flatten_expression(ast)
    names = ["a", "b"]
    rows = [(2.0, 3.0), (4.0, 5.0)]

    cpu_res, cpu_ms = cpu_eval_batch(ast, rows, names)
    gpu = None
    print(f"cpu ({cpu_ms:.4f} ms): {cpu_res}")
    print("reusable in a loop:")
    for i in range(3):
        samples = []
        reps = plan_reps(lambda: cpu_eval_batch(ast, rows, names),
                         max_reps=300, min_reps=5, budget_s=0.5)
        for _ in range(reps):
            _, ms = cpu_eval_batch(ast, rows, names)
            samples.append(ms)
        st = describe(samples)
        print(f"  trial {i}: {st['n']} reps, median {st['median']:.4f} ms "
              f"+/- {st['ci95']:.4f}")
