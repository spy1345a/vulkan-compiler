# Vulkan Compiler

A toy expression compiler that compiles arithmetic expressions to bytecode and
executes them on **both** the CPU (tree-walking evaluator) and the GPU
(Vulkan compute shader acting as a register machine), then verifies the two
results match.

## Pipeline

```
source text ──► Lexer ──► Parser ──► AST ──┬─► Evaluator ────────────► CPU result
                                           │
                                           ├─► toyc.py ──► .toyc file (JSON)
                                           │
                                           └─► Flattener ──► bytecode ──► GPUExecutor ──► GPU result
                                                                            (Vulkan + executor.comp)
```

| Stage | File | What it does |
|-------|------|--------------|
| Lexer | `compiler/lexer.py` | Tokenizes numbers, identifiers, keywords (`kernel`, `return`) and operators `+ - * /` |
| Parser | `compiler/parser.py` | Recursive-descent parser with correct precedence, builds the AST |
| AST | `compiler/ast_nodes.py` | `Number`, `Var`, `BinOp` nodes; serializable to/from JSON |
| Evaluator | `compiler/evaluator.py` | Tree-walking interpreter used as the CPU reference |
| Compiler | `compiler/toyc.py` | Saves compiled programs as `.toyc` JSON files and loads them back |
| Codegen | `compiler/gpu/flattener.py` | Lowers the AST to flat `[op, dest, src1, src2]` bytecode with register allocation |
| ISA | `compiler/gpu/instructions.py` | Opcodes: `ADD`, `SUB`, `MUL`, `DIV`, `LOAD` (constant), `VAR` (variable) |
| Shader | `compiler/gpu/shaders/executor.comp` | GLSL compute shader that interprets the bytecode on a 64-register file |
| Dispatch | `compiler/gpu/dispatch.py` | Vulkan boilerplate: buffers, descriptors, pipeline, dispatch, result read-back |

## Example

```python
from compiler import Lexer, Parser, Evaluator
from compiler.gpu import Flattener, GPUExecutor

code = "a + b * 2"
env  = {"a": 10.0, "b": 5.0}

# CPU
tokens = Lexer(code).tokenize()
ast    = Parser(tokens).parse()
cpu    = Evaluator(env).eval(ast)          # -> 20.0

# GPU
f = Flattener()
f.flatten(ast)
gpu = GPUExecutor()
g   = gpu.run(f.get_flat(), f.const_values, [10.0, 5.0])  # -> 20.0
```

Or just run the demo:

```bash
python main.py
```

Output:

```
GPU: <your device name>
Code:   a + b * 2
CPU:    20.0
GPU:    20.0
Match:  True
```

## Requirements

- Python 3.12+
- A GPU with Vulkan support and the Vulkan loader installed
- [`glslc`](https://github.com/KhronosGroup/shaderc) on your `PATH`
  (used at runtime to compile the compute shader to SPIR-V)

Install Python dependencies:

```bash
pip install -r requirement.txt
```

## Project status

v0.1.0 — expressions only (`+ - * /`, parentheses, variables, int/float
literals). The lexer already recognizes `kernel`/`return` keywords for future
kernel definitions.

## Benchmarks

Four benchmark scripts live in [`benchmark/`](benchmark/):

```bash
# sequential CPU vs GPU benchmark (10 expressions x 10 runs each)
python benchmark/benchmark.py

# GPU parallelism benchmark (multithreaded, pool of Vulkan instances)
python benchmark/gpu_parallel_benchmark.py            # N = 10..1000
python benchmark/gpu_parallel_benchmark.py --sizes 10 100 500 --workers-cap 32

# multithreaded CPU vs max-parallel GPU, N repetitions, full report
python benchmark/head_to_head_benchmark.py -n 5
python benchmark/head_to_head_benchmark.py -n 10 --cpu-workers 4 --gpu-workers 64

# batch-dispatch mode: one dispatch evaluates N instances data-parallel
python benchmark/batch_benchmark.py                   # N = 1k..200k
python benchmark/batch_benchmark.py --sizes 1000000 --repeats 5
```

Each exports its raw results as `.txt` files (`cpu_benchmark.txt`,
`gpu_benchmark.txt`, `gpu_parallel_benchmark.txt`, `head_to_head.txt`,
`batch_benchmark.txt`) with a detailed methodology + results report.

Results below were measured on an **AMD Radeon RX 580 2048SP** (RADV,
Linux) with Python 3.14.

### Sequential: CPU vs GPU

10 different arithmetic expressions, each executed 10 times (full pipeline:
lex → parse → eval on CPU; lex → parse → flatten → Vulkan dispatch on GPU,
with the GLSL→SPIR-V `glslc` compilation **cached** and timed separately).

GPU-side execution time is measured with **Vulkan timestamp queries**
(2 timestamps bracketing the dispatch, read back via
`vkCmdCopyQueryPoolResults`), not host wall-clock.

| Backend | Avg time / run | Breakdown (GPU) |
|---------|---------------:|-----------------|
| CPU     | **0.0108 ms**  | — |
| GPU     | **1.55 ms**    | compile ~0.002 · buffers ~0.67 · pipeline setup ~0.62 · host dispatch ~0.15 · **GPU exec ~0.014** · readback ~0.06 (all ms) |

The shader is compiled once by `glslc` and then reused from a disk +
in-memory cache (`compile_shader()` in `compiler/gpu/dispatch.py`), so the
per-run cost is now dominated by per-call buffer allocation and pipeline
creation, not compilation. Before caching, this number was ~186 ms/run
(~120x slower), because every single call re-invoked `glslc`.

The timestamp queries expose the key fact: the compute dispatch itself takes
only **~14 µs on the device** — over 99% of the 1.55 ms wall-clock is
host-side overhead (buffer allocation, pipeline/descriptor setup, submission).

### GPU parallelism (multithreading)

Same GPU test launched N times concurrently through a `ThreadPoolExecutor`,
where workers share a pre-built pool of up to 64 independent `GPUExecutor`
instances (one Vulkan instance each).

| N tasks | Workers | Wall time (ms) | Throughput (tasks/s) | Effective parallelism | GPU exec / task |
|--------:|--------:|---------------:|---------------------:|----------------------:|----------------:|
| 10      | 10      | 51.21          | 195                  | 6.35x                 | 0.017 ms        |
| 25      | 25      | 89.28          | 280                  | 11.12x                | 0.021 ms        |
| 50      | 50      | 95.53          | 523                  | 17.33x                | 0.018 ms       |
| 100     | 64      | 186.28         | 537                  | 28.12x                | 0.020 ms        |
| 250     | 64      | 483.25         | 517                  | 45.23x                | 0.019 ms        |
| 500     | 64      | 1,011.67       | 494                  | 54.43x                | 0.021 ms        |
| 750     | 64      | 1,639.48       | 457                  | 58.58x                | 0.020 ms        |
| 1000    | 64      | 2,287.09       | 437                  | 59.62x                | 0.022 ms        |

Pre-caching baselines are archived in
[`benchmark/archive/pre_shader_cache/`](benchmark/archive/pre_shader_cache/).

What we actually see:

- **Caching changed everything:** with the compile cached, throughput jumped
  from ~17 tasks/s to **~520 tasks/s (~30x)**, and effective parallelism now
  scales cleanly with worker count up to ~60x.
- **The GPU is never the bottleneck.** Timestamp queries show device-side
  execution stays flat at **~20 µs per task** even with 64 workers and
  N=1000 queued tasks, while wall-clock per task grows to ~136 ms — all of
  the slowdown is host-side setup + queue submission, not the device.
- **Parallelism helps until the queue saturates.** Past ~64 workers nothing
  improves; work just queues up on the single physical compute queue.
- **The next bottleneck is per-call setup**: each task still rebuilds
  buffers + descriptors + pipeline (~26–111 ms of the task time under
  contention). Reusing a persistent pipeline would push throughput much
  closer to the hardware limit.

### Batch dispatch: where the GPU actually wins

The single-value results above measure the GPU in its worst case — one
expression per dispatch. `GPUExecutor.run_batch()` flips that: all N variable
sets are uploaded, and **one** data-parallel dispatch evaluates every instance
(one shader invocation per instance, `local_size_x = 64`), with all N results
read back in a single call.

Workload: evaluate `((a * b) + (c * d)) / (a - b)` for N independent variable
assignments. CPU = sequential tree-walk; GPU = single batch dispatch; every
GPU result is verified against the CPU (max relative error ~1e-6, float32).

| N instances | CPU wall | CPU evals/s | GPU wall (best) | GPU evals/s | Speedup | Device exec |
|------------:|---------:|------------:|----------------:|------------:|--------:|------------:|
| 1,000       | 2.76 ms  | 362,617     | 1.85 ms         | 540,253     | 1.5x    | 0.020 ms    |
| 10,000      | 26.88 ms | 371,964     | 4.20 ms         | 2,379,238   | 6.4x    | 0.030 ms    |
| 50,000      | 135.80 ms| 368,196     | 12.24 ms        | 4,083,412   | 11.1x   | 0.096 ms    |
| 200,000     | 524.08 ms| 381,621     | 41.11 ms        | 4,864,405   | 12.7x   | 0.316 ms    |

What this shows:

- **The crossover is at N ≈ 1,000** — past it the GPU pulls away and keeps
  pulling: CPU stays flat at ~370k evals/s while GPU climbs to ~4.9M.
- **CPU cost is linear** (every instance costs another tree-walk); GPU cost
  is a fixed per-call overhead plus a tiny device kernel — device execution
  was only **0.32 ms of the 41 ms** batch at N=200k.
- Even at N=200k, >99% of GPU wall-clock is still host-side upload +
  pipeline build. A persistent pipeline would raise the ceiling further;
  the device itself is nowhere near saturated.

So: single expressions → use the CPU; bulk evaluation → batch on the GPU,
and the gap grows with workload size.

## License

This project is licensed under the [MIT License](LICENSE).
