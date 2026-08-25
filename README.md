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

Two benchmark scripts live in [`benchmark/`](benchmark/):

```bash
# sequential CPU vs GPU benchmark (10 expressions x 10 runs each)
python benchmark/benchmark.py

# GPU parallelism benchmark (multithreaded, pool of Vulkan instances)
python benchmark/gpu_parallel_benchmark.py            # N = 10..1000
python benchmark/gpu_parallel_benchmark.py --sizes 10 100 500 --workers-cap 32
```

Both export their raw results as `.txt` files (`cpu_benchmark.txt`,
`gpu_benchmark.txt`, `gpu_parallel_benchmark.txt`).

Results below were measured on an **AMD Radeon RX 580 2048SP** (RADV,
Linux) with Python 3.14.

### Sequential: CPU vs GPU

10 different arithmetic expressions, each executed 10 times (full pipeline:
lex → parse → eval on CPU; lex → parse → flatten → Vulkan dispatch on GPU,
with the GLSL→SPIR-V `glslc` compilation **cached** and timed separately).

| Backend | Avg time / run | Breakdown (GPU) |
|---------|---------------:|-----------------|
| CPU     | **0.0130 ms**  | — |
| GPU     | **1.58 ms**    | compile ~0.002 ms · buffers ~0.70 ms · pipeline setup ~0.63 ms · dispatch ~0.18 ms · readback ~0.04 ms |

The shader is compiled once by `glslc` and then reused from a disk +
in-memory cache (`compile_shader()` in `compiler/gpu/dispatch.py`), so the
per-run cost is now dominated by per-call buffer allocation and pipeline
creation, not compilation. Before caching, this number was ~186 ms/run
(~100x slower), because every single call re-invoked `glslc`.

### GPU parallelism (multithreading)

Same GPU test launched N times concurrently through a `ThreadPoolExecutor`,
where workers share a pre-built pool of up to 64 independent `GPUExecutor`
instances (one Vulkan instance each).

| N tasks | Workers | Wall time (ms) | Throughput (tasks/s) | Effective parallelism |
|--------:|--------:|---------------:|---------------------:|----------------------:|
| 10      | 10      | 34.54          | 290                  | 8.89x                 |
| 25      | 25      | 53.62          | 466                  | 12.46x                |
| 50      | 50      | 96.21          | 520                  | 18.14x                |
| 100     | 64      | 194.31         | 515                  | 28.38x                |
| 250     | 64      | 474.10         | 527                  | 45.78x                |
| 500     | 64      | 1,020.90       | 490                  | 52.10x                |
| 750     | 64      | 1,796.46       | 417                  | 57.37x                |
| 1000    | 64      | 2,542.68       | 393                  | 60.22x                |

Pre-caching baselines are archived in
[`benchmark/archive/pre_shader_cache/`](benchmark/archive/pre_shader_cache/).

What we actually see:

- **Caching changed everything:** with the compile cached, throughput jumped
  from ~17 tasks/s to **~520 tasks/s (~30x)**, and effective parallelism now
  scales cleanly with worker count up to ~60x.
- **Parallelism helps until the queue saturates.** Per-task latency grows
  from ~1.6 ms (solo) to ~153 ms at N=1000 because all instances submit to
  the same physical compute queue; past ~64 workers nothing improves.
- **The next bottleneck is per-call setup**, not the GPU: each task still
  rebuilds buffers + descriptors + pipeline (~23–132 ms of the task time
  under contention). Reusing a persistent pipeline would push throughput
  much closer to the hardware limit.

## License

This project is licensed under the [MIT License](LICENSE).
