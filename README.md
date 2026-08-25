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

## License

This project is licensed under the [MIT License](LICENSE).
