from compiler import Lexer, Parser, Evaluator
from compiler.vulkan import Flattener, GPUExecutor

def compile(code):
    tokens = Lexer(code).tokenize()
    return Parser(tokens).parse()

def run_cpu(code, env):
    return Evaluator(env).eval(compile(code))

def run_gpu(code, env, gpu):
    f = Flattener()
    f.flatten(compile(code))
    vars = [None] * len(f.var_map)
    for name, (_, idx) in f.var_map.items():
        vars[idx] = env[name]
    return gpu.run(f.get_flat(), f.const_values, vars)


# ── run ───────────────────────────────────────────
env  = {"a": 10.0, "b": 5.0}
code = "a + b * 2"
gpu  = GPUExecutor()

cpu = run_cpu(code, env)
g   = run_gpu(code, env, gpu)

print(f"Code:   {code}")
print(f"CPU:    {cpu}")
print(f"GPU:    {g}")
print(f"Match:  {abs(cpu - g) < 1e-4}")