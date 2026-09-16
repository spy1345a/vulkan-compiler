"""Minimal toyc example: single runs, batch runs, benches saved to CSV."""
from toyc import Cpu, GpuVulkan  # ty: ignore[unresolved-import]
from toyc.bench import bench, batch_bench, summarize, to_csv  # ty: ignore[unresolved-import]


# 1. single values, straight from strings
print("cpu single:", Cpu.run("1 + 2 * 7", silent=True))
print("gpu single:", GpuVulkan.run("1 + 2 * 7", silent=True))

# 2. variables
env = {"a": 10.0, "b": 5.0}
print("cpu vars:", Cpu.run("a + b * 2", env=env, silent=True))
print("gpu vars:", GpuVulkan.run("a + b * 2", env=env, silent=True))

# 3. batch: one expression, many variable sets
sets = [{"a": float(i), "b": 2.0} for i in range(10)]
print("gpu batch:", GpuVulkan.run_batch("a + b * 2", sets, silent=True))

# 4. bench single runs (both backends) + save
single_rows = bench("a + b * 2", backend="cpu", n=50, repeat=3, seed=1)
single_rows += bench("a + b * 2", backend="vulkan", n=50, repeat=3, seed=1)
to_csv(single_rows, "example_single.csv")
print("saved example_single.csv")

# 5. bench batch runs (both backends) + save
batch_rows = batch_bench("a + b * 2", backend="cpu", n=500, repeat=2,
                         seed=1)
batch_rows += batch_bench("a + b * 2", backend="vulkan", n=500, repeat=2,
                          seed=1)
to_csv(batch_rows, "example_batch.csv")
print("saved example_batch.csv")

# 6. quick summary
for s in summarize(single_rows + batch_rows):
    print(f"{s['backend']:8} {s['mode']:8} "
          f"mean_total={s['mean_total'] * 1e3:.3f} ms")

GpuVulkan.shutdown()
