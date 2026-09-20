# vulkan-compiler — experiments for a research paper

Benchmarks and test scripts for **[toyc](https://github.com/spy1345a/toyc-repo)**,
a toy expression compiler with CPU and Vulkan GPU backends. This repo
holds the experiments; the library under study lives in toyc.

## Run

```bash
pip install -r requirements.txt
```
```python
python example.py    # minimal usage demo, CSVs + summary
```

Results land as `bench_*.csv` / `example_*.csv` (one row per run:
backend, timings, worker/chunk columns), ready for pandas/matplotlib
analysis in the paper.

# Disclamer
"The Vulkan backend architecture, AST flattening strategy, VRAM-aware batch sizing, and pipeline design were designed by the author. AI tooling was used to assist with implementation of low-level Vulkan API boilerplate."