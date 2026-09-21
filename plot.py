import os
import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DB_PATH = os.getenv("BENCH_DB", "data/bench_data_100k.db")  # override via env var if needed

CPU_LABEL = os.getenv("CPU_LABEL", "Intel Core i5-4440")
GPU_LABEL = os.getenv("GPU_LABEL", "AMD RX 580 2048SP 8GB")
os.makedirs("data", exist_ok=True)
db = duckdb.connect(DB_PATH)

cpu_single_bench = db.execute("SELECT * FROM cpu_single_bench").fetch_df()
cpu_batch_bench  = db.execute("SELECT * FROM cpu_batch_bench").fetch_df()
gpu_single_bench = db.execute("SELECT * FROM gpu_single_bench").fetch_df()
gpu_batch_bench  = db.execute("SELECT * FROM gpu_batch_bench").fetch_df()

db.close()

# Hardware labels
CPU_LABEL = "Intel Core i5-4440"
GPU_LABEL = "AMD RX 580 2048SP 8GB"

def style_ax(ax, title, xlabel="n (evaluations)", ylabel="mean time per eval (s)"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

def add_hw_footnote(fig, extra=""):
    fig.text(0.5, -0.02,
        f"CPU: {CPU_LABEL}    GPU: {GPU_LABEL}    {extra}",
        ha="center", fontsize=8, color="gray"
    )

# ─── Average across equations ───────────────────────────────────────────────
cpu_single_avg = cpu_single_bench.groupby("n")["mean_per_eval"].mean().reset_index()
cpu_batch_avg  = cpu_batch_bench.groupby("n")["mean_per_eval"].mean().reset_index()
gpu_single_avg = gpu_single_bench.groupby("n")["mean_per_eval"].mean().reset_index()
gpu_batch_avg  = gpu_batch_bench.groupby("n")["mean_per_eval"].mean().reset_index()

# ─── Chart 1: Combined all 4 averaged ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cpu_single_avg["n"], cpu_single_avg["mean_per_eval"], label="CPU Single")
ax.plot(cpu_batch_avg["n"],  cpu_batch_avg["mean_per_eval"],  label="CPU Batch (4 threads)")
ax.plot(gpu_single_avg["n"], gpu_single_avg["mean_per_eval"], label="GPU Single")
ax.plot(gpu_batch_avg["n"],  gpu_batch_avg["mean_per_eval"],  label="GPU Batch")
style_ax(ax, "All modes — mean per eval (averaged across equations)")
add_hw_footnote(fig, "averaged across: 1+2, 10-5, 5*10, 500/10")
plt.tight_layout()
plt.savefig("data/chart_01_all_modes_avg.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Chart 2: Single — CPU vs GPU averaged ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cpu_single_avg["n"], cpu_single_avg["mean_per_eval"], label=f"CPU Single ({CPU_LABEL})")
ax.plot(gpu_single_avg["n"], gpu_single_avg["mean_per_eval"], label=f"GPU Single ({GPU_LABEL})")
style_ax(ax, "Single execution — CPU vs GPU")
add_hw_footnote(fig)
plt.tight_layout()
plt.savefig("data/chart_02_single_cpu_vs_gpu.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Chart 3: Batch — CPU vs GPU averaged ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cpu_batch_avg["n"], cpu_batch_avg["mean_per_eval"], label=f"CPU Batch 4T ({CPU_LABEL})")
ax.plot(gpu_batch_avg["n"], gpu_batch_avg["mean_per_eval"], label=f"GPU Batch ({GPU_LABEL})")
style_ax(ax, "Batch execution — CPU vs GPU")
add_hw_footnote(fig)
plt.tight_layout()
plt.savefig("data/chart_03_batch_cpu_vs_gpu.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Chart 4: Single vs Batch per backend ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cpu_single_avg["n"], cpu_single_avg["mean_per_eval"], label="Single")
axes[0].plot(cpu_batch_avg["n"],  cpu_batch_avg["mean_per_eval"],  label="Batch (4 threads)")
style_ax(axes[0], f"CPU — Single vs Batch\n{CPU_LABEL}")

axes[1].plot(gpu_single_avg["n"], gpu_single_avg["mean_per_eval"], label="Single")
axes[1].plot(gpu_batch_avg["n"],  gpu_batch_avg["mean_per_eval"],  label="Batch")
style_ax(axes[1], f"GPU — Single vs Batch\n{GPU_LABEL}")

add_hw_footnote(fig)
plt.tight_layout()
plt.savefig("data/chart_04_single_vs_batch_per_backend.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Charts 5–8: Per equation breakdown per bench type ──────────────────────
bench_configs = [
    (cpu_single_bench, "CPU Single",    f"CPU Single — per equation\n{CPU_LABEL}",    "chart_05_cpu_single_per_eq.png"),
    (gpu_single_bench, "GPU Single",    f"GPU Single — per equation\n{GPU_LABEL}",    "chart_06_gpu_single_per_eq.png"),
    (cpu_batch_bench,  "CPU Batch",     f"CPU Batch — per equation\n{CPU_LABEL}",     "chart_07_cpu_batch_per_eq.png"),
    (gpu_batch_bench,  "GPU Batch",     f"GPU Batch — per equation\n{GPU_LABEL}",     "chart_08_gpu_batch_per_eq.png"),
]

for df, mode_label, title, fname in bench_configs:
    fig, ax = plt.subplots(figsize=(10, 5))
    for program, group in df.groupby("program"):
        group_sorted = group.sort_values("n")
        ax.plot(group_sorted["n"], group_sorted["mean_per_eval"], label=program)
    style_ax(ax, title)
    add_hw_footnote(fig)
    plt.tight_layout()
    plt.savefig(f"data/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

# ─── Chart 9: GPU batch — num_batches vs n ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for program, group in gpu_batch_bench.groupby("program"):
    group_sorted = group.sort_values("n")
    ax.plot(group_sorted["n"], group_sorted["num_batches"], label=program)
ax.set_xscale("log")
ax.set_xlabel("n (evaluations)", fontsize=11)
ax.set_ylabel("num batches dispatched", fontsize=11)
ax.set_title(f"GPU Batch — dispatch count vs n\n{GPU_LABEL}", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
add_hw_footnote(fig, f"batch_size=1638 fixed")
plt.tight_layout()
plt.savefig("data/chart_09_gpu_batch_dispatch.png", dpi=150, bbox_inches="tight")
plt.close()


# ─── Chart 10: GPU batch — mean_per_eval + num_batches dual axis ────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

gpu_batch_avg_nb = gpu_batch_bench.groupby("n")[["mean_per_eval","num_batches"]].mean().reset_index()

ax1.plot(gpu_batch_avg_nb["n"], gpu_batch_avg_nb["mean_per_eval"], color="red",  label="mean per eval (s)")
ax2.plot(gpu_batch_avg_nb["n"], gpu_batch_avg_nb["num_batches"],   color="blue", linestyle="--", label="num batches")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("n (evaluations)", fontsize=11)
ax1.set_ylabel("mean time per eval (s)", fontsize=11, color="red")
ax2.set_ylabel("num batches dispatched", fontsize=11, color="blue")
ax1.set_title(f"GPU Batch — amortization effect\n{GPU_LABEL}", fontsize=13, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax1.grid(True, which="both", linestyle="--", alpha=0.4)
add_hw_footnote(fig, "shows how more dispatches = lower cost per eval")
plt.tight_layout()
plt.savefig("data/chart_10_gpu_batch_amortization.png", dpi=150, bbox_inches="tight")
plt.close()


# ─── Chart 11: std_total — variance/noise per mode ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for df, label in [
    (cpu_single_bench, "CPU Single"),
    (cpu_batch_bench,  "CPU Batch"),
    (gpu_single_bench, "GPU Single"),
    (gpu_batch_bench,  "GPU Batch"),
]:
    avg = df.groupby("n")["std_total"].mean().reset_index()
    ax.plot(avg["n"], avg["std_total"], label=label)
style_ax(ax, "Timing variance (std_total) across all modes", ylabel="std total (s)")
add_hw_footnote(fig, "lower = more consistent")
plt.tight_layout()
plt.savefig("data/chart_11_variance.png", dpi=150, bbox_inches="tight")
plt.close()


# ─── Chart 12: Speedup — CPU single as baseline ─────────────────────────────
merged = cpu_single_avg.rename(columns={"mean_per_eval": "cpu_single"})
merged = merged.merge(cpu_batch_avg.rename(columns={"mean_per_eval": "cpu_batch"}), on="n")
merged = merged.merge(gpu_single_avg.rename(columns={"mean_per_eval": "gpu_single"}), on="n")
merged = merged.merge(gpu_batch_avg.rename(columns={"mean_per_eval":  "gpu_batch"}),  on="n")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged["n"], merged["cpu_single"] / merged["cpu_batch"],  label="CPU Batch vs CPU Single")
ax.plot(merged["n"], merged["cpu_single"] / merged["gpu_single"], label="GPU Single vs CPU Single")
ax.plot(merged["n"], merged["cpu_single"] / merged["gpu_batch"],  label="GPU Batch vs CPU Single")
ax.set_xscale("log")
ax.set_xlabel("n (evaluations)", fontsize=11)
ax.set_ylabel("speedup (×)", fontsize=11)
ax.set_title("Speedup relative to CPU Single baseline", fontsize=13, fontweight="bold")
ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, label="baseline (1×)")
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
add_hw_footnote(fig, "higher = faster than CPU single")
plt.tight_layout()
plt.savefig("data/chart_12_speedup.png", dpi=150, bbox_inches="tight")
plt.close()
