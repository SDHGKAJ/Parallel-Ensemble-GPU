"""
plots.py – Generate all benchmark visualisation plots.
Reads results/results.csv and writes PNGs to results/plots/
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional

# Consistent colour palette
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def plot_latency_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: mean latency per batch for each mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    modes = df["mode"].tolist()
    latencies = df["mean_latency_ms"].tolist()
    bars = ax.barh(modes, latencies, color=PALETTE[:len(modes)])
    ax.set_xlabel("Mean Latency per Batch (ms)")
    ax.set_title("Latency Comparison Across Inference Modes")
    ax.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "latency_comparison.png"))


def plot_throughput_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: samples/sec for each mode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    modes = df["mode"].tolist()
    throughputs = df["throughput_samples_per_sec"].tolist()
    bars = ax.barh(modes, throughputs, color=PALETTE[:len(modes)])
    ax.set_xlabel("Throughput (samples / second)")
    ax.set_title("Throughput Comparison Across Inference Modes")
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "throughput_comparison.png"))


def plot_speedup_chart(df: pd.DataFrame, out_dir: str) -> None:
    """Speedup vs sequential CPU baseline."""
    col = "speedup_vs_seq_cpu"
    valid = df[df[col].notna()].copy()
    if valid.empty:
        warnings.warn("No speedup data available.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(valid["mode"].tolist(), valid[col].tolist(), color=PALETTE[:len(valid)])
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2, label="Baseline (1×)")
    ax.set_xlabel("Speedup (×)")
    ax.set_title("Speedup vs. Sequential CPU Ensemble")
    ax.bar_label(bars, fmt="%.2f×", padding=4, fontsize=9)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "speedup_chart.png"))


def plot_accuracy_f1_bar(df: pd.DataFrame, config: dict, out_dir: str) -> None:
    """Grouped bar chart: accuracy & macro-F1 for each mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in [
        (axes[0], "accuracy",  "Accuracy"),
        (axes[1], "macro_f1", "Macro-F1"),
    ]:
        modes = df["mode"].tolist()
        vals  = df[col].tolist()
        bars  = ax.bar(modes, vals, color=PALETTE[:len(modes)])
        ax.set_ylim(max(0, min(vals) - 0.05), 1.0)
        ax.set_xlabel("Mode")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Inference Mode")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.suptitle("Single-Model vs Ensemble – AG News Test Set", fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "accuracy_f1_bar.png"))


def plot_gpu_memory(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: peak GPU memory per mode."""
    gpu_df = df[df["peak_gpu_mem_mb"] > 0].copy()
    if gpu_df.empty:
        warnings.warn("No GPU memory data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(gpu_df["mode"].tolist(), gpu_df["peak_gpu_mem_mb"].tolist(), color=PALETTE[:len(gpu_df)])
    ax.set_xlabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory Usage per Inference Mode")
    ax.bar_label(bars, fmt="%.0f MB", padding=4, fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "gpu_memory.png"))


def plot_scaling_vs_batch_size(config: dict, out_dir: str) -> None:
    """
    Throughput vs batch size for sequential_gpu (if available).
    Requires a run with multiple batch sizes — uses simulated scaling if real data absent.
    """
    csv_path = os.path.join(config["paths"]["results"], "scaling_results.csv")
    if not os.path.exists(csv_path):
        warnings.warn("scaling_results.csv not found; skipping scaling chart.")
        return

    scale_df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, grp in scale_df.groupby("mode"):
        ax.plot(grp["batch_size"], grp["throughput_samples_per_sec"],
                marker="o", label=mode)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Throughput Scaling vs. Batch Size")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "scaling_vs_batch_size.png"))


def generate_all_plots(csv_path: str, config: dict) -> None:
    """Generate all benchmark plots from results CSV."""
    df = pd.read_csv(csv_path)
    out_dir = config["paths"]["plots"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating plots -> {out_dir}")
    plot_latency_comparison(df, out_dir)
    plot_throughput_comparison(df, out_dir)
    plot_speedup_chart(df, out_dir)
    plot_accuracy_f1_bar(df, config, out_dir)
    plot_gpu_memory(df, out_dir)
    plot_scaling_vs_batch_size(config, out_dir)


if __name__ == "__main__":
    from data import load_config
    cfg = load_config()
    csv = os.path.join(cfg["paths"]["results"], "results.csv")
    generate_all_plots(csv, cfg)
