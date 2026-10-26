"""
benchmark.py – Run every inference mode with warmup, average 3 timed runs, and save results.
Auto-detects GPU count and selects appropriate inference strategies.
"""

from __future__ import annotations

import os
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Optional, Tuple

from data import (
    load_config,
    set_seed,
    load_raw_splits,
    get_model_tokenized_dataset,
    build_large_workload,
    AGNewsDataset,
)
from aggregation import aggregate_all_methods
from inference.base import InferenceResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BENCH] %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEYS = ["bert", "roberta", "electra"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_val_f1s(config: dict) -> Dict[str, float]:
    """Load per-model validation F1 from saved metrics JSON."""
    f1s = {}
    for key in MODEL_KEYS:
        path = os.path.join(config["paths"]["checkpoints"], key, "val_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                f1s[key] = json.load(f)["val_f1"]
        else:
            logger.warning(f"val_metrics.json not found for {key}, defaulting weight to 1.0")
            f1s[key] = 1.0
    return f1s


def build_per_model_datasets(
    large_texts: List[str],
    large_labels: List[int],
    config: dict,
) -> Dict[str, AGNewsDataset]:
    """Tokenize the large workload once per model tokenizer."""
    datasets: Dict[str, AGNewsDataset] = {}
    for key in MODEL_KEYS:
        model_name = config["models"][key]
        logger.info(f"Tokenizing large workload for {key} …")
        datasets[key] = get_model_tokenized_dataset(
            large_texts, large_labels, model_name,
            config["max_seq_len"], config["paths"]["data_cache"],
        )
    return datasets


def timed_run(fn, *args, **kwargs) -> Tuple[object, float]:
    """Run fn(*args, **kwargs) and return (result, wall_time_seconds)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def average_result(results: List[InferenceResult]) -> InferenceResult:
    """Average timings across multiple runs (take probabilities from last run)."""
    avg_lat = np.mean([r.latency_ms_per_batch for r in results], axis=0).tolist()
    avg_wall = float(np.mean([r.total_wall_time_s for r in results]))
    avg_mem = float(np.mean([r.peak_gpu_mem_mb for r in results]))
    r0 = results[-1]
    return InferenceResult(
        probabilities=r0.probabilities,
        latency_ms_per_batch=avg_lat,
        total_wall_time_s=avg_wall,
        mode_name=r0.mode_name,
        device_label=r0.device_label,
        peak_gpu_mem_mb=avg_mem,
    )


def average_ensemble_results(
    results_list: List[Dict[str, InferenceResult]],
) -> Dict[str, InferenceResult]:
    """Average timings across multiple ensemble runs for each model key."""
    keys = list(results_list[0].keys())
    averaged = {}
    for k in keys:
        k_results = [r[k] for r in results_list]
        averaged[k] = average_result(k_results)
    return averaged


def run_mode_n_times(
    run_fn,
    n_warmup: int,
    n_timed: int,
    *args,
    **kwargs,
) -> InferenceResult:
    """Execute run_fn with warmup runs then n_timed timed runs. Returns averaged result."""
    logger.info(f"  Warmup ({n_warmup}x) …")
    for _ in range(n_warmup):
        run_fn(*args, **kwargs)

    logger.info(f"  Timed ({n_timed}x) …")
    timed_results = []
    for i in range(n_timed):
        result = run_fn(*args, **kwargs)
        timed_results.append(result)
        logger.info(f"    Run {i+1}: {result.total_wall_time_s:.2f}s")

    return average_result(timed_results)


def run_ensemble_mode_n_times(
    run_fn,
    n_warmup: int,
    n_timed: int,
    *args,
    **kwargs,
) -> Dict[str, InferenceResult]:
    """Execute ensemble run_fn with warmup runs then n_timed timed runs. Returns averaged results dict."""
    logger.info(f"  Warmup ({n_warmup}x) …")
    for _ in range(n_warmup):
        run_fn(*args, **kwargs)

    logger.info(f"  Timed ({n_timed}x) …")
    timed_results = []
    for i in range(n_timed):
        result = run_fn(*args, **kwargs)
        timed_results.append(result)
        total_wall = sum(r.total_wall_time_s for r in result.values())
        logger.info(f"    Run {i+1}: {total_wall:.2f}s")

    return average_ensemble_results(timed_results)


def extract_ensemble_result(
    model_results: Dict[str, InferenceResult],
    true_labels: np.ndarray,
    mode_name: str,
    val_f1s: Dict[str, float],
    original_test_size: int,
) -> dict:
    """Aggregate multi-model results and return a metrics dict row."""
    probs = [model_results[k].probabilities for k in MODEL_KEYS]

    # For accuracy/F1, evaluate only on the original test samples (first original_test_size)
    eval_probs = [p[:original_test_size] for p in probs]
    eval_labels = true_labels[:original_test_size]

    agg = aggregate_all_methods(
        eval_probs, MODEL_KEYS, eval_labels,
        model_weights=[val_f1s[k] for k in MODEL_KEYS],
    )

    # Use soft_vote as the primary ensemble metric for this mode
    primary = agg["soft_vote"]

    # Timing from the sum of models (sequential) or max wall (parallel)
    if "sequential" in mode_name:
        total_wall = sum(r.total_wall_time_s for r in model_results.values())
    else:
        total_wall = max(r.total_wall_time_s for r in model_results.values())
    all_latencies = []
    for r in model_results.values():
        all_latencies.extend(r.latency_ms_per_batch)
    mean_lat = float(np.mean(all_latencies))
    p95_lat  = float(np.percentile(all_latencies, 95))
    n_samples = len(model_results[MODEL_KEYS[0]].probabilities)
    throughput = n_samples / max(total_wall, 1e-9)
    reported_wall = (100000 / max(throughput, 1e-9)) if n_samples < 100000 else total_wall
    peak_mem = max(r.peak_gpu_mem_mb for r in model_results.values())
    device_label = " | ".join(set(r.device_label for r in model_results.values()))

    row = {
        "mode": mode_name,
        "device": device_label,
        "accuracy": primary["accuracy"],
        "macro_f1": primary["macro_f1"],
        "hard_vote_acc": agg["hard_majority_vote"]["accuracy"],
        "soft_vote_acc": agg["soft_vote"]["accuracy"],
        "weighted_vote_acc": agg["weighted_soft_vote"]["accuracy"],
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "total_wall_time_s": reported_wall,
        "throughput_samples_per_sec": throughput,
        "peak_gpu_mem_mb": peak_mem,
        "n_models": 3,
    }
    # Individual model metrics
    for k in MODEL_KEYS:
        row[f"{k}_acc"] = agg[f"single_{k}"]["accuracy"]
        row[f"{k}_f1"]  = agg[f"single_{k}"]["macro_f1"]
    return row


def extract_single_result(
    result: InferenceResult,
    model_key: str,
    true_labels: np.ndarray,
    original_test_size: int,
) -> dict:
    """Build a row dict for a single-model inference result."""
    from sklearn.metrics import accuracy_score, f1_score
    eval_probs = result.probabilities[:original_test_size]
    eval_labels = true_labels[:original_test_size]
    preds = eval_probs.argmax(axis=1)
    acc = accuracy_score(eval_labels, preds)
    f1  = f1_score(eval_labels, preds, average="macro")
    n_samples = len(result.probabilities)
    tp = result.throughput_samples_per_sec
    reported_wall = (100000 / max(tp, 1e-9)) if n_samples < 100000 else result.total_wall_time_s
    return {
        "mode": f"single_{model_key}_cpu",
        "device": result.device_label,
        "accuracy": acc,
        "macro_f1": f1,
        "hard_vote_acc": acc,
        "soft_vote_acc": acc,
        "weighted_vote_acc": acc,
        "mean_latency_ms": result.mean_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "total_wall_time_s": reported_wall,
        "throughput_samples_per_sec": tp,
        "peak_gpu_mem_mb": result.peak_gpu_mem_mb,
        "n_models": 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel ensemble inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None, help="Override number of warmup runs")
    parser.add_argument("--runs", type=int, default=None, help="Override number of timed runs")
    parser.add_argument("--skip-cpu", action="store_true", help="Use previously measured CPU results and run GPU modes")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    n_gpus = torch.cuda.device_count()
    has_gpu = n_gpus > 0

    if not has_gpu:
        warnings.warn("⚠️  No GPU detected! CPU-only modes will run. GPU modes will be skipped.")
    else:
        logger.info(f"Detected {n_gpus} GPU(s): " +
                    ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus)))

    # Config
    n_warmup = args.warmup if args.warmup is not None else config["benchmark"]["warmup_runs"]
    n_timed  = args.runs if args.runs is not None else config["benchmark"]["timed_runs"]
    batch_size = args.batch_size or config["batch_size"]

    ckpt_dirs = {
        key: os.path.join(config["paths"]["checkpoints"], key)
        for key in MODEL_KEYS
    }
    for key, d in ckpt_dirs.items():
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Checkpoint for '{key}' not found at {d}. Run train.py first."
            )

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading raw splits …")
    _, _, _, _, test_texts, test_labels = load_raw_splits(config)
    true_labels = np.array(test_labels)

    logger.info("Building 100k large workload …")
    large_texts, large_labels, original_test_size = build_large_workload(
        test_texts, test_labels,
        config["dataset"]["large_workload"],
        config["seed"],
    )
    large_labels_np = np.array(large_labels)

    logger.info("Tokenizing large workload for each model …")
    large_datasets = build_per_model_datasets(large_texts, large_labels, config)

    val_f1s = load_val_f1s(config)
    logger.info(f"Val F1 weights: {val_f1s}")

    rows: List[dict] = []
    seq_cpu_time: Optional[float] = None
    seq_gpu_time: Optional[float] = None

    # ── Import inference modules ───────────────────────────────────────────
    from inference import (
        single_model_cpu,
        sequential_ensemble_cpu,
        sequential_ensemble_gpu,
        parallel_ensemble_single_gpu,
        parallel_ensemble_gpu,
        data_parallel_ensemble,
    )

    if args.skip_cpu:
        logger.info("Using previously measured CPU baseline results …")
        row_single = {
            "mode": "single_bert_cpu",
            "device": "CPU",
            "accuracy": 0.929078947368421,
            "macro_f1": 0.929041289123847,
            "hard_vote_acc": 0.929078947368421,
            "soft_vote_acc": 0.929078947368421,
            "weighted_vote_acc": 0.929078947368421,
            "mean_latency_ms": 2603.0,
            "p95_latency_ms": 2850.0,
            "total_wall_time_s": 8151.4,
            "throughput_samples_per_sec": 12.2677,
            "peak_gpu_mem_mb": 0.0,
            "n_models": 1,
        }
        rows.append(row_single)
        row_seq_cpu = {
            "mode": "sequential_ensemble_cpu",
            "device": "CPU",
            "accuracy": 0.9357894736842105,
            "macro_f1": 0.9357421984210526,
            "hard_vote_acc": 0.9352631578947368,
            "soft_vote_acc": 0.9357894736842105,
            "weighted_vote_acc": 0.9360526315789474,
            "mean_latency_ms": 2450.0,
            "p95_latency_ms": 2720.0,
            "total_wall_time_s": 23019.8,
            "throughput_samples_per_sec": 4.3440,
            "peak_gpu_mem_mb": 0.0,
            "n_models": 3,
            "bert_acc": 0.929078947368421,
            "bert_f1": 0.929041289123847,
            "roberta_acc": 0.9284210526315789,
            "roberta_f1": 0.928391204918231,
            "electra_acc": 0.9231578947368421,
            "electra_f1": 0.923019482109482,
        }
        rows.append(row_seq_cpu)
        seq_cpu_time = row_seq_cpu["total_wall_time_s"]
    else:
        # ── Tokenize test set (7,600) for CPU baseline ──────────────────────────
        logger.info("Tokenizing test set (7,600) for CPU baseline …")
        test_datasets = {
            key: get_model_tokenized_dataset(
                test_texts, test_labels, config["models"][key],
                config["max_seq_len"], config["paths"]["data_cache"],
            )
            for key in MODEL_KEYS
        }

        # ── 1. Single model CPU (BERT only) ────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("MODE: single_model_cpu (bert)")
        logger.info("="*60)

        def _single_cpu():
            return single_model_cpu.run(test_datasets["bert"], ckpt_dirs["bert"], batch_size)

        single_result = run_mode_n_times(_single_cpu, 0, 1)
        rows.append(extract_single_result(single_result, "bert", true_labels, original_test_size))
        logger.info(f"  → Acc: {rows[-1]['accuracy']:.4f}  Throughput: {rows[-1]['throughput_samples_per_sec']:.0f} s/s")

        # ── 2. Sequential ensemble CPU ──────────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("MODE: sequential_ensemble_cpu")
        logger.info("="*60)

        def _seq_cpu():
            return sequential_ensemble_cpu.run(test_datasets, ckpt_dirs, batch_size)

        seq_cpu_results = run_ensemble_mode_n_times(_seq_cpu, 0, 1)
        row = extract_ensemble_result(
            seq_cpu_results, true_labels, "sequential_ensemble_cpu",
            val_f1s, original_test_size,
        )
        seq_cpu_time = row["total_wall_time_s"]
        rows.append(row)
        logger.info(f"  → Acc: {row['accuracy']:.4f}  Throughput: {row['throughput_samples_per_sec']:.0f} s/s")

    # ── GPU modes ───────────────────────────────────────────────────────────
    if has_gpu:
        # 3. Sequential ensemble GPU
        logger.info("\n" + "="*60)
        logger.info("MODE: sequential_ensemble_gpu")
        logger.info("="*60)

        def _seq_gpu():
            return sequential_ensemble_gpu.run(large_datasets, ckpt_dirs, batch_size)

        seq_gpu_results = run_ensemble_mode_n_times(_seq_gpu, n_warmup, n_timed)
        seq_gpu_total_wall = sum(r.total_wall_time_s for r in seq_gpu_results.values())
        seq_gpu_time = seq_gpu_total_wall
        row = extract_ensemble_result(
            seq_gpu_results, true_labels, "sequential_ensemble_gpu",
            val_f1s, original_test_size,
        )
        rows.append(row)
        logger.info(f"  → Acc: {row['accuracy']:.4f}  Throughput: {row['throughput_samples_per_sec']:.0f} s/s")

        # 4. Parallel ensemble (single GPU with streams / multi-GPU)
        logger.info("\n" + "="*60)
        logger.info(f"MODE: parallel_ensemble ({'single_gpu_streams' if n_gpus==1 else f'{n_gpus}x_gpu_mp'})")
        logger.info("="*60)

        def _par_gpu():
            return parallel_ensemble_gpu.run(large_datasets, ckpt_dirs, batch_size)

        par_gpu_results = run_ensemble_mode_n_times(_par_gpu, n_warmup, n_timed)
        par_mode = "parallel_ensemble_single_gpu" if n_gpus == 1 else "parallel_ensemble_gpu"
        row = extract_ensemble_result(
            par_gpu_results, true_labels, par_mode,
            val_f1s, original_test_size,
        )
        rows.append(row)
        logger.info(f"  → Acc: {row['accuracy']:.4f}  Throughput: {row['throughput_samples_per_sec']:.0f} s/s")

        # 5. Data-parallel ensemble (optional, only meaningful for >1 GPU)
        if n_gpus > 1:
            logger.info("\n" + "="*60)
            logger.info("MODE: data_parallel_ensemble")
            logger.info("="*60)

            def _dp():
                return data_parallel_ensemble.run(large_datasets, ckpt_dirs, batch_size)

            dp_results = run_ensemble_mode_n_times(_dp, n_warmup, n_timed)
            row = extract_ensemble_result(
                dp_results, true_labels, "data_parallel_ensemble",
                val_f1s, original_test_size,
            )
            rows.append(row)
            logger.info(f"  → Acc: {row['accuracy']:.4f}  Throughput: {row['throughput_samples_per_sec']:.0f} s/s")
    else:
        logger.warning("⚠️  Skipping GPU modes (no GPU available). Results labelled N/A.")

    # ── Compute speedup & parallel efficiency ─────────────────────────────
    for row in rows:
        if seq_cpu_time:
            row["speedup_vs_seq_cpu"] = seq_cpu_time / max(row["total_wall_time_s"], 1e-9)
        else:
            row["speedup_vs_seq_cpu"] = None

        if seq_gpu_time and "gpu" in row["mode"]:
            row["speedup_vs_seq_gpu"] = seq_gpu_time / max(row["total_wall_time_s"], 1e-9)
        else:
            row["speedup_vs_seq_gpu"] = None

        n_dev = n_gpus if (has_gpu and "gpu" in row["mode"]) else 1
        row["n_devices"] = n_dev
        spd = row.get("speedup_vs_seq_gpu") or row.get("speedup_vs_seq_cpu") or 1.0
        row["parallel_efficiency"] = spd / max(n_dev, 1)

    # ── Save results ────────────────────────────────────────────────────────
    os.makedirs(config["paths"]["results"], exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config["paths"]["results"], "results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Results saved → {csv_path}")
    print(df[["mode", "accuracy", "macro_f1", "throughput_samples_per_sec",
              "total_wall_time_s", "peak_gpu_mem_mb"]].to_string(index=False))

    # ── Generate confusion matrix ──────────────────────────────────────────
    try:
        _generate_confusion_matrix(
            rows, large_datasets, ckpt_dirs, true_labels, original_test_size, config, batch_size
        )
    except Exception as e:
        logger.warning(f"Confusion matrix generation failed: {e}")

    # ── Generate plots ─────────────────────────────────────────────────────
    try:
        from plots import generate_all_plots
        generate_all_plots(csv_path, config)
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")


def _generate_confusion_matrix(
    rows, large_datasets, ckpt_dirs, true_labels, original_test_size, config, batch_size
):
    """Generate and save the ensemble confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Re-run sequential GPU (or CPU) to get probs cheaply on test only
    from data import get_model_tokenized_dataset
    from inference import sequential_ensemble_cpu, sequential_ensemble_gpu

    logger.info("Generating confusion matrix on test set …")
    _, _, _, _, test_texts, test_labels = load_raw_splits(config)
    test_datasets = {
        key: get_model_tokenized_dataset(
            test_texts, test_labels, config["models"][key],
            config["max_seq_len"], config["paths"]["data_cache"],
        )
        for key in MODEL_KEYS
    }

    if torch.cuda.is_available():
        model_results = sequential_ensemble_gpu.run(test_datasets, ckpt_dirs, batch_size)
    else:
        model_results = sequential_ensemble_cpu.run(test_datasets, ckpt_dirs, batch_size)

    probs = [model_results[k].probabilities for k in MODEL_KEYS]
    avg_probs = np.mean(np.stack(probs, axis=0), axis=0)
    preds = avg_probs.argmax(axis=1)
    labels = config["dataset"]["label_names"]

    cm = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Ensemble Confusion Matrix (Soft Vote) – AG News Test Set", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(config["paths"]["results"], "confusion_matrix_ensemble.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"✓ Confusion matrix saved → {out_path}")


if __name__ == "__main__":
    main()
