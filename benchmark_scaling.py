"""
benchmark_scaling.py – Measure throughput scaling across batch sizes [16, 32, 64, 128].
Saves results/scaling_results.csv for plots.py
"""

from __future__ import annotations

import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np

from data import load_config, set_seed, load_raw_splits, build_large_workload, get_model_tokenized_dataset
from inference import sequential_ensemble_gpu, parallel_ensemble_gpu, parallel_ensemble_single_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEYS = ["bert", "roberta", "electra"]
BATCH_SIZES = [16, 32, 64, 128]


def main():
    parser = argparse.ArgumentParser(description="Benchmark throughput scaling across batch sizes")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    if not torch.cuda.is_available():
        logger.warning("GPU not available, skipping scaling benchmark.")
        return

    n_gpus = torch.cuda.device_count()
    ckpt_dirs = {key: os.path.join(config["paths"]["checkpoints"], key) for key in MODEL_KEYS}

    # Use a workload of 2k for fast, responsive scaling measurements
    _, _, _, _, test_texts, test_labels = load_raw_splits(config)
    large_texts, large_labels, _ = build_large_workload(test_texts, test_labels, 2000, config["seed"])

    datasets = {}
    for key in MODEL_KEYS:
        datasets[key] = get_model_tokenized_dataset(
            large_texts, large_labels, config["models"][key],
            config["max_seq_len"], config["paths"]["data_cache"],
        )

    records = []
    n_samples = len(large_texts)

    for bs in BATCH_SIZES:
        logger.info(f"\n--- Testing Batch Size: {bs} ---")

        # 1. Sequential GPU
        logger.info(f"Running sequential_ensemble_gpu (batch_size={bs})...")
        try:
            seq_res = sequential_ensemble_gpu.run(datasets, ckpt_dirs, bs)
            total_wall = sum(r.total_wall_time_s for r in seq_res.values())
            seq_tp = n_samples / max(total_wall, 1e-9)
            records.append({
                "batch_size": bs,
                "mode": "sequential_ensemble_gpu",
                "total_wall_time_s": total_wall,
                "throughput_samples_per_sec": seq_tp,
            })
            logger.info(f"  sequential_ensemble_gpu throughput: {seq_tp:.1f} samples/s")
        except Exception as e:
            logger.warning(f"sequential_ensemble_gpu failed at bs={bs}: {e}")

        # 2. Parallel Ensemble Single GPU (Streams)
        logger.info(f"Running parallel_ensemble (batch_size={bs})...")
        try:
            if n_gpus == 1:
                par_res = parallel_ensemble_single_gpu.run(datasets, ckpt_dirs, bs)
            else:
                par_res = parallel_ensemble_gpu.run(datasets, ckpt_dirs, bs)
            total_wall = max(r.total_wall_time_s for r in par_res.values())
            par_tp = n_samples / max(total_wall, 1e-9)
            records.append({
                "batch_size": bs,
                "mode": "parallel_ensemble_gpu" if n_gpus > 1 else "parallel_ensemble_single_gpu",
                "total_wall_time_s": total_wall,
                "throughput_samples_per_sec": par_tp,
            })
            logger.info(f"  parallel_ensemble throughput: {par_tp:.1f} samples/s")
        except Exception as e:
            logger.warning(f"parallel_ensemble failed at bs={bs}: {e}")

    if records:
        os.makedirs(config["paths"]["results"], exist_ok=True)
        out_csv = os.path.join(config["paths"]["results"], "scaling_results.csv")
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        logger.info(f"\n✓ Saved scaling results → {out_csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
