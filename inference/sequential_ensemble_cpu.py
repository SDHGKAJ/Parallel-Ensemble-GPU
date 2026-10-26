"""
inference/sequential_ensemble_cpu.py – All 3 models, one after another, CPU.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
from typing import Dict, List

from inference.base import (
    InferenceResult,
    load_model_for_inference,
    run_model_on_loader,
)
from data import AGNewsDataset, make_dataloader


def run(
    datasets: Dict[str, AGNewsDataset],
    checkpoint_dirs: Dict[str, str],
    batch_size: int = 32,
) -> Dict[str, InferenceResult]:
    """
    Run all 3 models sequentially on CPU.

    Args:
        datasets:        {model_key: AGNewsDataset} (each tokenized for its model).
        checkpoint_dirs: {model_key: checkpoint_path}.
        batch_size:      Inference batch size.

    Returns:
        Dict mapping model_key → InferenceResult.
    """
    device = torch.device("cpu")
    results: Dict[str, InferenceResult] = {}

    for key, ckpt_dir in checkpoint_dirs.items():
        print(f"  [sequential_ensemble_cpu] Running {key} …")
        model, _ = load_model_for_inference(ckpt_dir, device)
        loader = make_dataloader(datasets[key], batch_size, shuffle=False, num_workers=0)

        t_start = time.perf_counter()
        probs, latencies = run_model_on_loader(model, loader, device, use_fp16=False)
        t_end = time.perf_counter()

        results[key] = InferenceResult(
            probabilities=probs,
            latency_ms_per_batch=latencies,
            total_wall_time_s=t_end - t_start,
            mode_name="sequential_ensemble_cpu",
            device_label="CPU",
            peak_gpu_mem_mb=0.0,
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results
