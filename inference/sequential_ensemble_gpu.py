"""
inference/sequential_ensemble_gpu.py – All 3 models, one after another, on 1 GPU.
"""

from __future__ import annotations

import time
from typing import Dict

import torch

from inference.base import (
    InferenceResult,
    load_model_for_inference,
    run_model_on_loader,
    get_peak_gpu_mem_mb,
)
from data import AGNewsDataset, make_dataloader


def run(
    datasets: Dict[str, AGNewsDataset],
    checkpoint_dirs: Dict[str, str],
    batch_size: int = 32,
) -> Dict[str, InferenceResult]:
    """
    Run all 3 models sequentially on a single GPU.

    Args:
        datasets:        {model_key: AGNewsDataset} (each tokenized for its model).
        checkpoint_dirs: {model_key: checkpoint_path}.
        batch_size:      Inference batch size.

    Returns:
        Dict mapping model_key → InferenceResult.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("sequential_ensemble_gpu requires a CUDA GPU.")

    device = torch.device("cuda:0")
    results: Dict[str, InferenceResult] = {}

    for key, ckpt_dir in checkpoint_dirs.items():
        print(f"  [sequential_ensemble_gpu] Running {key} on {device} …")
        torch.cuda.reset_peak_memory_stats()

        model, _ = load_model_for_inference(ckpt_dir, device)
        loader = make_dataloader(datasets[key], batch_size, shuffle=False, num_workers=0)

        t_start = time.perf_counter()
        probs, latencies = run_model_on_loader(model, loader, device, use_fp16=True)
        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        peak_mem = get_peak_gpu_mem_mb()

        results[key] = InferenceResult(
            probabilities=probs,
            latency_ms_per_batch=latencies,
            total_wall_time_s=t_end - t_start,
            mode_name="sequential_ensemble_gpu",
            device_label=f"GPU:0 ({torch.cuda.get_device_name(0)})",
            peak_gpu_mem_mb=peak_mem,
        )
        del model
        torch.cuda.empty_cache()

    return results
