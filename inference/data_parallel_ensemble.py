"""
inference/data_parallel_ensemble.py
Optional: Full ensemble replicated per GPU, data sharded across GPUs.
Falls back gracefully to sequential_ensemble_gpu when only 1 GPU is available.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from inference.base import InferenceResult, load_model_for_inference, get_peak_gpu_mem_mb
from data import AGNewsDataset, make_dataloader, shard_dataset


def run(
    datasets: Dict[str, AGNewsDataset],
    checkpoint_dirs: Dict[str, str],
    batch_size: int = 32,
) -> Dict[str, InferenceResult]:
    """
    Data-parallel ensemble: replicate the full ensemble on each GPU, shard data.

    With N GPUs:
      - Each GPU hosts all 3 models.
      - Data is split into N shards; GPU i processes shard i for all models.
      - Results are concatenated in shard order.

    With 1 GPU: delegates to sequential_ensemble_gpu.

    Args:
        datasets:        {model_key: AGNewsDataset}.
        checkpoint_dirs: {model_key: checkpoint_path}.
        batch_size:      Inference batch size.

    Returns:
        Dict mapping model_key → InferenceResult.
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        raise RuntimeError("data_parallel_ensemble requires at least 1 GPU.")

    if n_gpus == 1:
        print("  [data_parallel_ensemble] Only 1 GPU — delegating to sequential_ensemble_gpu")
        from inference import sequential_ensemble_gpu
        return sequential_ensemble_gpu.run(datasets, checkpoint_dirs, batch_size)

    print(f"  [data_parallel_ensemble] Replicating ensemble across {n_gpus} GPUs …")

    # Shard each dataset across GPUs
    sharded: Dict[str, List[AGNewsDataset]] = {
        key: shard_dataset(ds, n_gpus) for key, ds in datasets.items()
    }

    # Per-GPU, per-model results
    shard_results: Dict[str, List[np.ndarray]] = {key: [] for key in checkpoint_dirs}
    shard_latencies: Dict[str, List[float]] = {key: [] for key in checkpoint_dirs}

    t_start = time.perf_counter()

    for gpu_id in range(n_gpus):
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.reset_peak_memory_stats()

        for key, ckpt_dir in checkpoint_dirs.items():
            model, _ = load_model_for_inference(ckpt_dir, device)
            shard_ds = sharded[key][gpu_id]
            loader = make_dataloader(shard_ds, batch_size, shuffle=False, num_workers=0)

            with torch.no_grad():
                for batch in loader:
                    t0 = time.perf_counter()
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)

                    from torch.amp import autocast
                    with autocast("cuda"):
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        ).logits

                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    shard_latencies[key].append((t1 - t0) * 1000.0)
                    probs = F.softmax(logits, dim=-1).cpu().float().numpy()
                    shard_results[key].append(probs)

            del model
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_wall = t_end - t_start

    results: Dict[str, InferenceResult] = {}
    for key in checkpoint_dirs:
        all_probs = np.vstack(shard_results[key])
        results[key] = InferenceResult(
            probabilities=all_probs,
            latency_ms_per_batch=shard_latencies[key],
            total_wall_time_s=total_wall,
            mode_name="data_parallel_ensemble",
            device_label=f"{n_gpus}x GPU",
            peak_gpu_mem_mb=get_peak_gpu_mem_mb(),
        )

    return results
