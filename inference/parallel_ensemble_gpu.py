"""
inference/parallel_ensemble_gpu.py
One model per GPU — one process per model, concurrent inference with multiprocessing queues.
Falls back to round-robin assignment for 2 GPUs; uses parallel_ensemble_single_gpu for 1 GPU.
"""

from __future__ import annotations

import os
import time
import queue
import threading
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from inference.base import InferenceResult, get_peak_gpu_mem_mb
from data import AGNewsDataset, make_dataloader


# ──────────────────────────────────────────────────────────────────────────────
# Worker function (runs in a subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _worker(
    gpu_id: int,
    checkpoint_dir: str,
    model_key: str,
    dataset_dict: dict,  # serialisable dict of tensors
    labels: list,
    batch_size: int,
    result_queue: mp.Queue,
) -> None:
    """
    Subprocess worker: loads model on gpu_id, runs inference, puts result in queue.
    dataset_dict: {'input_ids': Tensor, 'attention_mask': Tensor, ...}
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        model = model.to(device).eval()

        from data import AGNewsDataset, make_dataloader
        ds = AGNewsDataset(dataset_dict, labels)
        loader = make_dataloader(ds, batch_size, shuffle=False, num_workers=0)

        all_probs: List[np.ndarray] = []
        latencies: List[float] = []

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
                latencies.append((t1 - t0) * 1000.0)
                probs = F.softmax(logits, dim=-1).cpu().float().numpy()
                all_probs.append(probs)

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        result_queue.put({
            "model_key": model_key,
            "probs": np.vstack(all_probs),
            "latencies": latencies,
            "peak_gpu_mem_mb": peak_mem,
            "gpu_id": gpu_id,
        })
    except Exception as e:
        result_queue.put({"model_key": model_key, "error": str(e)})


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run(
    datasets: Dict[str, AGNewsDataset],
    checkpoint_dirs: Dict[str, str],
    batch_size: int = 32,
) -> Dict[str, InferenceResult]:
    """
    Parallel inference: spawn one subprocess per model, assign round-robin to GPUs.

    GPU assignment:
      >=3 GPUs  → one model per GPU (0,1,2)
       2 GPUs   → bert→0, roberta→1, electra→0 (round-robin)
       1 GPU    → delegates to parallel_ensemble_single_gpu

    Args:
        datasets:        {model_key: AGNewsDataset}.
        checkpoint_dirs: {model_key: checkpoint_path}.
        batch_size:      Inference batch size.

    Returns:
        Dict mapping model_key → InferenceResult.
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        raise RuntimeError("parallel_ensemble_gpu requires at least 1 GPU.")

    if n_gpus == 1:
        print("  [parallel_ensemble_gpu] Only 1 GPU — delegating to parallel_ensemble_single_gpu")
        from inference import parallel_ensemble_single_gpu
        return parallel_ensemble_single_gpu.run(datasets, checkpoint_dirs, batch_size)

    # Assign GPUs round-robin
    keys = list(checkpoint_dirs.keys())
    gpu_assignments = {key: i % n_gpus for i, key in enumerate(keys)}
    print(f"  [parallel_ensemble_gpu] GPU assignment: {gpu_assignments}")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()

    processes = []
    t_start = time.perf_counter()

    for key in keys:
        gpu_id = gpu_assignments[key]
        ds = datasets[key]
        # Convert to plain dict of cpu tensors for pickle-ability
        dataset_dict = {k: v.cpu() for k, v in ds.encodings.items()}
        labels = list(ds.labels)

        p = ctx.Process(
            target=_worker,
            args=(gpu_id, checkpoint_dirs[key], key, dataset_dict, labels, batch_size, result_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_wall = t_end - t_start

    # Collect results
    results: Dict[str, InferenceResult] = {}
    while not result_queue.empty():
        r = result_queue.get()
        if "error" in r:
            raise RuntimeError(f"Worker {r['model_key']} failed: {r['error']}")
        key = r["model_key"]
        gpu_id = r["gpu_id"]
        results[key] = InferenceResult(
            probabilities=r["probs"],
            latency_ms_per_batch=r["latencies"],
            total_wall_time_s=total_wall,
            mode_name="parallel_ensemble_gpu",
            device_label=f"GPU:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})",
            peak_gpu_mem_mb=r["peak_gpu_mem_mb"],
        )

    return results
