"""
inference/parallel_ensemble_single_gpu.py
Fallback for single-GPU machines: 3 models on one GPU using separate CUDA streams
and concurrent Python threads for asynchronous batching.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inference.base import (
    InferenceResult,
    load_model_for_inference,
    get_peak_gpu_mem_mb,
)
from data import AGNewsDataset, make_dataloader


def _infer_on_stream(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    stream: torch.cuda.Stream,
    result_container: List,
    use_fp16: bool = True,
) -> None:
    """
    Run inference in a dedicated CUDA stream; stores (probs, latencies) in result_container.
    """
    all_probs: List[np.ndarray] = []
    latencies: List[float] = []

    with torch.cuda.stream(stream):
        with torch.no_grad():
            for batch in loader:
                t0 = time.perf_counter()

                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device, non_blocking=True)

                if use_fp16:
                    from torch.amp import autocast
                    with autocast("cuda"):
                        logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        ).logits
                else:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    ).logits

                stream.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                probs = F.softmax(logits, dim=-1).cpu().float().numpy()
                all_probs.append(probs)

    result_container.append((np.vstack(all_probs), latencies))


def run(
    datasets: Dict[str, AGNewsDataset],
    checkpoint_dirs: Dict[str, str],
    batch_size: int = 32,
) -> Dict[str, InferenceResult]:
    """
    Parallel inference on a single GPU using one CUDA stream per model.
    Models are loaded onto the same GPU; threads issue work to separate streams.

    NOTE: True kernel-level parallelism is limited by GPU SM availability.
    This primarily overlaps CPU-side data transfers and async dispatch.

    Args:
        datasets:        {model_key: AGNewsDataset}.
        checkpoint_dirs: {model_key: checkpoint_path}.
        batch_size:      Inference batch size.

    Returns:
        Dict mapping model_key → InferenceResult.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("parallel_ensemble_single_gpu requires a CUDA GPU.")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    # Load all models onto the same GPU
    models: Dict[str, torch.nn.Module] = {}
    for key, ckpt_dir in checkpoint_dirs.items():
        print(f"  [parallel_ensemble_single_gpu] Loading {key} onto {device} …")
        model, _ = load_model_for_inference(ckpt_dir, device)
        models[key] = model

    # Create one CUDA stream per model
    streams = {key: torch.cuda.Stream(device=device) for key in checkpoint_dirs}
    loaders = {
        key: make_dataloader(datasets[key], batch_size, shuffle=False, num_workers=0)
        for key in checkpoint_dirs
    }

    result_containers: Dict[str, List] = {key: [] for key in checkpoint_dirs}
    threads: List[threading.Thread] = []

    t_start = time.perf_counter()

    for key in checkpoint_dirs:
        t = threading.Thread(
            target=_infer_on_stream,
            args=(
                models[key],
                loaders[key],
                device,
                streams[key],
                result_containers[key],
                True,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    peak_mem = get_peak_gpu_mem_mb()
    total_wall = t_end - t_start

    results: Dict[str, InferenceResult] = {}
    for key in checkpoint_dirs:
        probs, latencies = result_containers[key][0]
        results[key] = InferenceResult(
            probabilities=probs,
            latency_ms_per_batch=latencies,
            total_wall_time_s=total_wall,          # shared wall time (parallel)
            mode_name="parallel_ensemble_single_gpu",
            device_label=f"GPU:0 ({torch.cuda.get_device_name(0)}) [streams]",
            peak_gpu_mem_mb=peak_mem,
        )

    for model in models.values():
        del model
    torch.cuda.empty_cache()

    return results
