"""
inference/single_model_cpu.py – Single model, CPU inference.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from inference.base import (
    InferenceResult,
    load_model_for_inference,
    run_model_on_loader,
    get_peak_gpu_mem_mb,
)
from data import AGNewsDataset, make_dataloader


def run(
    dataset: AGNewsDataset,
    checkpoint_dir: str,
    batch_size: int = 32,
) -> InferenceResult:
    """
    Run inference with a single model on CPU.

    Args:
        dataset:        Pre-tokenized AGNewsDataset.
        checkpoint_dir: Path to the fine-tuned model checkpoint.
        batch_size:     Inference batch size.

    Returns:
        InferenceResult with probabilities and timing.
    """
    device = torch.device("cpu")
    model, _ = load_model_for_inference(checkpoint_dir, device)
    loader = make_dataloader(dataset, batch_size, shuffle=False, num_workers=0)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t_start = time.perf_counter()
    probs, latencies = run_model_on_loader(model, loader, device, use_fp16=False)
    t_end = time.perf_counter()

    return InferenceResult(
        probabilities=probs,
        latency_ms_per_batch=latencies,
        total_wall_time_s=t_end - t_start,
        mode_name="single_model_cpu",
        device_label="CPU",
        peak_gpu_mem_mb=0.0,
    )
