"""
inference/base.py – Common utilities for all inference modules.
Defines InferenceResult dataclass and shared model-loading helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class InferenceResult:
    """Container for inference outputs and timing metadata."""
    probabilities: np.ndarray           # (N, num_classes) float32
    latency_ms_per_batch: List[float]   # per-batch latency in ms
    total_wall_time_s: float            # total wall-clock seconds
    mode_name: str
    device_label: str
    peak_gpu_mem_mb: float = 0.0        # peak GPU memory in MB

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latency_ms_per_batch))

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latency_ms_per_batch, 95))

    @property
    def throughput_samples_per_sec(self) -> float:
        n = len(self.probabilities)
        return n / max(self.total_wall_time_s, 1e-9)


def load_model_for_inference(
    checkpoint_dir: str,
    device: torch.device,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a fine-tuned model and tokenizer from a checkpoint directory."""
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


def collate_batch(batch: List[dict]) -> dict:
    """Collate a list of dataset items into a batched dict."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def run_model_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run model inference on a DataLoader.
    Returns (probabilities array of shape (N, C), list of per-batch latencies in ms).
    """
    all_probs = []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            t0 = time.perf_counter()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            if use_fp16 and device.type == "cuda":
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

            if device.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            probs = F.softmax(logits, dim=-1).cpu().float().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs), latencies


def get_peak_gpu_mem_mb() -> float:
    """Return peak GPU memory allocated in MB (resets the counter)."""
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        return mem
    return 0.0
