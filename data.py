"""
data.py – Dataset loading, tokenization, large-workload construction, and batch sharding.
Uses fancyzhx/ag_news with a reproducible train/val/test split.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DATA] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AGNewsDataset(Dataset):
    """AG News dataset wrapper with pre-tokenized inputs."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_texts(
    texts: List[str],
    tokenizer_name: str,
    max_len: int,
    cache_dir: str,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of texts and return tensors."""
    logger.info(f"Tokenizing {len(texts)} texts with {tokenizer_name} (max_len={max_len})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc


def load_raw_splits(config: dict) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load fancyzhx/ag_news and produce train / val / test text+label lists.

    Strategy (matches user spec):
      - Shuffle full train (120k) with seed=42, select first 20k
      - Split 20k → 18k train / 2k val  (test_size=0.1)
      - Test split is the original 7,600-sample test set
    """
    seed = config["seed"]
    cache_dir = config["paths"]["data_cache"]

    logger.info("Loading fancyzhx/ag_news …")
    train_full = load_dataset("fancyzhx/ag_news", split="train", cache_dir=cache_dir)
    test_raw   = load_dataset("fancyzhx/ag_news", split="test",  cache_dir=cache_dir)

    subset = train_full.shuffle(seed=seed).select(range(20_000))
    splits = subset.train_test_split(test_size=0.1, seed=seed)
    train_data = splits["train"]   # 18 000
    valid_data = splits["test"]    # 2 000

    train_texts  = train_data["text"];  train_labels  = train_data["label"]
    val_texts    = valid_data["text"];  val_labels    = valid_data["label"]
    test_texts   = test_raw["text"];    test_labels   = test_raw["label"]

    logger.info(
        f"Split sizes → train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}"
    )
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def get_ag_news_splits(
    config: dict,
    tokenizer_name: Optional[str] = None,
) -> Tuple["AGNewsDataset", "AGNewsDataset", "AGNewsDataset", List[str], List[int]]:
    """
    Full pipeline: load raw splits → tokenize with *tokenizer_name* → return Datasets.

    Returns:
        train_dataset, val_dataset, test_dataset, test_texts (raw), test_labels (raw)
    """
    if tokenizer_name is None:
        tokenizer_name = config["models"]["bert"]

    max_len   = config["max_seq_len"]
    cache_dir = config["paths"]["data_cache"]

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_raw_splits(config)

    enc_train = tokenize_texts(train_texts, tokenizer_name, max_len, cache_dir)
    enc_val   = tokenize_texts(val_texts,   tokenizer_name, max_len, cache_dir)
    enc_test  = tokenize_texts(test_texts,  tokenizer_name, max_len, cache_dir)

    return (
        AGNewsDataset(enc_train, train_labels),
        AGNewsDataset(enc_val,   val_labels),
        AGNewsDataset(enc_test,  test_labels),
        test_texts,
        test_labels,
    )


def get_model_tokenized_dataset(
    texts: List[str],
    labels: List[int],
    model_name: str,
    max_len: int,
    cache_dir: str,
) -> "AGNewsDataset":
    """Tokenize texts with a specific model's tokenizer and return an AGNewsDataset."""
    enc = tokenize_texts(texts, model_name, max_len, cache_dir)
    return AGNewsDataset(enc, list(labels))


def build_large_workload(
    test_texts: List[str],
    test_labels: List[int],
    target_size: int,
    seed: int,
) -> Tuple[List[str], List[int], int]:
    """
    Repeat the test set to build a large workload of `target_size` texts.

    Returns:
        (large_texts, large_labels, original_test_size)
    """
    original_size = len(test_texts)
    times = (target_size // original_size) + 1
    large_texts  = (list(test_texts)  * times)[:target_size]
    large_labels = (list(test_labels) * times)[:target_size]
    logger.info(
        f"Built large workload: {len(large_texts):,} samples "
        f"(original test: {original_size:,})"
    )
    return large_texts, large_labels, original_size


def make_dataloader(
    dataset: "AGNewsDataset",
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from an AGNewsDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def shard_dataset(
    dataset: "AGNewsDataset",
    num_shards: int,
) -> List["AGNewsDataset"]:
    """Split a dataset into `num_shards` roughly equal parts."""
    size = len(dataset)
    shards = []
    for i in range(num_shards):
        start = (i * size) // num_shards
        end   = ((i + 1) * size) // num_shards
        shard_enc    = {k: v[start:end] for k, v in dataset.encodings.items()}
        shard_labels = dataset.labels[start:end]
        shards.append(AGNewsDataset(shard_enc, shard_labels))
    return shards


if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])

    train_ds, val_ds, test_ds, test_texts, test_labels = get_ag_news_splits(cfg)

    large_texts, large_labels, orig_size = build_large_workload(
        test_texts, test_labels,
        cfg["dataset"]["large_workload"],
        cfg["seed"],
    )

    print(f"✓ Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"✓ Large workload: {len(large_texts):,}")
    print(f"  Sample: {test_texts[0][:80]}… → label {test_labels[0]}")
