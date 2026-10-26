"""
train.py – Fine-tune BERT, RoBERTa, and ELECTRA on AG News.
Saves checkpoints to checkpoints/<model_key>/
Logs validation accuracy after each epoch.
"""

from __future__ import annotations

import os
import time
import json
import logging
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple

from data import (
    load_config,
    set_seed,
    load_raw_splits,
    tokenize_texts,
    AGNewsDataset,
    make_dataloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if use_fp16 and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            logger.info(
                f"  Epoch {epoch} | Step {step+1}/{n_batches} | Loss: {loss.item():.4f}"
            )

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> Tuple[float, float]:
    """Evaluate model. Returns (accuracy, macro_f1)."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if use_fp16 and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train_model(
    model_key: str,
    model_name: str,
    config: dict,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
) -> Dict[str, float]:
    """Fine-tune one model and save checkpoint. Returns best val metrics."""
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = config["training"]["fp16"] and device.type == "cuda"
    max_len  = config["max_seq_len"]
    cache_dir       = config["paths"]["data_cache"]
    checkpoint_dir  = os.path.join(config["paths"]["checkpoints"], model_key)
    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs       = config["training"]["epochs"]
    lr           = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    warmup_ratio = config["training"]["warmup_ratio"]
    train_bs     = config["training"]["per_device_train_batch_size"]
    eval_bs      = config["training"]["per_device_eval_batch_size"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_key} ({model_name})  →  {device}")
    logger.info(f"Train: {len(train_texts):,}  Val: {len(val_texts):,}")
    logger.info(f"{'='*60}")

    # Tokenize
    enc_train = tokenize_texts(train_texts, model_name, max_len, cache_dir)
    enc_val   = tokenize_texts(val_texts,   model_name, max_len, cache_dir)
    train_ds  = AGNewsDataset(enc_train, list(train_labels))
    val_ds    = AGNewsDataset(enc_val,   list(val_labels))

    train_loader = make_dataloader(train_ds, train_bs, shuffle=True,  num_workers=0)
    val_loader   = make_dataloader(val_ds,   eval_bs,  shuffle=False, num_workers=0)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["num_labels"],
        cache_dir=cache_dir,
    ).to(device)

    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(device="cuda", enabled=use_fp16)

    best_acc, best_f1, best_epoch = 0.0, 0.0, 0

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_fp16, epoch
        )
        val_acc, val_f1 = evaluate(model, val_loader, device, use_fp16)
        elapsed = time.time() - t0

        logger.info(
            f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc, best_f1, best_epoch = val_acc, val_f1, epoch
            model.save_pretrained(checkpoint_dir)
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir).save_pretrained(
                checkpoint_dir
            )
            logger.info(f"  ✓ Saved best checkpoint (epoch {epoch}) → {checkpoint_dir}")

    # Persist val metrics alongside checkpoint for use in weighted voting
    metrics_path = os.path.join(checkpoint_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"val_acc": best_acc, "val_f1": best_f1, "best_epoch": best_epoch}, f, indent=2)

    logger.info(
        f"\n{model_key} Best → Epoch {best_epoch} | Acc: {best_acc:.4f} | F1: {best_f1:.4f}\n"
    )
    return {"model_key": model_key, "val_acc": best_acc, "val_f1": best_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune transformer models on AG News")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "roberta", "electra", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    # Load raw text splits using the user-specified pattern
    logger.info("Loading fancyzhx/ag_news and building splits …")
    train_texts, train_labels, val_texts, val_labels, _, _ = load_raw_splits(config)

    models_cfg = config["models"]
    model_map  = {
        "bert":    models_cfg["bert"],
        "roberta": models_cfg["roberta"],
        "electra": models_cfg["electra"],
    }
    target_models = list(model_map.keys()) if "all" in args.models else args.models

    results = []
    for key in target_models:
        res = train_model(
            key, model_map[key], config,
            train_texts, train_labels, val_texts, val_labels,
        )
        results.append(res)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"  {r['model_key']:10s}  Val Acc: {r['val_acc']:.4f}  Val F1: {r['val_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
