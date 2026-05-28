#!/usr/bin/env python
# coding: utf-8
"""
Sliced-backbone GPN classifier training driver.

Variant of gpn.ss.run_classifier that supports:
  - training on the first N conv layers of the GPN trunk (N configurable)
  - correct weight-loading order: seed -> random init -> overlay pretrained,
    so seeded re-init never clobbers pretrained backbone weights.
  - optional freezing of the backbone
  - selectable classification head (lstm | mean_pooling)

The core data pipeline, tokenization, BalancedTrainer, and compute_metrics are
reused from gpn.ss.run_classifier; this file only replaces the model-build
block and the seeding block.
"""

import gc
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
)

import gpn.model  # noqa: F401 — registers ConvNetConfig with AutoConfig

from gpn.ss.run_classifier import (
    BalancedTrainer,
    CustomTrainingArguments,
    DataTrainingArguments,
    ModelArguments,
    NumericalTrialFailure,
    ParquetClassificationDataset,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)


@dataclass
class SliceArguments:
    """Extra flags for layer slicing, pretrained overlay, and head selection."""

    num_conv_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Use only the first N conv layers of the trunk. None = use full depth from config."},
    )
    pretrained_path: str = field(
        default="None",
        metadata={"help": "Pretrained checkpoint directory. Use the literal string 'None' for scratch runs."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "If true, freeze model.encoder (classifier head still trains)."},
    )
    classification_head: str = field(
        default="mean_pooling",
        metadata={"help": "Classification head to use: 'mean_pooling' or 'lstm'."},
    )


def seed_everything(seed: int) -> None:
    """Seed all RNGs that can affect model init, data sampling, and dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def _apply_runtime_config(cfg, data_args, training_args, slice_args):
    cfg.num_labels = 2
    cfg.hidden_dropout_prob = training_args.hidden_dropout_prob
    setattr(cfg, "conv_dropout_p", training_args.conv_dropout_p)
    setattr(cfg, "seq_len", data_args.max_seq_length)
    setattr(cfg, "lstm_pool_size", training_args.lstm_pool_size)
    setattr(cfg, "lstm_pool_stride", training_args.lstm_pool_stride)
    setattr(cfg, "classification_head", slice_args.classification_head)
    return cfg


def build_sliced_model(model_args, data_args, training_args, slice_args):
    """
    Build the model with the correct weight-loading order:

      1. Resolve config (from pretrained dir or fresh), apply slice depth.
      2. Seed all RNGs.
      3. Instantiate the model from_config — runs random init with the seed.
      4. If pretrained_path is provided, overlay matching keys from
         model.safetensors. Sliced trunks get only the keys they can accept.
         Classification-head keys are absent from the MLM checkpoint, so the
         head retains its seed-dependent init.

    DO NOT reorder step 2 after step 4 — that would re-init and destroy the
    pretrained overlay.
    """
    is_pretrained = slice_args.pretrained_path and slice_args.pretrained_path.lower() != "none"

    # 1. Config
    if is_pretrained:
        cfg = AutoConfig.from_pretrained(slice_args.pretrained_path)
    else:
        if model_args.model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model_type={model_args.model_type!r}. Available: {list(CONFIG_MAPPING.keys())}"
            )
        cfg = CONFIG_MAPPING[model_args.model_type]()

    if slice_args.num_conv_layers is not None:
        cfg.n_layers = int(slice_args.num_conv_layers)

    cfg = _apply_runtime_config(cfg, data_args, training_args, slice_args)

    # 2. Seed everything BEFORE init so random weights are seed-controlled
    seed_everything(training_args.seed)

    # 3. Instantiate + random init (consumes seeded RNG)
    model = AutoModelForSequenceClassification.from_config(cfg)
    logger.info(
        "Built model from_config: n_layers=%s hidden_size=%s head=%s pretrained=%s seed=%s",
        cfg.n_layers,
        cfg.hidden_size,
        getattr(cfg, "classification_head", "lstm"),
        is_pretrained,
        training_args.seed,
    )

    # 4. Overlay pretrained weights (if any) AFTER seeded init
    if is_pretrained:
        safetensors_path = os.path.join(slice_args.pretrained_path, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"Expected pretrained weights at {safetensors_path}"
            )
        full_sd = load_safetensors(safetensors_path)
        own_sd = model.state_dict()
        kept = {}
        shape_mismatched = []
        for key, tensor in full_sd.items():
            if key not in own_sd:
                continue
            if own_sd[key].shape != tensor.shape:
                shape_mismatched.append((key, tuple(own_sd[key].shape), tuple(tensor.shape)))
                continue
            kept[key] = tensor
        missing, unexpected = model.load_state_dict(kept, strict=False)
        backbone_loaded = sum(1 for k in kept if k.startswith("model.encoder."))
        logger.info(
            "Pretrained overlay: checkpoint_tensors=%d kept=%d backbone_layers_loaded=%d "
            "missing=%d unexpected=%d shape_mismatched=%d",
            len(full_sd),
            len(kept),
            backbone_loaded,
            len(missing),
            len(unexpected),
            len(shape_mismatched),
        )
        if shape_mismatched:
            logger.info("Shape-mismatched keys (first 5): %s", shape_mismatched[:5])
        if backbone_loaded == 0:
            raise RuntimeError(
                f"Pretrained overlay kept 0 backbone (model.encoder.*) tensors from "
                f"{safetensors_path}. Refusing to train on silent zero-overlay."
            )

    # 5. Optional freeze
    if slice_args.freeze_backbone:
        frozen = 0
        for p in model.model.encoder.parameters():
            p.requires_grad = False
            frozen += 1
        logger.info("Froze %d backbone parameter tensors (encoder only)", frozen)

    return cfg, model


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments, SliceArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, slice_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, slice_args = parser.parse_args_into_dataclasses()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed: %s, fp16: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Slice parameters %s", slice_args)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    train_pos_path = os.path.join(data_args.datadir, "train_positives.parquet")
    train_neg_path = os.path.join(data_args.datadir, "train_negatives.parquet")
    val_pos_path = os.path.join(data_args.datadir, "val_positives.parquet")
    val_neg_path = os.path.join(data_args.datadir, "val_negatives.parquet")

    logger.info("Loading classifier parquet datasets from %s", data_args.datadir)

    pos_dataset = ParquetClassificationDataset(train_pos_path, tokenizer, max_length=data_args.max_seq_length)
    neg_dataset = ParquetClassificationDataset(train_neg_path, tokenizer, max_length=data_args.max_seq_length)
    train_dataset = ConcatDataset([pos_dataset, neg_dataset])

    val_pos_dataset = ParquetClassificationDataset(val_pos_path, tokenizer, max_length=data_args.max_seq_length)
    val_neg_dataset = ParquetClassificationDataset(val_neg_path, tokenizer, max_length=data_args.max_seq_length)
    val_dataset = ConcatDataset([val_pos_dataset, val_neg_dataset])

    test_pos_path = os.path.join(data_args.datadir, "test_positives.parquet")
    test_neg_path = os.path.join(data_args.datadir, "test_negatives.parquet")
    if os.path.exists(test_pos_path) and os.path.exists(test_neg_path):
        test_pos_dataset = ParquetClassificationDataset(test_pos_path, tokenizer, max_length=data_args.max_seq_length)
        test_neg_dataset = ParquetClassificationDataset(test_neg_path, tokenizer, max_length=data_args.max_seq_length)
        test_dataset = ConcatDataset([test_pos_dataset, test_neg_dataset])
    else:
        test_dataset = None
        logger.warning("Test parquet files not found; skipping test metrics.")

    split_datasets = [
        ("train_pos", pos_dataset),
        ("train_neg", neg_dataset),
        ("val_pos", val_pos_dataset),
        ("val_neg", val_neg_dataset),
    ]
    if test_dataset is not None:
        split_datasets.extend([("test_pos", test_pos_dataset), ("test_neg", test_neg_dataset)])
    for split, ds in split_datasets:
        print(split, ds.df["label"].value_counts(), flush=True)

    world_size = max(1, training_args.world_size)
    eff_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
    print(
        f"Train examples: {len(train_dataset)}, world_size: {world_size}, "
        f"steps_per_epoch (optimizer steps): {steps_per_epoch}",
        flush=True,
    )

    train_labels = np.concatenate(
        [
            pos_dataset.df["label"].to_numpy(dtype=np.int64),
            neg_dataset.df["label"].to_numpy(dtype=np.int64),
        ]
    )
    class_counts = np.bincount(train_labels, minlength=2)
    inv_class_weights = np.zeros(2, dtype=np.float64)
    for cls_idx, count in enumerate(class_counts):
        if count > 0:
            inv_class_weights[cls_idx] = 1.0 / float(count)
    train_sample_weights = torch.tensor(inv_class_weights[train_labels], dtype=torch.double)
    logger.info(
        "Balanced sampler configured with class counts=%s inverse weights=%s",
        class_counts.tolist(),
        inv_class_weights.tolist(),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.asarray(labels)
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        if not torch.isfinite(logits_t).all():
            raise NumericalTrialFailure("non-finite logits during evaluation")
        probs_t = torch.softmax(logits_t, dim=-1)
        per_ex_xent_t = F.cross_entropy(logits_t, labels_t, reduction="none")
        probs = probs_t.cpu().numpy()[:, 1]
        preds = logits_t.argmax(dim=-1).cpu().numpy()
        per_ex_xent = per_ex_xent_t.cpu().numpy()
        pos_mask = labels == 1
        neg_mask = labels == 0
        xent_pos = per_ex_xent[pos_mask].mean() if pos_mask.any() else float("nan")
        xent_neg = per_ex_xent[neg_mask].mean() if neg_mask.any() else float("nan")
        xent_balanced = 0.5 * (xent_pos + xent_neg) if (pos_mask.any() and neg_mask.any()) else float("nan")
        return {
            "accuracy": accuracy_score(labels, preds),
            "auroc": roc_auc_score(labels, probs),
            "auprc": average_precision_score(labels, probs),
            "mcc": matthews_corrcoef(labels, preds),
            "random_auprc": labels.mean(),
            "cross_entropy_pos": xent_pos,
            "cross_entropy_neg": xent_neg,
            "cross_entropy_balanced": xent_balanced,
        }

    def model_init(trial=None):
        _, model = build_sliced_model(model_args, data_args, training_args, slice_args)
        return model

    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer = BalancedTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_sample_weights=train_sample_weights,
        test_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    # Using "4" here makes sense to me if I am using 1000 steps per eval. 

    print("Starting sliced-backbone training with args:", trainer.args, flush=True)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete and model saved to %s", training_args.output_dir)
    if trainer.state.best_model_checkpoint:
        logger.info("Best checkpoint on disk: %s", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
