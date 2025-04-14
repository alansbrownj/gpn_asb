#!/usr/bin/env python
# coding: utf-8

## 2025-04-08 Tuesday W15.2 
## Writing this script for sequence classification

import logging
import math
import numpy as np
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
import torch
import torch.nn as nn
import argparse
import time
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets

# import evaluate
## Adding EarlyStoppingCallback -- 2025-03-18 Tuesday W12.2
## needed AutoModelForSequenceClassification
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.modeling_outputs import SequenceClassifierOutput

from Bio.Seq import Seq
import gpn.model
import numpy as np
import pandas as pd
from scipy.stats import geom
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, ConcatDataset, WeightedRandomSampler, Dataset
from tqdm import tqdm

## These imports are *mostly* the same as the run_mlm.py script. 

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)

class ParquetClassificationDataset(Dataset):
    def __init__(self, parquet_file: str, tokenizer, max_length: int = 512):
        """
        Args:
            parquet_file: path to the Parquet file.
            tokenizer: a Hugging Face tokenizer.
            max_length: maximum sequence length for tokenization.
        """
        self.df = pd.read_parquet(parquet_file)
        # Ensure the dataframe has a sequence column and a label column
        # You can also check for other columns if needed.
        assert 'seq' in self.df.columns and 'label' in self.df.columns, "Missing required columns"
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the row as a dict
        row = self.df.iloc[idx]
        seq = row['seq']
        # Tokenize using the provided tokenizer; modify as needed for your sequence type.
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=False
        )
        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # Use the label from the Parquet; convert to torch tensor
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item

# ----- Main training function ----- #
def main():
    parser = argparse.ArgumentParser(description="Run GPN-based Sequence Classification")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--model_type", type=str, default="ConvNet",
                        help="Model type to use when training from scratch (e.g., ConvNet, bytenet, roformer)")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Pre-trained model path")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name/path")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Initialize model configuration and model
    if args.model_name_or_path is not None and args.model_name_or_path.lower() != "none":
        # Load pretrained configuration and model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = 2  # binary classification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        logger.info(f"Loaded pre-trained model from {args.model_name_or_path}")
    else:
        # Initialize the model from scratch using the provided model type.
        # Make sure the model type exists in CONFIG_MAPPING.
        if args.model_type not in CONFIG_MAPPING:
            raise ValueError(f"Unknown model type: {args.model_type}. Available types: {list(CONFIG_MAPPING.keys())}")
        config = CONFIG_MAPPING[args.model_type]()
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_config(config)
        logger.info(f"Initialized model from scratch using the configuration for {args.model_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the positive and negative datasets for training
    pos_dataset = ParquetClassificationDataset(f"{args.datadir}/train_positives.parquet", tokenizer, max_length=512)
    neg_dataset = ParquetClassificationDataset(f"{args.datadir}/train_negatives.parquet", tokenizer, max_length=512)

    # Combine datasets
    combined_dataset = ConcatDataset([pos_dataset, neg_dataset])

    # Create balanced sampling weights: equal weight for each class
    labels = []
    for ds in [pos_dataset, neg_dataset]:
        for i in range(len(ds)):
            sample = ds[i]
            labels.append(sample["labels"].item())
    labels_tensor = torch.tensor(labels)
    class_sample_count = torch.tensor(
        [(labels_tensor == t).sum() for t in torch.unique(labels_tensor, sorted=True)]
    )
    # Inverse frequency weighting
    weights = 1. / class_sample_count.float()
    sample_weights = weights[labels_tensor]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataloader = DataLoader(combined_dataset, batch_size=128, sampler=sampler, num_workers=4)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    model.train()
    global_step = 0
    for epoch in tqdm(range(args.num_train_epochs)):
        for step, batch in tqdm(enumerate(train_dataloader)):
            # if "token_type_ids" in batch:
            #     batch.pop("token_type_ids")
            # print("PRINTING BATCH")
            # print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs: SequenceClassifierOutput = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                logger.info(f"Epoch {epoch+1} Step {global_step}/{num_training_steps} Loss: {loss.item():.4f}")
            
            if global_step % args.save_steps == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    main()