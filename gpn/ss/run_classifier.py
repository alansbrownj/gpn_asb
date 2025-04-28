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

from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             average_precision_score,
                             precision_recall_curve,
                             roc_curve,
                             matthews_corrcoef)
# import evaluate
## Adding EarlyStoppingCallback -- 2025-03-18 Tuesday W12.2
## needed AutoModelForSequenceClassification
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
import pandas as pd
from scipy.stats import geom
# from torch.utils.data import DataLoader, IterableDataset, get_worker_info, ConcatDataset, WeightedRandomSampler, Dataset
from torch.utils.data import ConcatDataset, WeightedRandomSampler, Dataset
from tqdm import tqdm

## These imports are *mostly* the same as the run_mlm.py script. 
# help(TrainingArguments)

print("Python version:", sys.version)
# print(f"I think these are the available models: {list(CONFIG_MAPPING.keys())}")

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

## define dataclass for arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained model or shortcut name. If not specified, a new model will be created."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(CONFIG_MAPPING.keys())},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pre-trained tokenizer name or path"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )

    # same as from run_mlm.py
    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "(Not sure if this still applies...) The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    datadir : Optional[str] = field(
        default=None,
        metadata={"help": "The input data directory. Should contain the training files for the model."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    def __post_init__(self):
        if(
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )

# ----- Main training function ----- #
def main():
    # Initialize argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    ## copied from run_mlm.py:
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## also copying from the mlm py
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classifier", model_args, data_args)
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # if data_args.min_lr_rate is not None:
    #     print(f"Setting min_lr_rate to {data_args.min_lr_rate}")
    #     training_args.lr_scheduler_kwargs["min_lr_rate"] = data_args.min_lr_rate

    # Set seed for reproducibility
    torch.manual_seed(training_args.seed)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    # Initialize model configuration and model
    if model_args.model_name_or_path is not None and model_args.model_name_or_path.lower() != "none":
        # Load pretrained configuration and model
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = 2  # binary classification
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
        logger.info(f"Loaded pre-trained model from {model_args.model_name_or_path}")
    else:
        # Initialize the model from scratch using the provided model type.
        # Make sure the model type exists in CONFIG_MAPPING.
        if model_args.model_type not in CONFIG_MAPPING:
            raise ValueError(f"Unknown model type: {model_args.model_type}. Available types: {list(CONFIG_MAPPING.keys())}")
        config = CONFIG_MAPPING[model_args.model_type]()
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_config(config)
        logger.info(f"Initialized model from scratch using the configuration for {model_args.model_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the positive and negative datasets for training
    pos_dataset = ParquetClassificationDataset(f"{data_args.datadir}/train_positives.parquet", tokenizer, max_length=512)
    neg_dataset = ParquetClassificationDataset(f"{data_args.datadir}/train_negatives.parquet", tokenizer, max_length=512)

    # Combine datasets
    # combined_dataset = ConcatDataset([pos_dataset, neg_dataset])
    train_dataset = ConcatDataset([pos_dataset, neg_dataset])

    ## this change is faster? 
    # labels_tensor = torch.tensor(pd.concat([pos_dataset.df["label"], neg_dataset.df["label"]]).values)
    # class_sample_count = torch.tensor(
    #     [(labels_tensor == t).sum() for t in torch.unique(labels_tensor, sorted=True)]
    # )
    # # Inverse frequency weighting
    # weights = 1. / class_sample_count.float()
    # sample_weights = weights[labels_tensor]

    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # train_dataloader = DataLoader(combined_dataset, batch_size=training_args.per_device_train_batch_size, sampler=sampler, num_workers=2)

    # VALIDATION DATA
    val_pos_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/val_positives.parquet", tokenizer, max_length=512
    )
    val_neg_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/val_negatives.parquet", tokenizer, max_length=512
    )
    val_dataset = ConcatDataset([val_pos_dataset, val_neg_dataset])
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=training_args.per_device_eval_batch_size or 256,
    #     shuffle=False,
    #     num_workers=2,
    # )

    # def evaluate(model, dataloader):
    #     model.eval()
    #     losses, correct, total = [], 0, 0
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             batch = {k: v.to(device) for k, v in batch.items()}
    #             outputs = model(**batch)
    #             losses.append(outputs.loss.item())
    #             preds = outputs.logits.argmax(dim=-1)
    #             correct += (preds == batch["labels"]).sum().item()
    #             total += batch["labels"].size(0)
    #     model.train()
    #     return np.mean(losses), correct / total
    # best_eval_loss = float("inf")
    print("Printing data splits: ", flush=True)
    for split, ds in zip(
        ["train_pos","train_neg","val_pos","val_neg"],
        [pos_dataset, neg_dataset, val_pos_dataset, val_neg_dataset],
    ):
        print(split, ds.df["label"].value_counts(), flush=True)

    class BalancedTrainer(transformers.Trainer):
        def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            # build the same sampler you used before
            labels = torch.tensor(
                pd.concat([pos_dataset.df["label"], neg_dataset.df["label"]]).values
            )
            class_sample_count = torch.tensor(
                [(labels == t).sum() for t in torch.unique(labels, sorted=True)]
            )
            weights = 1.0 / class_sample_count.float()
            sample_weights = weights[labels]
            return torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = logits.argmax(axis=-1)

        return {
            "accuracy" : accuracy_score(labels, preds),
            "auroc"    : roc_auc_score(labels, probs),
            "auprc"    : average_precision_score(labels, probs),
            "mcc"      : matthews_corrcoef(labels, preds),
        }

    hf_args = TrainingArguments(
        output_dir           = training_args.output_dir,
        overwrite_output_dir = True,
        num_train_epochs     = training_args.num_train_epochs,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        per_device_eval_batch_size  = training_args.per_device_eval_batch_size,
        eval_strategy        = training_args.eval_strategy,
        eval_steps           = training_args.eval_steps,
        save_steps           = training_args.save_steps,
        save_total_limit     = training_args.save_total_limit,
        learning_rate        = training_args.learning_rate,
        warmup_steps         = training_args.warmup_steps,
        logging_dir          = os.path.join(training_args.output_dir, "logs"),
        logging_steps        = training_args.logging_steps,
        load_best_model_at_end = True,
        metric_for_best_model  = training_args.metric_for_best_model,
        report_to            = ["tensorboard"],
        fp16                 = training_args.fp16,
        optim                = training_args.optim,
    )

    trainer = BalancedTrainer(
        model          = model,
        args           = hf_args,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        tokenizer      = tokenizer,
        compute_metrics= compute_metrics,
    )
    # # Set up optimizer and scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    # num_update_steps_per_epoch = len(train_dataloader)
    # num_training_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # lr_scheduler = get_scheduler(
    #     name="cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=int(0.1 * num_training_steps),
    #     num_training_steps=num_training_steps,
    # )
    # print(f"Number of training steps: {num_training_steps}")
    # print(f"Number of update steps per epoch: {num_update_steps_per_epoch}")
    # # Training loop
    # model.train()
    # global_step = 0
    # for epoch in tqdm(range(int(training_args.num_train_epochs))):
    #     for step, batch in tqdm(enumerate(train_dataloader)):
    #         # if "token_type_ids" in batch:
    #         #     batch.pop("token_type_ids")
    #         # print("PRINTING BATCH")
    #         # print(batch)
    #         optimizer.zero_grad(set_to_none=True)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs: SequenceClassifierOutput = model(**batch)
    #         loss = outputs.loss
    #         loss.backward() ## where is forward?
    #         optimizer.step()
    #         lr_scheduler.step()
            
    #         global_step += 1
    #         if global_step % training_args.logging_steps == 0:
    #             logger.info(f"Epoch {epoch+1} Step {global_step}/{num_training_steps} Loss: {loss.item():.4f}")
            
    #         if global_step % training_args.save_steps == 0:
    #             os.makedirs(training_args.output_dir, exist_ok=True)
    #             save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
    #             model.save_pretrained(save_path)
    #             logger.info(f"Saved checkpoint to {save_path}")
            
    #         ## evaluate every eval_steps
    #         if global_step % training_args.eval_steps == 0:
    #             eval_loss, eval_acc = evaluate(model, val_dataloader)
    #             logger.info(
    #                 f"[EVAL]  Step {global_step} • loss {eval_loss:.4f} • acc {eval_acc:.4%}"
    #             )

    #             # save best model so far
    #             if eval_loss < best_eval_loss:
    #                 best_eval_loss = eval_loss
    #                 best_path = os.path.join(training_args.output_dir, "checkpoint-best")
    #                 os.makedirs(best_path, exist_ok=True)
    #                 model.save_pretrained(best_path)
    #                 tokenizer.save_pretrained(best_path)
    #                 logger.info(f"New best model saved to {best_path} (eval_loss {eval_loss:.4f})")

    # Final save
    os.makedirs(training_args.output_dir, exist_ok=True)
    # model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete and model saved.")
    best_ckpt = trainer.state.best_model_checkpoint
    logger.info("best checkpoint on disk:", best_ckpt)      # e.g. .../checkpoint-4500
    
    

if __name__ == "__main__":
    main()