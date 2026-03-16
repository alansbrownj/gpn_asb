#!/usr/bin/env python
# coding: utf-8

## 2025-04-08 Tuesday W15.2 
## Writing this script for sequence classification
import optuna
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from transformers import Adafactor

import plotly.io as pio
import joblib

import csv
import fcntl
import logging
import math
import json
import numpy as np
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from transformers.trainer_utils import HPSearchBackend, get_last_checkpoint
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


class NumericalTrialFailure(RuntimeError):
    """Raised when a trial produces non-finite tensors or metrics."""

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
    hyperparameter_search: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run hyperparameter search or not. If set to True, the script will not train the model but will only run the search."
        },
    )
    hpo_total_trials: int = field(
        default=100,
        metadata={"help": "Target total number of HPO trials across all workers."},
    )
    hpo_worker_count: int = field(
        default=1,
        metadata={"help": "Number of parallel HPO workers sharing one study."},
    )
    hpo_worker_index: int = field(
        default=0,
        metadata={"help": "0-based worker index for this process."},
    )
    hpo_study_name: str = field(
        default="convnet_classifier_hpo",
        metadata={"help": "Stable Optuna study name for shared-worker tuning."},
    )
    hpo_storage: Optional[str] = field(
        default=None,
        metadata={"help": "Optuna storage URL. Defaults to sqlite:///<output_dir>/optuna.db"},
    )
    train_best_after_search: bool = field(
        default=False,
        metadata={"help": "If true, train once after HPO using best hyperparameters."},
    )

    # same as from run_mlm.py
    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        if self.hpo_total_trials < 0:
            raise ValueError("--hpo_total_trials must be >= 0")
        if self.hpo_worker_count <= 0:
            raise ValueError("--hpo_worker_count must be > 0")
        if self.hpo_worker_index < 0 or self.hpo_worker_index >= self.hpo_worker_count:
            raise ValueError("--hpo_worker_index must be in [0, hpo_worker_count)")

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
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization (padding/truncation)."}
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

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Extend HF TrainingArguments with CyclicLR knobs. Keep defaults so existing
    behavior is unchanged unless explicitly enabled.
    """
    use_cyclic_lr: bool = field(
        default=False,
        metadata={"help": "If true, use torch.optim.lr_scheduler.CyclicLR instead of the HF scheduler."},
    )
    base_lr: float = field(
        default=1e-5,
        metadata={"help": "CyclicLR base_lr (min LR)."},
    )
    max_lr: float = field(
        default=1e-3,
        metadata={"help": "CyclicLR max_lr (peak LR)."},
    )
    cycle_steps_up: int = field(
        default=1000,
        metadata={"help": "CyclicLR step_size_up."},
    )
    cycle_steps_down: Optional[int] = field(
        default=None,
        metadata={"help": "CyclicLR step_size_down; defaults to cycle_steps_up if None."},
    )
    cycle_mode: str = field(
        default="triangular2",
        metadata={"help": "CyclicLR mode (e.g., triangular, triangular2, exp_range)."},
    )
    cycle_gamma: float = field(
        default=1.0,
        metadata={"help": "CyclicLR gamma (used by exp_range mode)."},
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the ConvNet classification head."},
    )
    conv_dropout_p: float = field(
        default=0.35,
        metadata={"help": "Dropout probability for ConvLayer residual dropout."},
    )
    lstm_pool_size: int = field(
        default=4,
        metadata={"help": "MaxPool1d kernel size applied before the LSTM classification head."},
    )
    lstm_pool_stride: int = field(
        default=4,
        metadata={"help": "MaxPool1d stride applied before the LSTM classification head."},
    )

# ----- Main training function ----- #
def main():
    # Initialize argument parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    ## copied from run_mlm.py:
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ## also copying from the mlm py
    send_example_telemetry("run_classifier", model_args, data_args)
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility 
    SEED = training_args.seed
    print(f"seed used for training: {SEED}", flush=True)
    torch.manual_seed(SEED)

    # Load tokenizer and model builder
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    def _apply_runtime_config(cfg, runtime_args):
        cfg.num_labels = 2  # binary classification
        cfg.hidden_dropout_prob = runtime_args.hidden_dropout_prob
        setattr(cfg, "conv_dropout_p", runtime_args.conv_dropout_p)
        setattr(cfg, "seq_len", data_args.max_seq_length)
        setattr(cfg, "lstm_pool_size", runtime_args.lstm_pool_size)
        setattr(cfg, "lstm_pool_stride", runtime_args.lstm_pool_stride)
        return cfg

    def _build_model_with_runtime_config(runtime_args):
        if model_args.model_name_or_path is not None and model_args.model_name_or_path.lower() != "none":
            cfg = AutoConfig.from_pretrained(model_args.model_name_or_path)
            cfg = _apply_runtime_config(cfg, runtime_args)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, config=cfg
            )
            logger.info(
                "Loaded pre-trained model from %s with hidden_dropout_prob=%.4f conv_dropout_p=%.4f",
                model_args.model_name_or_path,
                cfg.hidden_dropout_prob,
                getattr(cfg, "conv_dropout_p", float("nan")),
            )
            return cfg, model

        if model_args.model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model type: {model_args.model_type}. Available types: {list(CONFIG_MAPPING.keys())}"
            )
        cfg = CONFIG_MAPPING[model_args.model_type]()
        cfg = _apply_runtime_config(cfg, runtime_args)
        model = AutoModelForSequenceClassification.from_config(cfg)
        logger.info(
            "Initialized model from scratch for %s with hidden_dropout_prob=%.4f conv_dropout_p=%.4f",
            model_args.model_type,
            cfg.hidden_dropout_prob,
            getattr(cfg, "conv_dropout_p", float("nan")),
        )
        return cfg, model

    # Load the positive and negative datasets for training
    pos_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/train_positives.parquet", tokenizer, max_length=data_args.max_seq_length
    )
    neg_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/train_negatives.parquet", tokenizer, max_length=data_args.max_seq_length
    )

    ## Going to try weighting the loss of the positives. 
    ## MIGHT NEED TO MOVE THIS CODE. 
    # pos = len(pos_dataset)
    # neg = len(neg_dataset)
    # # w_pos = math.sqrt(neg / max(pos, 1))
    # w_pos = neg / max(pos, 1)
    # class_weight = torch.tensor([1.0, w_pos], dtype=torch.float)
    # print(f"Class weights for loss: {class_weight.numpy()}", flush=True)

    # Combine datasets
    # combined_dataset = ConcatDataset([pos_dataset, neg_dataset])
    train_dataset = ConcatDataset([pos_dataset, neg_dataset])

    # VALIDATION DATA
    val_pos_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/val_positives.parquet", tokenizer, max_length=data_args.max_seq_length
    )
    val_neg_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/val_negatives.parquet", tokenizer, max_length=data_args.max_seq_length
    )
    val_dataset = ConcatDataset([val_pos_dataset, val_neg_dataset])

    # TEST DATA (optional): if missing, training/eval continues without test metrics.
    test_pos_path = os.path.join(data_args.datadir, "test_positives.parquet")
    test_neg_path = os.path.join(data_args.datadir, "test_negatives.parquet")
    if os.path.exists(test_pos_path) and os.path.exists(test_neg_path):
        test_pos_dataset = ParquetClassificationDataset(
            test_pos_path, tokenizer, max_length=data_args.max_seq_length
        )
        test_neg_dataset = ParquetClassificationDataset(
            test_neg_path, tokenizer, max_length=data_args.max_seq_length
        )
        test_dataset = ConcatDataset([test_pos_dataset, test_neg_dataset])
    else:
        test_dataset = None
        logger.warning(
            "Test parquet files not found at %s and %s. Test metrics (including test_auprc) will be skipped.",
            test_pos_path,
            test_neg_path,
        )
 
    print("Printing data splits: ", flush=True)
    split_datasets = [
        ("train_pos", pos_dataset),
        ("train_neg", neg_dataset),
        ("val_pos", val_pos_dataset),
        ("val_neg", val_neg_dataset),
    ]
    if test_dataset is not None:
        split_datasets.extend(
            [
                ("test_pos", test_pos_dataset),
                ("test_neg", test_neg_dataset),
            ]
        )
    for split, ds in split_datasets:
        print(split, ds.df["label"].value_counts(), flush=True)

    # Estimate steps per epoch (optimizer steps), accounting for grad accumulation and DDP
    world_size = max(1, training_args.world_size)  # number of processes
    eff_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
    print(
        f"Train examples: {len(train_dataset)}, world_size: {world_size}, "
        f"per_device_batch: {training_args.per_device_train_batch_size}, "
        f"grad_accum: {training_args.gradient_accumulation_steps}, "
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
        "Balanced sampler configured with class counts=%s and inverse weights=%s",
        class_counts.tolist(),
        inv_class_weights.tolist(),
    )

    class BalancedTrainer(transformers.Trainer):
        def __init__(self, *args, train_sample_weights=None, test_dataset=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_sample_weights = train_sample_weights
            self.test_dataset = test_dataset
            # [CODEX CHANGE 2026-02-27] Track per-trial peak/final eval AUPRC for HPO objective reporting.
            self.hpo_peak_eval_auprc = None
            self.hpo_peak_step = None
            self.hpo_final_eval_auprc = None

        # [CODEX CHANGE 2026-02-27] Use trial-scoped TensorBoard directories.
        def _hp_search_setup(self, trial):
            super()._hp_search_setup(trial)
            if self.hp_search_backend is None or trial is None:
                return
            trial_number = getattr(trial, 'number', None)
            if trial_number is None:
                return

            # Close previous TensorBoard writer so the next trial uses its own log dir.
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, 'tb_writer') and callback.tb_writer is not None:
                    try:
                        callback.tb_writer.close()
                    except Exception:
                        pass
                    callback.tb_writer = None

            run_dir = os.path.join(self.args.output_dir, f'run-{trial_number}')
            tb_dir = os.path.join(run_dir, 'tb')
            os.makedirs(tb_dir, exist_ok=True)
            self.args.logging_dir = tb_dir
            self.hpo_peak_eval_auprc = None
            self.hpo_peak_step = None
            self.hpo_final_eval_auprc = None
            logger.info('HPO trial %s TensorBoard log dir: %s', trial_number, tb_dir)

        @staticmethod
        def _tensor_debug_summary(tensor: Optional[Union[torch.Tensor, np.ndarray]]) -> str:
            if tensor is None:
                return 'n/a'
            data = torch.as_tensor(tensor).detach().float().cpu()
            finite = data[torch.isfinite(data)]
            min_value = float(finite.min().item()) if finite.numel() else None
            max_value = float(finite.max().item()) if finite.numel() else None
            return (
                f"shape={tuple(data.shape)} finite={int(finite.numel())}/{data.numel()} "
                f"min={min_value} max={max_value}"
            )

        def _current_learning_rate(self) -> float:
            try:
                learning_rate = self._get_learning_rate()
                if learning_rate is not None:
                    return float(learning_rate)
            except Exception:
                pass
            return float(self.args.learning_rate)

        def _raise_numerical_failure(self, reason: str, *, logits=None, probs=None, loss=None):
            trial = getattr(self, '_trial', None)
            trial_number = getattr(trial, 'number', 'n/a')
            message = (
                f"Trial {trial_number} numerical failure: {reason}; "
                f"global_step={int(self.state.global_step)}; "
                f"learning_rate={self._current_learning_rate()}; "
                f"hidden_dropout_prob={self.args.hidden_dropout_prob}; "
                f"conv_dropout_p={self.args.conv_dropout_p}; "
                f"logits={self._tensor_debug_summary(logits)}; "
                f"probs={self._tensor_debug_summary(probs)}; "
                f"loss={self._tensor_debug_summary(loss)}"
            )
            logger.error(message)
            raise NumericalTrialFailure(message)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            if not torch.isfinite(logits).all():
                self._raise_numerical_failure("non-finite logits during training", logits=logits)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss).all():
                self._raise_numerical_failure("non-finite loss during training", logits=logits, loss=loss)
            return (loss, outputs) if return_outputs else loss

        def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
            if self.args.world_size > 1:
                return super()._get_train_sampler()
            if self.train_sample_weights is None:
                return super()._get_train_sampler()
            return WeightedRandomSampler(
                self.train_sample_weights,
                num_samples=len(self.train_sample_weights),
                replacement=True,
            )

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            """
            Override to allow a CyclicLR schedule when requested; otherwise fall
            back to the HF default scheduler creation.
            """
            if getattr(self.args, "use_cyclic_lr", False):
                opt = optimizer if optimizer is not None else self.optimizer
                if opt is None:
                    raise ValueError("Optimizer must be set before creating the scheduler.")
                self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                    opt,
                    base_lr=self.args.base_lr,
                    max_lr=self.args.max_lr,
                    step_size_up=self.args.cycle_steps_up,
                    step_size_down=self.args.cycle_steps_down or self.args.cycle_steps_up,
                    mode=self.args.cycle_mode,
                    gamma=self.args.cycle_gamma,
                    cycle_momentum=False,  # AdamW/Adafactor do not use momentum buffers here
                )
                self._created_lr_scheduler = True
                return self.lr_scheduler
            return super().create_scheduler(num_training_steps, optimizer)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """
            Keep standard evaluation behavior, and additionally compute/log
            train auPRC with the same compute_metrics logic used for eval.
            """
            eval_metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            # Avoid recursion if someone explicitly runs a train-prefixed evaluate.
            if metric_key_prefix == "train":
                return eval_metrics

            # [CODEX CHANGE 2026-02-27] Track running peak eval_auprc and expose it to Optuna objective.
            if metric_key_prefix == "eval":
                eval_auprc = eval_metrics.get("eval_auprc")
                if eval_auprc is not None:
                    self.hpo_final_eval_auprc = float(eval_auprc)
                    if self.hpo_peak_eval_auprc is None or eval_auprc > self.hpo_peak_eval_auprc:
                        self.hpo_peak_eval_auprc = float(eval_auprc)
                        self.hpo_peak_step = int(self.state.global_step)
                    eval_metrics["eval_auprc_peak"] = self.hpo_peak_eval_auprc
                    eval_metrics["eval_auprc_peak_step"] = float(self.hpo_peak_step)
                    eval_metrics["eval_auprc_final"] = self.hpo_final_eval_auprc

            if metric_key_prefix == "eval" and self.test_dataset is not None:
                test_metrics = self.predict(
                    self.test_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="test",
                ).metrics
                test_auprc = test_metrics.get("test_auprc")
                if test_auprc is not None:
                    self.log({"test_auprc": test_auprc})
                    eval_metrics["test_auprc"] = test_auprc

            return eval_metrics


    # [CODEX CHANGE 2026-02-27] Helpers for study-level reporting and per-trial summaries.
    def _checkpoint_step_from_path(path_value: Optional[str]) -> Optional[int]:
        if not path_value or 'checkpoint-' not in str(path_value):
            return None
        try:
            return int(str(path_value).rsplit('checkpoint-', 1)[1])
        except Exception:
            return None

    # [CODEX CHANGE 2026-02-27] Read the latest trainer_state for a run directory.
    def _latest_trainer_state_path(run_dir: str) -> Optional[str]:
        if not os.path.isdir(run_dir):
            return None
        latest_step = -1
        latest_state_path = None
        for name in os.listdir(run_dir):
            if not name.startswith('checkpoint-'):
                continue
            try:
                step = int(name.split('-', 1)[1])
            except Exception:
                continue
            state_path = os.path.join(run_dir, name, 'trainer_state.json')
            if step > latest_step and os.path.exists(state_path):
                latest_step = step
                latest_state_path = state_path
        return latest_state_path

    # [CODEX CHANGE 2026-02-27] Extract final/peak metrics from trainer_state.json.
    def _extract_run_metrics(run_dir: str) -> Dict[str, Any]:
        out = {
            'peak_eval_auprc': None,
            'peak_step': None,
            'final_eval_auprc': None,
            'best_checkpoint': None,
        }
        state_path = _latest_trainer_state_path(run_dir)
        if state_path is None:
            return out

        with open(state_path) as f:
            state = json.load(f)

        out['best_checkpoint'] = state.get('best_model_checkpoint')
        out['peak_eval_auprc'] = state.get('best_metric')
        out['peak_step'] = _checkpoint_step_from_path(out['best_checkpoint'])

        eval_history = [entry for entry in state.get('log_history', []) if 'eval_auprc' in entry]
        if eval_history:
            out['final_eval_auprc'] = eval_history[-1].get('eval_auprc')
            if out['peak_eval_auprc'] is None:
                best_entry = max(eval_history, key=lambda x: x.get('eval_auprc', float('-inf')))
                out['peak_eval_auprc'] = best_entry.get('eval_auprc')
            if out['peak_step'] is None and out['peak_eval_auprc'] is not None:
                best_entry = max(eval_history, key=lambda x: x.get('eval_auprc', float('-inf')))
                out['peak_step'] = best_entry.get('step')

        return out

    # [CODEX CHANGE 2026-02-27] Single-objective Optuna helper.
    def _trial_objective_value(trial: optuna.trial.FrozenTrial) -> Optional[float]:
        if trial.values is None or len(trial.values) == 0:
            return None
        return trial.values[0]

    def _normalize_hpo_params(params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params)
        pool_size = normalized.get("lstm_pool_size")
        stride_divisor = normalized.pop("lstm_pool_stride_divisor", None)
        if pool_size is not None and stride_divisor is not None:
            normalized["lstm_pool_stride"] = int(pool_size) // int(stride_divisor)
        return normalized

    # [CODEX CHANGE 2026-02-27] Study-level CSV plus per-run JSON summaries.
    def write_study_trial_metrics(study: optuna.Study, output_dir: str):
        csv_path = os.path.join(output_dir, 'trial_metrics.csv')
        fieldnames = [
            'trial_number',
            'state',
            'objective',
            'peak_eval_auprc',
            'peak_step',
            'final_eval_auprc',
            'best_checkpoint',
            'retry_of_trial',
        ]

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for trial in sorted(study.get_trials(deepcopy=False), key=lambda t: t.number):
                run_dir = os.path.join(output_dir, f'run-{trial.number}')
                run_metrics = _extract_run_metrics(run_dir)
                row = {
                    'trial_number': trial.number,
                    'state': trial.state.name,
                    'objective': _trial_objective_value(trial),
                    'peak_eval_auprc': run_metrics['peak_eval_auprc'],
                    'peak_step': run_metrics['peak_step'],
                    'final_eval_auprc': run_metrics['final_eval_auprc'],
                    'best_checkpoint': run_metrics['best_checkpoint'],
                    'retry_of_trial': trial.user_attrs.get('retry_of_trial', ''),
                }
                writer.writerow(row)

                if os.path.isdir(run_dir):
                    trial_summary = dict(row)
                    trial_summary['params'] = _normalize_hpo_params(trial.params)
                    trial_summary['user_attrs'] = dict(trial.user_attrs)
                    trial_summary['system_attrs_keys'] = sorted(trial.system_attrs.keys())
                    with open(os.path.join(run_dir, 'trial_summary.json'), 'w') as f:
                        json.dump(trial_summary, f, indent=2)


    def hp_space(trial):
        lstm_pool_size = trial.suggest_categorical("lstm_pool_size", [2, 4, 8, 16])
        lstm_pool_stride = lstm_pool_size // trial.suggest_categorical(
            "lstm_pool_stride_divisor", [1, 2]
        )
        return {
            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.7),
            "conv_dropout_p": trial.suggest_float("conv_dropout_p", 0.0, 0.7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "lstm_pool_size": lstm_pool_size,
            "lstm_pool_stride": lstm_pool_stride,
            # "lr_scheduler_type": trial.suggest_categorical(
            #     "lr_scheduler_type", ["cosine", "linear", "constant"]
            # ),
            "warmup_steps": 0,
            "warmup_ratio": 0.0,
            "use_cyclic_lr": False,
        }

    # choose *one* metric to optimise for hyperparameter search
    def objective(metrics: Dict[str, float]) -> float:
        # [CODEX CHANGE 2026-02-27] Old behavior kept for reference:
        # return metrics["eval_auprc"]
        # New behavior: optimize peak eval_auprc observed during the trial.
        if "eval_auprc_peak" in metrics:
            return metrics["eval_auprc_peak"]
        return metrics["eval_auprc"]

    # [CODEX CHANGE 2026-02-27] Normalize output path so run-*/tb/trial_summary paths are stable.
    hpo_output_dir = os.path.abspath(training_args.output_dir)
    training_args.output_dir = hpo_output_dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    base_training_args = replace(training_args)

    # Does this log to slurm? (yes! but adding a file handler too)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    log_path = os.path.join(
        training_args.output_dir, f"optuna_trials_worker{model_args.hpo_worker_index}.log"
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    optuna.logging.get_logger("optuna").addHandler(fh)

    def _is_known_numerical_failure(exc: Exception) -> bool:
        if isinstance(exc, (NumericalTrialFailure, FloatingPointError)):
            return True
        if isinstance(exc, ValueError):
            message = str(exc).lower()
            if "input contains nan" in message or "input contains infinity" in message:
                return True
            if "not finite" in message or "non-finite" in message:
                return True
        return False

    def _build_trainer(run_args):
        trainer_holder: Dict[str, BalancedTrainer] = {}

        def compute_metrics(eval_pred):
            trainer = trainer_holder["trainer"]
            logits, labels = eval_pred
            labels = np.asarray(labels)

            logits_t = torch.as_tensor(logits, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)

            if not torch.isfinite(logits_t).all():
                trainer._raise_numerical_failure("non-finite logits during evaluation", logits=logits_t)

            probs_t = torch.softmax(logits_t, dim=-1)
            if not torch.isfinite(probs_t).all():
                trainer._raise_numerical_failure(
                    "non-finite probabilities during evaluation", logits=logits_t, probs=probs_t
                )

            per_ex_xent_t = F.cross_entropy(logits_t, labels_t, reduction="none")
            if not torch.isfinite(per_ex_xent_t).all():
                trainer._raise_numerical_failure(
                    "non-finite per-example cross entropy during evaluation",
                    logits=logits_t,
                    probs=probs_t,
                    loss=per_ex_xent_t,
                )

            probs = probs_t.cpu().numpy()[:, 1]
            preds = logits_t.argmax(dim=-1).cpu().numpy()
            per_ex_xent = per_ex_xent_t.cpu().numpy()

            pos_mask = labels == 1
            neg_mask = labels == 0

            xent_pos = per_ex_xent[pos_mask].mean() if pos_mask.any() else np.nan
            xent_neg = per_ex_xent[neg_mask].mean() if neg_mask.any() else np.nan
            xent_balanced = 0.5 * (xent_pos + xent_neg) if (pos_mask.any() and neg_mask.any()) else np.nan

            pos_frac = labels.mean()

            return {
                "accuracy": accuracy_score(labels, preds),
                "auroc": roc_auc_score(labels, probs),
                "auprc": average_precision_score(labels, probs),
                "mcc": matthews_corrcoef(labels, preds),
                "random_auprc": pos_frac,
                "cross_entropy_pos": xent_pos,
                "cross_entropy_neg": xent_neg,
                "cross_entropy_balanced": xent_balanced,
            }

        def model_init(trial=None):
            if trial is not None and hasattr(trial, "params"):
                logger.info("Optuna trial %s params=%s", getattr(trial, "number", "n/a"), trial.params)
            logger.info(
                "model_init effective args: hidden_dropout_prob=%.4f conv_dropout_p=%.4f "
                "lstm_pool_size=%s lstm_pool_stride=%s learning_rate=%s lr_scheduler_type=%s use_cyclic_lr=%s",
                run_args.hidden_dropout_prob,
                run_args.conv_dropout_p,
                run_args.lstm_pool_size,
                run_args.lstm_pool_stride,
                run_args.learning_rate,
                run_args.lr_scheduler_type,
                run_args.use_cyclic_lr,
            )
            _, model = _build_model_with_runtime_config(run_args)
            return model

        trainer = BalancedTrainer(
            model_init=model_init,
            args=run_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            train_sample_weights=train_sample_weights,
            test_dataset=test_dataset,
        )
        trainer_holder["trainer"] = trainer
        return trainer

    def _cleanup_trainer(trainer):
        if trainer is None:
            return
        try:
            from accelerate.utils.memory import release_memory

            trainer.model_wrapped, trainer.model = release_memory(trainer.model_wrapped, trainer.model)
        except Exception:
            pass
        try:
            trainer.accelerator.clear()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_or_load_study(study_name: str, storage: str) -> optuna.Study:
        return optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            pruner=HyperbandPruner(
                min_resource=max(1, steps_per_epoch),
                max_resource="auto",
                reduction_factor=2,
            ),
        )

    counted_trial_states = {TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    terminal_trial_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    claim_lock_path = os.path.join(training_args.output_dir, "optuna_claim.lock")

    def _claim_next_hpo_trial(study_name: str, storage: str):
        with open(claim_lock_path, "a+") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            study = _create_or_load_study(study_name, storage)
            trials = study.get_trials(deepcopy=False)
            counted_trials = sum(1 for trial in trials if trial.state in counted_trial_states)
            terminal_trials = sum(1 for trial in trials if trial.state in terminal_trial_states)
            if counted_trials >= model_args.hpo_total_trials:
                return None, None, terminal_trials, counted_trials

            trial = study.ask()
            trial.set_user_attr("worker_index", model_args.hpo_worker_index)
            trial.set_user_attr("worker_pid", os.getpid())
            logger.info(
                "HPO worker %s/%s claimed trial %s counted=%s/%s terminal=%s",
                model_args.hpo_worker_index,
                model_args.hpo_worker_count,
                trial.number,
                counted_trials + 1,
                model_args.hpo_total_trials,
                terminal_trials,
            )
            return study, trial, terminal_trials, counted_trials + 1

    trainer = None

    #  HYPER-PARAMETER SEARCH BRANCH
    if model_args.hyperparameter_search:
        study_name = model_args.hpo_study_name
        storage = model_args.hpo_storage
        if storage is None:
            storage = f"sqlite:///{os.path.abspath(os.path.join(training_args.output_dir, 'optuna.db'))}"

        logger.info(
            "HPO worker %s/%s using dynamic shared scheduling total_trials=%s study_name=%s storage=%s",
            model_args.hpo_worker_index,
            model_args.hpo_worker_count,
            model_args.hpo_total_trials,
            study_name,
            storage,
        )
        print(
            f"HPO worker {model_args.hpo_worker_index}/{model_args.hpo_worker_count} "
            f"using dynamic shared scheduling; total_trials={model_args.hpo_total_trials}; "
            f"study={study_name}; storage={storage}",
            flush=True,
        )

        claimed_trials = 0
        completed_trials = 0
        pruned_trials = 0
        failed_trials = 0

        while True:
            study, trial, terminal_trials, counted_trials = _claim_next_hpo_trial(study_name, storage)
            if trial is None:
                logger.info(
                    "HPO worker %s/%s found no remaining unclaimed trials. terminal=%s counted=%s target=%s",
                    model_args.hpo_worker_index,
                    model_args.hpo_worker_count,
                    terminal_trials,
                    counted_trials,
                    model_args.hpo_total_trials,
                )
                break

            claimed_trials += 1
            trainer = None
            try:
                trainer = _build_trainer(replace(base_training_args))
                trainer.hp_search_backend = HPSearchBackend.OPTUNA
                trainer.hp_space = hp_space
                trainer.compute_objective = objective
                trainer.objective = None

                trainer.train(resume_from_checkpoint=None, trial=trial)
                if getattr(trainer, "objective", None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)

                if trainer.objective is None:
                    raise NumericalTrialFailure(f"Trial {trial.number} did not produce an objective value.")

                objective_value = float(trainer.objective)
                if not math.isfinite(objective_value):
                    raise NumericalTrialFailure(
                        f"Trial {trial.number} produced non-finite objective {objective_value}."
                    )

                study.tell(trial, objective_value, state=TrialState.COMPLETE)
                completed_trials += 1
                logger.info(
                    "Worker %s/%s completed trial %s objective=%s params=%s",
                    model_args.hpo_worker_index,
                    model_args.hpo_worker_count,
                    trial.number,
                    objective_value,
                    _normalize_hpo_params(trial.params),
                )
            except optuna.TrialPruned as exc:
                trial.set_user_attr("trial_outcome", "PRUNED")
                trial.set_user_attr("pruned_reason", str(exc))
                study.tell(trial, state=TrialState.PRUNED)
                pruned_trials += 1
                logger.info(
                    "Worker %s/%s pruned trial %s params=%s reason=%s",
                    model_args.hpo_worker_index,
                    model_args.hpo_worker_count,
                    trial.number,
                    trial.params,
                    exc,
                )
            except Exception as exc:
                if trial is not None and _is_known_numerical_failure(exc):
                    trial.set_user_attr("trial_outcome", "FAIL")
                    trial.set_user_attr("failure_reason", str(exc)[:2000])
                    study.tell(trial, state=TrialState.FAIL)
                    failed_trials += 1
                    logger.warning(
                        "Worker %s/%s marked trial %s as FAIL due to numerical issue: %s",
                        model_args.hpo_worker_index,
                        model_args.hpo_worker_count,
                        trial.number,
                        exc,
                    )
                else:
                    raise
            finally:
                _cleanup_trainer(trainer)
                trainer = None

        # Save study
        study = optuna.load_study(study_name=study_name, storage=storage)
        joblib.dump(study, os.path.join(training_args.output_dir, "optuna_study.pkl"))
        # [CODEX CHANGE 2026-02-27] Write study-level CSV and per-trial JSON summaries.
        write_study_trial_metrics(study, training_args.output_dir)

        dashboard_builders = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "parallel_coordinate": plot_parallel_coordinate,
        }
        for name, build_figure in dashboard_builders.items():
            try:
                fig = build_figure(study)
                html_path = os.path.join(training_args.output_dir, f"{name}.html")
                fig.write_html(html_path)
                # optional PNG (needs `pip install kaleido` which is in my "deep" conda env)
                try:
                    png_path = os.path.join(training_args.output_dir, f"{name}.png")
                    pio.write_image(fig, png_path, format="png", width=1000, height=600)
                except Exception as e:
                    logger.warning(f"PNG export failed for {name}: {e}")
            except Exception as e:
                logger.warning(f"Dashboard export failed for {name}: {e}")

        state_counts = dict(Counter(trial.state.name for trial in study.trials))
        try:
            study_best_value = study.best_value
            study_best_trial = study.best_trial.number
            study_best_params = _normalize_hpo_params(study.best_params)
        except ValueError:
            study_best_value = None
            study_best_trial = None
            study_best_params = {}

        best_payload = {
            "worker_index": model_args.hpo_worker_index,
            "worker_count": model_args.hpo_worker_count,
            "claimed_trials": claimed_trials,
            "completed_trials": completed_trials,
            "pruned_trials": pruned_trials,
            "failed_trials": failed_trials,
            "study_name": study_name,
            "storage": storage,
            "best_run": {
                "run_id": str(study_best_trial) if study_best_trial is not None else None,
                "objective": study_best_value,
                "hyperparameters": study_best_params,
            },
            "study_best": {
                "trial_number": study_best_trial,
                "value": study_best_value,
                "params": study_best_params,
            },
        }
        with open(os.path.join(training_args.output_dir, "hpo_best_params.json"), "w") as f:
            json.dump(best_payload, f, indent=2)

        summary_payload = {
            "worker_index": model_args.hpo_worker_index,
            "worker_count": model_args.hpo_worker_count,
            "claimed_trials": claimed_trials,
            "completed_trials": completed_trials,
            "pruned_trials": pruned_trials,
            "failed_trials": failed_trials,
            "requested_total_trials": model_args.hpo_total_trials,
            "study_name": study_name,
            "storage": storage,
            "num_trials_recorded": len(study.trials),
            "state_counts": state_counts,
            "best_trial_number": study_best_trial,
            "best_value": study_best_value,
            "objective_mode": "peak_eval_auprc",
        }
        with open(os.path.join(training_args.output_dir, "hpo_study_summary.json"), "w") as f:
            json.dump(summary_payload, f, indent=2)

        if study_best_trial is not None:
            logger.info(
                "Study best trial=%s objective=%s params=%s",
                study_best_trial,
                study_best_value,
                study_best_params,
            )

        if not model_args.train_best_after_search:
            logger.info(
                "Hyperparameter search complete. train_best_after_search=False, exiting without final training."
            )
            return

        if not study_best_params:
            raise ValueError("No completed hyperparameter trial is available for final training.")

        training_args = replace(base_training_args, **study_best_params)
        logger.info(
            "Re-initializing trainer with tuned hyperparameters for final training. params=%s",
            study_best_params,
        )
        trainer = _build_trainer(training_args)
    else:
        training_args = replace(base_training_args)
        trainer = _build_trainer(training_args)

    # ----------------------------------------------------------------------------------
    #  NORMAL TRAINING / FINAL TRAIN WITH BEST PARAMS
    # ----------------------------------------------------------------------------------
    print("Starting training with the following parameters:", trainer.args, flush=True)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete and model saved.")
    best_ckpt = trainer.state.best_model_checkpoint
    logger.info(f"best checkpoint on disk: {best_ckpt}")      # e.g. .../checkpoint-4500


if __name__ == "__main__":
    main()
