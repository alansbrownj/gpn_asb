#!/usr/bin/env python
# coding: utf-8

## 2025-04-08 Tuesday W15.2 
## Writing this script for sequence classification
import optuna
from optuna.pruners import HyperbandPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from transformers import Adafactor

import plotly.io as pio
import joblib

import logging
import math
import numpy as np
import os
import sys
from dataclasses import dataclass, field
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
    hyperparameter_search: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run hyperparameter search or not. If set to True, the script will not train the model but will only run the search."
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
    pos_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/train_positives.parquet", tokenizer, max_length=data_args.max_seq_length
    )
    neg_dataset = ParquetClassificationDataset(
        f"{data_args.datadir}/train_negatives.parquet", tokenizer, max_length=data_args.max_seq_length
    )

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
    ## Going to try weighting the loss of the positives. 
    pos = len(pos_dataset)
    neg = len(neg_dataset)
    # w_pos = math.sqrt(neg / max(pos, 1))
    w_pos = neg / max(pos, 1)
    class_weight = torch.tensor([1.0, w_pos], dtype=torch.float)
    print(f"Class weights for loss: {class_weight.numpy()}", flush=True)

    class BalancedTrainer(transformers.Trainer):
        def __init__(self, *args, class_weight=None, test_dataset=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weight = class_weight  # tensor([w_neg, w_pos]) or None
            self.test_dataset = test_dataset
        
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            ## changing from below, because maybe I should not mutate the dict?
            # labels = inputs.pop("labels")
            # outputs = model(**inputs)
            logits = outputs.logits

            # w = None
            if self.class_weight is not None:
                w = self.class_weight.to(logits.device, dtype=torch.float32)
            else:
                w = None
            
            # loss = F.cross_entropy(logits.view(-1, model.config.num_labels), labels.view(-1), weight=w)
            loss = F.cross_entropy(logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

        # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        #     # build the same sampler you used before
        #     labels = torch.tensor(
        #         pd.concat([pos_dataset.df["label"], neg_dataset.df["label"]]).values
        #     )
        #     class_sample_count = torch.tensor(
        #         [(labels == t).sum() for t in torch.unique(labels, sorted=True)]
        #     )
        #     weights = 1.0 / class_sample_count.float()
        #     sample_weights = weights[labels]
        #     return torch.utils.data.WeightedRandomSampler(
        #         sample_weights, num_samples=len(sample_weights), replacement=True
        #     )

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

            # TRAIN auPRC LOGGING BLOCK (comment out to disable)
            # if self.train_dataset is not None:
            #     train_metrics = self.predict(
            #         self.train_dataset,
            #         ignore_keys=ignore_keys,
            #         metric_key_prefix="train",
            #     ).metrics
            #     train_auprc = train_metrics.get("train_auprc")
            #     if train_auprc is not None:
            #         self.log({"train_auprc": train_auprc})
            #         eval_metrics["train_auprc"] = train_auprc

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

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = np.asarray(labels)

        logits_t = torch.tensor(logits)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # probs for AUPRC
        probs = torch.softmax(logits_t, dim=-1).numpy()[:, 1]
        preds = logits_t.argmax(dim=-1).numpy()

        # per-example cross entropy
        per_ex_xent = F.cross_entropy(logits_t, labels_t, reduction="none").numpy()

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


    def hp_space(trial):
        # Tune CyclicLR
        base_lr = trial.suggest_float("base_lr", 1e-8, 5e-7, log=True)
        max_lr = trial.suggest_float("max_lr", base_lr * 5.0, base_lr * 200.0, log=True)
        cycle_steps_up = trial.suggest_int("cycle_steps_up", 500, 4000, step=250)
        cycle_steps_down = trial.suggest_int("cycle_steps_down", 500, 8000, step=250)
        cycle_gamma = trial.suggest_float("cycle_gamma", 0.9995, 0.99999)

        return {
            "use_cyclic_lr": True,
            "learning_rate": base_lr,
            "base_lr": base_lr,
            "max_lr": max_lr,
            "cycle_steps_up": cycle_steps_up,
            "cycle_steps_down": cycle_steps_down,
            "cycle_gamma": cycle_gamma,
        }

    # choose *one* metric to optimise for hyperparameter search
    def objective(metrics: Dict[str, float]) -> float:
        return metrics["eval_loss"]
        # return metrics["auprc"]        # change if you prefer auroc/accuracy

    # Does this log to slurm? (yes! but adding a file handler too)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    log_path = os.path.join(training_args.output_dir, "optuna_trials.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    optuna.logging.get_logger("optuna").addHandler(fh)

    hf_args = training_args # redefining it was a mistake.... caused some problems.
    # hf_args = TrainingArguments(
    #     output_dir           = training_args.output_dir,
    #     overwrite_output_dir = True,
    #     num_train_epochs     = training_args.num_train_epochs,
    #     per_device_train_batch_size = training_args.per_device_train_batch_size,
    #     per_device_eval_batch_size  = training_args.per_device_eval_batch_size,
    #     eval_strategy        = training_args.eval_strategy,
    #     eval_steps           = training_args.eval_steps,
    #     save_steps           = training_args.save_steps,
    #     save_total_limit     = training_args.save_total_limit,
    #     learning_rate        = training_args.learning_rate,
    #     warmup_steps         = training_args.warmup_steps,
    #     logging_dir          = os.path.join(training_args.output_dir, "logs"),
    #     logging_steps        = training_args.logging_steps,
    #     load_best_model_at_end = True,
    #     metric_for_best_model  = training_args.metric_for_best_model,
    #     report_to            = ["tensorboard"],
    #     fp16                 = training_args.fp16,
    #     optim                = training_args.optim,
    #     weight_decay         = training_args.weight_decay,
    # )

    def model_init():
        """
        Return a *new* instance of the model each time Optuna asks
        for a trial.  It must have the *same* initial weights for every
        trial when the seed is fixed, otherwise results aren’t comparable.
        """
        if model_args.model_name_or_path is not None and model_args.model_name_or_path.lower() != "none":
            cfg = AutoConfig.from_pretrained(model_args.model_name_or_path)
            cfg.num_labels = 2
            return AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                    config=cfg)
        else:
            cfg = CONFIG_MAPPING[model_args.model_type]()
            cfg.num_labels = 2
            return AutoModelForSequenceClassification.from_config(cfg)

    trainer = BalancedTrainer(
        model_init     = model_init,
        args           = hf_args,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        tokenizer      = tokenizer,
        compute_metrics= compute_metrics,
        class_weight   = class_weight,
        test_dataset   = test_dataset,
    )
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    # model.save_pretrained(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)

    ## Test the tuning stuff if the flag is set
        # ----------------------------------------------------------------------------------
    #  HYPER-PARAMETER SEARCH BRANCH
    # ----------------------------------------------------------------------------------
    if model_args.hyperparameter_search:
        # Prepare Optuna study (SQLite so multiple jobs can resume)
        study_name = f"optuna_{int(time.time())}"
        storage    = f"sqlite:///{os.path.join(training_args.output_dir, 'optuna.db')}"

        best_run = trainer.hyperparameter_search(
            n_trials          = 150,
            direction         = "minimize",
            hp_space          = hp_space,
            compute_objective = objective,
            backend           = "optuna",
            study_name       = study_name,
            storage           = storage,
            pruner            = HyperbandPruner(
                min_resource=steps_per_epoch * 3.5,
                max_resource="auto",
                reduction_factor=2,
            ),

        )

        # Save study
        study = optuna.load_study(study_name=study_name, storage=storage)
        joblib.dump(study, os.path.join(training_args.output_dir, "optuna_study.pkl"))

        dashboards = {
            "optimization_history": plot_optimization_history(study),
            "param_importances":    plot_param_importances(study),
            "parallel_coordinate":  plot_parallel_coordinate(study),
        }
        for name, fig in dashboards.items():
            html_path = os.path.join(training_args.output_dir, f"{name}.html")
            fig.write_html(html_path)
            # optional PNG (needs `pip install kaleido` which is in my "deep" conda env)
            try:
                png_path = os.path.join(training_args.output_dir, f"{name}.png")
                pio.write_image(fig, png_path, format="png", width=1000, height=600)
            except Exception as e:
                logger.warning(f"PNG export failed for {name}: {e}")

        logger.info(
            f"Best trial run_id={best_run.run_id} "
            f"objective={best_run.objective:.4f} "
            f"params={best_run.hyperparameters}"
            )

        from dataclasses import replace, asdict

        # build a new TrainingArguments object that merges in the tuned params
        # new_args_dict = asdict(hf_args)
        # new_args_dict.update(best_run.hyperparameters)
        # hf_args = TrainingArguments(**new_args_dict) ## fails because some args are not in datasets
        hf_args = replace(hf_args, **best_run.hyperparameters)
        # re-initialize the trainer with the new args
        logger.info("Re-initializing trainer with tuned hyperparameters.")
        trainer = BalancedTrainer(
                model_init     = model_init,
                args           = hf_args,
                train_dataset  = train_dataset,
                eval_dataset   = val_dataset,
                tokenizer      = tokenizer,
                compute_metrics= compute_metrics,
                class_weight   = class_weight,
                test_dataset   = test_dataset,
            )
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
