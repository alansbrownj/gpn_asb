import argparse
from Bio import SeqIO, bgzf
from Bio.Seq import Seq
from datasets import load_dataset
import gzip
import numpy as np
import os
import pandas as pd
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments

import gpn.model
from gpn.utils import Genome, load_dataset_from_file_or_dir, token_input_id


def center(X):
    return X - torch.unsqueeze(torch.mean(X, dim=1), 1)


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_path, id_a, id_c, id_g, id_t):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.id_a = id_a
        self.id_c = id_c
        self.id_g = id_g
        self.id_t = id_t

    def get_logits(self, input_ids, pos):
        logits = self.model.forward(input_ids=input_ids).logits
        logits = logits[torch.arange(len(pos)), pos]
        return logits

    def forward(
        self,
        input_ids_fwd=None,
        pos_fwd=None,
        input_ids_rev=None,
        pos_rev=None,
    ):
        id_a = self.id_a
        id_c = self.id_c
        id_g = self.id_g
        id_t = self.id_t
        logits_fwd = center(
            self.get_logits(input_ids_fwd, pos_fwd)[:, [id_a, id_c, id_g, id_t]]
        )
        logits_rev = center(
            self.get_logits(input_ids_rev, pos_rev)[:, [id_t, id_g, id_c, id_a]]
        )
        # TODO: not sure about averaging logits... makes some sense to average
        # probabilities, but logits, not sure...
        return (logits_fwd+logits_rev)/2


def get_logits(
    positions, genome, window_size, tokenizer, model,
    n_prefix=0, per_device_batch_size=8,
):
    n_positions = len(list(
        positions.map(lambda examples: {"chrom": examples["chrom"]}, batched=True)
    ))
    original_cols = list(positions.take(1))[0].keys()

    def tokenize(seqs):
        return tokenizer(
            seqs,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]

    def get_tokenized_seq(vs):
        # we convert from 1-based coordinate (standard in VCF) to 
        # 0-based, to use with Genome
        chrom = np.array(vs["chrom"])
        n = len(chrom)
        pos = np.array(vs["pos"]) - 1
        start = pos - window_size//2
        end = pos + window_size//2
        seq_fwd, seq_rev = zip(*(
            genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n)
        ))
        seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd])
        seq_rev = np.array([list(seq.upper()) for seq in seq_rev])
        assert seq_fwd.shape[1] == window_size
        assert seq_rev.shape[1] == window_size
        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

        def prepare_output(seq, pos):
            seq[:, pos] = tokenizer.mask_token
            return (
                tokenize(["".join(x) for x in seq]),
                [pos + n_prefix for _ in range(n)],
            )

        vs["input_ids_fwd"], vs["pos_fwd"] = prepare_output(seq_fwd, pos_fwd)
        vs["input_ids_rev"], vs["pos_rev"] = prepare_output(seq_rev, pos_rev)
        return vs

    positions = positions.map(get_tokenized_seq, remove_columns=original_cols, batched=True)
    # Ugly hack to be able to display a progress bar
    # Warning: this will override len() for all instances of datasets.IterableDataset
    # Didn't find a way to just override for this instance
    positions.__class__.__len__ = lambda self: n_positions
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=positions).predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get logits with AutoModelForMaskedLM"
    )
    parser.add_argument(
        "positions_path", type=str,
        help="Positions path. Needs the following columns: chrom,pos",
    )
    parser.add_argument(
        "genome_path", type=str, help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")
    parser.add_argument(
        "model_path", help="Model path (local or on HF hub)", type=str
    )
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per-device-batch-size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer-path", type=str,
        help="Tokenizer path (optional, else will use model_path)",
    )
    parser.add_argument(
        "--n-prefix", type=int, default=0, help="Number of prefix tokens (e.g. CLS)."
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split",
    )
    parser.add_argument(
        "--is-file", action="store_true", help="positions_PATH is a file, not directory",
    )
    parser.add_argument(
        "--format", type=str, default="parquet",
        help="If is-file, specify format (parquet, csv, json)",
    )
    args = parser.parse_args()

    positions = load_dataset_from_file_or_dir(
        args.positions_path, streaming=True, split=args.split, is_file=args.is_file,
        format=args.format,
    )
    genome = Genome(args.genome_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = MLMforLogitsModel(
        args.model_path,
        *[token_input_id(nuc, tokenizer, args.n_prefix) for nuc in "ACGT"]
    )
    pred = get_logits(
        positions, genome, args.window_size, tokenizer, model,
        per_device_batch_size=args.per_device_batch_size, n_prefix=args.n_prefix,
    )
    directory = os.path.dirname(args.output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # TODO: could add chrom,pos,ref,alt besides the score, as in CADD output
    # or make it an option
    # could also compress the output with bgzip, as tsv, so it can be indexed with tabix
    pd.DataFrame(pred, columns=list("ACGT")).to_parquet(args.output_path, index=False)
