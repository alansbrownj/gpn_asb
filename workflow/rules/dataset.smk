from gpn.data import Genome, make_windows, get_seq
import math
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

## removing this part... 
# split_proportions = [config["split_proportion"][split] for split in splits]
# assert np.isclose(sum(split_proportions), 1)
target_intervals = config["target_intervals"]

## make training assemblies... (these should use the non_blacklist intervals)
rule make_dataset_assembly_train:
    input:
        lambda wildcards: f"results/intervals/{wildcards['assembly']}/{config['target_intervals']}.parquet",
        "results/genome/{assembly}.fa.gz",
        ## using this to make all training sets. 
    output:
        temp("results/dataset_assembly/{assembly}/train.parquet"),
        # temp(expand("results/dataset_assembly/{{assembly}}/{split}.parquet", split=splits)),
    threads: 2
    run:
        intervals = pd.read_parquet(input[0])
        genome = Genome(input[1])
        intervals = make_windows(
            intervals, config["window_size"], config["step_size"], config["add_rc"],
        )
        intervals = intervals.sample(frac=1.0, random_state=42)
        intervals["assembly"] = wildcards["assembly"]
        intervals = intervals[["assembly", "chrom", "start", "end", "strand"]]
        intervals = get_seq(intervals, genome)

        # chroms = intervals.chrom.unique()

        # Create a default split assignment
        # chrom_split = pd.Series("train", index=chroms)

        # Assign whitelist chromosomes explicitly
        # chrom_split.loc[np.isin(chroms, config["whitelist_validation_chroms"])] = "validation"
        # chrom_split.loc[np.isin(chroms, config["whitelist_test_chroms"])] = "test"

        # Ensure non-whitelisted chromosomes remain "train"
        # non_whitelist_chroms = ~chrom_split.index.isin(
            # config["whitelist_validation_chroms"] + config["whitelist_test_chroms"]
        # )
        # chrom_split.loc[non_whitelist_chroms] = "train"

        # Map split assignment back to intervals
        # intervals["split"] = chrom_split[intervals.chrom].values
        intervals.to_parquet(output[0], index=False) ## added to just make one file
        # for path, split in zip(output, splits):
        #     intervals[intervals["split"] == split].to_parquet(path, index=False)

rule make_dataset_whitelist_test:
    input:
        fasta=lambda wildcards: f"results/for_making_blacklist/{wildcards.assembly}",
        intervals=lambda wildcards: f"results/intervals/{wildcards.assembly}/{config['target_intervals']}_white.parquet",
    output:
        "results/dataset_assembly/{assembly}/test.parquet"
    threads: 2
    run:
        genome = Genome(input.fasta)
        intervals = pd.read_parquet(input.intervals)
        intervals = make_windows(
            intervals, config["window_size"], config["step_size"], config["add_rc"]
        )
        intervals = intervals.sample(frac=1.0, random_state=42)
        intervals["assembly"] = wildcards.assembly
        intervals = intervals[["assembly", "chrom", "start", "end", "strand"]]
        intervals = get_seq(intervals, genome)
        intervals.to_parquet(output[0], index=False)

rule make_dataset_whitelist_validation:
    input:
        fasta=lambda wildcards: f"results/for_making_blacklist/{wildcards.assembly}",
        intervals=lambda wildcards: f"results/intervals/{wildcards.assembly}/{config['target_intervals']}_white.parquet",
    output:
        "results/dataset_assembly/{assembly}/validation.parquet"
    threads: 2
    run:
        genome = Genome(input.fasta)
        intervals = pd.read_parquet(input.intervals)
        intervals = make_windows(
            intervals, config["window_size"], config["step_size"], config["add_rc"]
        )
        intervals = intervals.sample(frac=1.0, random_state=42)
        intervals["assembly"] = wildcards.assembly
        intervals = intervals[["assembly", "chrom", "start", "end", "strand"]]
        intervals = get_seq(intervals, genome)
        intervals.to_parquet(output[0], index=False)


rule merge_datasets:
    input:
        lambda wildcards: (
            expand("results/dataset_assembly/{assembly}/train.parquet", assembly=training_assemblies)
            if wildcards.split == "train" else
            expand("results/dataset_assembly/{assembly}/test.parquet", assembly=config["whitelist_test_hold_out"])
            if wildcards.split == "test" else
            expand("results/dataset_assembly/{assembly}/validation.parquet", assembly=config["whitelist_validation_hold_out"])
            if wildcards.split == "validation" else
            []
        )
    output:
        directory("results/dataset/{target_intervals}/data/{split}")
    threads: workflow.cores
    run:
        import os
        import math
        from tqdm import tqdm
        # Read and concatenate only the files for this split
        intervals = pd.concat(
            tqdm((pd.read_parquet(path) for path in input), total=len(input)),
            ignore_index=True,
        ).sample(frac=1, random_state=42)
        if "split" in intervals.columns:
            intervals = intervals.drop(columns="split")
            
        ## a subsampling idea. Not implemented. 
        # if wildcards.split == "train":
        #     n_target = (intervals.assembly == config["target_assembly"]).sum()
        #     intervals = intervals.groupby("assembly").sample(
        #         n=n_target, random_state=42
        #     ).sample(frac=1, random_state=42)

        n_shards = math.ceil(len(intervals) / config["samples_per_file"])
        assert n_shards < 10000
        os.makedirs(output[0], exist_ok=True)
        for i in tqdm(range(n_shards)):
            path = os.path.join(output[0], f"shard_{i:05}.jsonl.zst")
            intervals.iloc[i::n_shards].to_json(
                path, orient="records", lines=True,
                compression={'method': 'zstd', 'threads': -1}
            )
