from gpn.data import (
    Genome, load_table, get_balanced_intervals, filter_length,
    filter_annotation_features, get_only_feature_and_promoter_intervals,
    get_only_promoter_intervals,
)

whitelist_gff = config["whitelist_annotation_file"]
# another rule can make the blacklist 
rule minimap2_and_merge_blacklist:
    input:
        test="results/for_making_blacklist/chr14_NC_037283.1.fa.gz",
        validation="results/for_making_blacklist/chr13_NC_004331.3.fa.gz",
        reference="results/genome/{assembly}.fa.gz"
    output:
        merged_bed="results/blacklist_beds/{assembly}_blacklist.bed",
        temp_sam=temp("results/blacklist_beds/{assembly}_tmp.sam"),
        temp_bam=temp("results/blacklist_beds/{assembly}_tmp.sorted.bam")
    params:
        minimap2="/home/asb5975/group/lab/alan/deep_learning_sequence_prediction/minimap_stuff/minimap2/minimap2",
        threshold=1,
        output_dir="results/blacklist_beds"
    run:
        import os

        # Create temporary blacklist file path
        bed = os.path.join(params.output_dir, f"{wildcards.assembly}_tmp_blacklist.bed")
        os.makedirs(params.output_dir, exist_ok=True)

        # Remove existing blacklist file if it exists
        if os.path.exists(bed):
            os.remove(bed)

        # Iterate over test and validation chromosomes
        for chrom in [input.test, input.validation]:
            sam = output.temp_sam
            bam = output.temp_bam

            # Run minimap2, samtools, and bedtools
            shell(f"""
                {params.minimap2} -ax asm5 {chrom} {input.reference} > {sam}
                samtools sort -o {bam} {sam}
                bedtools bamtobed -i {bam} >> {bed}
            """)

        # Sort and merge the blacklist file
        shell(f"""
            sort -k1,1 -k2,2n {bed} > {bed}.sorted
            mv {bed}.sorted {bed}
            bedtools merge -i {bed} | awk '($3 - $2) > {params.threshold}' > {output.merged_bed}
            rm {bed}
        """)

rule make_non_blacklist_intervals:
    input:
        genome="results/genome/{assembly}.fa.gz",
        merged_bed="results/blacklist_beds/{assembly}_blacklist.bed"
    output:
        "results/intervals/{assembly}/non_blacklist.parquet",
        "results/intervals/{assembly}/non_blacklist.csv"
    threads: 2
    run:
        # Initialize the Genome object with the blacklist
        genome = Genome(input["genome"], blacklist_path=input["merged_bed"])
        intervals = genome.get_non_blacklist_intervals()
        intervals = filter_length(intervals, config["window_size"])
        intervals.to_parquet(output[0], index=False)
        intervals.to_csv(output[1], index=False)


rule make_all_intervals:
    input:
        "results/genome/{assembly}.fa.gz",
    output:
        "results/intervals/{assembly}/all.parquet",
    threads: 2
    run:
        I = Genome(input[0]).get_all_intervals()
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)


rule make_defined_intervals:
    input:
        "results/genome/{assembly}.fa.gz",
    output:
        "results/intervals/{assembly}/defined.parquet",
    threads: 2
    run:
        I = Genome(input[0]).get_defined_intervals()
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)


rule make_annotation_intervals:
    input:
        "results/intervals/{assembly}/defined.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/annotation_{feature}.parquet",
    run:
        I = pd.read_parquet(input[0])
        annotation = load_table(input[1])
        include_flank = config.get(
            "annotation_features_include_flank", config["window_size"] // 2
        )
        add_jiter = config.get("annotation_features_add_jitter", 100)
        I = filter_annotation_features(
            I, annotation, wildcards.feature,
            include_flank=include_flank, jitter=add_jitter,
        )
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)

rule make_balanced_v1_intervals:
    input:
        "results/intervals/{assembly}/non_blacklist.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/balanced_v1.parquet",
    run:
        defined_intervals = load_table(input[0])
        annotation = load_table(input[1])
        intervals = get_balanced_intervals(
            defined_intervals, annotation, config["window_size"],
            config.get("promoter_upstream", 1000),
        )
        intervals.to_parquet(output[0], index=False)

rule make_only_feature_and_promoter_intervals:
    input:
        "results/intervals/{assembly}/non_blacklist.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/feature_and_promoter_only.parquet",
    run:
        defined_intervals = load_table(input[0])
        annotation = load_table(input[1])
        intervals = get_only_feature_and_promoter_intervals(
            defined_intervals, annotation, config["window_size"],
            config.get("promoter_upstream", 1000),
            config.get("promoter_downstream", 1000),
        )
        intervals.to_parquet(output[0], index=False)

rule make_only_promoter_intervals:
    input:
        "results/intervals/{assembly}/non_blacklist.parquet",
        "results/annotation/{assembly}.gff.gz",
    output:
        "results/intervals/{assembly}/only_promoter_intervals.parquet",
    run:
        defined_intervals = load_table(input[0])
        annotation = load_table(input[1])
        intervals = get_only_promoter_intervals(
            defined_intervals, annotation, config["window_size"],
            config.get("promoter_upstream", 1000),
            config.get("promoter_downstream", 1000),
        )
        intervals.to_parquet(output[0], index=False)

rule make_whitelist_base_intervals:
    # the assembly should already have the .fa.gz part on it (in config file)
    input:
        "results/for_making_blacklist/{assembly}"
    output:
        "results/intervals/{assembly}/non_blacklist_white.parquet"
    threads: 2
    run:
        I = Genome(input[0]).get_all_intervals()
        I = filter_length(I, config["window_size"])
        I.to_parquet(output[0], index=False)

rule make_whitelist_intervals_feature_and_promoter:
    input:
        whitelist_all="results/intervals/{assembly}/non_blacklist_white.parquet",
        annotation=lambda wildcards: f"results/annotation/{config['whitelist_annotation_file']}",
    output:
        "results/intervals/{assembly}/feature_and_promoter_only_white.parquet"
    run:
        defined_intervals = pd.read_parquet(input.whitelist_all)
        annotation = load_table(input.annotation)
        intervals = get_only_feature_and_promoter_intervals(
            defined_intervals,
            annotation,
            config["window_size"],
            config.get("promoter_upstream", 1000),
        )
        intervals.to_parquet(output[0], index=False)