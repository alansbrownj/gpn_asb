#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=94GB
#SBATCH --job-name=TrainBestFromStudy
#SBATCH --output=slurm-%x-%j.out

# [CODEX CHANGE 2026-02-27]
# Single-GPU launcher that reuses an existing Optuna study and performs exactly
# one final training run using the study-best hyperparameters.

set -o pipefail
umask 007

date +"%F W%V.%u %T"
source ~/anaconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER-}"
conda activate deep
set -euo pipefail

# Positional args:
# 1) pretrained_model
# 2) dataset_name
# 3) study_output_dir (where optuna.db is stored)
# 4) data_dir
# 5) max_steps (optional, default: 8000)
# 6) seq_len (optional, default: 512)
# 7) study_name (optional, default: convnet_classifier_tuning_superLongRun)
# 8) final_output_dir (optional)
pretrained_model=$1
datasetname=$2
study_output_dir=$3
data_dir=$4
max_steps=${5:-8000}
seq_len=${6:-512}
study_name=${7:-convnet_classifier_tuning_superLongRun}
final_output_dir=${8:-"${study_output_dir}/final_best_from_study_$(date +%Y%m%d_%H%M%S)"}

study_output_dir=$(readlink -f "$study_output_dir")
final_output_dir=$(readlink -m "$final_output_dir")
storage_url="sqlite:///${study_output_dir}/optuna.db"

mkdir -p "$final_output_dir"
mkdir -p "$final_output_dir/logs"

echo "pretrained model: ${pretrained_model}"
echo "dataset name: ${datasetname}"
echo "study output dir: ${study_output_dir}"
echo "final output dir: ${final_output_dir}"
echo "data directory: ${data_dir}"
echo "study name: ${study_name}"
echo "storage: ${storage_url}"
echo "max steps: ${max_steps}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/asb5975/group/lab/alan/deep_learning_sequence_prediction
export PYTHONPATH="/home/asb5975/group/lab/alan/deep_learning_sequence_prediction/gpn_asb:${PYTHONPATH:-}"

echo "Bootstrapping Optuna study (create/load once)"
python - <<PY
import optuna
optuna.create_study(
    direction="maximize",
    study_name="${study_name}",
    storage="${storage_url}",
    load_if_exists=True,
)
print("Optuna study ready:", "${study_name}", flush=True)
PY

echo "Launching single worker to train best hyperparameters from existing study"
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=Classifier_tune python -m gpn.ss.run_classifier \
    --datadir "$data_dir" --fp16 \
    --model_name_or_path "$pretrained_model" \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir "$final_output_dir" \
    --model_type ConvNet \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --dataset_name "$datasetname" \
    --max_seq_length "$seq_len" \
    --report_to tensorboard \
    --save_strategy steps --save_steps 200 --eval_strategy steps \
    --save_total_limit 40 \
    --eval_steps 200 --logging_steps 200 \
    --max_steps "$max_steps" \
    --overwrite_output_dir \
    --load_best_model_at_end True \
    --metric_for_best_model auprc --greater_is_better True \
    --optim adamw_torch --weight_decay 0.01 \
    --hyperparameter_search True \
    --train_best_after_search True \
    --hpo_total_trials 0 \
    --hpo_worker_count 1 \
    --hpo_worker_index 0 \
    --hpo_study_name "$study_name" \
    --hpo_storage "$storage_url" \
    > "$final_output_dir/final_train.log" 2>&1

echo "Final training finished."
date +"%F W%V.%u %T"
