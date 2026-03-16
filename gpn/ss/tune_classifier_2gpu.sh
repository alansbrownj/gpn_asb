#!/bin/bash
#SBATCH --time=8-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=94GB
#SBATCH --job-name=TuneClassifier2GPU
#SBATCH --output=slurm-%x-%j.out

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
# 3) output_dir
# 4) data_dir
# 5) total_trials (optional, default: 100)
# 6) max_steps (optional, default: 8000)
# 7) seq_len (optional, default: 512)
# 8) study_name (optional, default: convnet_classifier_hpo)
pretrained_model=$1
datasetname=$2
out_dir=$3
data_dir=$4
total_trials=${5:-100}
max_steps=${6:-8000}
seq_len=${7:-512}
study_name=${8:-convnet_classifier_tuning}

mkdir -p "$out_dir/logs"
out_dir=$(readlink -f "$out_dir")
storage_url="sqlite:///${out_dir}/optuna.db"

echo "pretrained model: ${pretrained_model}"
echo "dataset name: ${datasetname}"
echo "output directory: ${out_dir}"
echo "data directory: ${data_dir}"
echo "study name: ${study_name}"
echo "storage: ${storage_url}"
echo "total trials: ${total_trials}"
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

echo "Launching worker 0 on GPU 0"
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=Classifier_tune python -m gpn.ss.run_classifier \
    --datadir "$data_dir" --fp16 \
    --model_name_or_path "$pretrained_model" \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir "$out_dir" \
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
    --train_best_after_search False \
    --hpo_total_trials "$total_trials" \
    --hpo_worker_count 2 \
    --hpo_worker_index 0 \
    --hpo_study_name "$study_name" \
    --hpo_storage "$storage_url" \
    > "$out_dir/worker0.log" 2>&1 &
pid0=$!

sleep 2

echo "Launching worker 1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=Classifier_tune python -m gpn.ss.run_classifier \
    --datadir "$data_dir" --fp16 \
    --model_name_or_path "$pretrained_model" \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --output_dir "$out_dir" \
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
    --train_best_after_search False \
    --hpo_total_trials "$total_trials" \
    --hpo_worker_count 2 \
    --hpo_worker_index 1 \
    --hpo_study_name "$study_name" \
    --hpo_storage "$storage_url" \
    > "$out_dir/worker1.log" 2>&1 &
pid1=$!

wait "$pid0"
status0=$?
wait "$pid1"
status1=$?

echo "worker0 exit code: $status0"
echo "worker1 exit code: $status1"

if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
    echo "At least one HPO worker failed."
    exit 1
fi

echo "Both HPO workers completed successfully."
date +"%F W%V.%u %T"
