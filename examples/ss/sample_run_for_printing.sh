#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=training
#SBATCH --output=slurm-%x-%j.out
umask 007

# srun -n1 -c24 --gres=gpu:1 --mem=128G bash sample_run_for_printing.sh
# srun -n1 -c4 --gres=gpu:1 --mem=64G bash sample_run_for_printing.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep

export OMP_NUM_THREADS=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out_dir="test_run_for_printing"

data_dir="/home/asb5975/group/lab/alan/deep_learning_sequence_prediction/results/dataset/feature_and_promoter_only"

echo "starting model training"
echo "output directory: ${out_dir}"

# free -m
# nvidia-smi

date
# --report_to wandb
# --master_port=29501 
# changed --remove_unused_columns from True to False
# soft weights were 0.1, but I need to check what they are doing
## early runs used --model_type ConvNet but I think it should be GPN
# --rdzv_endpoint=localhost:29501

## need to update 
WANDB_PROJECT=asb_test torchrun  --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') \
    -m gpn.ss.run_mlm --do_train --do_eval \
    --fp16 --report_to tensorboard --logging_dir "${out_dir}/logs" --prediction_loss_only True --remove_unused_columns False \
    --dataset_name $data_dir --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 1 --soft_masked_loss_weight_evaluation 0.1 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 1 --seed 42 \
    --save_strategy steps --save_steps 1 --evaluation_strategy steps \
    --eval_steps 1 --logging_steps 1 --max_steps 5 --warmup_steps 1 \
    --learning_rate 1e-3 --lr_scheduler_type cosine \
    --run_name testing --output_dir $out_dir --model_type ConvNet \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
    --torch_compile --overwrite_output_dir  --load_best_model_at_end True --metric_for_best_model eval_loss \
    --greater_is_better False  --save_total_limit 20

echo "finished"
date
