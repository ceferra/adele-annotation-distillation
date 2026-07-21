#!/bin/bash
#SBATCH --job-name=delean-distill-lora
#SBATCH --partition=batch
#SBATCH --nodelist=vrhpc6.dsic.upv.es
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/data/$USER/delean_run/slurm-distill-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate distill-ft

cd /data/$USER/delean_run

accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
    train_distill_lora.py \
    --train-file /data/$USER/delean_run/distillation_dataset/train.jsonl \
    --val-file /data/$USER/delean_run/distillation_dataset/val.jsonl \
    --output-dir /data/$USER/delean_run/distill_lora_out \
    --per-device-batch-size 1 \
    --grad-accum 16

echo "=== TRAINING DONE ==="
