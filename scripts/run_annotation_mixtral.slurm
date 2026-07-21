#!/bin/bash
#SBATCH --job-name=delean-algebra-mixtral
#SBATCH --partition=batch
#SBATCH --nodelist=vrhpc5.dsic.upv.es
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/$USER/delean_run/slurm-mixtral-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rcb-vllm06

export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

BASE="/data/$USER/delean_run/batch_runs/algebra_mixtral"
cd "$BASE"

for dir in */; do
    acronym="${dir%/}"
    if [ -f "${acronym}/output.jsonl" ]; then
        echo "=== Skipping ${acronym}, output already exists ==="
        continue
    fi
    echo "=== Processing rubric: ${acronym} ($(date)) ==="
    python -m vllm.entrypoints.openai.run_batch \
        -i "${acronym}/input.jsonl" \
        -o "${acronym}/output.jsonl" \
        --model MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ \
        --quantization awq_marlin \
        --tensor-parallel-size 4 \
        --max-model-len 4096 \
        --enforce-eager \
        --gpu-memory-utilization 0.95
    echo "=== Finished rubric: ${acronym} ($(date)) ==="
done

echo "=== ALL RUBRICS DONE ==="
