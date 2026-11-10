#!/bin/bash
# Retrain LoRA v1 -> v2 -> v3 as three dependent Slurm jobs, with the settings of docker/train.sh (each round starts from the
# adapter of the previous one, only after it finished well). Needs the packaged SFT files in data/sft/ (scripts/pack_training_code.py).
#   bash slurm/train_chain.sh [extra sbatch options, e.g. --partition=a100 --gres=gpu:a100:1]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
R=runs/lora
j1=$(sbatch --parsable "$@" --job-name=lora-v1 --export=ALL,DATA=data/sft/sql_rft_v1.jsonl,OUT=$R/sql_rft_v1,INIT=,MAX_LEN=16384,LR=1e-4 slurm/train_lora.sbatch)
j2=$(sbatch --parsable "$@" --job-name=lora-v2 --dependency=afterok:"$j1" \
    --export=ALL,DATA=data/sft/sql_rft_v2.jsonl,OUT=$R/sql_rft_v2,INIT=$R/sql_rft_v1/adapter,MAX_LEN=20000,LR=5e-5 slurm/train_lora.sbatch)
j3=$(sbatch --parsable "$@" --job-name=lora-v3 --dependency=afterok:"$j2" \
    --export=ALL,DATA=data/sft/sql_rft_v3_notes.jsonl,OUT=$R/sql_rft_v3,INIT=$R/sql_rft_v2/adapter,MAX_LEN=32768,LR=5e-5 slurm/train_lora.sbatch)
echo "submitted v1=$j1 v2=$j2 (after v1) v3=$j3 (after v2); squeue -u ${USER:-$(whoami)} to follow, logs in logs/"
