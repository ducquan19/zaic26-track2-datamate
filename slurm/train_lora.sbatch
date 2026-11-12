#!/bin/bash
# One QLoRA round on a Slurm cluster (training/train_lora.py), resumable across time limits. See slurm/README.md.
#   sbatch slurm/train_lora.sbatch                                   # LoRA v4: continue v3 on data/sft/sql_rft_v3_notes.jsonl
#   sbatch --export=ALL,DATA=...,OUT=...,INIT=...,MAX_LEN=...,LR=... slurm/train_lora.sbatch
#   sbatch --export=ALL,EXTRA="--limit-rows 50 --max-steps 5" --time=00:30:00 slurm/train_lora.sbatch   # quick setup check
# Set for the team's cluster (partition batch, account compute-account, A100-40GB nodes gpu01-03). Another cluster: edit the
# #SBATCH lines below or override them on the sbatch command line. gpu04 holds V100s, which have no bf16: excluded.
# Memory: the cluster allows up to 32 GB per job; LoRA v3 was trained on a machine with 30 GB of RAM.
#SBATCH --job-name=lora
#SBATCH --partition=batch
#SBATCH --account=compute-account
#SBATCH --exclude=gpu04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# no CUDA module needed: the torch wheel brings its CUDA runtime (cu126 on a 12.x driver, see slurm/README.md)

DATA=${DATA:-data/sft/sql_rft_v3_notes.jsonl}
OUT=${OUT:-runs/lora/sql_rft_v4}
INIT=${INIT-adapters/sql_rft_v3}          # empty (INIT=) = start from the base model, as LoRA v1 did
MAX_LEN=${MAX_LEN:-32768}
LR=${LR:-5e-5}
MODEL=${MODEL:-models/Qwen3.5-9B}
PY=${PYTHON:-.venv-train/bin/python}
EXTRA=${EXTRA:-}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID:-local}"
mkdir -p logs "$OUT"

# Near the time limit: put the job back in the queue. The next start continues from the newest checkpoint (--resume).
trap 'echo "[slurm] time limit near: requeue ${SLURM_JOB_ID:-}"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

echo "[slurm] job ${SLURM_JOB_ID:-local} on $(hostname): data=$DATA out=$OUT init=${INIT:-base} max_len=$MAX_LEN lr=$LR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The fla kernels (Gated DeltaNet layers) must match the PyTorch reference before they are trusted for training
"$PY" training/check_fla.py --T 4096

# shellcheck disable=SC2086
"$PY" training/train_lora.py "$DATA" --out "$OUT" --model "$MODEL" ${INIT:+--init-adapter "$INIT"} --max-len "$MAX_LEN" \
    --r 16 --alpha 32 --lr "$LR" --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking --completion-logits \
    --save-steps 50 --resume $EXTRA &
wait $!   # in the background, so the USR1 trap runs at once instead of after the training step
