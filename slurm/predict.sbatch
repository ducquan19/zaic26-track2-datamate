#!/bin/bash
# The inference pipeline on the cluster with the flags of the submitted setup (scripts/flags.sh), one shard of the shops per GPU
# (job array task i of n = --shard i/n). See slurm/README.md, inference. Submit from the repository root:
#
#   sbatch --array=0 --export=ALL,EXTRA="--limit 20",OUT=runs/smoke slurm/predict.sbatch          # smoke test: 20 public questions
#   sbatch slurm/predict.sbatch                                                                     # public test, 4 shards, saved v9 layers
#   sbatch --export=ALL,FRESH=1,OUT=runs/public_fresh slurm/predict.sbatch                          # drafts every layer, as the private run
#   sbatch --export=ALL,SPLIT=train,IDS=splits/holdout600.txt,OUT=runs/holdout slurm/predict.sbatch # holdout of train (scored after merge)
#
# Then, when every task has finished (login node, CPU only):
#   $PYTHON -m datamate.submission merge $OUT --questions data/$SPLIT/questions.jsonl
#
# Environment (export before sbatch, or in --export): PYTHON (the vLLM environment, default .venv/bin/python), MODEL_DIR
# (bf16 base, default models/Qwen3.5-9B), LORA_DIR (default adapters/sql_rft_v3), SPLIT, IDS, OUT, FRESH, EXTRA (more
# predict flags, e.g. "--stock-views --value-lookup"), ENGINE / QUANT / GPU_MEM (scripts/flags.sh).
# vLLM 0.32.0+cu129 runs on the cluster's 555 driver (CUDA 12.5) through CUDA 12 minor-version compatibility.
#SBATCH --job-name=datamate
#SBATCH --partition=batch
#SBATCH --account=compute-account
#SBATCH --exclude=gpu04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source scripts/flags.sh

PY=${PYTHON:-.venv/bin/python}
SPLIT=${SPLIT:-public_test}
OUT=${OUT:-runs/public}
I=${SLURM_ARRAY_TASK_ID:-0}
N=${SLURM_ARRAY_TASK_COUNT:-1}
LAYERS=()
if [ "$SPLIT" = public_test ] && [ "${FRESH:-0}" != 1 ]; then
  LAYERS=(--views-from configs/public_v9)   # the views and debt layers of the 0.683 run, as scripts/run_public.sh
fi
IDS_ARG=()
[ -n "${IDS:-}" ] && IDS_ARG=(--ids "$IDS")
mkdir -p "$OUT" logs

echo "== shard $I/$N of $SPLIT -> $OUT  job ${SLURM_JOB_ID:-local}  host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), '| vllm', vllm.__version__)"

# --deadline: about 45 min below the 12 h limit, the run skips thinking samples for a shop once it is behind pace
# shellcheck disable=SC2086
"$PY" -m datamate.predict --data "data/$SPLIT" --output "$OUT/shard$I" --shard "$I/$N" "${IDS_ARG[@]}" \
  --model "${MODEL_DIR:-models/Qwen3.5-9B}" --lora "${LORA_DIR:-adapters/sql_rft_v3}" --fewshot assets/fewshot_bank.jsonl \
  "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" --save-prompts --deadline "${DEADLINE:-40500}" "${LAYERS[@]}" ${EXTRA:-}
echo "== done $(date)"
