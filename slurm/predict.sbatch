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
# vLLM 0.32.0+cu129 needs a recent enough driver: gpu03 (580) runs it, gpu01 (525, CUDA 12.0) fails at the first CUDA call
# ("named symbol not found"), gpu04 holds V100s. Both are excluded; see slurm/README.md for each node's driver.
#SBATCH --job-name=datama
#SBATCH --partition=batch
#SBATCH --account=compute-account
#SBATCH --exclude=gpu01,gpu04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# An fp8 KV cache needs SM 8.9+ (the Triton attention backend refuses it on an A100, SM 8.0): below that, keep the submitted
# engine settings but with the KV cache in the model's dtype. The weights stay fp8 (QUANT). An ENGINE given by the caller wins.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
if [ -z "${ENGINE:-}" ] && [ -n "$CAP" ] && awk -v c="$CAP" 'BEGIN { exit !(c < 8.9) }'; then
  export ENGINE='{"max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
  echo "== GPU compute capability $CAP < 8.9: KV cache in bf16 (no fp8 KV cache)"
fi
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

# Compute nodes cannot write $HOME (Triton, vLLM and torch.compile cache there by default): keep every cache under CACHE_ROOT,
# shared by the shards so the compiled kernels are reused by later runs.
CACHE_ROOT=${CACHE_ROOT:-/media/$USER/cache}
export TRITON_HOME="$CACHE_ROOT" TRITON_CACHE_DIR="$CACHE_ROOT/triton" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" \
  XDG_CACHE_HOME="$CACHE_ROOT/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor" HF_HUB_OFFLINE=1
mkdir -p "$CACHE_ROOT"

# vLLM reads the safetensors through mmap plus a background prefetch, which made no progress in an hour on the cluster's
# NFS. Copy the model once per node to its local disk (sequential read) and load it from there; later jobs on the node
# reuse the copy. STAGE=0 loads straight from MODEL_DIR; STAGE_DIR picks the local directory (default /tmp/$USER-models).
MODEL=${MODEL_DIR:-models/Qwen3.5-9B}
if [ "${STAGE:-1}" = 1 ]; then
  BASE=${STAGE_DIR:-/tmp/$USER-models}
  LOCAL=$BASE/$(basename "$MODEL")
  NEED=$(du -sk "$MODEL" | cut -f1)
  mkdir -p "$BASE"
  (
    command -v flock >/dev/null && flock 9   # one copy per node, whatever the number of shards started on it
    if [ "$(cat "$LOCAL/.staged" 2>/dev/null)" != "$NEED" ]; then
      FREE=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
      if [ "$FREE" -gt $((NEED + 2097152)) ]; then
        echo "== staging $MODEL -> $LOCAL ($((NEED / 1048576)) GB), $(date +%T)"
        PART="$LOCAL.part.${SLURM_JOB_ID:-$BASHPID}"   # own temporary copy, renamed in one step: a concurrent job never sees a half copy
        rm -rf "$PART"; [ -f "$LOCAL/.staged" ] || rm -rf "$LOCAL"
        if cp -r "$MODEL" "$PART" && echo "$NEED" > "$PART/.staged" && mv -T "$PART" "$LOCAL" 2>/dev/null; then
          echo "== staged at $(date +%T)"
        else
          rm -rf "$PART"; echo "== not staged by this job (failed, or another job finished first)"
        fi
      else
        echo "== $BASE has $((FREE / 1048576)) GB free, $((NEED / 1048576)) GB needed: loading from $MODEL"
      fi
    fi
  ) 9>"$BASE/.stage.lock"
  [ "$(cat "$LOCAL/.staged" 2>/dev/null)" = "$NEED" ] && MODEL=$LOCAL
fi
echo "== model $MODEL"

echo "== shard $I/$N of $SPLIT -> $OUT  job ${SLURM_JOB_ID:-local}  host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), '| vllm', vllm.__version__)"

# --deadline: about 45 min below the 12 h limit, the run skips thinking samples for a shop once it is behind pace
# shellcheck disable=SC2086
"$PY" -m datamate.predict --data "data/$SPLIT" --output "$OUT/shard$I" --shard "$I/$N" "${IDS_ARG[@]}" \
  --model "$MODEL" --lora "${LORA_DIR:-adapters/sql_rft_v3}" --fewshot assets/fewshot_bank.jsonl \
  "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" --save-prompts --deadline "${DEADLINE:-40500}" "${LAYERS[@]}" ${EXTRA:-}
echo "== done $(date)"
