# Train LoRA trên cluster Slurm

Viết cho cluster của đội; cluster khác thì sửa các dòng `#SBATCH` trong [train_lora.sbatch](train_lora.sbatch).

| | Cluster của đội | Ảnh hưởng |
| --- | --- | --- |
| Partition, account | `batch`, `compute-account`, QOS `gpu_limit` (tối đa 4 GPU cùng lúc mỗi người) | đã điền trong `train_lora.sbatch` |
| GPU | gpu01 (8×), gpu02-03 (4×) A100 40GB; gpu04 V100 32GB | V100 không có bf16: `--exclude=gpu01,gpu04` |
| Driver | mỗi node một khác: gpu01 525.147 (CUDA 12.0), gpu02 555.42 (12.5), gpu03 580.82 (13.0) | gpu01 quá cũ cho torch cu126/cu129 (`CUDA error: named symbol not found` ngay lệnh CUDA đầu tiên): bị loại. `torch` mặc định trên PyPI là bản CUDA 13: cài bản `+cu126` (train), `+cu129` (vLLM) |
| Python | hệ thống chỉ có 3.10; `trl` cần ≥ 3.11 | dùng CPython 3.12 dựng sẵn, giải nén vào thư mục của bạn |
| Thời gian | tối đa 2 ngày một job, requeue được bật | job 12 h, tự requeue khi sắp hết giờ |
| Bộ nhớ, đĩa | tối đa 32 GB RAM một job; `/media02` dùng chung và gần đầy | `--mem=32G`; giữ model và checkpoint càng nhỏ càng tốt |

Một vòng train cần 1 GPU bf16 từ 24 GB. LoRA v3 cần tối đa 15,6 GiB và mất 8,2 h trên RTX PRO 4000 Blackwell (chưa đo trên A100).

## 1. Cài đặt trên login node (một lần, cần mạng)

Login node có mạng, compute node có thể không: mọi thứ cần tải đều làm ở đây. Mọi đường dẫn nằm dưới `/media02/lhbac34`
(compute node phải thấy thư mục này).

```bash
cd /media02/lhbac34
# Python 3.12 không cần root: bản CPython dựng sẵn (python-build-standalone), giải nén là dùng
mkdir -p opt && curl -L -o /tmp/py312.tar.gz \
  "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar -xzf /tmp/py312.tar.gz -C opt && mv opt/python opt/python3.12
opt/python3.12/bin/python3.12 --version                        # Python 3.12.16

cd datamate
git pull && git lfs install && git lfs pull                       # code và adapters/sql_rft_v3
/media02/lhbac34/opt/python3.12/bin/python3.12 -m venv .venv-train
.venv-train/bin/pip install --upgrade pip
# torch cho CUDA 12.6 TRƯỚC: bản mặc định (CUDA 13) không chạy trên driver 555. pip sẽ coi torch==2.14.1 của requirements là đã có.
.venv-train/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv-train/bin/pip install -r training/requirements.txt
.venv-train/bin/pip install fla-core==0.6.0 --no-deps
# requirements kéo torchvision mới nhất từ PyPI (bản cho CUDA 13): toán tử của nó không nạp được với torch cu126, unsloth báo lỗi khi import
.venv-train/bin/pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 "torchvision==0.29.1"
.venv-train/bin/python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__)"   # phải là 2.14.1+cu126 12.6 0.29.1+cu126

export HF_HOME=/media02/lhbac34/hf
.venv-train/bin/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
```

Nếu dòng kiểm tra in ra bản khác `+cu126` (một gói khác đã kéo torch mặc định về), cài lại torch bằng lệnh `--index-url` ở trên.

Dữ liệu (`data/`, gồm `data/sft/*.jsonl`) không có trong git. Chép từ máy của đội: `rsync -av data/ lhbac34@<login>:/media02/lhbac34/datamate/data/`.
Các file SFT đã dùng nằm trong `build/training_code/data/sft/` (tạo bằng `python3 scripts/pack_training_code.py`).

## 2. Chạy

```bash
cd /media02/lhbac34/datamate
# kiểm tra GPU, driver, kernel fla và bộ nhớ trong vài phút, trước khi chạy job dài
sbatch --export=ALL,EXTRA="--limit-rows 50 --max-steps 5" --time=00:30:00 slurm/train_lora.sbatch

# LoRA v4: train tiếp từ v3 (mặc định của script)
sbatch slurm/train_lora.sbatch

# vòng khác: đổi dữ liệu, thư mục ra, adapter khởi đầu, max-len, lr
sbatch --export=ALL,DATA=data/sft/sql_rft_notes.jsonl,OUT=runs/lora/sql_rft_v5,INIT=runs/lora/sql_rft_v4/adapter slurm/train_lora.sbatch

# train lại cả chuỗi v1 -> v2 -> v3 (3 job phụ thuộc nhau, tham số như docker/train.sh)
bash slurm/train_chain.sh
```

Job kiểm tra in tên GPU (phải là A100) rồi chạy `check_fla.py`, dòng cuối phải là `fla check PASS`. Adapter được ghi vào
`$OUT/adapter`, checkpoint mỗi 50 bước nằm trong `$OUT`.

Theo dõi: `squeue -u $USER` (cột cuối ghi lý do đang chờ, ví dụ `Resources` khi mọi GPU đều bận), `tail -f logs/<tên job>-<jobid>.out`,
`srun --jobid <jobid> --overlap nvidia-smi`, `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`.

## 3. Hết giờ và chạy tiếp

Script nhận tín hiệu `USR1` 10 phút trước khi hết giờ, rồi tự xếp lại job vào hàng đợi (`scontrol requeue`). Khi chạy lại,
`--resume` tiếp tục từ checkpoint mới nhất. `sbatch` lại đúng lệnh cũ cũng cho kết quả như vậy.

## 4. Lỗi hay gặp

- `CUDA driver version is insufficient` hoặc `torch.cuda.is_available()` là False: torch không phải bản `+cu126`, xem mục 1.
- `check_fla.py` in `fla check FAIL`: dừng lại, đừng train. Thường do phiên bản torch, triton hoặc fla-core không khớp.
- Job chạy trên V100 (log in `Tesla V100`): thiếu `--exclude=gpu01,gpu04`.
- `CUDA error: named symbol not found` ngay khi khởi tạo: job rơi vào node có driver quá cũ (gpu01, 525). Dòng
  `nvidia-smi` đầu log in driver; xem `--exclude` ở trên.
- Hết bộ nhớ (OOM): giảm `MAX_LEN`. Mỗi dòng SFT dài 9k đến 19k token, nên dưới khoảng 16k thì phần lớn dữ liệu bị bỏ.
- Tạo lại dữ liệu SFT (chạy `datamate.predict` trên train với `--save-prompts`, rồi `build_sft.py` và `add_notes.py`) cần vLLM:
  dùng môi trường vLLM `+cu129` của mục 5 (`slurm/predict.sbatch` với `SPLIT=train IDS=splits/rft_train24.txt`). `add_notes.py`
  chạy trên A100 với `--quant fp8` như mục 5; nếu lỗi thì `--quant none` (bf16, cần nhiều bộ nhớ GPU hơn).

## 5. Inference (chạy pipeline và chấm điểm) trên cluster

vLLM 0.32.0 của bản nộp có bản build cho CUDA 12.9 (`+cu129`, đi cùng `torch 2.13.0+cu129`), chạy trên driver 555 (CUDA 12.5)
nhờ tương thích phiên bản phụ của CUDA 12: wheel có sẵn mã máy cho A100 (sm_80), Triton biên dịch kernel ra mã máy, nên driver
không phải dịch PTX. `scripts/flags.sh` đã chọn backend attention Triton và tắt sampler FlashInfer. Trên A100 (không có fp8
phần cứng), `--quant fp8` chạy theo kiểu chỉ lượng tử hoá trọng số (Marlin).

`/media02` gần đầy: môi trường vLLM (khoảng 10 GB) và model bf16 (19,3 GB, inference cần bản bf16, không dùng bản 4-bit của
bước train) đặt ở `/media/lhbac34`.

```bash
# một lần, trên login node
source /media02/lhbac34/datamate.env
mkdir -p /media/lhbac34/envs /media/lhbac34/models
/media02/lhbac34/opt/python3.12/bin/python3.12 -m venv /media/lhbac34/envs/datamate-vllm
V=/media/lhbac34/envs/datamate-vllm/bin
$V/pip install --upgrade pip
$V/pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129
$V/pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
# vLLM kéo torchcodec mới nhất từ PyPI (bản CUDA 13): libtorchcodec_image.so không nạp được, `import vllm` lỗi. Thay bằng bản cu129
$V/pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu129 "torchcodec==0.16.0"
cd /media02/lhbac34/datamate && $V/pip install -e .          # datamate (vllm đã có, không cài lại)
$V/python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"   # 2.13.0+cu129 12.9 0.32.0+cu129
$V/pip cache purge
HF_HUB_DISABLE_XET=1 $V/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --local-dir /media/lhbac34/models/Qwen3.5-9B
```

Chạy (từ thư mục gốc repo; `logs/` phải có trước khi `sbatch`, vì Slurm mở file log trước khi script chạy):

```bash
cd /media02/lhbac34/datamate && source /media02/lhbac34/datamate.env && mkdir -p logs
export PYTHON=/media/lhbac34/envs/datamate-vllm/bin/python MODEL_DIR=/media/lhbac34/models/Qwen3.5-9B

# thử 20 câu public: GPU, driver, vLLM, LoRA, fp8 trên A100
sbatch --array=0 --time=01:00:00 --export=ALL,EXTRA="--limit 20",OUT=runs/smoke slurm/predict.sbatch
tail -f logs/datamate-<jobid>_0.out

# public test, 4 shard (QOS cho tối đa 4 GPU), rồi gộp file nộp
sbatch slurm/predict.sbatch
$PYTHON -m datamate.submission merge runs/public --questions data/public_test/questions.jsonl

# holdout của train, có chấm điểm
sbatch --export=ALL,SPLIT=train,IDS=splits/holdout600.txt,OUT=runs/holdout slurm/predict.sbatch
$PYTHON -m datamate.submission merge runs/holdout --questions data/train/questions.jsonl
grep -F -f splits/holdout600.txt data/train/questions.jsonl > runs/holdout/questions.jsonl
$PYTHON -m datamate.scoring --data data/train --pred runs/holdout/submission.jsonl --questions runs/holdout/questions.jsonl --output runs/holdout/score.json
```

Thêm cải tiến thử nghiệm bằng `EXTRA`, ví dụ `--export=ALL,OUT=runs/public_lots,EXTRA="--stock-views --value-lookup"`.
Lỗi hay gặp:
- `PYTHON` và `MODEL_DIR` phải được export trước `sbatch` (nên ghi vào `datamate.env`); thiếu thì job lỗi ngay với
  `.venv/bin/python: No such file or directory` (mã 127).
- `Permission denied: '/home/lhbac34/.triton'`: compute node không ghi được `/home`. `predict.sbatch` đặt mọi cache (Triton, vLLM,
  torch.compile) dưới `CACHE_ROOT` (mặc định `/media/$USER/cache`).
- `OSError: Could not load this library: .../libtorchcodec_image.so`: torchcodec là bản CUDA 13, cài lại bản cu129 như trên.
- `CUDA driver version is insufficient`: torch không phải bản `+cu129`, cài lại torch như trên.
- Lỗi khi dùng KV cache fp8 trên A100: chạy với `ENGINE='{"max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'`
  (bỏ `kv_cache_dtype`), export trước khi `sbatch`.
- Hết bộ nhớ GPU: giảm `max_num_seqs` trong `ENGINE`, hoặc `GPU_MEM=0.85`.
