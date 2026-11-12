# Train LoRA trên cluster Slurm

Viết cho cluster của đội; cluster khác thì sửa các dòng `#SBATCH` trong [train_lora.sbatch](train_lora.sbatch).

| | Cluster của đội | Ảnh hưởng |
| --- | --- | --- |
| Partition, account | `batch`, `compute-account`, QOS `gpu_limit` (tối đa 4 GPU cùng lúc mỗi người) | đã điền trong `train_lora.sbatch` |
| GPU | gpu01 (8×), gpu02-03 (4×) A100 40GB; gpu04 V100 32GB | V100 không có bf16: `--exclude=gpu04` |
| Driver | 555.42.02 (CUDA ≤ 12.5) | `torch` mặc định trên PyPI là bản CUDA 13 (cần driver ≥ 580): cài bản `+cu126` |
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
.venv-train/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"   # phải là 2.14.1+cu126 12.6

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
- Job chạy trên V100 (log in `Tesla V100`): thiếu `--exclude=gpu04`.
- Hết bộ nhớ (OOM): giảm `MAX_LEN`. Mỗi dòng SFT dài 9k đến 19k token, nên dưới khoảng 16k thì phần lớn dữ liệu bị bỏ.
- Tạo lại dữ liệu SFT (chạy `datamate.predict` trên train rồi `add_notes.py`) cần vLLM 0.32 và fp8 (`scripts/flags.sh`).
  Cluster này không hợp cho bước đó: A100 không có fp8 phần cứng và driver 555 cũ hơn bản CUDA của vLLM. Nên chạy bước đó trên máy đã
  dùng cho bản nộp, rồi chép file SFT sang đây để train.
