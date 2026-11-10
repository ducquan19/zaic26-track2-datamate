# Train LoRA trên cluster Slurm

Một vòng train cần 1 GPU hỗ trợ bf16 (Ampere trở lên, ví dụ A100, L40S, H100), từ 24 GB VRAM. LoRA v3 cần tối đa 15,6 GiB và
mất 8,2 giờ trên RTX PRO 4000 Blackwell. V100 và T4 không chạy được. Tên partition, loại GPU và module là của từng cluster:
sửa các dòng `#SBATCH` trong [train_lora.sbatch](train_lora.sbatch), hoặc truyền ghi đè khi gọi `sbatch`.

## 1. Cài đặt trên login node (một lần)

Compute node thường không có mạng, nên mọi thứ cần tải đều làm ở bước này.

```bash
git clone https://github.com/ducquan19/zaic26-track2-datamate.git datamate && cd datamate
git lfs install && git lfs pull                          # adapters/sql_rft_v3
uv venv .venv-train --python 3.12                        # chưa có uv: pip install --user uv
uv pip install --python .venv-train -r training/requirements.txt
uv pip install --python .venv-train fla-core==0.6.0 --no-deps
export HF_HOME=$SCRATCH/hf                               # cache trên ổ scratch, không để trong $HOME
.venv-train/bin/hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/Qwen3.5-9B
```

Dữ liệu (`data/`, gồm `data/sft/*.jsonl`) không có trong git. Chép từ máy của đội bằng `rsync -av data/ user@cluster:~/datamate/data/`.
Các file SFT đã dùng nằm trong `build/training_code/data/sft/` (tạo bằng `python3 scripts/pack_training_code.py`).

## 2. Chạy

```bash
# kiểm tra môi trường, kernel fla và bộ nhớ trong vài phút, trước khi chạy job dài
sbatch --export=ALL,EXTRA="--limit-rows 50 --max-steps 5" --time=00:30:00 slurm/train_lora.sbatch

# LoRA v4: train tiếp từ v3 (mặc định của script)
sbatch slurm/train_lora.sbatch

# vòng khác: đổi dữ liệu, thư mục ra, adapter khởi đầu, max-len, lr
sbatch --export=ALL,DATA=data/sft/sql_rft_notes.jsonl,OUT=runs/lora/sql_rft_v5,INIT=runs/lora/sql_rft_v4/adapter slurm/train_lora.sbatch

# train lại cả chuỗi v1 -> v2 -> v3 (3 job phụ thuộc nhau, tham số như docker/train.sh)
bash slurm/train_chain.sh --partition=a100 --gres=gpu:a100:1
```

Adapter được ghi vào `$OUT/adapter`, checkpoint mỗi 50 bước nằm trong `$OUT`. Khi chạy inference thì trỏ `LORA_DIR` vào đó.

Theo dõi: `squeue -u $USER`, `tail -f logs/<tên job>-<jobid>.out` (loss theo bước), `srun --jobid <jobid> --overlap nvidia-smi`,
`sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`.

## 3. Hết giờ và chạy tiếp

Script nhận tín hiệu `USR1` 10 phút trước khi hết giờ, rồi tự xếp lại job vào hàng đợi (`scontrol requeue`). Khi chạy lại,
`--resume` tiếp tục từ checkpoint mới nhất. Nếu cluster không cho requeue, chỉ cần `sbatch` lại đúng lệnh cũ.

## 4. Lỗi hay gặp

- `check_fla.py` in `fla check FAIL`: dừng lại, đừng train. Thường do phiên bản torch, triton hoặc fla-core không khớp, hoặc GPU không hỗ trợ.
- Hết bộ nhớ (OOM): giảm `MAX_LEN`. Mỗi dòng SFT dài 9k đến 19k token, nên dưới khoảng 16k thì phần lớn dữ liệu bị bỏ.
- Tạo lại dữ liệu SFT (chạy `datamate.predict` trên train rồi `add_notes.py`) cần môi trường vLLM riêng: `uv venv .venv && uv pip install --python .venv -e ".[llm]"`.
  `scripts/flags.sh` dùng fp8: chọn GPU Ada, Hopper hoặc Blackwell. Trên A100 thì `add_notes.py` cần `--quant none`, còn `flags.sh` phải sửa dòng `--quant`.
