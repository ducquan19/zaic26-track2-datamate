"""Save the 4-bit base that QLoRA trains on, so a machine short of disk keeps about a third of the bf16 weights (19.3 GB).

    .venv-train/bin/python training/save_4bit.py --model models/Qwen3.5-9B --out models/Qwen3.5-9B-bnb-4bit

The base is loaded exactly as training/train_lora.py loads it (Unsloth, load_in_4bit=True), then saved with its quantization
config. train_lora.py --model models/Qwen3.5-9B-bnb-4bit then starts from the same 4-bit weights it would have made itself.
Needs a CUDA GPU (bitsandbytes quantizes on it) and the training environment (training/requirements.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from unsloth import FastModel  # noqa: I001  (unsloth patches transformers on import: keep it first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--out", type=Path, default=Path("models/Qwen3.5-9B-bnb-4bit"))
    ap.add_argument("--max-len", type=int, default=32768)
    a = ap.parse_args()
    model, tokenizer = FastModel.from_pretrained(model_name=a.model, max_seq_length=a.max_len, load_in_4bit=True, full_finetuning=False)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tokenizer.save_pretrained(a.out)
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file()) / 1e9
    print(f"4-bit base saved to {a.out} ({size:.1f} GB)")


if __name__ == "__main__":
    main()
