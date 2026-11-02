"""ローカル LLM (Qwen2.5-Instruct GGUF, 4bit 量子化) を models/ にダウンロードする."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),     # 約 2.1GB (既定)
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),  # 約 1.1GB
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=MODELS, default="3b")
    repo, fname = MODELS[ap.parse_args().size]
    print(hf_hub_download(repo, fname, local_dir=Path(__file__).parent / "models"))
