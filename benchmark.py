"""Measure cached and uncached generation speed for model M."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from generate import generate
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = GPTConfig(
        vocab_size=1024,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    )
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def measure(model, prompt_ids, use_cache):
    if model.config.block_size < 120:
        raise ValueError("block_size must be at least 120")
    prompt_ids = prompt_ids[-(model.config.block_size - 120):]
    if not prompt_ids:
        raise ValueError("the prompt needs at least one token")

    generate(model, prompt_ids, 2, temperature=0, use_cache=use_cache)
    started = time.perf_counter()
    generate(model, prompt_ids, 120, temperature=0, use_cache=use_cache)
    elapsed = time.perf_counter() - started
    return 120 / elapsed


def main() -> None:
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(root / "checkpoints" / "M.pt", device)
    tokenizer = BPETokenizer.load(str(root / "tokenizer.json"))
    prompt_ids = tokenizer.encode("ROMEO:")

    cached = measure(model, prompt_ids, use_cache=True)
    uncached = measure(model, prompt_ids, use_cache=False)
    speedup = cached / uncached

    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "cached_tokens_per_sec", "uncached_tokens_per_sec", "speedup"])
        writer.writerow(["M", cached, uncached, speedup])
    print(f"Cached: {cached:.2f} tokens/s")
    print(f"Uncached: {uncached:.2f} tokens/s")
    print(f"Speed-up: {speedup:.2f}x")


if __name__ == "__main__":
    main()

