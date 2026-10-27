"""Count-based bigram baseline with add-one smoothing."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from train import append_run, prepare_data


def bigram_validation_loss(train_ids, val_ids, vocab_size=1024) -> float:
    counts = torch.ones((vocab_size, vocab_size), dtype=torch.float64)
    for current_id, next_id in zip(train_ids[:-1], train_ids[1:]):
        counts[current_id, next_id] += 1

    row_totals = counts.sum(dim=1)
    total_loss = 0.0
    transitions = len(val_ids) - 1
    for current_id, next_id in zip(val_ids[:-1], val_ids[1:]):
        probability = counts[current_id, next_id] / row_totals[current_id]
        total_loss -= math.log(probability.item())
    return total_loss / transitions


def main() -> None:
    project_root = Path(__file__).resolve().parent
    _, train_ids, val_ids = prepare_data(project_root)
    loss = bigram_validation_loss(train_ids, val_ids)
    print(f"Bigram validation loss: {loss:.6f} nats/token")
    append_run(project_root / "runs.csv", {
        "run_id": "bigram",
        "n_layer": "",
        "n_head": "",
        "n_embd": "",
        "block_size": "",
        "max_lr": "",
        "max_steps": "",
        "ablation": "count_based",
        "params": 1024 * 1024,
        "best_val_loss": loss,
        "final_train_loss": "",
        "tokens_per_sec": "",
        "wall_time_min": "",
    })


if __name__ == "__main__":
    main()

