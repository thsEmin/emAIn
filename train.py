"""Training pipeline for the HW1 GPT models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


LOG_COLUMNS = ["step", "train_loss", "val_loss", "lr", "tokens_seen"]
RUN_COLUMNS = [
    "run_id", "n_layer", "n_head", "n_embd", "block_size", "max_lr",
    "max_steps", "ablation", "params", "best_val_loss", "final_train_loss",
    "tokens_per_sec", "wall_time_min",
]


def learning_rate(step: int, max_steps: int, warmup_steps: int, max_lr: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if max_steps <= warmup_steps:
        return 0.1 * max_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (0.1 + 0.9 * cosine)


def prepare_data(project_root: Path):
    data_path = project_root / "data" / "input.txt"
    if not data_path.exists():
        raise FileNotFoundError(
            "Place Tiny Shakespeare at data/input.txt before training."
        )

    tokenizer_path = project_root / "tokenizer.json"
    cache_directory = project_root / "cache"
    train_cache = cache_directory / "train_ids.pt"
    val_cache = cache_directory / "val_ids.pt"

    if tokenizer_path.exists() and train_cache.exists() and val_cache.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        return tokenizer, torch.load(train_cache), torch.load(val_cache)

    text = data_path.read_text(encoding="utf-8")
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=1024)
        tokenizer.save(str(tokenizer_path))

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    cache_directory.mkdir(parents=True, exist_ok=True)
    torch.save(train_ids, train_cache)
    torch.save(val_ids, val_cache)
    return tokenizer, train_ids, val_ids


def random_batch(ids, batch_size, block_size, device, generator):
    highest_start = len(ids) - block_size - 1
    if highest_start < 0:
        raise ValueError("token sequence is shorter than block_size + 1")
    starts = torch.randint(highest_start + 1, (batch_size,), generator=generator)
    x = torch.stack([ids[start:start + block_size] for start in starts])
    y = torch.stack([ids[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


def estimate_loss(model, ids, config, device, generator, batches=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = random_batch(
                ids, config["batch_size"], config["block_size"], device, generator
            )
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def make_model_config(config):
    ablation = config.get("ablation", "none")
    return GPTConfig(
        vocab_size=1024,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        use_position_embedding=ablation != "no_position",
        norm_type="post" if ablation == "post_ln" else "pre",
    )


def append_run(path: Path, row: dict) -> None:
    old_rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            old_rows = [
                old_row for old_row in csv.DictReader(file)
                if old_row["run_id"] != row["run_id"]
            ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def train(config_path: str) -> None:
    project_root = Path(__file__).resolve().parent
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    _, train_ids, val_ids = prepare_data(project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(make_model_config(config)).to(device)

    decay = []
    no_decay = []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config["max_lr"],
        betas=(0.9, 0.95),
    )

    train_generator = torch.Generator().manual_seed(config["seed"])
    val_generator = torch.Generator().manual_seed(config["seed"] + 1)
    log_directory = project_root / "logs"
    checkpoint_directory = project_root / "checkpoints"
    log_directory.mkdir(exist_ok=True)
    checkpoint_directory.mkdir(exist_ok=True)
    log_path = log_directory / f"{config['run_id']}.csv"

    best_val_loss = float("inf")
    tokens_seen = 0
    final_train_loss = float("nan")
    started = time.perf_counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
        writer.writeheader()

        for step in range(config["max_steps"]):
            completed_step = step + 1
            lr = learning_rate(
                completed_step,
                config["max_steps"],
                config["warmup_steps"],
                config["max_lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = random_batch(
                train_ids, config["batch_size"], config["block_size"],
                device, train_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            final_train_loss = loss.item()
            tokens_seen += config["batch_size"] * config["block_size"]

            should_evaluate = (
                completed_step % config["eval_interval"] == 0
                or completed_step == config["max_steps"]
            )
            if should_evaluate:
                val_loss = estimate_loss(
                    model, val_ids, config, device, val_generator, batches=50
                )
                writer.writerow({
                    "step": completed_step,
                    "train_loss": final_train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                })
                log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "config": config,
                            "model_state": model.state_dict(),
                            "step": completed_step,
                            "val_loss": val_loss,
                        },
                        checkpoint_directory / f"{config['run_id']}.pt",
                    )

    wall_time = time.perf_counter() - started
    append_run(project_root / "runs.csv", {
        "run_id": config["run_id"],
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "max_lr": config["max_lr"],
        "max_steps": config["max_steps"],
        "ablation": config.get("ablation", "none"),
        "params": model.num_params(),
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "tokens_per_sec": tokens_seen / wall_time,
        "wall_time_min": wall_time / 60,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

