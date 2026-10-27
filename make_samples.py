"""Generate the three samples required for reflection R4."""

from __future__ import annotations

from pathlib import Path

import torch

from benchmark import load_model
from generate import generate
from tokenizer import BPETokenizer


def main() -> None:
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(root / "checkpoints" / "L.pt", device)
    tokenizer = BPETokenizer.load(str(root / "tokenizer.json"))
    prompt_ids = tokenizer.encode("ROMEO:")

    samples = []
    for sample_number in range(3):
        output_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=120,
            temperature=0.8,
            use_cache=True,
            seed=42 + sample_number,
        )
        samples.append(
            f"Sample {sample_number + 1} (seed {42 + sample_number})\n"
            + tokenizer.decode(output_ids)
        )

    (root / "samples.txt").write_text("\n\n".join(samples), encoding="utf-8")


if __name__ == "__main__":
    main()

