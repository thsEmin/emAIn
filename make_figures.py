"""Create the two figures required by the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    figure_directory = root / "figures"
    figure_directory.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    for log_path in sorted((root / "logs").glob("*.csv")):
        log = pd.read_csv(log_path)
        plt.plot(log["step"], log["val_loss"], label=log_path.stem)
    plt.xlabel("Training step")
    plt.ylabel("Validation loss (nats/token)")
    plt.title("Validation loss by run")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figure_directory / "validation_curves.png", dpi=180)
    plt.close()

    runs = pd.read_csv(root / "runs.csv")
    scaling = runs[runs["run_id"].isin(["S", "M", "L"])].copy()
    if len(scaling) != 3:
        raise ValueError("runs.csv must contain completed S, M, and L rows")
    scaling = scaling.set_index("run_id").loc[["S", "M", "L"]].reset_index()

    plt.figure(figsize=(6, 4.5))
    plt.plot(scaling["params"], scaling["best_val_loss"], marker="o")
    for _, row in scaling.iterrows():
        plt.annotate(row["run_id"], (row["params"], row["best_val_loss"]), xytext=(5, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Trainable parameters (log scale)")
    plt.ylabel("Best validation loss (nats/token)")
    plt.title("Scaling result")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_directory / "scaling.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()

