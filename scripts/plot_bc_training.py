"""Plot the behavior-cloning training curves for slide 1 ("BC reproduces MCTS").

Two composed panels from one finished BC run (default: lucky-music-7 / s0ga84ri,
the cash-token 64x32 run that hit 0.73 joint accuracy):

  LEFT  — train vs val loss, OVERLAID on one axes. The point isn't "loss went
          down" but "val tracks train" -> it generalized, no overfitting. This is
          the visual foil for the PPO-divergence slide that follows.
  RIGHT — final per-head val accuracy as a bar chart (asset / verb / target /
          joint). Shows WHERE it learned: verbs are easy, asset selection is
          hardest -> the lead-in to "asset is where PPO has the most room".

History is pulled live from the Weights & Biases API (read-only).

    poetry run python scripts/plot_bc_training.py
    poetry run python scripts/plot_bc_training.py --run nicksrusso/battleboats-data/s0ga84ri
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-head series -> (legend label, color index in tab10).
HEAD_SERIES = [
    ("val_acc/joint", "joint (all 3 heads correct)", 3),
    ("val_acc/verb", "verb", 0),
    ("val_acc/target", "target", 1),
    ("val_acc/asset", "asset", 2),
]


def _fetch(run_path: str):
    """Return the run's full history as a DataFrame (one row per logged step)."""
    import wandb

    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    # Unrestricted scan: passing keys=[...] would only return steps where EVERY
    # key is present, but train/loss (per-step) and val_acc/* (per-epoch) never
    # share a step -> that intersection is empty. Fetch all rows, filter per-key.
    rows = list(run.scan_history())
    return run, rows


def _series(rows, xkey, ykey):
    """(x, y) for one metric, keeping only steps where ykey was actually logged."""
    xs, ys = [], []
    for r in rows:
        y = r.get(ykey)
        x = r.get(xkey)
        if y is not None and x is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        default="nicksrusso/battleboats-data/s0ga84ri",
        help="W&B run path entity/project/run_id (default: lucky-music-7).",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/bc_training.png"))
    args = parser.parse_args()

    run, rows = _fetch(args.run)
    cmap = plt.get_cmap("tab10")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- LEFT: train vs val loss, overlaid (loss logged per-step / per-epoch) ----
    tx, ty = _series(rows, "_step", "train/loss")
    vx, vy = _series(rows, "_step", "val/loss")
    ax_loss.plot(tx, ty, color=cmap(0), lw=1.6, alpha=0.85, label="train loss")
    ax_loss.plot(vx, vy, color=cmap(3), lw=2.2, marker="o", ms=5, label="val loss")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss  (− joint log-prob of expert action)")
    ax_loss.set_title("Loss Plot", fontsize=13, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=10)
    ax_loss.grid(True, alpha=0.25)

    # ---- RIGHT: per-head val accuracy over training, as LINES (shows the flatness:
    # accuracy plateaus within epoch 0 and barely moves for the rest of training).
    final_joint = None
    for ykey, label, ci in HEAD_SERIES:
        xs, ys = _series(rows, "_step", ykey)
        if not ys:
            continue
        final = ys[-1]
        if ykey == "val_acc/joint":
            final_joint = final
        lw = 2.6 if ykey == "val_acc/joint" else 1.8
        ax_acc.plot(xs, ys, color=cmap(ci), lw=lw, marker="o", ms=4, label=f"{label}  ({final:.2f})")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_xlabel("training step")
    ax_acc.set_ylabel("validation accuracy  (matches MCTS expert)")
    ax_acc.set_title("Per-Head Accuracy", fontsize=13, fontweight="bold")
    ax_acc.legend(loc="lower right", fontsize=10, title="head (final acc)")
    ax_acc.grid(True, alpha=0.25)

    headline = "Behavior Cloning Initial Results"
    if final_joint is not None:
        headline += f"  —  {final_joint:.0%} Joint Accuracy"
    fig.suptitle(headline, fontsize=16, fontweight="bold")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}  (run {run.name}/{run.id}, joint={final_joint})")


if __name__ == "__main__":
    main()
