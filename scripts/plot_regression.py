"""Visualize a heuristic linear regression (fit to MCTS root values).

Produces two presentation figures from a fitted weights JSON + its training
harvest, loaded EXACTLY the way the fit saw it (same decisive-only filter,
same mcts_root_value target, same by-game split) via test_train.load_rows:

  1. docs/regression_calibration.png  — predicted vs. true MCTS value on the
     held-out split, with R^2 / MSE. "How well does the linear model fit?"
  2. docs/regression_importance.png   — standardized coefficients
     (weight x feature_std = contribution in MCTS-value units), so features on
     different scales are comparable. "What did the regression decide matters?"

    poetry run python scripts/plot_regression.py runs/weights/v5.json
    poetry run python scripts/plot_regression.py runs/weights/v5.json --include-truncated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_train import load_rows, game_level_split, to_xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="?", default=Path("runs/weights/v5.json"),
                        help="Fitted weights JSON (default: runs/weights/v5.json).")
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--include-truncated", action="store_true",
                        help="Match a fit done with truncated games included (default: decisive-only).")
    args = parser.parse_args()

    cfg = json.loads(args.weights.read_text())
    keys = cfg["feature_keys"]
    w = np.array([cfg["weights"][k] for k in keys], dtype=float)
    b = float(cfg["intercept"])
    test_frac = cfg.get("split", {}).get("test_frac", 0.1)
    seed = cfg.get("split", {}).get("seed", 42)
    tag = args.weights.stem

    # Load the source harvest(s) exactly as the fit did.
    sources = [Path(s) for s in cfg["source_files"]]
    rows = load_rows(sources, decisive_only=not args.include_truncated)
    train_rows, test_rows = game_level_split(rows, test_frac=test_frac, seed=seed)
    X_all, _ = to_xy(rows, keys)
    Xte, yte = to_xy(test_rows, keys)
    pred_te = Xte @ w + b

    ss_res = float(((yte - pred_te) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    mse = float(((yte - pred_te) ** 2).mean())
    mae = float(np.abs(yte - pred_te).mean())
    print(f"[{tag}] held-out: n={len(yte):,}  R2={r2:.3f}  MSE={mse:.3f}  MAE={mae:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================== Fig 1: calibration
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    hb = ax.hexbin(yte, pred_te, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = (-1.15, 1.15)
    ax.plot(lim, lim, color="#d62728", lw=1.8, ls="--", label="perfect fit (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true MCTS root value", fontsize=12)
    ax.set_ylabel("heuristic linear-model prediction", fontsize=12)
    ax.set_title("How Accurately the Heuristic Estimates Game Value", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.text(
        0.97, 0.06,
        f"held-out ({test_frac:.0%} by game)\n$R^2$ = {r2:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nn = {len(yte):,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("point density (log)", fontsize=10)
    fig.tight_layout()
    out1 = args.out_dir / f"regression_calibration_{tag}.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    print(f"saved {out1}")

    # ===================================================== Fig 2: importance
    std = X_all.std(axis=0)
    importance = w * std
    order = np.argsort(np.abs(importance))  # ascending; barh draws bottom->top
    labels = [keys[i].replace("_", " ") for i in order]
    vals = importance[order]
    colors = ["#2e9e3f" if v >= 0 else "#d62728" for v in vals]

    fig2, ax2 = plt.subplots(figsize=(9, max(4, 0.5 * len(keys) + 2)))
    ax2.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(vals)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("standardized coefficient  (weight × feature std)  →  contribution to value", fontsize=11)
    ax2.set_title(
        "What Drives the Heuristic's Value Estimate\n(green favors the player, red works against)",
        fontsize=13, fontweight="bold",
    )
    ax2.margins(y=0.01)
    fig2.tight_layout()
    out2 = args.out_dir / f"regression_importance_{tag}.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
