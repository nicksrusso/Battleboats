"""Plot heuristic weight trajectories across tuning iterations.

The heuristic-tuning loop is a fixed-point iteration (fit a value model on play
guided by the previous model). This plot is the convergence diagnostic: one line
per feature, x = iteration, y = STANDARDIZED coefficient (weight × feature_std).

Why standardized: raw weights span orders of magnitude (0.00008 → 0.07), so they
are unreadable on one axis; multiplying by a common reference feature_std puts
every feature in comparable value-units, and the only thing varying across
iterations is the weight itself.

Read it as:
  - lines flattening                       -> converged (stop iterating)
  - a see-saw between collinear features    -> near-converged but noisy
    (e.g. has_landing <-> landing_pressure)
  - a line drifting monotonically to an
    implausible sign                        -> noise-chasing (stop, keep last sane set)

    poetry run python scripts/plot_weight_history.py \\
        runs/weights/v0_handtuned.json runs/weights/v4.json runs/weights/v5.json

Pass weight JSONs in iteration order. The reference feature_std is taken from the
LAST file's source harvest (the most recent data); override with --ref-source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_train import load_rows, to_xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="+", help="Weight JSONs in iteration order.")
    parser.add_argument("--ref-source", type=Path, default=None,
                        help="Harvest dir/file for the reference feature_std (default: last file's source_files).")
    parser.add_argument("--out", type=Path, default=Path("docs/weight_history.png"))
    args = parser.parse_args()

    cfgs = [json.loads(p.read_text()) for p in args.weights]
    labels = [p.stem for p in args.weights]

    # Canonical feature order = the LAST file's feature_keys (the current set);
    # earlier files missing a feature contribute 0 for it (they didn't fit it).
    keys = list(cfgs[-1]["feature_keys"])

    # Reference feature_std from the most recent harvest, so standardized
    # coefficients are comparable across iterations (only the weight varies).
    ref = args.ref_source
    if ref is None:
        ref = Path(cfgs[-1]["source_files"][0])
    try:
        rows = load_rows([ref], decisive_only=True)
        X, _ = to_xy(rows, keys)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        std_note = f"std from {Path(ref).name}"
    except Exception as e:  # noqa: BLE001 — fall back to raw weights if source missing
        print(f"  WARNING: could not load reference source ({e}); plotting RAW weights.")
        std = np.ones(len(keys))
        std_note = "raw weights (no reference std)"

    # standardized[iter, feature]
    M = np.array([[cfg["weights"].get(k, 0.0) * std[j] for j, k in enumerate(keys)]
                  for cfg in cfgs], dtype=float)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels) + 4), 6.5))
    cmap = plt.get_cmap("tab10")
    for j, k in enumerate(keys):
        ax.plot(x, M[:, j], marker="o", lw=2, color=cmap(j % 10), label=k.replace("_", " "))
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("tuning iteration")
    ax.set_ylabel("raw weight" if "raw" in std_note
                  else "standardized weight  (contribution to value estimate)")
    ax.set_title("How the Heuristic's Weights Settle Across Tuning Iterations\n"
                 "standardized contributions — lines flattening out indicate convergence",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, title="feature")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
