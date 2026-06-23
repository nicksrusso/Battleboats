"""Tiled heatmaps of the decisiveness sweep: decisive-rate and attack-rate over
the board x iterations grid, with the BC-harvest source cells marked.

Reads runs/sweep/decisiveness_results.json (written by decisiveness_sweep.py).
Rows = board size (y-tick also shows measured density, ships/1000 tiles);
cols = MCTS iterations. Each cell annotated with its value. The cells our
behavior-cloning data was harvested from are outlined so you can point to where
on the decisiveness landscape BC trained.

    poetry run python scripts/plot_decisiveness_sweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Where the cloned data came from: (board key, iters, label). 160x80 BC ran at
# iters=50 (on-grid); 64x32 BC ran at iters=250 (OFF the right edge of the sweep
# — marked on the nearest cell with a note). Conditions also differed in
# cash/max_turns, so these mark the REGION, not an identical setting.
BC_SOURCES = [
    ("160x80", 50, "BC 160×80 (0.26)"),
    ("64x32", 250, "BC 64×32 (0.73)"),
]


def _grid(results, metric, boards, iters):
    M = np.full((len(boards), len(iters)), np.nan)
    for i, (w, h) in enumerate(boards):
        for j, it in enumerate(iters):
            c = results["cells"].get(f"{w}x{h}|{it}")
            if c is not None:
                M[i, j] = c[metric]
    return M


def _heatmap(ax, M, boards, iters, density, title, cmap, fmt, vmin, vmax):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels([str(i) for i in iters])
    ax.set_yticks(range(len(boards)))
    ax.set_yticklabels([f"{w}×{h}\n({density.get(f'{w}x{h}', {}).get('ships_per_1000_tiles', float('nan')):.1f}/1k tiles)"
                        for (w, h) in boards], fontsize=9)
    ax.set_xlabel("MCTS iterations")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            frac = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="white" if frac < 0.5 else "black", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _mark_bc(ax, boards, iters):
    """Outline the BC-source cells (solid = on-grid, dashed = off-grid iters).
    Returns (handles, labels) for a shared figure legend (no on-cell text, which
    clips against the title/edges on a small grid)."""
    from matplotlib.patches import Patch

    row_of = {f"{w}x{h}": i for i, (w, h) in enumerate(boards)}
    handles, labels = [], []
    for board, it, label in BC_SOURCES:
        if board not in row_of:
            continue
        row = row_of[board]
        on_grid = it in iters
        col = iters.index(it) if on_grid else len(iters) - 1
        ls = "-" if on_grid else "--"
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                               edgecolor="#00e5ff", lw=3, ls=ls, zorder=5))
        handles.append(Patch(facecolor="none", edgecolor="#00e5ff", lw=3, ls=ls))
        labels.append(label if on_grid else f"{label}  (iters={it}, off-grid →)")
    return handles, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("runs/sweep/decisiveness_results.json"))
    p.add_argument("--out", type=Path, default=Path("docs/decisiveness_sweep.png"))
    args = p.parse_args()

    results = json.loads(args.results.read_text())
    boards = [tuple(b) for b in results["config"]["boards"]]
    iters = results["config"]["iters"]
    density = results.get("density", {})

    fig, (ax_d, ax_a) = plt.subplots(1, 2, figsize=(13, 5.5))

    M_dec = _grid(results, "decisive_rate", boards, iters)
    _heatmap(ax_d, M_dec, boards, iters, density,
             "Decisive Rate (1 − draw rate)", cmap="RdYlGn", fmt="{:.2f}", vmin=0.0, vmax=1.0)
    bc_handles, bc_labels = _mark_bc(ax_d, boards, iters)

    M_atk = _grid(results, "attack_rate", boards, iters)
    amax = float(np.nanmax(M_atk)) if np.isfinite(np.nanmax(M_atk)) else 1.0
    _heatmap(ax_a, M_atk, boards, iters, density,
             "Attack Rate (attacks / MCTS decision)", cmap="viridis", fmt="{:.3f}", vmin=0.0, vmax=max(amax, 1e-3))
    _mark_bc(ax_a, boards, iters)

    if bc_handles:
        fig.legend(bc_handles, bc_labels, loc="lower center", ncol=len(bc_handles),
                   fontsize=9, title="BC harvest source", frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Decisiveness & Aggression Across Board Size × Search Depth", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Self-play MCTS · v5 heuristic · cash=500 · max_turns=100 · 30 games/cell.  "
             "Outlined = where BC data was harvested (different cash/turns; region, not identical setting).",
             ha="center", fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
