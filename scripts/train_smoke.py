"""Smoke test for HarvestDataset — load a harvest, print summary stats, dump 10 rows.

Use this every time you produce a new harvest file to confirm shapes,
ranges, and feature ordering before any training runs.

Usage:
    poetry run python scripts/train_smoke.py \\
        --harvest runs/harvests/harvest_20260530_095655.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from battleboats.training.dataset import HarvestDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", type=Path, required=True, help="Path to harvest JSONL.")
    parser.add_argument("--n-print", type=int, default=10, help="How many rows to dump.")
    args = parser.parse_args()

    ds = HarvestDataset(args.harvest)
    n = len(ds)
    n_games = len(ds.unique_game_ids())
    n_features = ds.phi.shape[1]

    print(f"Loaded: {n} rows  |  {n_games} games  |  {n_features} features")
    print(f"Feature keys ({n_features}): {ds.feature_keys}")
    print()

    # Per-feature ranges — fast sanity check that no column is constant
    # or absurdly out of bounds (e.g. NaNs, inf, accidental scale issues).
    print("Per-feature stats (min / mean / max / std):")
    for i, key in enumerate(ds.feature_keys):
        col = ds.phi[:, i]
        print(f"  {key:30s}  {col.min():+10.3f}  {col.mean():+10.3f}  {col.max():+10.3f}  {col.std():10.3f}")
    print()

    # Target distribution: should be roughly balanced ±1 in self-play.
    targets = ds.targets
    pos = int((targets > 0).sum())
    neg = int((targets < 0).sum())
    zero = int((targets == 0).sum())
    print(f"Targets: +1={pos}  -1={neg}  0={zero}  (balance ratio: {pos / max(neg, 1):.3f})")
    print()

    # Dump N items as (idx, game_id, target, first-3-features).
    print(f"First {args.n_print} rows:")
    for i in range(min(args.n_print, n)):
        phi, target = ds[i]
        gid = int(ds.game_ids[i])
        head = ", ".join(f"{x:+.2f}" for x in phi[:3].tolist())
        print(
            f"  [{i:4d}] game={gid:4d}  target={target.item():+.0f}  "
            f"phi.shape={tuple(phi.shape)}  phi[:3]=[{head}]  dtype={phi.dtype}"
        )


if __name__ == "__main__":
    main()
