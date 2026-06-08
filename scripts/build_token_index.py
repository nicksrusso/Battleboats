"""Build a compact token index for a harvest so HarvestDataset can SKIP the slow
per-line JSON scan (~40 min over the full harvest) on every training run.

The scan's expensive part is json.loads-ing ~140 GB of token-laden text just to
extract a few small fields per row. This runs that ONCE and serializes the result
— per-row (shard, byte_offset, label, game_id, step) — to a tiny .npz. Training
then loads the .npz in seconds; token bytes stay in the original shards and are
read lazily at __getitem__ time. Re-run only when the harvest changes.

Build over the WHOLE harvest (no game filter) so one index serves any split — the
dataset filters to its split's game_idxs in memory at load.

    # BC index (expert action labels):
    poetry run python scripts/build_token_index.py \\
        --harvest runs/harvests/harvest_20260605_111638 \\
        --out runs/indexes/harvest_20260605_111638_bc.npz --bc

    # Value index (token regression):
    poetry run python scripts/build_token_index.py \\
        --harvest runs/harvests/<dir> --out runs/indexes/<name>.npz \\
        --target-key mcts_root_value
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from battleboats.training.dataset import HarvestDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Where to write the .npz index.")
    p.add_argument("--bc", action="store_true", help="BC mode (expert action labels). Omit for value mode.")
    p.add_argument("--target-key", default="target", choices=["target", "mcts_root_value"])
    args = p.parse_args()

    t0 = time.time()
    # game_idxs=None -> scan every game; load_tokens=True so token rows are indexed.
    ds = HarvestDataset(args.harvest, game_idxs=None, bc=args.bc, load_tokens=True, target_key=args.target_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_index(args.out)
    print(
        f"scanned {len(ds)} rows in {time.time() - t0:.0f}s  "
        f"(bc={args.bc}, target_key={args.target_key}, token_dim={ds.token_dim})\n"
        f"wrote index -> {args.out}"
    )


if __name__ == "__main__":
    main()
