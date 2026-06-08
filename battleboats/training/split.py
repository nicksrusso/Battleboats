"""By-game train/val split.

States within a single game share the same terminal label (modulo
perspective sign), so a random row-level split would put near-identical
labels on both sides of the train/val boundary and produce inflated
held-out numbers. Splitting at the game level is the only honest
evaluation protocol for this dataset.

Splits are deterministic given the seed and persistable to JSON so
every downstream model (linear baseline, MLP, transformer) is evaluated
on the exact same held-out games. Comparisons across models are only
meaningful when the val set is identical.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

# game_idx is always the FIRST key of a harvest row (harvest.py builds each row
# from a base dict that starts with it), so a prefix match pulls it out without
# json.loads-ing the rest of the line. On a token harvest that's the difference
# between seconds and many minutes — every row carries a ~100 KB token array we
# don't need just to read one integer.
_GAME_IDX_PREFIX = re.compile(r'^\s*\{\s*"game_idx"\s*:\s*(-?\d+)')


def _extract_game_idx(line: str) -> int:
    """game_idx from a row, cheaply. Falls back to a full parse if the prefix
    layout ever changes, so correctness never depends on the fast path."""
    m = _GAME_IDX_PREFIX.match(line)
    if m:
        return int(m.group(1))
    return json.loads(line)["game_idx"]


def read_game_ids(jsonl_path: str | Path, max_games: Optional[int] = None) -> Set[int]:
    """Return the set of distinct game_idx values in a harvest.

    Accepts either form of harvest output:
      - a DIRECTORY of per-game shards (`game_<idx>.jsonl`): the game ids ARE
        the filenames, so this reads ZERO content — instant even on a huge
        corpus.
      - a single .jsonl FILE (legacy): streams line-by-line, extracting only
        game_idx (no full JSON parse of the token arrays — see
        _extract_game_idx).

    If `max_games` is given, stops once that many distinct games are seen. For a
    file, games are contiguous so this reads only the FRONT (the lever for not
    choking on a 25 GB harvest); for a shard dir it just takes that many
    filenames. Either way the subset is effectively arbitrary (fine for a
    memory-bounded experiment). Omit it to take everything.
    """
    path = Path(jsonl_path)
    ids: Set[int] = set()

    if path.is_dir():
        for shard in sorted(path.glob("game_*.jsonl")):
            stem = shard.stem
            if not stem.startswith("game_"):
                continue
            try:
                ids.add(int(stem[len("game_") :]))
            except ValueError:
                continue
            if max_games is not None and len(ids) >= max_games:
                break
        return ids

    with open(path) as f:
        for line in f:
            ids.add(_extract_game_idx(line))
            if max_games is not None and len(ids) >= max_games:
                break
    return ids


def split_by_game(
    game_ids: Iterable[int],
    val_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[Set[int], Set[int]]:
    """Deterministic train/val partition of game_idxs.

    Uses a local RNG (not the global one) so calling this never
    perturbs other randomness in the program.

    With val_frac=0.2 and 100 games you get 80 train / 20 val. The
    smallest val set we tolerate is 1 game — if val_frac rounds to
    zero we still split off one game so val is never empty.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    ids = sorted(game_ids)  # sort first → seed alone determines order
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val = set(ids[:n_val])
    train = set(ids[n_val:])
    return train, val


def save_split(
    path: str | Path,
    train_ids: Set[int],
    val_ids: Set[int],
    harvest_path: str | Path,
    seed: int,
    val_frac: float,
) -> None:
    """Persist a split + its provenance to JSON.

    Sorted lists in the file (not sets) so it diffs cleanly when
    re-saved and so the JSON ordering is deterministic.
    """
    payload: Dict[str, object] = {
        "harvest_path": str(harvest_path),
        "seed": seed,
        "val_frac": val_frac,
        "n_train_games": len(train_ids),
        "n_val_games": len(val_ids),
        "train_game_ids": sorted(train_ids),
        "val_game_ids": sorted(val_ids),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(path: str | Path) -> Tuple[Set[int], Set[int], Dict[str, object]]:
    """Reload a previously saved split. Returns (train_ids, val_ids, metadata)."""
    with open(path) as f:
        payload = json.load(f)
    train = set(payload["train_game_ids"])
    val = set(payload["val_game_ids"])
    if train & val:
        # If this ever fires, the file on disk is corrupt — refuse
        # rather than silently train on contaminated data.
        raise ValueError(f"Split file has overlapping train/val ids: {sorted(train & val)[:10]}")
    return train, val, payload


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create or inspect a by-game train/val split.")
    parser.add_argument("--harvest", type=Path, required=True, help="Path to harvest JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the split JSON.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Cap total games in the split (train+val). Bounds downstream RAM "
        "when token mode loads each game's tokens — and lets the scan early-exit "
        "instead of reading the whole harvest. Omit to use every game.",
    )
    args = parser.parse_args()

    ids = read_game_ids(args.harvest, max_games=args.max_games)
    train, val = split_by_game(ids, val_frac=args.val_frac, seed=args.seed)
    save_split(args.out, train, val, args.harvest, args.seed, args.val_frac)

    print(f"Harvest:        {args.harvest}")
    print(f"Max games:      {args.max_games if args.max_games is not None else 'all'}")
    print(f"Total games:    {len(ids)}")
    print(f"Train games:    {len(train)}  ({sorted(train)})")
    print(f"Val games:      {len(val)}    ({sorted(val)})")
    print(f"Overlap:        {len(train & val)}  (must be 0)")
    print(f"Wrote split to: {args.out}")


if __name__ == "__main__":
    _cli()
