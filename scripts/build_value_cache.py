#!/usr/bin/env python3
"""Build the value-net memmap cache from a value harvest (scripts/harvest_value.py).

    poetry run python scripts/build_value_cache.py --config configs/value_net_v0.5.0.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from battleboats.training.value_data import build_value_cache  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/value_net_v0.5.0.yaml")
    ap.add_argument("--harvest-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--k-pool", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-games", type=int)
    a = ap.parse_args(argv)
    d = yaml.safe_load(Path(a.config).read_text())["data"]
    t0 = time.time()
    meta = build_value_cache(
        a.harvest_dir or d["harvest_dir"],
        a.cache_dir or d["cache_dir"],
        k_pool=a.k_pool or d["k_pool"],
        seed=d.get("seed", 0),
        val_frac=d["val_frac"],
        test_frac=d["test_frac"],
        workers=a.workers or d.get("cache_workers", 2),
        max_games=a.max_games,
        weight_floor=float(d.get("sample_weight_floor", 1.0)),
        weight_power=float(d.get("sample_weight_power", 1.0)),
    )
    meta["build_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
