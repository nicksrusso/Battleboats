#!/usr/bin/env python3
"""Evaluate a value-net checkpoint on a cache split against the gate baselines.

    poetry run python scripts/eval_value.py --checkpoint runs/value_net_v0.5.0/<ts>/best.pt \
        --cache runs/value_harvest_48x24_fc_v0.5.0/value_cache_k64 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from battleboats.training.value_data import ValueCache  # noqa: E402
from battleboats.training.value_eval import Baselines, evaluate_split, format_report  # noqa: E402
from battleboats.training.value_net import load_value_checkpoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--baseline-fit-rows", type=int, default=60000)
    ap.add_argument("--out", help="write the full JSON result here")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    model, payload = load_value_checkpoint(a.checkpoint)
    cache = ValueCache(a.cache)
    bl = Baselines(cache, cache.split_rows("train"), max_fit_rows=a.baseline_fit_rows)
    res = evaluate_split(model, cache, a.split, baselines=bl)
    print(format_report(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0 if res.get("gate", {}).get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
