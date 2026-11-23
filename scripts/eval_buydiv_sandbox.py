#!/usr/bin/env python3
"""Unmasked buy-diversity sandbox validation (sampled policy, p=0).

Runs a checkpoint for ``--episodes`` games (default 128, even split across
16x8 and 20x10), reports buy counts, gold share, diversity mean with a Wald
95% CI, and min/max type counts. Optionally runs the same protocol on a
baseline (frozen v0.4.0) for comparison.

    PYTHONPATH=<repo> python scripts/eval_buydiv_sandbox.py \
        --checkpoint runs/sandbox_buydiv_v0.4.0/update_300.pt \
        --baseline runs/checkpoints/v0.4.0_8ad2530.pt \
        --output-dir runs/sandbox_buydiv_v0.4.0/eval_unmasked
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from battleboats.training.buydiv import (BUYDIV_HULLS, BuyDivConfig, _env_for,
                                         run_unmasked_eval,
                                         summarize_validation)
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.manifest import append_eval
from battleboats.training.policy import set_dropout


def _print_report(label: str, summary: Dict[str, Any]) -> None:
    div = summary["diversity"]
    print(f"=== {label} ===", flush=True)
    print(
        f"  episodes={summary['episodes']}  "
        f"diversity mean={div['mean']:.3f}  95% CI [{div['lo']:.3f}, {div['hi']:.3f}]  "
        f"std={div['std']:.3f}  range=[{div['min']:.3f}, {div['max']:.3f}]",
        flush=True,
    )
    print(f"  buy_counts={summary['buy_counts']}", flush=True)
    buy_mean = {k: round(summary["buy_mean"][k], 2) for k in BUYDIV_HULLS}
    print(f"  buy_mean/ep={buy_mean}", flush=True)
    share = {k: round(summary["gold_share"][k], 3) for k in BUYDIV_HULLS}
    print(f"  gold_share={share}", flush=True)
    print(
        f"  type totals min={summary['type_counts_min']} {summary['type_counts_min_types']}  "
        f"max={summary['type_counts_max']} {summary['type_counts_max_types']}  "
        f"max/min={summary['type_counts_ratio_max_min']}  "
        f"mean={summary['type_counts_mean']:.1f}",
        flush=True,
    )
    emin, emax = summary["episode_type_min"], summary["episode_type_max"]
    print(
        f"  per-episode type min mean={emin['mean']:.2f}  "
        f"type max mean={emax['mean']:.2f}  "
        f"roughly_even={summary['roughly_even']}",
        flush=True,
    )


def _run_one(
    path: Path, label: str, cfg: BuyDivConfig, n: int, seed: int
) -> Dict[str, Any]:
    device = torch.device("cpu")
    policy, _ = load_checkpoint(path, device)
    set_dropout(policy, 0.0)
    policy.eval()
    sizes = [(int(a), int(b)) for a, b in cfg.sizes]
    envs = {(w, h): _env_for(cfg, w, h) for w, h in sizes}
    rng = random.Random(seed)
    torch.manual_seed(seed)
    records, _ = run_unmasked_eval(
        policy, cfg, envs, rng, device, n_episodes=int(n), seed0=int(seed)
    )
    for env in envs.values():
        env.close()
    summary = summarize_validation(records)
    summary["label"] = label
    summary["checkpoint"] = str(path)
    summary["seed"] = int(seed)
    _print_report(label, summary)
    # Keep raw per-episode scores and counts for the written payload, not the
    # full token rollouts.
    summary["episodes_raw"] = [
        {
            "diversity": r["diversity"],
            "purchases": r["purchases"],
            "gold_share": r["gold_share"],
            "width": r["width"],
            "height": r["height"],
            "turn": r["turn"],
        }
        for r in records
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", type=str, default="buydiv")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--baseline-label", type=str, default="v0.4.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = BuyDivConfig(
        run_dir=str(args.run_dir or args.output_dir),
        horizon_turns=int(args.horizon),
        sizes=[[16, 8], [20, 10]],
        coverage_coef=0.25,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "episodes": int(args.episodes),
        "horizon_turns": int(args.horizon),
        "sizes": cfg.sizes,
        "seed": int(args.seed),
        "mask_p": 0.0,
        "sampled": True,
    }
    payload["checkpoint"] = _run_one(
        args.checkpoint, args.label, cfg, args.episodes, args.seed
    )
    if args.baseline is not None:
        payload["baseline"] = _run_one(
            args.baseline, args.baseline_label, cfg, args.episodes, args.seed
        )
    # Drop bulky raw lists from the short summary file.
    short = dict(payload)
    if "checkpoint" in short:
        short["checkpoint"] = {
            k: v for k, v in short["checkpoint"].items() if k != "episodes_raw"
        }
    if "baseline" in short:
        short["baseline"] = {
            k: v for k, v in short["baseline"].items() if k != "episodes_raw"
        }
    (out / "summary.json").write_text(json.dumps(short, indent=2))
    (out / "episodes.json").write_text(json.dumps(payload, indent=2))
    append_eval(
        short.get("checkpoint") or short,
        checkpoint=args.checkpoint,
        name=args.label,
    )
    if args.baseline is not None:
        append_eval(
            short.get("baseline") or {},
            checkpoint=args.baseline,
            name=args.baseline_label,
        )
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
