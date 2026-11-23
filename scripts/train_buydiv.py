#!/usr/bin/env python3
"""Fine-tune a PPO warm start in the buy-diversity sandbox (no opponent).

    PYTHONPATH=<repo> python scripts/train_buydiv.py \
        --config configs/sandbox_buydiv.yaml \
        --run-dir runs/sandbox_buydiv_v0.4.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from battleboats.training.buydiv import (BuyDivConfig, load_buydiv_config,
                                         run_buydiv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--decisions", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--p-start", type=float, default=None)
    parser.add_argument("--p-end", type=float, default=None)
    parser.add_argument("--coverage-coef", type=float, default=None)
    parser.add_argument("--anchor-coef", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args()

    cfg = load_buydiv_config(args.config) if args.config else BuyDivConfig()
    if args.init is not None:
        cfg.init_checkpoint = str(args.init)
    if args.run_dir is not None:
        cfg.run_dir = str(args.run_dir)
    if args.updates is not None:
        cfg.updates = int(args.updates)
    if args.decisions is not None:
        cfg.decisions_per_update = int(args.decisions)
    if args.horizon is not None:
        cfg.horizon_turns = int(args.horizon)
    if args.p_start is not None:
        cfg.p_start = float(args.p_start)
    if args.p_end is not None:
        cfg.p_end = float(args.p_end)
    if args.coverage_coef is not None:
        cfg.coverage_coef = float(args.coverage_coef)
    if args.anchor_coef is not None:
        cfg.anchor_coef = float(args.anchor_coef)
    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.eval_every is not None:
        cfg.eval_every = int(args.eval_every)
    if args.eval_episodes is not None:
        cfg.eval_episodes = int(args.eval_episodes)
    if args.max_minutes is not None:
        cfg.max_minutes = float(args.max_minutes)
    summary = run_buydiv(cfg)
    print(
        f"stop={summary['stop_reason']} updates={summary['updates_run']} "
        f"min={summary['elapsed_min']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
