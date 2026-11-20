"""Full-game PPO against the last few snapshots, plus a random-opponent mix.

    poetry run python scripts/train_selfplay.py --config configs/selfplay.yaml
    poetry run python scripts/train_selfplay.py \\
        --config configs/selfplay_smoke.yaml \\
        --init-checkpoint runs/curriculum_smoke/latest.pt

The opponent ring is uniform over recent checkpoints. There is no Elo, no
PFSP, and no exploiter. ``RecentSnapshots.sample`` is the place a later
quality-diversity archive would plug in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleboats.training.selfplay import load_selfplay_config, run_selfplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Override the config device: cpu, cuda, cuda:N, or auto.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_selfplay_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.init_checkpoint is not None:
        cfg.init_checkpoint = args.init_checkpoint
    if args.wandb:
        cfg.wandb = True

    result = run_selfplay(cfg, resume=not args.no_resume)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
