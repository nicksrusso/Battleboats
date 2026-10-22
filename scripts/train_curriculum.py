"""Train the skill curriculum (merchant, landing, combat, builder, then full game).

    poetry run python scripts/train_curriculum.py --config configs/curriculum.yaml
    poetry run python scripts/train_curriculum.py --config configs/curriculum_smoke.yaml

Exit status 2 when a stage misses its gate and ``allow_unmet`` is off.
Checkpoints are plain ``{model_state, model_config}`` files. A later
quality-diversity run can load ``latest.pt`` without a ratings table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleboats.training.curriculum import load_curriculum_config, run_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Override the config device: cpu, cuda, cuda:N, or auto.",
    )
    parser.add_argument(
        "--allow-unmet",
        action="store_true",
        help="Hand the checkpoint on even if a stage misses its gate.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_curriculum_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.allow_unmet:
        cfg.allow_unmet = True
    if args.init_checkpoint is not None:
        cfg.init_checkpoint = args.init_checkpoint
    if args.wandb:
        cfg.wandb = True

    result = run_curriculum(cfg, resume=not args.no_resume)
    summary = {k: v for k, v in result.items() if k != "state"}
    print(json.dumps(summary, default=str))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
