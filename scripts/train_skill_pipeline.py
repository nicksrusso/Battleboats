"""Train one policy on every sandbox, then on the full game with skill refreshers.

    poetry run python scripts/train_skill_pipeline.py --config configs/skill_pipeline.yaml
    poetry run python scripts/train_skill_pipeline.py --config configs/skill_pipeline_smoke.yaml

The policy to play is ``<run_dir>/final.pt`` (default ``runs/skill_pipeline/final.pt``).
``latest.pt`` is the resume point and matches ``final.pt`` when the run ends.

Exit status 2 when a stage misses its gate and ``allow_unmet`` is off.
Weights & Biases is off unless the config sets ``wandb: true`` or the
environment variable ``BATTLEBOATS_WANDB`` is ``1``. A missing install or
login does not fail the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleboats.training.skill_pipeline import load_pipeline_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument(
        "--allow-unmet",
        action="store_true",
        help="Hand the checkpoint on even if a stage misses its gate.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.allow_unmet:
        cfg.allow_unmet = True
    if args.init_checkpoint is not None:
        cfg.init_checkpoint = args.init_checkpoint
    if args.wandb:
        cfg.wandb = True

    result = run_pipeline(cfg, resume=not args.no_resume)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, default=str))
    print(f"[skill_pipeline] checkpoint {result['checkpoint']}")
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
