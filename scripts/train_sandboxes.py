"""Train every sandbox at once as a warm-up for the full-game pointer net.

No skill one-hot. ``latest.pt`` loads in self-play with no shape change:

    poetry run python scripts/train_sandboxes.py --config configs/sandboxes.yaml --device cpu
    poetry run python scripts/train_selfplay.py --config configs/selfplay.yaml \\
        --init-checkpoint runs/sandboxes/latest.pt --device cpu

``<run_dir>/<skill>/size_<N>.pt`` and ``<skill>/final.pt`` mark the size gates.
``<run_dir>/latest.pt`` is the resume point and the handoff weights.
``<run_dir>/summary.json`` holds per-sandbox, per-size success rates.

Weights & Biases is off unless ``wandb: true`` or ``BATTLEBOATS_WANDB=1``.
A missing install or login does not stop training. Project defaults to
``battleboats-skill-pipeline``.

Exit status 2 when a size misses its gate and ``allow_unmet`` is off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleboats.training.sandbox_curriculum import load_sandbox_config, run_sandboxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument("--allow-unmet", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_sandbox_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.allow_unmet:
        cfg.allow_unmet = True
    if args.init_checkpoint is not None:
        cfg.init_checkpoint = args.init_checkpoint
    if args.wandb:
        cfg.wandb = True

    result = run_sandboxes(cfg, resume=not args.no_resume)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
