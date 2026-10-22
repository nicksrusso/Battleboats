"""Play a checkpoint against the random or scripted baseline.

    poetry run python scripts/eval_agent.py \\
        --checkpoint runs/selfplay/latest.pt --opponent scripted --games 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleboats.training.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("random", "scripted"), default="random")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--map-count", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto. ROCm reports the AMD GPU as cuda.",
    )
    args = parser.parse_args()
    stats = evaluate_checkpoint(
        args.checkpoint,
        opponent=args.opponent,
        games=args.games,
        width=args.width,
        height=args.height,
        map_count=args.map_count,
        max_turns=args.max_turns,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
