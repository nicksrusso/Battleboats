"""Local page to play a full game against random, scripted, or a checkpoint.

    poetry run python scripts/play_vs_agent.py --checkpoint-dir runs --port 8765

Open the printed URL. One game at a time. The scripted opponent sees the
whole board; a checkpoint uses the same fog and action masks as training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from battleboats.ui.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto. ROCm reports the AMD GPU as cuda.",
    )
    args = parser.parse_args()
    serve(args.host, args.port, args.checkpoint_dir, args.device)


if __name__ == "__main__":
    main()
