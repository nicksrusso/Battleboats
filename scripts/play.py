"""Launch the local Battleboats web UI for a game against an agent.

    poetry run python scripts/play.py --map 16x8 --opponent tactical --seed 0

Opens a local HTTP server (127.0.0.1) with an interactive HTML5 Canvas grid,
strict fog-of-war per config/visibilityMatrix.csv, ship/port actions, and
combat/turn logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer the repo containing this script (worktrees included).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from battleboats.ui.maps import list_map_options  # noqa: E402
from battleboats.ui.server import serve  # noqa: E402
from battleboats.ui.session import PlaySession  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        default="16x8",
        help="Map preset (16x8, 24x12, 32x16, 48x24, small, training, medium, large), "
        "WxH, pool map name, or path to a .json map.",
    )
    parser.add_argument(
        "--opponent",
        choices=("tactical", "scripted", "random", "checkpoint"),
        default=None,
        help="Opponent agent (default: tactical; or checkpoint if --checkpoint is set).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional policy checkpoint (.pt) for opponent=checkpoint.",
    )
    parser.add_argument(
        "--side",
        "--human-side",
        dest="human_side",
        type=int,
        choices=(0, 1),
        default=0,
        help="Human player seat: 0 (first) or 1.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Game & map seed.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Turn limit (default: scaled from map home distance).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, cuda:N, or auto.",
    )
    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="List available map presets and discovered pool files, then exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_maps:
        for opt in list_map_options():
            print(f"{opt['value']:24s} {opt['label']}")
        return

    opponent = args.opponent
    if opponent is None:
        opponent = "checkpoint" if args.checkpoint is not None else "tactical"

    map_dir = args.checkpoint_dir / "play_maps"
    session = PlaySession(
        opponent=opponent,
        human_side=args.human_side,
        map_spec=args.map,
        seed=args.seed,
        max_turns=args.max_turns,
        map_dir=map_dir,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(
        f"Map: {session.map_name} ({session.engine.map.width}x{session.engine.map.height}) · "
        f"Opponent: {session.opponent_label} · Seat: Player {session.human_side} · "
        f"Max turns: {session.max_turns}",
        flush=True,
    )
    serve(
        args.host,
        args.port,
        args.checkpoint_dir,
        args.device,
        initial_session=session,
        auto_play=True,
    )


if __name__ == "__main__":
    main()
