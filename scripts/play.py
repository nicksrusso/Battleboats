"""Launch the local Battleboats web UI for a game against an agent.

    poetry run python scripts/play.py --map 16x8 --opponent tactical --seed 0
    poetry run python scripts/play.py --map 16x8 --opponent scripted --seed 0
    poetry run python scripts/play.py --map 32x16 --opponent scripted_agent --seed 0
    poetry run python scripts/play.py --map 16x8 --opponent inkling --seed 0
    poetry run python scripts/play.py --map 16x8 --opponent llm --model thinkingmachines/inkling --reasoning high
    poetry run python scripts/play.py --map 16x8 --opponent inkling --reasoning max

Opens a local HTTP server (127.0.0.1) with an interactive HTML5 Canvas grid,
strict fog-of-war per config/visibilityMatrix.csv, ship/port actions, and
combat/turn logs. An Inkling / LLM opponent needs OPENROUTER_API_KEY in the
environment or a repo-root .env file (never committed). LLM turns run in the
background; the page polls and shows "Inkling thinking…". Every game is
logged incrementally to runs/human_games/<timestamp>_<map>_<opponent>_seed<N>.jsonl
(the path is printed at startup and on the game-over screen).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer the repo containing this script (worktrees included).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from battleboats.llm.openrouter import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    INKLING_MODEL_ID,
)
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
        choices=("tactical", "scripted_agent", "fog_commander", "scripted", "random", "checkpoint", "llm", "inkling"),
        default=None,
        help="Opponent agent (default: tactical; scripted_agent is the fog-limited doctrine agent; "
        "fog_commander is a deprecated alias for scripted_agent; "
        "scripted is the full-information "
        "baseline; checkpoint if --checkpoint is set; inkling is an alias for llm + "
        "the Inkling flagship at high reasoning). You always have fog of war.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional policy checkpoint (.pt) for opponent=checkpoint.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model id for --opponent llm/inkling "
        f"(default: {INKLING_MODEL_ID}).",
    )
    parser.add_argument(
        "--reasoning",
        default=DEFAULT_REASONING,
        choices=("max", "high", "medium", "low", "minimal", "none"),
        help="OpenRouter reasoning.effort (default: high).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per LLM decision, seconds (default: 600).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Completion budget per LLM decision, including reasoning tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM sampling temperature (default: 0.3, same as the harness).",
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
        if args.checkpoint is not None:
            opponent = "checkpoint"
        else:
            opponent = "tactical"

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
        llm_model=args.model,
        llm_reasoning=args.reasoning,
        llm_timeout=args.timeout,
        llm_max_tokens=args.max_tokens,
        llm_temperature=args.temperature,
    )
    print(
        f"Map: {session.map_name} ({session.engine.map.width}x{session.engine.map.height}) · "
        f"Opponent: {session.opponent_label} · Seat: Player {session.human_side} · "
        f"Max turns: {session.max_turns}",
        flush=True,
    )
    print(f"Game log: {session.log_path}", flush=True)
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
