"""Harvest full self-play games from a local llama-server.

The model is not downloaded or started here. Start llama.cpp first, then:

    poetry run python scripts/llm_selfplay.py --games 10 --seed 0
"""

from __future__ import annotations

import argparse

from battleboats.llm.play import run_games


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM vs LLM battleboats self-play logger."
    )
    parser.add_argument("--games", type=int, default=1, help="Number of full games.")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="llama-server base URL (OpenAI /v1/chat/completions).",
    )
    parser.add_argument("--output-dir", default="runs/llm_selfplay")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First game seed; later games add the index.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=200, help="Round cap (env truncation)."
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--horizon", type=int, default=4, help="Interaction lookahead in turns."
    )
    parser.add_argument(
        "--max-edges", type=int, default=20, help="Cap on the interactions block."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=48, help="Reply budget. Keep small."
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="HTTP timeout per reply, seconds."
    )
    parser.add_argument(
        "--model", default="inkling", help="Model field sent to llama-server."
    )
    parser.add_argument(
        "--map", default=None, help="Map JSON. Default: core/config/map.json."
    )
    args = parser.parse_args()
    run_games(
        games=args.games,
        server_url=args.server_url,
        output_dir=args.output_dir,
        seed=args.seed,
        max_turns=args.max_turns,
        temperature=args.temperature,
        map_path=args.map,
        horizon=args.horizon,
        max_edges=args.max_edges,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
