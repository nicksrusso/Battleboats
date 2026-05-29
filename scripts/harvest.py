"""Self-play data harvester — collects (phi, MC-return) rows for V_θ regression.

Runs N games at a fixed MCTS iteration count using the same unified-work-queue
parallel dispatch as `trade_study.py`, filters out truncated games (no clean
Monte Carlo return is available for those), and flattens the surviving game
trajectories into per-state training rows.

Each row is a dict with:
    phi:    feature vector (one entry per FEATURE_KEYS), from MCTS-player POV
    target: Monte Carlo return from MCTS-player POV
            (+1 if MCTS won, -1 if random won)
    game_idx, seed, iterations, step, turn, actor: provenance / debug fields

The output JSONL streams to disk as games complete, so a killed run still
leaves the partial dataset usable. A sidecar `<...>_meta.json` records the
run config and final tallies.

Usage:
    poetry run python scripts/harvest.py
    poetry run python scripts/harvest.py --num-games 100 --iterations 50 \\
        --workers 16 --max-turns 400
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Sibling import — works because `python scripts/harvest.py` puts scripts/
# on sys.path. _worker_run_game is the picklable top-level wrapper around
# _play_one_game, identical to what trade_study.py uses.
from benchmark_godmode_mcts import MAP_JSON, _worker_run_game

DEFAULT_OUTPUT_DIR = Path("/home/nick/Desktop/repos/Battleboats/runs/harvests")
DEFAULT_NUM_GAMES = 100
DEFAULT_ITERATIONS = 50


def _flatten_to_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert one terminated-game record into a list of training rows.

    Emits TWO rows per trajectory step — one from each player's POV —
    so the regression sees both winning and losing examples from the
    same trajectory. Without this symmetry the dataset has no target
    variance (every kept game has MCTS winning, every row would have
    target=+1, and the regression collapses to "predict the mean").

    Returns empty list for truncated games (winner is None), since
    those have no clean MC return for either side. Skips per-step
    entries where either phi is empty — those are terminal-state or
    dead-agent entries with no feature signal.

    Targets are anchored to absolute player id:
        target_p0 = +1 if player 0 won, -1 if player 1 won
        target_p1 = -target_p0

    The MCTS-player identity is recorded on the row (via
    `mcts_player_id`) but not used by the regression — it's pure
    provenance for downstream filtering / debug.
    """
    winner = record["winner"]
    if winner is None:
        return []

    target_p0 = 1.0 if winner == 0 else -1.0
    target_p1 = -target_p0

    rows: List[Dict[str, Any]] = []
    for step in record["trajectory"]:
        phi_p0 = step.get("phi_p0")
        phi_p1 = step.get("phi_p1")
        if not phi_p0 or not phi_p1:
            continue
        base = {
            "game_idx": record["game_idx"],
            "seed": record["seed"],
            "iterations": record["iterations"],
            "mcts_player_id": record["mcts_player_id"],
            "step": step["step"],
            "turn": step["turn"],
            "actor": step["actor"],
        }
        # One row per perspective. `perspective` field marks which player
        # the phi/target pair refers to so post-hoc analysis can filter
        # (e.g. "only learn from MCTS-perspective rows" if desired).
        rows.append({**base, "perspective": 0, "phi": phi_p0, "target": target_p0})
        rows.append({**base, "perspective": 1, "phi": phi_p1, "target": target_p1})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest (phi, MC-return) data for V_theta regression.")
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES, help="Total games to run.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="MCTS iterations per move.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel game workers. Default: all logical CPUs incl. SMT.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=400,
        help="Env turn cap. Larger than the benchmark default to convert near-wins into terminations.",
    )
    parser.add_argument("--step-budget", type=int, default=10_000, help="Per-game step cap.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; per-game seed = base + game_idx.")
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=Path("runs/scenarios/scenarios_500.json"),
        help="JSON of pre-generated scenarios (default: scenarios_500). Cycles through them.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write JSONL + meta.")
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Both players use MCTS with the current heuristic weights "
        "(symmetric self-play). When unset, MCTS plays vs. random_agent.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.output_dir / f"harvest_{timestamp}.jsonl"
    meta_path = args.output_dir / f"harvest_{timestamp}_meta.json"

    # Clamp workers — no point spawning more processes than there are jobs.
    workers = max(1, min(args.workers, args.num_games))

    # Load scenarios (cycles through scenarios_500.json for varied starts)
    scenarios = json.loads(args.scenarios_file.read_text())
    print(f"Loaded {len(scenarios)} scenarios for harvest.")

    # Build the global work list. All games at the same iteration count
    # but with distinct seeds + cycled scenarios.
    work_items: List[Dict[str, Any]] = []
    for game_idx in range(args.num_games):
        work_items.append(
            {
                "game_idx": game_idx,
                "map_json_path": None,  # overridden by scenario
                "seed": args.seed + game_idx,
                "scenario": scenarios[game_idx % len(scenarios)],
                "mcts_player_id": game_idx % 2,
                "iterations": args.iterations,
                "max_turns": args.max_turns,
                "step_budget": args.step_budget,
                "self_play": args.self_play,
                "verbose": False,
                "debug_plot": False,
                "debug_plot_mcts_only": False,
            }
        )

    mode_label = "self-play (MCTS vs MCTS)" if args.self_play else "vs random_agent"
    print(f"Harvest: {args.num_games} games at iter={args.iterations} using {len(scenarios)} scenarios [{mode_label}]")
    print(f"  max_turns:   {args.max_turns}")
    print(f"  workers:     {workers}")
    print(f"  scenarios:   {args.scenarios_file}")
    print(f"  output:      {jsonl_path}")
    print(f"  meta:        {meta_path}")
    print()

    overall_t0 = time.perf_counter()
    completed = 0
    kept_games = 0
    truncated_games = 0
    # In self-play, "MCTS won" is always true if anyone wins — so split by
    # seat (p0 vs p1) instead. In random-opponent mode, split by whether the
    # MCTS-occupying seat or the random-occupying seat won.
    p0_wins = 0
    p1_wins = 0
    mcts_wins = 0
    random_wins = 0
    total_rows = 0

    # Stream rows to disk as games finish — keeps memory flat regardless of
    # corpus size, and a killed run still leaves a usable partial JSONL.
    with open(jsonl_path, "w") as fout, mp.Pool(processes=workers) as pool:
        for record in pool.imap_unordered(_worker_run_game, work_items):
            completed += 1
            rows = _flatten_to_rows(record)

            mp_id = record["mcts_player_id"]
            winner = record["winner"]
            if winner is None:
                outcome = "truncated"
                truncated_games += 1
            elif args.self_play:
                outcome = f"p{winner} won"
                if winner == 0:
                    p0_wins += 1
                else:
                    p1_wins += 1
            elif winner == mp_id:
                outcome = "MCTS won"
                mcts_wins += 1
            else:
                outcome = "random won"
                random_wins += 1

            if rows:
                kept_games += 1
                total_rows += len(rows)
                for row in rows:
                    fout.write(json.dumps(row, default=str) + "\n")
                fout.flush()

            print(
                f"  [{completed:3d}/{args.num_games}] game_idx={record['game_idx']:3d}  "
                f"seed={record['seed']:3d}  player_{mp_id}  "
                f"turn={record['final_turn']:3d}  steps={record['steps']:5d}  "
                f"-> {outcome:10s}  ({record['wall_time_s']:.1f}s, {len(rows):5d} rows)",
                flush=True,
            )

    overall_elapsed = time.perf_counter() - overall_t0

    results_block: Dict[str, Any] = {
        "completed_games": completed,
        "kept_games": kept_games,
        "truncated_games": truncated_games,
        "total_rows": total_rows,
        "total_wall_time_s": round(overall_elapsed, 1),
    }
    if args.self_play:
        results_block["p0_wins"] = p0_wins
        results_block["p1_wins"] = p1_wins
    else:
        results_block["mcts_wins"] = mcts_wins
        results_block["random_wins"] = random_wins

    metadata = {
        "config": {
            "num_games": args.num_games,
            "iterations": args.iterations,
            "self_play": args.self_play,
            "max_turns": args.max_turns,
            "step_budget": args.step_budget,
            "seed": args.seed,
            "workers": workers,
            "scenarios_file": str(args.scenarios_file),
            "scenarios_used": len(scenarios),
            "map_json": None,  # per-scenario
        },
        "results": results_block,
        "jsonl_path": str(jsonl_path),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print(f"Harvest complete in {overall_elapsed:.1f}s  [{mode_label}]")
    print(f"  games:     completed={completed}  kept={kept_games}  truncated={truncated_games}")
    if args.self_play:
        print(f"  outcomes:  p0_wins={p0_wins}  p1_wins={p1_wins}")
    else:
        print(f"  outcomes:  mcts_wins={mcts_wins}  random_wins={random_wins}")
    print(f"  rows:      {total_rows}")
    print(f"  jsonl:     {jsonl_path}")
    print(f"  meta:      {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
