"""MCTS iteration-count trade study with a single unified work queue.

Builds (iterations × game_idx) work items for all settings up front and
dispatches them to one shared worker pool. This eliminates the dead time
between settings in the shell-script version: when a fast game finishes
on a worker, that worker immediately picks up the next job from the
*global* queue regardless of which iteration setting it belongs to.
Critical when game times vary 10× within a setting (a single slow game
no longer blocks the rest of the batch from starting the next setting).

Outputs:
  runs/benchmarks/trade_study_<timestamp>/iter_<N>.json — per-setting,
      same shape as the regular benchmark JSON.
  runs/benchmarks/trade_study_<timestamp>/summary.json — aggregated
      list of dicts: [{iter_count, num_wins, avg_runtime_s, ...}, ...].

Usage:
    poetry run python scripts/trade_study.py
    poetry run python scripts/trade_study.py \\
        --iterations 25 50 75 100 125 150 \\
        --games-per-setting 8 --workers 16
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

# Sibling import — works because `python scripts/trade_study.py` puts
# scripts/ on sys.path. Reusing _play_one_game / _worker_run_game keeps
# the actual game-running code in one place.
from benchmark_godmode_mcts import MAP_JSON, OUTPUT_DIR, _worker_run_game

DEFAULT_ITERATIONS = [25, 50, 75, 100, 125, 150]
DEFAULT_GAMES_PER_SETTING = 8


def _outcome_for(record: Dict[str, Any]) -> str:
    """Translate a per-game record's winner field into a short label."""
    mp_id = record["mcts_player_id"]
    winner = record["winner"]
    if winner == mp_id:
        return "MCTS won"
    if winner == (1 - mp_id):
        return "random won"
    return "truncated"


def _write_setting_json(
    out_dir: Path,
    iters: int,
    games: List[Dict[str, Any]],
    args: argparse.Namespace,
    overall_t0: float,
) -> None:
    """Flush the per-setting JSON to disk; called after each game completes.

    Same shape as benchmark_godmode_mcts.py's output, so existing tools
    that parse those files (jq, etc.) work unchanged. Overwrites the
    file each call with the current games-so-far so a killed run still
    leaves a usable partial log.
    """
    mcts_wins = sum(1 for g in games if g["winner"] == g["mcts_player_id"])
    random_wins = sum(1 for g in games if g["winner"] == (1 - g["mcts_player_id"]))
    truncated = sum(1 for g in games if g["winner"] is None)
    sorted_games = sorted(games, key=lambda g: g["game_idx"])
    summary = {
        "config": {
            "iterations": iters,
            "num_games": args.games_per_setting,
            "max_turns": args.max_turns,
            "step_budget": args.step_budget,
            "seed": args.seed,
            "workers": args.workers,
            "map_json": MAP_JSON,
        },
        "results": {
            "mcts_wins": mcts_wins,
            "random_wins": random_wins,
            "truncated": truncated,
            "completed_games": len(games),
            "total_wall_time_s": time.perf_counter() - overall_t0,
        },
        "games": sorted_games,
    }
    with open(out_dir / f"iter_{iters}.json", "w") as f:
        json.dump(summary, f, default=str)


def _print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    """Pretty-print the per-setting summary as a fixed-width table."""
    header = f"{'iter':>6s}  {'wins':>4s}  {'loss':>4s}  {'trunc':>5s}  {'avg_runtime_s':>14s}  {'avg_turns':>10s}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['iter_count']:>6d}  "
            f"{row['num_wins']:>4d}  "
            f"{row['num_losses']:>4d}  "
            f"{row['num_truncated']:>5d}  "
            f"{row['avg_runtime_s']:>14.1f}  "
            f"{row['avg_turns']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS iter-count trade study (unified work queue).")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=DEFAULT_ITERATIONS,
        help="List of MCTS iteration counts to sweep. Default: 25 50 75 100 125 150.",
    )
    parser.add_argument(
        "--games-per-setting",
        type=int,
        default=DEFAULT_GAMES_PER_SETTING,
        help="Games per iteration setting. Default: 8.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel game workers. Default: os.cpu_count() (= all logical CPUs incl. SMT).",
    )
    parser.add_argument("--max-turns", type=int, default=250, help="Env truncation turn cap.")
    parser.add_argument("--step-budget", type=int, default=50_000, help="Per-game step cap.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; per-game seed = base + game_idx.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"trade_study_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the global work list: one job per (iter, game_idx) pair. All jobs
    # go into one queue; workers consume the next-available job regardless of
    # which setting it belongs to, so a slow game in one setting doesn't
    # block progress on other settings.
    work_items: List[Dict[str, Any]] = []
    for iters in args.iterations:
        for game_idx in range(args.games_per_setting):
            work_items.append(
                {
                    "game_idx": game_idx,
                    "map_json_path": MAP_JSON,
                    "seed": args.seed + game_idx,
                    "mcts_player_id": game_idx % 2,
                    "iterations": iters,
                    "max_turns": args.max_turns,
                    "step_budget": args.step_budget,
                    "verbose": False,
                    "debug_plot": False,
                    "debug_plot_mcts_only": False,
                }
            )
    total_jobs = len(work_items)

    print(f"Trade study: {len(args.iterations)} settings × {args.games_per_setting} games = {total_jobs} jobs")
    print(f"  iterations: {args.iterations}")
    print(f"  workers:    {args.workers}")
    print(f"  out_dir:    {out_dir}")
    print()

    results_by_iters: Dict[int, List[Dict[str, Any]]] = {it: [] for it in args.iterations}
    overall_t0 = time.perf_counter()

    completed = 0
    with mp.Pool(processes=args.workers) as pool:
        for record in pool.imap_unordered(_worker_run_game, work_items):
            completed += 1
            iters = record["iterations"]
            results_by_iters[iters].append(record)

            print(
                f"  [{completed:3d}/{total_jobs}] iter={iters:4d}  game_idx={record['game_idx']:2d}  "
                f"seed={record['seed']:3d}  player_{record['mcts_player_id']}  "
                f"-> {_outcome_for(record):10s}  "
                f"(steps={record['steps']}, turn={record['final_turn']}, "
                f"{record['wall_time_s']:.1f}s)",
                flush=True,
            )
            _write_setting_json(out_dir, iters, results_by_iters[iters], args, overall_t0)

    overall_elapsed = time.perf_counter() - overall_t0

    # Build the final summary across all settings.
    summary_rows: List[Dict[str, Any]] = []
    for iters in args.iterations:
        games = results_by_iters[iters]
        if not games:
            continue
        num_wins = sum(1 for g in games if g["winner"] == g["mcts_player_id"])
        num_losses = sum(1 for g in games if g["winner"] == (1 - g["mcts_player_id"]))
        num_truncated = sum(1 for g in games if g["winner"] is None)
        avg_runtime = sum(g["wall_time_s"] for g in games) / len(games)
        avg_turns = sum(g["final_turn"] for g in games) / len(games)
        summary_rows.append(
            {
                "iter_count": iters,
                "num_wins": num_wins,
                "num_losses": num_losses,
                "num_truncated": num_truncated,
                "avg_runtime_s": round(avg_runtime, 1),
                "avg_turns": round(avg_turns, 1),
            }
        )

    # Persist the aggregated summary too.
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": {
                    "iterations": args.iterations,
                    "games_per_setting": args.games_per_setting,
                    "workers": args.workers,
                    "max_turns": args.max_turns,
                    "step_budget": args.step_budget,
                    "seed": args.seed,
                    "total_wall_time_s": round(overall_elapsed, 1),
                },
                "rows": summary_rows,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"Trade study complete in {overall_elapsed:.1f}s")
    print(f"Per-setting JSONs: {out_dir}/iter_*.json")
    print(f"Summary JSON:      {out_dir}/summary.json")
    print("=" * 70)
    print()

    # The user-requested list-of-dicts view (per-line for readability).
    print("Summary (list of dicts):")
    for row in summary_rows:
        print(f"  {row}")
    print()

    # And a wider table view for quick scanning.
    print("Summary (table):")
    _print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
