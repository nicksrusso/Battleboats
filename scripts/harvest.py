"""Self-play data harvester — collects per-state value-regression rows.

Runs N games at a fixed MCTS iteration count using the same unified-work-queue
parallel dispatch as `trade_study.py`. Each worker STREAMS its game's rows
directly to its own shard file (`harvest_<ts>/game_<idx>.jsonl`) as the game
plays — never accumulating the trajectory in RAM — which keeps worker memory
flat and prevents the OOM that token-heavy trajectories caused.

Output layout: a directory `harvest_<ts>/` of per-game shards. Each shard is
JSONL with two row types:
    data rows (2 per kept step, one per perspective):
        phi:             feature vector (one entry per FEATURE_KEYS), that POV
        tokens:          variable-length (N, TOKEN_DIM) entity-token set
        mcts_root_value: search value at that state from the acting POV
                         (null on the non-acting perspective)
        game_idx, seed, iterations, mcts_player_id, step, turn, actor, perspective
    one footer row (`_type=game_footer`) carrying `winner` (0/1/None).

NOTE: the flat terminal ±1 target is NOT written per-row (the winner is only
known at game end). HarvestDataset derives it from the footer's `winner` at
load time. Truncated games (winner None) are KEPT — their mcts_root_value rows
are valid regardless of outcome; only the derived flat target is undefined.

A sidecar `<...>_meta.json` records the run config and final tallies.

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
DEFAULT_ITERATIONS = 250


def _shard_complete(path: Path) -> bool:
    """True if `path` is a FINISHED game shard — exists and ends with a footer
    line. A worker killed mid-game leaves a footerless (partial) shard, which
    reads as incomplete here and gets re-run. Reads only the file tail (the footer
    is the last, short line) so scanning a dir of large shards stays fast.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))  # footer is tiny; last 4 KB always holds it
        tail = f.read().decode("utf-8", "ignore")
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return False
    try:
        return json.loads(lines[-1]).get("_type") == "game_footer"
    except json.JSONDecodeError:
        return False  # truncated final line -> partial shard


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
        "--resume",
        type=Path,
        default=None,
        help="Resume into an EXISTING shard dir: skip game_idx whose shard already "
        "finished (has a footer), re-run partial/missing ones. Reuses the run's "
        "config from <dir>/_run_config.json if present; otherwise the gameplay args "
        "you pass MUST match the original run (--seed/--num-games/--scenarios-file/"
        "--iterations/--max-turns/--step-budget) or game_idx maps to different games.",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Both players use MCTS with the current heuristic weights "
        "(symmetric self-play). When unset, MCTS plays vs. random_agent.",
    )
    parser.add_argument(
        "--debug-mcts",
        action="store_true",
        help="Log the per-decision MCTS root (N, Q) table to each row's "
        "`mcts_root` field, for offline search inspection. Off by default — the "
        "full table per step would bloat a normal harvest. Re-run the specific "
        "game/seed you want to inspect with this flag set.",
    )
    args = parser.parse_args()

    # Output is a DIRECTORY of per-game shards (game_<idx>.jsonl), each written
    # by its own worker as the game plays. Streaming per-game keeps worker
    # memory flat (no whole-trajectory accumulation -> no OOM) and lets the
    # dataset/split later read only the games a split references.
    resuming = args.resume is not None
    if resuming:
        shard_dir = args.resume
        if not shard_dir.is_dir():
            parser.error(f"--resume directory not found: {shard_dir}")
        meta_path = Path(str(shard_dir) + "_meta.json")
        cfg_path = shard_dir / "_run_config.json"
        if cfg_path.exists():
            # Adopt the original run's gameplay args so game_idx maps to the SAME
            # games. Only --workers may safely differ between original and resume.
            cfg = json.loads(cfg_path.read_text())
            args.num_games = cfg["num_games"]
            args.iterations = cfg["iterations"]
            args.self_play = cfg["self_play"]
            args.max_turns = cfg["max_turns"]
            args.step_budget = cfg["step_budget"]
            args.seed = cfg["seed"]
            args.scenarios_file = Path(cfg["scenarios_file"])
            print(f"Resuming {shard_dir}/ with config from {cfg_path.name}")
        else:
            print(
                f"Resuming {shard_dir}/  (no _run_config.json — older run). Using the "
                "CLI args as given; they MUST match the original run or game_idx will "
                "map to different games."
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_dir = args.output_dir / f"harvest_{timestamp}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        meta_path = args.output_dir / f"harvest_{timestamp}_meta.json"

    # Load scenarios (cycles through scenarios_500.json for varied starts)
    scenarios = json.loads(args.scenarios_file.read_text())
    print(f"Loaded {len(scenarios)} scenarios for harvest.")

    # Persist the gameplay config so a future --resume into this dir maps game_idx
    # to identical games without the user re-specifying args. (Older dirs lack
    # this; resume falls back to the CLI args, which must match.)
    if not resuming:
        (shard_dir / "_run_config.json").write_text(
            json.dumps(
                {
                    "num_games": args.num_games,
                    "iterations": args.iterations,
                    "self_play": args.self_play,
                    "max_turns": args.max_turns,
                    "step_budget": args.step_budget,
                    "seed": args.seed,
                    "scenarios_file": str(args.scenarios_file),
                },
                indent=2,
            )
        )

    # Build the global work list. All games at the same iteration count but with
    # distinct seeds + cycled scenarios. On resume, skip game_idx whose shard is
    # already finished (footer present) — partial/missing ones get (re)run; a
    # worker reopening a shard with "w" truncates any partial cleanly.
    skipped = 0
    work_items: List[Dict[str, Any]] = []
    for game_idx in range(args.num_games):
        shard_path = shard_dir / f"game_{game_idx}.jsonl"
        if resuming and _shard_complete(shard_path):
            skipped += 1
            continue
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
                "emit_tokens": True,
                "debug_mcts": args.debug_mcts,
                "shard_path": str(shard_path),
            }
        )
    if resuming:
        print(f"Resume: {skipped} games already complete, {len(work_items)} to (re)run.")

    # Clamp workers to the jobs actually queued (post resume-skip) — no point
    # spawning more processes than there are games left to run.
    workers = max(1, min(args.workers, len(work_items)))

    mode_label = "self-play (MCTS vs MCTS)" if args.self_play else "vs random_agent"
    print(f"Harvest: {args.num_games} games at iter={args.iterations} using {len(scenarios)} scenarios [{mode_label}]")
    print(f"  max_turns:   {args.max_turns}")
    print(f"  workers:     {workers}")
    print(f"  scenarios:   {args.scenarios_file}")
    print(f"  shards:      {shard_dir}/")
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

    # Workers stream each game's rows to its own shard file; the parent never
    # holds rows — it only tallies the lean records they return. This is what
    # keeps memory flat regardless of corpus size or worker count.
    with mp.Pool(processes=workers) as pool:
        for record in pool.imap_unordered(_worker_run_game, work_items):
            completed += 1
            kept = record.get("kept_rows", 0)

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

            # Truncated games are now KEPT — their mcts_root_value rows are
            # valid (search value doesn't depend on the final outcome); only
            # the flat terminal target is undefined for them. So a game can be
            # both "truncated" (winner None) and "kept" (rows > 0).
            if kept > 0:
                kept_games += 1
                total_rows += kept

            print(
                f"  [{skipped + completed:3d}/{args.num_games}] game_idx={record['game_idx']:3d}  "
                f"seed={record['seed']:3d}  player_{mp_id}  "
                f"turn={record['final_turn']:3d}  steps={record['steps']:5d}  "
                f"-> {outcome:10s}  ({record['wall_time_s']:.1f}s, {kept:5d} rows)",
                flush=True,
            )

    overall_elapsed = time.perf_counter() - overall_t0

    results_block: Dict[str, Any] = {
        "resumed": resuming,
        "skipped_already_done": skipped,
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
        "shard_dir": str(shard_dir),
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
    print(f"  shards:    {shard_dir}/")
    print(f"  meta:      {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
