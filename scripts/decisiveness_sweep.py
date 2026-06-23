"""Decisiveness sweep: does shrinking the board (or adding search) reduce draws?

Isolates board size x MCTS iterations on game decisiveness, holding everything
else fixed (per-player cash budget, max_turns, heuristic weights, self-play,
seeds). This is the controlled experiment for the thesis point that shrinking
160x80 -> 64x32 -> 24x12 made games LESS decisive.

Design:
  - boards x iters grid (default 3x3 = 9 cells).
  - Per BOARD: generate N random maps (random land/ports) + a random fleet per
    side at a FIXED cash budget. These N scenarios are REUSED across the three
    iteration settings, so within a board, iterations is the only thing that
    changes (clean isolation). Density is a MEASURED covariate (fixed budget on
    a smaller board => higher ships/tile — reported, not controlled, because the
    44x area range makes density-constant degenerate; see docs/bc_findings.md).
  - Self-play MCTS vs MCTS with the CURRENT heuristic weights (point the
    heuristic at v5 before running — NOT v6_handfit).
  - max_turns scaled? No: a single flat max_turns is used (per the run spec).
    Note in the writeup that turn budget is constant in absolute terms, not
    scaled to traversal — a smaller board gets proportionally more turns.

Metrics per cell: decisive-rate (winner != None), mean final turn, and
attack-rate (attacks / MCTS decisions) as a mechanism readout.

    poetry run python scripts/decisiveness_sweep.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling imports
from scenario_generator import generate_scenario  # noqa: E402
from benchmark_godmode_mcts import _worker_run_game  # noqa: E402
from battleboats.core.config.generate_map import generate_map_json  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO / "runs" / "sweep"

# Defaults per the run spec.
BOARDS = [(160, 80), (64, 32), (24, 12)]
ITERS = [25, 50, 100]
CASH = 500          # per-player budget (fixed across boards; density is the covariate)
MAX_TURNS = 100
N_SCEN = 30
LAND_FRACTION = 0.01
NUM_PORTS = 4
STEP_BUDGET = 20000  # generous engine-step safety cap; max_turns is the real bound


def build_scenarios(w: int, h: int) -> tuple[list[dict], float]:
    """Generate N random maps + one random-fleet scenario each for a w x h board.

    Returns (scenarios, mean_ships_per_player). Maps/scenarios are written under
    runs/sweep/ for inspection + reproducibility (seeded)."""
    board_dir = SWEEP_DIR / f"maps_{w}x{h}"
    board_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = SWEEP_DIR / f"map_{w}x{h}.yaml"
    yaml_path.write_text(f"Map:\n  sizeX: {w}\n  sizeY: {h}\n  landFraction: {LAND_FRACTION}\n  numPorts: {NUM_PORTS}\n")

    rng = random.Random(0)
    scns = []
    for i in range(N_SCEN):
        map_path = board_dir / f"map_{i:02d}.json"
        generate_map_json(str(yaml_path), str(map_path), seed=i)
        rel = str(map_path.relative_to(REPO))
        scn = generate_scenario(Path(rel), CASH, i, rng)
        scns.append(scn)

    out = SWEEP_DIR / f"scenarios_{w}x{h}.json"
    out.write_text(json.dumps(scns))
    fleet = [len(s["player_0"]["ships"]) + len(s["player_1"]["ships"]) for s in scns]
    return scns, statistics.mean(fleet) / 2.0


def run_cell(scns: list[dict], iters: int, workers: int) -> list[dict]:
    """Run one (board, iters) cell: one self-play game per scenario."""
    items = [{
        "game_idx": i,
        "seed": i,                  # same seed across iter levels -> isolates iters
        "mcts_player_id": 0,
        "iterations": iters,
        "max_turns": MAX_TURNS,
        "step_budget": STEP_BUDGET,
        "scenario": scn,
        "self_play": True,
        "verbose": False,
        "debug_plot": False,
        "debug_plot_mcts_only": False,
        "emit_tokens": False,       # outcomes only — keep it light/fast
        "debug_mcts": False,
        "shard_path": None,
    } for i, scn in enumerate(scns)]
    with Pool(workers) as pool:
        return list(pool.imap_unordered(_worker_run_game, items))


def summarize(records: list[dict]) -> dict:
    """Decisive-rate, mean final turn, and attack-rate for a cell's games."""
    n = len(records)
    decisive = sum(1 for r in records if r["winner"] is not None)
    turns = [r["final_turn"] for r in records]
    # attack-rate: AttackAction labels among MCTS decisions in the trajectory.
    atk, mcts_moves = 0, 0
    for r in records:
        for t in r.get("trajectory", []):
            if t.get("actor") == "mcts" and t.get("action_type") not in (None, "None"):
                mcts_moves += 1
                if t["action_type"] == "AttackAction":
                    atk += 1
    return {
        "games": n,
        "decisive": decisive,
        "decisive_rate": round(decisive / n, 3) if n else 0.0,
        "mean_turns": round(statistics.mean(turns), 1) if turns else 0.0,
        "attack_rate": round(atk / mcts_moves, 4) if mcts_moves else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=min((os.cpu_count() or 2) - 1, 8))
    p.add_argument("--out", type=Path, default=SWEEP_DIR / "decisiveness_results.json")
    args = p.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Decisiveness sweep: boards={BOARDS} iters={ITERS} cash={CASH} max_turns={MAX_TURNS} "
          f"scenarios/board={N_SCEN} workers={args.workers}\n")

    results = {"config": {"boards": BOARDS, "iters": ITERS, "cash": CASH, "max_turns": MAX_TURNS,
                          "n_scenarios": N_SCEN, "land_fraction": LAND_FRACTION, "num_ports": NUM_PORTS},
               "cells": {}, "density": {}}
    t0 = time.perf_counter()
    for (w, h) in BOARDS:
        scns, ships_per_player = build_scenarios(w, h)
        area = w * h
        results["density"][f"{w}x{h}"] = {
            "ships_per_player": round(ships_per_player, 2),
            "ships_per_1000_tiles": round(2 * ships_per_player / area * 1000, 3),
        }
        density_k = 2 * ships_per_player / area * 1000
        print(f"[{w}x{h}] {ships_per_player:.1f} ships/player  ({density_k:.2f} ships/1000 tiles)")
        for iters in ITERS:
            ct = time.perf_counter()
            recs = run_cell(scns, iters, args.workers)
            s = summarize(recs)
            results["cells"][f"{w}x{h}|{iters}"] = s
            print(f"    iters={iters:4d}  decisive={s['decisive_rate']:.2f} "
                  f"({s['decisive']}/{s['games']})  mean_turns={s['mean_turns']:.0f}  "
                  f"attack_rate={s['attack_rate']:.3f}  [{time.perf_counter() - ct:.0f}s]")

    results["wall_time_s"] = round(time.perf_counter() - t0, 1)
    args.out.write_text(json.dumps(results, indent=2))

    # Decisive-rate table.
    print("\nDECISIVE-RATE TABLE (rows=board, cols=iters)")
    print("board       " + "".join(f"  i={i:<5d}" for i in ITERS))
    for (w, h) in BOARDS:
        row = "".join(f"   {results['cells'][f'{w}x{h}|{i}']['decisive_rate']:.2f} " for i in ITERS)
        print(f"{w}x{h:<8}" + row)
    print(f"\nsaved {args.out}  ({results['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
