"""Run god-mode MCTS vs random agent, with per-step telemetry.

Mirrors the shape of tests/test_godmode_mcts.py::test_mcts_beats_random but
runs as a plain script (no pytest plumbing), so it can be invoked directly
and its stdout / log files are clean.

Per-game telemetry is collected in memory and dumped to a timestamped JSON
file under runs/benchmarks/. Live stdout prints one line per MCTS move
showing turn, wall time, heuristic value, and the action type.

Usage:
    poetry run python scripts/benchmark_godmode_mcts.py
    poetry run python scripts/benchmark_godmode_mcts.py --iterations 250 \\
        --num-games 1 --max-turns 250 --step-budget 500000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from battleboats.agents.debug_plot import plot_state
from battleboats.agents.godmode_mcts import godmode_mcts_action
from battleboats.agents.heuristics import decompose, heuristic_eval
from battleboats.agents.random_agent import random_action
from battleboats.core.actions import MoveAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.battleboats_aec import BattleboatsAEC

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"
OUTPUT_DIR = Path("/home/nick/Desktop/repos/Battleboats/runs/benchmarks")


def _describe_action(action, engine) -> str:
    """Return a short, human-readable label for an action.

    For MoveActions, includes the ship type so we can spot when MCTS is
    bouncing Merchants / Builders around (which produce zero heuristic
    signal). For other action types, just returns the class name.
    """
    if action is None:
        return "None"
    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        if ship is None:
            return "Move(?)"
        return f"Move({ship.type.value})"
    return type(action).__name__


def _format_inventory(engine, player_id: int) -> str:
    """Format the player's ship inventory as 'Type:count - Type:count - ...'.

    Iterates ShipType in declaration order so columns are stable across
    turns. Types with zero count are omitted to keep the line compact.
    """
    counts: Dict[ShipType, int] = {}
    for sid in engine.players[player_id].owned_ship_ids:
        t = engine.ships[sid].type
        counts[t] = counts.get(t, 0) + 1
    return " - ".join(f"{st.value}:{counts[st]}" for st in ShipType if counts.get(st))


def _format_sightings(engine, player_id: int) -> str:
    """Format `player_id`'s fresh enemy ship sightings as 'Type:count - ...'.

    Only counts sightings marked `fresh` (currently in view per fog rules).
    Stale sightings (last-known but currently out of view) are not included
    — this is the "what's actively spotted" view. Same column ordering as
    _format_inventory for visual consistency.
    """
    counts: Dict[ShipType, int] = {}
    for sighting in engine.players[player_id].sightings.values():
        if not sighting.fresh:
            continue
        counts[sighting.type] = counts.get(sighting.type, 0) + 1
    return " - ".join(f"{st.value}:{counts[st]}" for st in ShipType if counts.get(st))


def _min_distance_to_enemy_home(engine, player_id: int) -> str:
    """Min Manhattan distance from any of player's ships to the enemy home port.

    Returns '--' if the player has no ships. Uses god-mode access to the
    enemy's home_port directly (legitimate here: benchmark, not the agent).
    """
    opp_home = engine.players[1 - player_id].home_port
    my_ships = engine.players[player_id].owned_ship_ids
    if not my_ships:
        return "--"
    return str(min(engine.map.manhattan(engine.ships[sid].position, opp_home) for sid in my_ships))


def _min_distance_to_enemy_ship(engine, player_id: int) -> str:
    """Min Manhattan distance from any of player's ships to any enemy ship.

    Returns '--' if either side has no ships. God-mode access — used to
    track whether MCTS is closing the gap to the opponent's fleet.
    """
    my_ships = engine.players[player_id].owned_ship_ids
    opp_ships = engine.players[1 - player_id].owned_ship_ids
    if not my_ships or not opp_ships:
        return "--"
    return str(
        min(engine.map.manhattan(engine.ships[s].position, engine.ships[o].position) for s in my_ships for o in opp_ships)
    )


def _play_one_game(
    game_idx: int,
    map_json_path: Optional[str] = None,
    seed: int = 0,
    mcts_player_id: int = 0,
    iterations: int = 100,
    max_turns: int = 250,
    step_budget: int = 50000,
    scenario: Optional[Dict[str, Any]] = None,
    self_play: bool = False,
    verbose: bool = True,
    debug_plot: bool = False,
    debug_plot_mcts_only: bool = False,
) -> Dict[str, Any]:
    """Play one game; return a record with trajectory + outcome + timing.

    Self-contained for multiprocessing: takes only picklable primitives and
    constructs its own env + rng inside the function body. The `verbose`
    flag gates per-step stdout — useful to keep on for serial single-game
    debugging, off in parallel benchmark runs where 8 workers interleaving
    step prints is unreadable.

    When `self_play=True`, BOTH players use `godmode_mcts_action` (with the
    current `DEFAULT_WEIGHTS` baked into `heuristic_eval`). `mcts_player_id`
    in that mode just selects whose POV the diagnostic `value` field is
    computed from — gameplay is symmetric.
    """
    map_path = scenario["map_path"] if scenario else map_json_path
    env = BattleboatsAEC(map_json_path=map_path, max_turns=max_turns)
    if scenario:
        env.reset(seed=seed, options={"scenario": scenario})
    else:
        env.reset(seed=seed)
    rng = random.Random(seed)

    trajectory: List[Dict[str, Any]] = []
    game_t0 = time.perf_counter()
    steps = 0

    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            action = None
            actor = "dead"
            elapsed = 0.0
            action_name = "None"
        else:
            pid = env._player_id(agent)
            use_mcts = self_play or (pid == mcts_player_id)
            if use_mcts:
                t0 = time.perf_counter()
                action = godmode_mcts_action(env.engine, pid, rng, iterations=iterations)
                elapsed = time.perf_counter() - t0
                actor = "mcts"
                action_name = _describe_action(action, env.engine)
                if verbose:
                    # Print is from the *acting* player's POV so self-play
                    # produces interpretable traces from each side as they
                    # take their turns.
                    value = heuristic_eval(env.engine, pid)
                    inventory = _format_inventory(env.engine, pid)
                    opp_inventory = _format_inventory(env.engine, 1 - pid)
                    spotted = _format_sightings(env.engine, pid)
                    cash = env.engine.players[pid].cash
                    opp_cash = env.engine.players[1 - pid].cash
                    d_home = _min_distance_to_enemy_home(env.engine, pid)
                    d_enemy = _min_distance_to_enemy_ship(env.engine, pid)
                    label = f"mcts(p{pid})" if self_play else "mcts"
                    print(
                        f"  step={steps:6d} turn={env.engine.turn:4d}  {label} {elapsed:6.2f}s  "
                        f"value={value:+.6f}  $me={cash:5d}  $opp={opp_cash:5d}  "
                        f"d_home={d_home:>3}  d_enemy={d_enemy:>3}  "
                        f"{action_name:<20}  "
                        f"mine=[{inventory}]  opp=[{opp_inventory}]  spot=[{spotted}]",
                        flush=True,
                    )
            else:
                action = random_action(env.engine, pid, rng)
                actor = "random"
                elapsed = 0.0
                action_name = _describe_action(action, env.engine)

        debug_plot_mcts_only = True
        debug_plot = False
        if debug_plot and (not debug_plot_mcts_only or actor == "mcts"):
            heur_val = heuristic_eval(env.engine, mcts_player_id)
            plot_state(
                env.engine,
                action,
                actor,
                mcts_player_id,
                step=steps,
                value=heur_val,
            )
            breakpoint()  # <-- inspect the plot here; 'c' to continue.
        # ---------------------------------------------------------------------

        # Telemetry entry — record state from BOTH players' perspectives so
        # the harvest can emit per-perspective training rows with opposite
        # MC targets (winner's POV = +1, loser's POV = -1). Without this
        # symmetry the dataset has no contrast — every kept-game row would
        # share the same target and regression learns nothing.
        #
        # `value` stays MCTS-side for diagnostic prints. `phi_p0` / `phi_p1`
        # are always indexed by absolute player id (not by MCTS-or-not) so
        # downstream code doesn't need to know which seat MCTS occupied.
        # On terminal states both phi dicts are {} by design (decompose
        # short-circuits); the harvest filters those out.
        value, phi_mcts, _ = decompose(env.engine, mcts_player_id)
        _, phi_opp, _ = decompose(env.engine, 1 - mcts_player_id)
        phi_p0 = phi_mcts if mcts_player_id == 0 else phi_opp
        phi_p1 = phi_opp if mcts_player_id == 0 else phi_mcts
        trajectory.append(
            {
                "step": steps,
                "turn": env.engine.turn,
                "actor": actor,
                "value": value,
                "phi_p0": phi_p0,
                "phi_p1": phi_p1,
                "elapsed_s": elapsed,
                "action_type": action_name,
            }
        )

        env.step(action)
        steps += 1
        if steps > step_budget:
            if verbose:
                print(f"  [step budget {step_budget} hit; aborting game]", flush=True)
            break

    return {
        "game_idx": game_idx,
        "iterations": iterations,
        "self_play": self_play,
        "mcts_player_id": mcts_player_id,
        "seed": seed,
        "winner": env.engine.winner,  # 0 / 1 / None
        "steps": steps,
        "final_turn": env.engine.turn,
        "wall_time_s": time.perf_counter() - game_t0,
        "trajectory": trajectory,
    }


def _worker_run_game(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level multiprocessing entry point.

    A bare wrapper around `_play_one_game(**kwargs)` so that
    `multiprocessing.Pool.imap_unordered` can dispatch it — closures and
    nested functions don't pickle reliably across spawn/fork boundaries.
    """
    return _play_one_game(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS vs random benchmark with telemetry.")
    parser.add_argument("--iterations", type=int, default=2500, help="MCTS iterations per move.")
    parser.add_argument("--num-games", type=int, default=4, help="Total games (sides alternate).")
    parser.add_argument("--max-turns", type=int, default=250, help="Env truncation turn cap.")
    parser.add_argument("--step-budget", type=int, default=50_000, help="Per-game step cap.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; per-game seed = base + i.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel game workers. 1 = serial (with per-step prints). "
        "Default = os.cpu_count(). Larger values oversubscribe physical cores.",
    )
    parser.add_argument(
        "--debug-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each action selection, render full-state plot and hit a "
        "breakpoint() so you can inspect the game state. Pass --no-debug-plot "
        "to disable.",
    )
    parser.add_argument(
        "--debug-plot-mcts-only",
        action="store_true",
        help="With --debug-plot, only pause for MCTS-chosen actions (skip random).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"mcts_vs_random_{timestamp}.json"

    # Clamp workers so we never spawn more processes than there are games.
    workers = max(1, min(args.workers, args.num_games))

    print(f"Running {args.num_games} games at {args.iterations} iters/move.")
    print(f"max_turns={args.max_turns}, step_budget={args.step_budget}, workers={workers}")
    print(f"Output: {out_path}")
    print()

    # Build all work items up front. `verbose=True` only in serial mode —
    # interleaved per-step prints from 8 workers are unreadable.
    work_items: List[Dict[str, Any]] = []
    for game_idx in range(args.num_games):
        work_items.append(
            {
                "game_idx": game_idx,
                "map_json_path": MAP_JSON,
                "seed": args.seed + game_idx,
                "mcts_player_id": game_idx % 2,
                "iterations": args.iterations,
                "max_turns": args.max_turns,
                "step_budget": args.step_budget,
                "verbose": workers == 1,
                "debug_plot": args.debug_plot,
                "debug_plot_mcts_only": args.debug_plot_mcts_only,
            }
        )

    mcts_wins = 0
    random_wins = 0
    truncated = 0
    games: List[Dict[str, Any]] = []
    overall_t0 = time.perf_counter()

    def _flush_summary() -> None:
        """Write the current cumulative summary to disk.

        Called after every game completes so a long benchmark leaves a
        partial log even if it's killed mid-batch. Games are sorted by
        `game_idx` for deterministic output ordering — under parallel
        execution `imap_unordered` delivers them in completion order, not
        dispatch order.
        """
        sorted_games = sorted(games, key=lambda g: g["game_idx"])
        summary = {
            "config": {
                "iterations": args.iterations,
                "num_games": args.num_games,
                "max_turns": args.max_turns,
                "step_budget": args.step_budget,
                "seed": args.seed,
                "workers": workers,
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
        with open(out_path, "w") as f:
            json.dump(summary, f, default=str)

    def _process_record(record: Dict[str, Any]) -> None:
        nonlocal mcts_wins, random_wins, truncated
        games.append(record)
        mcts_player = record["mcts_player_id"]
        winner = record["winner"]
        if winner == mcts_player:
            mcts_wins += 1
            outcome = "MCTS won"
        elif winner == (1 - mcts_player):
            random_wins += 1
            outcome = "random won"
        else:
            truncated += 1
            outcome = "truncated"
        print(
            f"  [{len(games):3d}/{args.num_games}] game_idx={record['game_idx']:3d}  "
            f"player_{mcts_player}  seed={record['seed']}  "
            f"-> {outcome:10s}  "
            f"({record['steps']} steps, turn {record['final_turn']}, {record['wall_time_s']:.1f}s)",
            flush=True,
        )
        _flush_summary()

    if workers == 1:
        for kwargs in work_items:
            print(
                f"=== Game {kwargs['game_idx'] + 1}/{args.num_games}  "
                f"(MCTS plays as player_{kwargs['mcts_player_id']}, seed={kwargs['seed']}) ==="
            )
            _process_record(_play_one_game(**kwargs))
            print()
    else:
        # imap_unordered yields records as workers finish, so progress prints
        # and the on-disk summary update in completion order. The pool context
        # manager handles join/terminate on exception.
        with mp.Pool(processes=workers) as pool:
            for record in pool.imap_unordered(_worker_run_game, work_items):
                _process_record(record)

    overall_elapsed = time.perf_counter() - overall_t0

    print()
    print("=" * 60)
    print(f"Summary: mcts={mcts_wins}  random={random_wins}  truncated={truncated}")
    print(f"Total wall time: {overall_elapsed:.1f}s")
    print(f"Log: {out_path}")


if __name__ == "__main__":
    main()
