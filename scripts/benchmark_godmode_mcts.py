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
from battleboats.agents.godmode_mcts import godmode_mcts_action, godmode_mcts_search, godmode_mcts_search_debug
from battleboats.agents.heuristics import decompose, heuristic_eval
from battleboats.agents.random_agent import random_action
from battleboats.core.actions import MoveAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.observation import build_entity_tokens

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
    emit_tokens: bool = False,
    debug_mcts: bool = False,
    shard_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Play one game; return a record with trajectory + outcome + timing.

    Streaming mode (shard_path set, used by the harvest): instead of
    accumulating the whole trajectory in RAM and returning it, write each
    step's per-perspective rows straight to `shard_path` as they're produced,
    then a `_type=game_footer` line carrying the winner. Worker memory stays
    flat regardless of game length — this is what keeps a token harvest from
    OOM-ing (a full token trajectory is ~1-2 GB; six workers holding one each
    blows past RAM). The returned record then omits `trajectory` and just
    carries outcome/tally fields. The terminal ±1 target is NOT written
    per-row (the winner is only known at game end); the footer's winner lets
    HarvestDataset derive it at load time.

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

    # Streaming mode: open this game's own shard file. No shared handle across
    # workers -> no locking. Closed (and flushed) at game end; a worker killed
    # mid-game just loses its partial file, completed games are safe on disk.
    shard_file = open(shard_path, "w") if shard_path is not None else None
    base_provenance = {
        "game_idx": game_idx,
        "seed": seed,
        "iterations": iterations,
        "mcts_player_id": mcts_player_id,
    }
    kept_rows = 0

    trajectory: List[Dict[str, Any]] = []
    game_t0 = time.perf_counter()
    steps = 0

    for agent in env.agent_iter():
        # Reset per-step MCTS labels; only set in the use_mcts branch below.
        mcts_root_value_step: Optional[float] = None
        mcts_root_stats_step: Optional[List[Dict[str, Any]]] = None
        acting_pid: Optional[int] = None
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
                if debug_mcts:
                    action, mcts_root_value_step, root_stats = godmode_mcts_search_debug(
                        env.engine, pid, rng, iterations=iterations
                    )
                    # Stringify each child's raw Action via the same labeller used
                    # for the chosen action, so the logged table lines up move-for-move
                    # with the heuristic move-score panel. Done pre-step (engine still
                    # at the decision state the search ran from).
                    mcts_root_stats_step = [
                        {"action": _describe_action(s["action"], env.engine), "visits": s["visits"], "q": s["q_root_pov"]}
                        for s in root_stats
                    ]
                else:
                    action, mcts_root_value_step = godmode_mcts_search(env.engine, pid, rng, iterations=iterations)
                elapsed = time.perf_counter() - t0
                actor = "mcts"
                acting_pid = pid
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
                acting_pid = pid
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
        # Per-perspective entity tokens for the transformer encoder — the
        # variable-length (N, TOKEN_DIM) set that `phi` aggregates away.
        # Indexed by absolute player id, same convention as phi_p*. Built
        # only when harvesting (emit_tokens); the benchmark/trade-study
        # callers leave it off to avoid the per-step tokenizer cost.
        # Stored as nested lists so the row stays JSON-serializable.
        tokens_p0 = build_entity_tokens(env.engine, 0).tolist() if emit_tokens else None
        tokens_p1 = build_entity_tokens(env.engine, 1).tolist() if emit_tokens else None

        # Behavior-cloning label: the MCTS-chosen action as factored
        # (asset_idx, verb_idx, target_idx) indices into the ACTING player's token
        # order (target_idx -1 = no target). Recorded only for expert (MCTS) moves;
        # attached below to the acting player's perspective row (it was decided from
        # that POV, so it aligns with that POV's tokens). Computed pre-step, on the
        # same engine state the tokens were built from. See action_masks.ActionMasks.
        action_idx = None
        if actor == "mcts" and acting_pid is not None and action is not None:
            action_idx = list(ActionMasks(env.engine, acting_pid).factor(action))

        if shard_file is not None:
            # Stream the two per-perspective rows immediately, then discard —
            # never accumulate. mcts_root_value is attached only to the acting
            # player's perspective (the POV search ran from); the other gets
            # null, same convention the harvest used. Skip terminal/dead steps
            # where decompose returned empty phi. No per-row terminal target —
            # the footer's winner lets the dataset derive it.
            if phi_p0 and phi_p1:
                rv_p0 = mcts_root_value_step if acting_pid == 0 else None
                rv_p1 = mcts_root_value_step if acting_pid == 1 else None
                # Root (N, Q) table rides the same acting-player perspective as
                # mcts_root_value. None unless --debug-mcts was set (flag-gated;
                # full tables would bloat a normal harvest).
                mr_p0 = mcts_root_stats_step if acting_pid == 0 else None
                mr_p1 = mcts_root_stats_step if acting_pid == 1 else None
                # Action label rides the same perspective as mcts_root_value: the
                # acting player's row gets it, the other gets null.
                act_p0 = action_idx if acting_pid == 0 else None
                act_p1 = action_idx if acting_pid == 1 else None
                # Per-player cash (a global, not in the entity tokens). Logged so
                # a state can be reconstructed steppably from a row — notably to
                # know which build_ship moves are affordable. Indexed by absolute
                # player id [p0, p1], same convention as phi_p*/tokens_p*.
                cash = [env.engine.players[0].cash, env.engine.players[1].cash]
                step_base = {**base_provenance, "step": steps, "turn": env.engine.turn, "actor": actor, "cash": cash}
                row0 = {**step_base, "perspective": 0, "phi": phi_p0, "tokens": tokens_p0, "mcts_root_value": rv_p0, "action": act_p0, "mcts_root": mr_p0}
                row1 = {**step_base, "perspective": 1, "phi": phi_p1, "tokens": tokens_p1, "mcts_root_value": rv_p1, "action": act_p1, "mcts_root": mr_p1}
                shard_file.write(json.dumps(row0, default=str) + "\n")
                shard_file.write(json.dumps(row1, default=str) + "\n")
                kept_rows += 2
        else:
            trajectory.append(
                {
                    "step": steps,
                    "turn": env.engine.turn,
                    "actor": actor,
                    "value": value,
                    "phi_p0": phi_p0,
                    "phi_p1": phi_p1,
                    "tokens_p0": tokens_p0,
                    "tokens_p1": tokens_p1,
                    "elapsed_s": elapsed,
                    "action_type": action_name,
                    # MCTS root-value label: mean backed-up value at the root from
                    # the acting player's POV. Lower-variance complement to the
                    # terminal MC return. None when the actor wasn't MCTS this
                    # step. `acting_pid` lets the harvest attach the label to the
                    # correct per-perspective row.
                    "mcts_root_value": mcts_root_value_step,
                    "acting_pid": acting_pid,
                    # Factored BC label of the MCTS action (None for non-MCTS steps).
                    "action": action_idx,
                }
            )

        env.step(action)
        steps += 1
        if steps > step_budget:
            if verbose:
                print(f"  [step budget {step_budget} hit; aborting game]", flush=True)
            break

    winner = env.engine.winner  # 0 / 1 / None
    record = {
        "game_idx": game_idx,
        "iterations": iterations,
        "self_play": self_play,
        "mcts_player_id": mcts_player_id,
        "seed": seed,
        "winner": winner,
        "steps": steps,
        "final_turn": env.engine.turn,
        "wall_time_s": time.perf_counter() - game_t0,
    }

    if shard_file is not None:
        # Footer carries the winner so the dataset can derive the ±1 terminal
        # target at load time (None for truncated games -> no flat target,
        # but their mcts_root_value rows are still usable).
        footer = {"_type": "game_footer", **base_provenance, "winner": winner,
                  "steps": steps, "final_turn": env.engine.turn}
        shard_file.write(json.dumps(footer, default=str) + "\n")
        shard_file.close()
        record["kept_rows"] = kept_rows
        return record

    record["trajectory"] = trajectory
    return record


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
