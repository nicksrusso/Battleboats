"""One harvest game on the Rust engine, observation, and scripted agent.

Python agents other than scripted agent still need the reference engine, so
this path is the scripted-agent self-play hot path. Random seats sample the
Rust legal list with the caller's ``random.Random``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import battleboats_core
from battleboats.agents.registry import is_scripted_agent_kind
from battleboats.envs.observation import GLOBALS_DIM, TOKEN_DIM


def _outcome(winner: Optional[int], seat: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == seat else -1.0


def play_harvest_game(job: Dict[str, Any]) -> Dict[str, Any]:
    """Play ``job`` on Rust and write the same npz harvest_value.py writes."""
    game_idx = int(job["game_idx"])
    seed = int(job["seed"])
    map_spec = str(job["map_spec"])
    actors = {0: str(job["actors"][0]), 1: str(job["actors"][1])}
    gamma = float(job["gamma"])
    out_path = Path(job["out_path"])
    commit = str(job["commit"])
    forced = job.get("doctrines") or {}

    from battleboats.ui.maps import resolve_map

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rm = resolve_map(map_spec, seed=seed, map_dir=Path(td))
        map_text = rm.path.read_text()
        max_turns = int(rm.recommended_max_turns)
        max_steps = int(job.get("max_steps") or (8000 * max_turns // 80))
        game = battleboats_core.Game(str(rm.path))
        game.reset(seed)

    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    agents = []
    for pid in (0, 1):
        if is_scripted_agent_kind(actors[pid]):
            doctrine = forced.get(pid) or forced.get(str(pid))
            agents.append(battleboats_core.ScriptedAgent(doctrine))
        else:
            agents.append(None)

    tokens_list: List[np.ndarray] = []
    globals_list: List[np.ndarray] = []
    seat_list: List[int] = []
    acting_list: List[int] = []
    turn_list: List[int] = []
    step_list: List[int] = []
    encoded_actions: List[list] = []
    buys = {0: Counter(), 1: Counter()}
    per_turn: Counter = Counter()
    illegal = {0: 0, 1: 0}
    forced_end = {0: 0, 1: 0}
    steps = 0
    winner = None
    final_turn = 0

    while steps < max_steps:
        winner = game.winner()
        final_turn = int(game.turn())
        if winner is not None or final_turn >= max_turns:
            break
        pid = int(game.current_player())
        legal = game.enumerate_legal(pid)
        for seat in (0, 1):
            flat, globs, n = game.observation(seat)
            if n:
                tokens = np.asarray(flat, dtype=np.float32).reshape(n, TOKEN_DIM).astype(np.float16)
            else:
                tokens = np.zeros((0, TOKEN_DIM), dtype=np.float16)
            tokens_list.append(tokens)
            globals_list.append(np.asarray(globs, dtype=np.float32))
            seat_list.append(seat)
            acting_list.append(pid)
            turn_list.append(final_turn)
            step_list.append(steps)
        if agents[pid] is not None:
            action = list(agents[pid].act(game, pid, rngs[pid]))
        else:
            action = list(rngs[pid].choice(legal))
        per_turn[(pid, final_turn)] += 1
        if action not in legal and not (action[0] == "b" and any(
            a[0] == "b" and a[1:3] == action[1:3] and a[5] == action[5] for a in legal
        )):
            illegal[pid] += 1
            action = ["e"]
        elif per_turn[(pid, final_turn)] > 400 and action != ["e"]:
            forced_end[pid] += 1
            action = ["e"]
        if action[0] == "b":
            buys[pid][action[5]] += 1
        encoded_actions.append(action)
        game.step(action)
        steps += 1
    else:
        winner = game.winner()
        final_turn = int(game.turn())

    n_dec = steps
    n_rows = len(tokens_list)
    outcome_by_seat = {0: _outcome(winner, 0), 1: _outcome(winner, 1)}
    lengths = [int(t.shape[0]) for t in tokens_list]
    splits = np.zeros(n_rows + 1, dtype=np.int32)
    if n_rows:
        splits[1:] = np.cumsum(lengths, dtype=np.int32)
        tokens_data = np.concatenate(tokens_list, axis=0)
        globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    else:
        tokens_data = np.zeros((0, TOKEN_DIM), dtype=np.float16)
        globals_arr = np.zeros((0, GLOBALS_DIM), dtype=np.float32)
    seat_arr = np.asarray(seat_list, dtype=np.int8)
    acting_arr = np.asarray(acting_list, dtype=np.int8)
    turn_arr = np.asarray(turn_list, dtype=np.int16)
    step_arr = np.asarray(step_list, dtype=np.int32)
    outcome_arr = np.asarray([outcome_by_seat[s] for s in seat_list], dtype=np.float32)
    if n_dec > 0:
        expo = np.asarray([(n_dec - 1) - st for st in step_list], dtype=np.float32)
        returns_arr = outcome_arr * (np.float32(gamma) ** expo)
    else:
        returns_arr = np.zeros((0,), dtype=np.float32)
    doctrines = {pid: agents[pid].doctrine_for(pid) if agents[pid] is not None else None for pid in (0, 1)}
    reason = "home_capture" if winner is not None else (
        "turn_cap" if final_turn >= max_turns else "step_cap"
    )
    meta = {
        "game_idx": game_idx,
        "kind": job["kind"],
        "seed": seed,
        "map_spec": map_spec,
        "width": int(game.width),
        "height": int(game.height),
        "max_turns": max_turns,
        "max_steps": max_steps,
        "actors": actors,
        "doctrines": doctrines,
        "winner": winner,
        "reason": reason,
        "turns": final_turn,
        "num_decisions": n_dec,
        "num_rows": n_rows,
        "outcome_by_seat": outcome_by_seat,
        "gamma": gamma,
        "buys": {pid: dict(buys[pid]) for pid in (0, 1)},
        "illegal": illegal,
        "forced_end": forced_end,
        "obs_encoding": "v0.5.0-fog-coastline-fix",
        "commit": commit,
        "backend": "rust",
        "path": str(out_path.relative_to(Path(job["out_root"]))),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        tokens_data=tokens_data,
        row_splits=splits,
        globals=globals_arr,
        seat=seat_arr,
        acting_seat=acting_arr,
        turn=turn_arr,
        step_idx=step_arr,
        outcome=outcome_arr,
        returns=returns_arr,
        actions_json=np.asarray(json.dumps(encoded_actions, separators=(",", ":"))),
        map_json=np.asarray(map_text),
        meta_json=np.asarray(json.dumps(meta, separators=(",", ":"))),
    )
    tmp_path.replace(out_path)
    meta["status"] = "ok"
    meta["bytes"] = int(out_path.stat().st_size)
    return meta
