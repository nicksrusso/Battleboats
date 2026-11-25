"""Generate compact golden seeded replay fixtures under tests/golden/fixtures/.

Usage:
    poetry run python tests/golden/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import gzip
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from battleboats.agents.fog_commander import DOCTRINE_NAMES
from battleboats.agents.registry import make_agent
from battleboats.core.actions import EndTurnAction
from battleboats.envs.map_pool import PoolSize, build_map_pool
from battleboats.envs.observation import build_observation
from tests.golden.backend import PythonEngineBackend, encode_action

FIXTURE_PATH = Path(__file__).resolve().parent / "replays.json.gz"
MAX_ACTIONS_PER_TURN = 400


def _obs_summary(inst, player_id: int) -> Dict[str, Any]:
    obs = build_observation(inst.raw, player_id)
    tokens = obs["entity_tokens"]
    globs = obs["globals"]
    return {
        "player_id": player_id,
        "turn": int(inst.raw.turn),
        "visible_enemy_ships": inst.visible_enemy_ships(player_id),
        "known_enemy_ships": inst.known_enemy_ships(player_id),
        "visible_enemy_ports": inst.visible_enemy_ports(player_id),
        "known_enemy_ports": inst.known_enemy_ports(player_id),
        "obs_token_count": int(tokens.shape[0]),
        "obs_globals": [round(float(x), 6) for x in globs.tolist()],
    }


def _play_one_game(
    game_id: str,
    map_preset: str,
    map_data: Dict[str, Any],
    seed: int,
    p0_kind: str,
    p0_doctrine: str | None,
    p1_kind: str,
    p1_doctrine: str | None,
    max_turns: int,
) -> Dict[str, Any]:
    backend = PythonEngineBackend()
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "map.json"
        mp.write_text(json.dumps(map_data))
        inst = backend.create(mp, kill_curve_k=2.0)
        inst.reset(seed=seed)

    # Record RNG combat outcomes during fixture generation only.
    combat_outcomes: List[List[Any]] = []
    step_counter = {"idx": 0}
    orig_resolve = inst.raw._resolve_attack

    def _recording_resolve(att, dfn) -> bool:
        killed = bool(orig_resolve(att, dfn))
        combat_outcomes.append([step_counter["idx"], killed])
        return killed

    inst.raw._resolve_attack = _recording_resolve  # type: ignore[method-assign]

    agents = [
        make_agent(p0_kind, doctrine=p0_doctrine),
        make_agent(p1_kind, doctrine=p1_doctrine),
    ]
    agent_rngs = [random.Random(seed * 101 + 1), random.Random(seed * 101 + 2)]

    initial_hash = inst.state_hash()
    actions: List[List[Any]] = []
    step_hashes: List[str] = []
    obs_checkpoints: List[Dict[str, Any]] = []

    actions_this_turn = 0
    prev_turn = inst.raw.turn
    obs_turns_captured = set()

    while not inst.is_terminal() and inst.raw.turn < max_turns:
        pid = inst.raw.current_player
        if actions_this_turn >= MAX_ACTIONS_PER_TURN:
            act = EndTurnAction()
        else:
            act = agents[pid].act(inst.raw, pid, agent_rngs[pid])
        actions.append(encode_action(act))
        inst.step(act)
        h = inst.state_hash()
        step_hashes.append(h)

        if isinstance(act, EndTurnAction):
            actions_this_turn = 0
        else:
            actions_this_turn += 1

        cur_turn = inst.raw.turn
        if cur_turn != prev_turn:
            prev_turn = cur_turn
            if cur_turn in (2, 10, 25, 50) and cur_turn not in obs_turns_captured:
                obs_turns_captured.add(cur_turn)
                obs_checkpoints.append(
                    {
                        "step_idx": step_counter["idx"],
                        "seats": [_obs_summary(inst, 0), _obs_summary(inst, 1)],
                    }
                )

        step_counter["idx"] += 1

    # Always include final obs checkpoint
    if actions:
        obs_checkpoints.append(
            {
                "step_idx": len(actions) - 1,
                "seats": [_obs_summary(inst, 0), _obs_summary(inst, 1)],
            }
        )

    matchup = f"{p0_doctrine or p0_kind}_vs_{p1_doctrine or p1_kind}"
    return {
        "id": game_id,
        "matchup": matchup,
        "map_preset": map_preset,
        "map": map_data,
        "seed": seed,
        "kill_curve_k": 2.0,
        "actions": actions,
        "initial_hash": initial_hash,
        "step_hashes": step_hashes,
        "combat_outcomes": combat_outcomes,
        "obs_checkpoints": obs_checkpoints,
        "final_turn": int(inst.raw.turn),
        "winner": None if inst.raw.winner is None else int(inst.raw.winner),
        "final_state_hash": step_hashes[-1] if step_hashes else initial_hash,
    }


def generate_all(out_path: Path = FIXTURE_PATH) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        pool_dir = Path(td)
        paths = build_map_pool(
            pool_dir,
            [
                PoolSize(16, 8, count=4, ports=2),
                PoolSize(32, 16, count=6, ports=3),
                PoolSize(48, 24, count=6, ports=4),
            ],
            seed=20261125,
            land_fraction=0.02,
            min_port_enemy_home_water=7,
        )
        maps_by_size: Dict[str, List[Dict[str, Any]]] = {"16x8": [], "32x16": [], "48x24": []}
        for p in paths:
            d = json.loads(Path(p).read_text())
            key = f"{d['width']}x{d['height']}"
            maps_by_size[key].append(d)

    doctrines = list(DOCTRINE_NAMES)
    specs: List[Tuple[str, int, str, str | None, str, str | None, int]] = []
    # 12 games on 16x8: 8 FC-vs-FC, 2 FC-vs-Random, 2 Random-vs-Random
    for i in range(8):
        d0 = doctrines[i % len(doctrines)]
        d1 = doctrines[(i + 2) % len(doctrines)]
        specs.append(("16x8", i % 4, "fog_commander", d0, "fog_commander", d1, 80))
    specs.append(("16x8", 0, "fog_commander", doctrines[0], "random", None, 60))
    specs.append(("16x8", 1, "random", None, "fog_commander", doctrines[3], 60))
    specs.append(("16x8", 2, "random", None, "random", None, 25))
    specs.append(("16x8", 3, "random", None, "random", None, 25))

    # 24 games on 32x16: 18 FC-vs-FC, 4 FC-vs-Random, 2 Random-vs-Random
    for i in range(18):
        d0 = doctrines[i % len(doctrines)]
        d1 = doctrines[(i * 2 + 1) % len(doctrines)]
        specs.append(("32x16", i % 6, "fog_commander", d0, "fog_commander", d1, 100))
    for i in range(4):
        d = doctrines[i % len(doctrines)]
        if i % 2 == 0:
            specs.append(("32x16", i % 6, "fog_commander", d, "random", None, 70))
        else:
            specs.append(("32x16", i % 6, "random", None, "fog_commander", d, 70))
    specs.append(("32x16", 4, "random", None, "random", None, 25))
    specs.append(("32x16", 5, "random", None, "random", None, 25))

    # 24 games on 48x24: 20 FC-vs-FC, 4 FC-vs-Random
    for i in range(20):
        d0 = doctrines[i % len(doctrines)]
        d1 = doctrines[(i + 3) % len(doctrines)]
        specs.append(("48x24", i % 6, "fog_commander", d0, "fog_commander", d1, 110))
    for i in range(4):
        d = doctrines[(i + 2) % len(doctrines)]
        if i % 2 == 0:
            specs.append(("48x24", i % 6, "fog_commander", d, "random", None, 80))
        else:
            specs.append(("48x24", i % 6, "random", None, "fog_commander", d, 80))

    games: List[Dict[str, Any]] = []
    for idx, (preset, map_idx, k0, d0, k1, d1, max_turns) in enumerate(specs):
        game_id = f"g{idx:03d}_{preset}"
        g = _play_one_game(
            game_id=game_id,
            map_preset=preset,
            map_data=maps_by_size[preset][map_idx],
            seed=1000 + idx,
            p0_kind=k0,
            p0_doctrine=d0,
            p1_kind=k1,
            p1_doctrine=d1,
            max_turns=max_turns,
        )
        games.append(g)

    payload = {
        "schema_version": 1,
        "num_games": len(games),
        "games": games,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))
    elapsed = time.perf_counter() - t0
    total_actions = sum(len(g["actions"]) for g in games)
    wins = sum(1 for g in games if g["winner"] is not None)
    print(
        f"Generated {len(games)} golden replay fixtures ({total_actions} total steps, "
        f"{wins}/{len(games)} decisive) -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB) in {elapsed:.1f}s"
    )
    return payload


if __name__ == "__main__":
    generate_all()
