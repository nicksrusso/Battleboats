"""Observation, scripted agent, batch, and harvest-flag checks.

Skipped when the extension is not built.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np
import pytest

bb = pytest.importorskip("battleboats_core")

from battleboats.agents.scripted_agent import ScriptedAgent, ScriptedAgentConfig  # noqa: E402
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.envs.observation import TOKEN_DIM, build_observation  # noqa: E402
from tests.golden.backend import decode_action, encode_action  # noqa: E402


def _games():
    path = Path(__file__).resolve().parent / "golden" / "fixtures" / "replays.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["games"]


def test_observation_matches_python_on_fixture_checkpoints(tmp_path: Path) -> None:
    games = [g for g in _games() if g["map_preset"] == "16x8"][:2]
    for game in games:
        map_path = tmp_path / f"{game['id']}.json"
        map_path.write_text(json.dumps(game["map"]))
        python = gameEngine(str(map_path))
        python.reset(seed=int(game["seed"]))
        rust = bb.Game(str(map_path))
        rust.reset(int(game["seed"]))
        rust.set_combat_outcomes(game["combat_outcomes"])
        checkpoints = {0} | {int(cp["step_idx"]) for cp in game["obs_checkpoints"]}
        for step, record in enumerate(game["actions"]):
            if step in checkpoints:
                for pid in (0, 1):
                    obs = build_observation(python, pid)
                    flat, globs, n = rust.observation(pid)
                    got = (
                        np.asarray(flat, dtype=np.float32).reshape(n, TOKEN_DIM)
                        if n
                        else np.zeros((0, TOKEN_DIM), np.float32)
                    )
                    assert np.array_equal(obs["entity_tokens"], got), game["id"]
                    assert np.array_equal(obs["globals"], np.asarray(globs, dtype=np.float32))
                    assert n == obs["entity_tokens"].shape[0]
            python.step(decode_action(record))
            rust.step(record)


def test_scripted_agent_actions_match_on_a_fixture_game(tmp_path: Path) -> None:
    game = next(g for g in _games() if g["id"].startswith("g000_"))
    parts = game["matchup"].split("_vs_")
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(game["map"]))
    python = gameEngine(str(map_path))
    python.reset(seed=int(game["seed"]))
    rust = bb.Game(str(map_path))
    rust.reset(int(game["seed"]))
    rust.set_combat_outcomes(game["combat_outcomes"])
    py_agents = [
        ScriptedAgent(ScriptedAgentConfig(doctrine=parts[0])),
        ScriptedAgent(ScriptedAgentConfig(doctrine=parts[1])),
    ]
    rs_agents = [bb.ScriptedAgent(parts[0]), bb.ScriptedAgent(parts[1])]
    py_rng = [random.Random(int(game["seed"]) * 101 + 1), random.Random(int(game["seed"]) * 101 + 2)]
    rs_rng = [random.Random(int(game["seed"]) * 101 + 1), random.Random(int(game["seed"]) * 101 + 2)]
    for record in game["actions"]:
        pid = python.current_player
        py_action = encode_action(py_agents[pid].act(python, pid, py_rng[pid]))
        rs_action = list(rs_agents[pid].act(rust, pid, rs_rng[pid]))
        assert py_action == rs_action
        python.step(decode_action(py_action))
        rust.step(rs_action)


def test_batch_tracks_single_games(tmp_path: Path) -> None:
    maps = []
    for i, game in enumerate(_games()[:3]):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(game["map"]))
        maps.append(str(path))
    batch = bb.Batch(maps, 2.0)
    seeds = [1, 2, 3]
    batch.reset(seeds)
    singles = []
    for path, seed in zip(maps, seeds):
        game = bb.Game(path, kill_curve_k=2.0)
        game.reset(seed)
        singles.append(game)
    for index, game in enumerate(singles):
        assert batch.state_hash(index) == game.state_hash()
        batch.step(index, ["e"])
        game.step(["e"])
        assert batch.state_hash(index) == game.state_hash()


def test_fog_commander_export_is_the_scripted_agent() -> None:
    assert bb.FogCommander is bb.ScriptedAgent


def test_rust_harvest_shard(tmp_path: Path) -> None:
    from battleboats.rust_play import play_harvest_game

    meta = play_harvest_game(
        {
            "game_idx": 0,
            "kind": "self_play",
            "seed": 3,
            "map_spec": "16x8",
            "actors": {0: "scripted_agent", 1: "scripted_agent"},
            "doctrines": {0: "RAIDER_BLITZ", 1: "ECONOMIC_BOOM"},
            "gamma": 0.99,
            "out_root": str(tmp_path),
            "out_path": str(tmp_path / "game_0.npz"),
            "commit": "test",
            "max_steps": 25,
        }
    )
    assert meta["status"] == "ok"
    assert meta["backend"] == "rust"
    assert meta["num_decisions"] == 25
    assert (tmp_path / "game_0.npz").is_file()
