"""Rust MCTS engine: no heuristic hydration; observation matches training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

bb = pytest.importorskip("battleboats_core")

from battleboats.agents.godmode_mcts import godmode_mcts_action  # noqa: E402
from battleboats.agents.rust_mcts_engine import RustMctsEngine, encode_action, observation_of  # noqa: E402
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.envs.observation import TOKEN_DIM, build_observation  # noqa: E402
from battleboats.envs.skill_maps import write_full_map  # noqa: E402
from battleboats.training.value_net import ValueNet  # noqa: E402
from tests.golden.backend import decode_action  # noqa: E402


def _tiny_net() -> ValueNet:
    return ValueNet(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0).eval()


def test_observation_matches_python_build_observation(tmp_path: Path) -> None:
    map_path = write_full_map(tmp_path / "m.json", 8, 8, seed=3)
    python = gameEngine(str(map_path))
    python.reset(seed=3)
    rust = RustMctsEngine.create(str(map_path), seed=3)

    def _assert_parity() -> None:
        for pid in (0, 1):
            flat, globs, n = rust.observation(pid)
            tokens, rust_globs = observation_of(rust, pid)
            obs = build_observation(python, pid)
            got = (
                np.asarray(flat, dtype=np.float32).reshape(n, TOKEN_DIM)
                if n
                else np.zeros((0, TOKEN_DIM), np.float32)
            )
            assert np.array_equal(obs["entity_tokens"], got)
            assert np.array_equal(obs["globals"], np.asarray(globs, dtype=np.float32))
            assert np.array_equal(tokens, obs["entity_tokens"])
            assert np.array_equal(rust_globs, obs["globals"])
            assert n == obs["entity_tokens"].shape[0]

    _assert_parity()
    act = encode_action(python.enumerate_legal(python.current_player)[0])
    python.step(decode_action(act))
    rust.step(act)
    _assert_parity()


def test_wrapper_has_no_canonical_hydration() -> None:
    source = Path(__file__).resolve().parents[1] / "battleboats" / "agents" / "rust_mcts_engine.py"
    text = source.read_text()
    assert "canonical_json(" not in text
    assert "_hydrate" not in text
    assert "view_from_canonical" not in text
    assert "json.loads" not in text


def test_rust_mcts_picks_a_legal_action(tmp_path: Path) -> None:
    map_path = write_full_map(tmp_path / "m.json", 6, 6, seed=1)
    rust = RustMctsEngine.create(str(map_path), seed=1)
    pid = rust.current_player
    legal = rust.enumerate_legal(pid)
    action = godmode_mcts_action(
        rust, pid, random.Random(0), _tiny_net(), iterations=8, leaf_batch=4, rollout_steps=8
    )
    assert action in legal


def test_ships_and_ports_are_god_mode_snapshots(tmp_path: Path) -> None:
    map_path = write_full_map(tmp_path / "m.json", 8, 8, seed=3)
    rust = RustMctsEngine.create(str(map_path), seed=3)
    ports = rust.list_ports()
    players = rust.list_players()
    assert rust.list_ships() == []
    assert any(p["is_home"] and p["owner"] == 0 for p in ports)
    assert any(p["is_home"] and p["owner"] == 1 for p in ports)
    assert {p["id"] for p in players} == {0, 1}
    # Buy a landing so the god-mode fleet snapshot is non-empty.
    buy = next(a for a in rust.enumerate_legal(0) if a[0] == "b" and a[-1] == "Landing")
    rust.step(buy)
    ships = rust.list_ships()
    assert len(ships) == 1
    assert ships[0]["type"] == "Landing"
    assert ships[0]["owner"] == 0
