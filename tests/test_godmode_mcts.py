"""God-mode MCTS uses the value net as the only leaf evaluator."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from battleboats.agents.godmode_mcts import _evaluate, _evaluate_batch, godmode_mcts_action, load_mcts_value_net
from battleboats.core.gameEngine import gameEngine
from battleboats.envs.observation import build_observation
from battleboats.envs.skill_maps import write_full_map
from battleboats.training.value_net import ValueNet, save_value_checkpoint, value_of_observation, value_of_observations


def test_heuristics_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        import battleboats.agents.heuristics  # noqa: F401


def _tiny_net() -> ValueNet:
    return ValueNet(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0).eval()


class _Terminal:
    def __init__(self, winner):
        self.winner = winner

    def is_terminal(self) -> bool:
        return True


def test_evaluate_terminal_short_circuits_without_the_net() -> None:
    net = _tiny_net()
    rng = random.Random(0)
    assert _evaluate(_Terminal(0), 0, rng, net) == 1.0
    assert _evaluate(_Terminal(0), 1, rng, net) == -1.0
    assert _evaluate(_Terminal(None), 0, rng, net) == 0.0


def test_evaluate_nonterminal_is_value_net_on_fogged_obs(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    net = _tiny_net()
    value = _evaluate(engine, 0, random.Random(0), net, rollout_steps=0)
    obs = build_observation(engine, 0)
    expected = value_of_observation(net, obs["entity_tokens"], obs["globals"])
    assert value == pytest.approx(expected, abs=1e-6)
    assert -1.0 <= value <= 1.0


def test_evaluate_batch_matches_singletons_and_mixes_terminals(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    net = _tiny_net()
    clone = engine.clone()
    mixed = _evaluate_batch(
        [clone, _Terminal(1), engine.clone()],
        [0, 0, 1],
        net,
        rollout_steps=0,
    )
    assert mixed[1] == -1.0
    o0 = build_observation(engine, 0)
    o1 = build_observation(engine, 1)
    assert mixed[0] == pytest.approx(value_of_observation(net, o0["entity_tokens"], o0["globals"]), abs=1e-5)
    assert mixed[2] == pytest.approx(value_of_observation(net, o1["entity_tokens"], o1["globals"]), abs=1e-5)


def test_mcts_picks_a_legal_action(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    pid = engine.current_player
    legal = engine.enumerate_legal(pid)
    action = godmode_mcts_action(
        engine, pid, random.Random(0), _tiny_net(), iterations=8, leaf_batch=4, rollout_steps=4
    )
    assert action in legal


def test_batched_search_picks_a_legal_action(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    pid = engine.current_player
    legal = engine.enumerate_legal(pid)
    action = godmode_mcts_action(
        engine, pid, random.Random(1), _tiny_net(), iterations=16, leaf_batch=16, rollout_steps=0
    )
    assert action in legal


def test_load_mcts_value_net_caches_and_requires_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mcts_value_net(tmp_path / "missing.pt", device="cpu")
    path = tmp_path / "leaf.pt"
    save_value_checkpoint(path, _tiny_net())
    first = load_mcts_value_net(path, device="cpu")
    second = load_mcts_value_net(path, device="cpu")
    assert first is second
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    action = godmode_mcts_action(
        engine, engine.current_player, random.Random(1), first, iterations=4, rollout_steps=0, leaf_batch=2
    )
    assert action is not None


def test_value_of_observations_matches_singletons(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "m.json", 6, 6, seed=1))
    engine.reset(seed=1)
    net = _tiny_net()
    o0 = build_observation(engine, 0)
    o1 = build_observation(engine, 1)
    batched = value_of_observations(
        net, [o0["entity_tokens"], o1["entity_tokens"]], [o0["globals"], o1["globals"]]
    )
    s0 = value_of_observation(net, o0["entity_tokens"], o0["globals"])
    s1 = value_of_observation(net, o1["entity_tokens"], o1["globals"])
    assert batched[0] == pytest.approx(s0, abs=1e-5)
    assert batched[1] == pytest.approx(s1, abs=1e-5)
