"""Smoke tests for the heuristic eval module.

Correctness sanity checks, not benchmarks. Coverage:
  - pk() math matches the engine's PK formula and edge cases
  - _ship_value() handles types and ranges correctly
  - _home_threat() math + skip list
  - heuristic_eval() is zero-sum, bounded, terminal-aware, and crash-free
    over random play

Run with: poetry run pytest tests/test_heuristics.py -v
"""

import random

import pytest

from battleboats.agents.heuristics import (
    HOME_THREAT_WEIGHT_LANDING,
    MAT_BUILDER_BONUS,
    MAT_LANDING_BONUS,
    MAT_PORT_VALUE,
    _home_threat,
    _ship_value,
    heuristic_eval,
    pk,
)
from battleboats.agents.random_agent import random_action
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship import Ship
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"


# --------------------------------------------------------------- helpers


def _make_ship(
    ship_type: ShipType,
    strength: float,
    attack_range: int = 1,
    position=(0, 0),
    owner: int = 0,
) -> Ship:
    """Build a Ship with custom stats for unit testing."""
    return Ship(
        id=-1,
        type=ship_type,
        stats=ShipStats(
            speed=1,
            cost=1,
            strength=strength,
            attack_range=attack_range,
            visibility=1.0,
            scouting=1.0,
        ),
        owner=owner,
        position=position,
    )


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@pytest.fixture
def fresh_engine() -> gameEngine:
    engine = gameEngine(map_json_path=MAP_JSON)
    engine.reset(seed=0)
    return engine


# ---------------------------------------------------------------- pk tests


def test_pk_equal_strength_gives_half() -> None:
    """Cruiser-vs-Cruiser has modifier 1.0 and equal strength → x=1, pk=0.5 for any k."""
    atk = _make_ship(ShipType.CRUISER, strength=100)
    defn = _make_ship(ShipType.CRUISER, strength=100)
    assert pk(atk, defn, kill_curve_k=2.0) == pytest.approx(0.5)


def test_pk_strong_vs_weak_is_high() -> None:
    """Battleship strength 125 vs Destroyer strength 25 → pk close to 1."""
    atk = _make_ship(ShipType.BATTLESHIP, strength=125)
    defn = _make_ship(ShipType.DESTROYER, strength=25)
    assert pk(atk, defn, kill_curve_k=2.0) > 0.9


def test_pk_weak_vs_strong_is_low() -> None:
    """Destroyer vs Battleship → pk close to 0."""
    atk = _make_ship(ShipType.DESTROYER, strength=25)
    defn = _make_ship(ShipType.BATTLESHIP, strength=125)
    assert pk(atk, defn, kill_curve_k=2.0) < 0.1


def test_pk_zero_strength_defender_always_destroyed() -> None:
    """Engine's _resolve_attack returns True (destroyed) when defender.strength <= 0.

    The heuristic's pk must mirror that with pk = 1.0. If this fails, the
    helper and the engine have diverged on the strength-0 edge case.
    """
    atk = _make_ship(ShipType.CRUISER, strength=100)
    defn = _make_ship(ShipType.LANDING, strength=0)
    assert pk(atk, defn, kill_curve_k=2.0) == 1.0


def test_pk_higher_k_is_more_decisive() -> None:
    """Higher kill_curve_k → more decisive outcomes (pk moves further from 0.5)."""
    atk = _make_ship(ShipType.BATTLESHIP, strength=125)
    defn = _make_ship(ShipType.DESTROYER, strength=100)  # slight advantage
    pk_low_k = pk(atk, defn, kill_curve_k=1.0)
    pk_high_k = pk(atk, defn, kill_curve_k=4.0)
    assert pk_high_k > pk_low_k  # advantage amplified by higher k


# -------------------------------------------------------- _ship_value tests


def test_ship_value_combat_uses_strength_times_range_plus_one() -> None:
    s = _make_ship(ShipType.CRUISER, strength=100, attack_range=3)
    assert _ship_value(s) == 100 * (1 + 3)


def test_ship_value_landing_adds_bonus() -> None:
    """Landing has strength 0 / range 0 in config — value = bonus only."""
    s = _make_ship(ShipType.LANDING, strength=0, attack_range=0)
    assert _ship_value(s) == MAT_LANDING_BONUS


def test_ship_value_builder_adds_bonus() -> None:
    s = _make_ship(ShipType.BUILDER, strength=0, attack_range=0)
    assert _ship_value(s) == MAT_BUILDER_BONUS


def test_ship_value_merchant_no_bonus() -> None:
    s = _make_ship(ShipType.MERCHANT, strength=0, attack_range=0)
    assert _ship_value(s) == 0


# -------------------------------------------------------- _home_threat tests


def test_home_threat_empty_input_is_zero() -> None:
    assert _home_threat((0, 0), [], _manhattan) == 0


def test_home_threat_skips_merchant_and_builder() -> None:
    merchant = _make_ship(ShipType.MERCHANT, strength=0, position=(0, 1))
    builder = _make_ship(ShipType.BUILDER, strength=0, position=(0, 1))
    assert _home_threat((0, 0), [merchant, builder], _manhattan) == 0


def test_home_threat_landing_uses_landing_weight_not_strength() -> None:
    """Landing has strength 0 but contributes LANDING_WEIGHT to home threat."""
    landing = _make_ship(ShipType.LANDING, strength=0, position=(1, 0))
    expected = HOME_THREAT_WEIGHT_LANDING / (1 + 1)
    assert _home_threat((0, 0), [landing], _manhattan) == pytest.approx(expected)


def test_home_threat_combat_uses_strength() -> None:
    """A Battleship at distance 4 contributes 125 / (1 + 4) = 25."""
    bs = _make_ship(ShipType.BATTLESHIP, strength=125, position=(4, 0))
    assert _home_threat((0, 0), [bs], _manhattan) == pytest.approx(125 / 5)


def test_home_threat_distance_inverse() -> None:
    """Same ship closer to home = more threat."""
    near = _make_ship(ShipType.CRUISER, strength=75, position=(1, 0))
    far = _make_ship(ShipType.CRUISER, strength=75, position=(10, 0))
    assert _home_threat((0, 0), [near], _manhattan) > _home_threat((0, 0), [far], _manhattan)


# ------------------------------------------------------ heuristic_eval tests


def test_heuristic_initial_state_zero_sum(fresh_engine: gameEngine) -> None:
    """Symmetric initial state → heuristic values sum to ~0 across players."""
    h0 = heuristic_eval(fresh_engine, 0)
    h1 = heuristic_eval(fresh_engine, 1)
    assert h0 + h1 == pytest.approx(0.0, abs=1e-9), (
        f"expected zero-sum at symmetric start, got h0={h0}, h1={h1}"
    )


def test_heuristic_initial_state_bounded(fresh_engine: gameEngine) -> None:
    for me in (0, 1):
        h = heuristic_eval(fresh_engine, me)
        assert -1.0 <= h <= 1.0, f"heuristic out of [-1, +1] for me={me}: {h}"


def test_heuristic_deterministic(fresh_engine: gameEngine) -> None:
    a = heuristic_eval(fresh_engine, 0)
    b = heuristic_eval(fresh_engine, 0)
    assert a == b


def test_heuristic_terminal_winner_pm1(fresh_engine: gameEngine) -> None:
    """When engine is terminal, return +1 for winner, -1 for loser."""
    fresh_engine.winner = 0
    assert heuristic_eval(fresh_engine, 0) == 1.0
    assert heuristic_eval(fresh_engine, 1) == -1.0


def test_heuristic_survives_random_play(fresh_engine: gameEngine) -> None:
    """50 random steps from initial state, then heuristic must not crash and stay bounded."""
    rng = random.Random(0)
    for _ in range(50):
        if fresh_engine.is_terminal():
            break
        action = random_action(fresh_engine, fresh_engine.current_player, rng)
        fresh_engine.step(action)
    h0 = heuristic_eval(fresh_engine, 0)
    h1 = heuristic_eval(fresh_engine, 1)
    assert -1.0 <= h0 <= 1.0
    assert -1.0 <= h1 <= 1.0


def test_heuristic_zero_sum_after_random_play(fresh_engine: gameEngine) -> None:
    """After arbitrary play, h(state, 0) + h(state, 1) must remain ~0."""
    rng = random.Random(42)
    for _ in range(100):
        if fresh_engine.is_terminal():
            break
        action = random_action(fresh_engine, fresh_engine.current_player, rng)
        fresh_engine.step(action)
    h0 = heuristic_eval(fresh_engine, 0)
    h1 = heuristic_eval(fresh_engine, 1)
    assert h0 + h1 == pytest.approx(0.0, abs=1e-9)
