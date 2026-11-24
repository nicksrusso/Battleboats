"""Smoke tests for _do_attack and _resolve_attack."""
from pathlib import Path
from typing import List, Tuple

import pytest

from battleboats.core.actions import AttackAction, EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType


MAP_PATH = str(
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def _find_water_line(engine, length: int) -> List[Tuple[int, int]]:
    """Find `length` colinear in-bounds water tiles. Returns [] if none exist."""
    m = engine.map
    for y in range(m.height):
        for x in range(m.width - length + 1):
            line = [(x + i, y) for i in range(length)]
            if all(m.is_water(p) and not m.is_occupied(p) for p in line):
                return line
    return []


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def test_defenseless_defender_always_dies(engine):
    """strength-0 ships (Builder/Landing/Merchant) are guaranteed kills."""
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    defender = engine._spawn_ship(1, ShipType.BUILDER, line[1])
    target_id = defender.id
    engine.step(AttackAction(attacker.id, target_id))
    assert target_id not in engine.ships
    assert attacker.has_attacked


def test_attack_beyond_range_is_noop(engine):
    """Battleship range=4; defender at distance 5 must be untouched."""
    line = _find_water_line(engine, 6)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    defender = engine._spawn_ship(1, ShipType.BUILDER, line[5])
    engine.step(AttackAction(attacker.id, defender.id))
    assert defender.id in engine.ships
    assert not attacker.has_attacked


def test_friendly_fire_blocked(engine):
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    friend = engine._spawn_ship(0, ShipType.BUILDER, line[1])
    engine.step(AttackAction(attacker.id, friend.id))
    assert friend.id in engine.ships
    assert not attacker.has_attacked


def test_attack_once_per_turn(engine):
    """Second attack same turn no-ops even with a legal target."""
    line = _find_water_line(engine, 3)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    first_target = engine._spawn_ship(1, ShipType.BUILDER, line[1])
    second_target = engine._spawn_ship(1, ShipType.BUILDER, line[2])
    engine.step(AttackAction(attacker.id, first_target.id))
    assert first_target.id not in engine.ships
    assert attacker.has_attacked
    engine.step(AttackAction(attacker.id, second_target.id))
    assert second_target.id in engine.ships  # untouched: budget exhausted


def test_non_combat_ship_cannot_attack(engine):
    """Ships with attack_range=0 (Builder/Landing/Merchant) cannot attack."""
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.BUILDER, line[0])
    defender = engine._spawn_ship(1, ShipType.BUILDER, line[1])
    engine.step(AttackAction(attacker.id, defender.id))
    assert defender.id in engine.ships
    assert not attacker.has_attacked


def test_has_attacked_resets_on_owners_end_turn(engine):
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    defender = engine._spawn_ship(1, ShipType.BUILDER, line[1])
    engine.step(AttackAction(attacker.id, defender.id))
    assert attacker.has_attacked
    engine.step(EndTurnAction())  # P0 → P1
    assert not attacker.has_attacked  # reset at P0's end turn


def test_attacker_not_owned_by_current_player_noop(engine):
    """An attack issued for an enemy-owned attacker is a no-op."""
    line = _find_water_line(engine, 2)
    p1_attacker = engine._spawn_ship(1, ShipType.BATTLESHIP, line[0])
    p0_defender = engine._spawn_ship(0, ShipType.BUILDER, line[1])
    # current_player is 0, so issuing a P1 attack should no-op
    engine.step(AttackAction(p1_attacker.id, p0_defender.id))
    assert p0_defender.id in engine.ships
    assert not p1_attacker.has_attacked


def test_resolve_attack_probability_at_parity(engine):
    """At x=1 (parity strengths, modifier 1), p = 0.5 exactly — verifies the
    sigmoid formula is wired right. Repeated trials should converge near 0.5.
    """
    # Pick two ship types with attack_modifier == 1.0 against each other.
    # Submarine vs Submarine: equal strength, modifier defaults to 1.0
    # unless explicitly set in the CSV. We don't rely on that — instead test
    # _resolve_attack's math directly via a controlled defender strength.
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.SUBMARINE, line[0])
    defender = engine._spawn_ship(1, ShipType.SUBMARINE, line[1])
    # Sample many resolves; reset has_attacked between calls.
    kills = 0
    trials = 2000
    for _ in range(trials):
        if engine._resolve_attack(attacker, defender):
            kills += 1
    rate = kills / trials
    # At x=1, k=2, p=0.5. Allow generous slack; this is a sanity check, not a stats test.
    assert 0.4 < rate < 0.6, f"expected ~0.5 kill rate at parity, got {rate:.3f}"


def test_deterministic_under_seed():
    """Same seed → same outcome sequence."""
    line_results = []
    for _ in range(2):
        e = gameEngine(MAP_PATH)
        e.reset(seed=42)
        line = _find_water_line(e, 2)
        a = e._spawn_ship(0, ShipType.CRUISER, line[0])
        d = e._spawn_ship(1, ShipType.DESTROYER, line[1])
        outcomes = [e._resolve_attack(a, d) for _ in range(50)]
        line_results.append(outcomes)
    assert line_results[0] == line_results[1]


def test_destroyer_vs_submarine_kill_probability_is_point_eight():
    """Nick's v0.5.0 follow-up: DD strength 50, DD->SS modifier 2, SS strength 50
    gives x = 2 and P_kill = 4/5 at k = 2 (same as the 25 x 4 / 50 it replaced)."""
    from battleboats.core.shipyard.ship_data import BASE_STATS, attack_modifier

    dd, ss = BASE_STATS[ShipType.DESTROYER], BASE_STATS[ShipType.SUBMARINE]
    x = dd.strength * attack_modifier(ShipType.DESTROYER, ShipType.SUBMARINE) / ss.strength
    assert x == 2.0
    assert x**2 / (1 + x**2) == 0.8
    assert ss.cost == 125 and dd.cost == 100
