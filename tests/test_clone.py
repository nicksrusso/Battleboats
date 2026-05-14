"""Tests for clone() equivalence and reset() determinism."""
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
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
    m = engine.map
    for y in range(m.height):
        for x in range(m.width - length + 1):
            line = [(x + i, y) for i in range(length)]
            if all(m.is_water(p) and not m.is_occupied(p) for p in line):
                return line
    return []


def _assert_states_equal(s1, s2):
    assert np.array_equal(s1["terrain"], s2["terrain"])
    assert np.array_equal(s1["port_owner"], s2["port_owner"])
    assert np.array_equal(s1["ship_at"], s2["ship_at"])
    assert {sid: asdict(s) for sid, s in s1["ships"].items()} == \
           {sid: asdict(s) for sid, s in s2["ships"].items()}
    assert [vars(p) for p in s1["players"]] == [vars(p) for p in s2["players"]]
    assert s1["current_player"] == s2["current_player"]
    assert s1["turn"] == s2["turn"]
    assert s1["winner"] == s2["winner"]


@pytest.fixture
def engine_with_ships():
    e = gameEngine(MAP_PATH)
    e.reset(seed=42)
    line = _find_water_line(e, 4)
    e._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    e._spawn_ship(0, ShipType.MERCHANT, line[1])
    e._spawn_ship(1, ShipType.CRUISER, line[2])
    e._spawn_ship(1, ShipType.BUILDER, line[3])
    return e


def test_clone_immediate_state_matches(engine_with_ships):
    e1 = engine_with_ships
    e2 = e1.clone()
    _assert_states_equal(e1.get_state(), e2.get_state())


def test_clone_mutation_isolated(engine_with_ships):
    """Mutating a clone must not propagate back to the original."""
    e1 = engine_with_ships
    e2 = e1.clone()

    some_sid = next(iter(e2.ships))
    e2.ships[some_sid].position = (0, 0)
    e2.players[0].cash = -9999
    e2.map.terrain[0, 0] = 99

    assert e1.ships[some_sid].position != (0, 0)
    assert e1.players[0].cash != -9999
    assert e1.map.terrain[0, 0] != 99

    # RNG independence: advance e2's RNG, then verify e1's RNG is still at
    # the original state by drawing the same value e2 drew.
    e2_draw = e2.rng.random()
    e1_draw = e1.rng.random()
    assert e1_draw == e2_draw


def test_clone_step_trajectories_match():
    """Identical action sequences on a clone produce identical state.

    Exercises RNG cloning (combat is probabilistic) and dispatch rebinding —
    if handlers were still bound to e1, stepping e2 would silently mutate e1.
    """
    e1 = gameEngine(MAP_PATH)
    e1.reset(seed=42)
    line = _find_water_line(e1, 3)
    atk = e1._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    target = e1._spawn_ship(1, ShipType.SUBMARINE, line[1])

    e2 = e1.clone()
    pre_e1_state = e1.get_state()

    actions = [
        AttackAction(atk.id, target.id),
        EndTurnAction(),
        EndTurnAction(),
    ]

    # Step e2 only. If dispatch was bound to e1, this mutates e1.
    for a in actions:
        e2.step(a)
    _assert_states_equal(pre_e1_state, e1.get_state())

    # Now step e1 through the same actions; final states must match.
    for a in actions:
        e1.step(a)
    _assert_states_equal(e1.get_state(), e2.get_state())


def test_reset_determinism_under_seed():
    """Same seed + same action sequence → identical final state across runs.
    Broader than test_combat::test_deterministic_under_seed which exercises
    _resolve_attack in a tight loop rather than step().
    """
    final_states = []
    for _ in range(2):
        e = gameEngine(MAP_PATH)
        e.reset(seed=1234)
        line = _find_water_line(e, 3)
        atk = e._spawn_ship(0, ShipType.BATTLESHIP, line[0])
        target = e._spawn_ship(1, ShipType.SUBMARINE, line[1])
        for a in [
            AttackAction(atk.id, target.id),
            EndTurnAction(),
            EndTurnAction(),
            AttackAction(atk.id, target.id) if target.id in e.ships else EndTurnAction(),
        ]:
            e.step(a)
        final_states.append(e.get_state())
    _assert_states_equal(final_states[0], final_states[1])
