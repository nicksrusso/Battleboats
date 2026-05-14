"""Smoke test for gameEngine.get_state — verifies the returned snapshot is
independent of engine internals."""
from pathlib import Path
from typing import List, Tuple

import pytest

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


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def test_get_state_returns_independent_snapshot(engine):
    line = _find_water_line(engine, 4)
    p0_ship = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    p0_merchant = engine._spawn_ship(0, ShipType.MERCHANT, line[1])
    p1_ship = engine._spawn_ship(1, ShipType.CRUISER, line[2])
    p1_builder = engine._spawn_ship(1, ShipType.BUILDER, line[3])

    state = engine.get_state()

    expected_keys = {
        "terrain", "port_owner", "ship_at",
        "ships", "players",
        "current_player", "turn", "winner",
    }
    assert set(state.keys()) == expected_keys

    assert set(state["ships"].keys()) == {
        p0_ship.id, p0_merchant.id, p1_ship.id, p1_builder.id,
    }

    assert state["current_player"] == engine.current_player
    assert state["turn"] == engine.turn
    assert state["winner"] == engine.winner

    # Grid copies are independent.
    state["terrain"][0, 0] = 99
    assert engine.map.terrain[0, 0] != 99

    # Ship copies are independent — different objects and mutation-isolated.
    assert state["ships"][p0_ship.id] is not engine.ships[p0_ship.id]
    original_position = engine.ships[p0_ship.id].position
    state["ships"][p0_ship.id].position = (0, 0)
    assert engine.ships[p0_ship.id].position == original_position

    # Player copies are independent.
    assert state["players"][0] is not engine.players[0]
    original_cash = engine.players[0].cash
    state["players"][0].cash = -9999
    assert engine.players[0].cash == original_cash
