"""Smoke tests for _do_build_port."""
from pathlib import Path
from typing import List, Tuple

import pytest

from battleboats.core.actions import BuildPortAction, BuildShipAction
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


def _find_water_adjacent_to_empty_land(engine):
    m = engine.map
    for x in range(m.width):
        for y in range(m.height):
            target = (x, y)
            if not m.is_land(target) or m.is_port(target):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                w = (x + dx, y + dy)
                if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
                    return target, w
    return None, None


def _adjacent_water(engine, pos):
    m = engine.map
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        w = (pos[0] + dx, pos[1] + dy)
        if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
            return w
    return None


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def test_builder_builds_port_on_adjacent_empty_land(engine):
    target, spawn = _find_water_adjacent_to_empty_land(engine)
    assert target is not None, "map should contain water-adjacent empty land"
    builder = engine._spawn_ship(0, ShipType.BUILDER, spawn)
    builder_id = builder.id
    engine.step(BuildPortAction(builder_id, target))
    assert engine.map.is_port(target)
    assert int(engine.map.port_owner[target]) == 0
    assert target in engine.players[0].owned_port_positions
    assert engine.ports[target].stockpile == 0
    assert engine.ports[target].owner == 0
    assert engine.ports[target].is_home is False
    # Builder consumed
    assert builder_id not in engine.ships
    assert builder_id not in engine.players[0].owned_ship_ids


def test_non_builder_ship_cannot_build_port(engine):
    target, spawn = _find_water_adjacent_to_empty_land(engine)
    destroyer = engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine.step(BuildPortAction(destroyer.id, target))
    assert not engine.map.is_port(target)
    assert destroyer.id in engine.ships


def test_build_port_on_existing_port_noop(engine):
    own_port = engine.players[0].home_port
    spawn = _adjacent_water(engine, own_port)
    assert spawn is not None
    builder = engine._spawn_ship(0, ShipType.BUILDER, spawn)
    engine.step(BuildPortAction(builder.id, own_port))
    assert builder.id in engine.ships
    # ownership of the existing port unchanged
    assert int(engine.map.port_owner[own_port]) == 0


def test_build_port_on_water_noop(engine):
    line = _find_water_line(engine, 2)
    builder = engine._spawn_ship(0, ShipType.BUILDER, line[0])
    engine.step(BuildPortAction(builder.id, line[1]))
    assert not engine.map.is_port(line[1])
    assert builder.id in engine.ships


def test_build_port_out_of_range_noop(engine):
    """Builder must be Manhattan-1 adjacent; distance-2 attempt is rejected."""
    # Find empty land with a 2-tile water gap to a buildable spawn.
    m = engine.map
    builder_spawn = None
    target = None
    for x in range(m.width):
        for y in range(m.height):
            t = (x, y)
            if not m.is_land(t) or m.is_port(t):
                continue
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                w = (x + dx, y + dy)
                # Need both the spawn tile and the intermediate to be water,
                # but only spawn matters for placement. Manhattan(spawn,target)=2.
                if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
                    builder_spawn = w
                    target = t
                    break
            if target:
                break
        if target:
            break
    assert target is not None
    builder = engine._spawn_ship(0, ShipType.BUILDER, builder_spawn)
    engine.step(BuildPortAction(builder.id, target))
    assert not engine.map.is_port(target)
    assert builder.id in engine.ships


def test_builder_owned_by_enemy_cannot_build_on_p0_turn(engine):
    target, spawn = _find_water_adjacent_to_empty_land(engine)
    builder = engine._spawn_ship(1, ShipType.BUILDER, spawn)  # P1's builder
    # current_player is 0
    engine.step(BuildPortAction(builder.id, target))
    assert not engine.map.is_port(target)
    assert builder.id in engine.ships


def test_new_port_is_enumerated_for_build_ship(engine):
    """After construction, the new port appears in legal BuildShipActions."""
    target, spawn = _find_water_adjacent_to_empty_land(engine)
    builder = engine._spawn_ship(0, ShipType.BUILDER, spawn)
    engine.step(BuildPortAction(builder.id, target))
    legal = engine.enumerate_legal(0)
    builds_from_new_port = [
        a for a in legal if isinstance(a, BuildShipAction) and a.port == target
    ]
    assert len(builds_from_new_port) > 0


def test_witnessed_builder_self_destruct_clears_enemy_sighting(engine):
    """If P1 has a fresh sighting of P0's builder, the build-port consumption
    counts as a witnessed death and clears P1's sighting.
    """
    target, spawn = _find_water_adjacent_to_empty_land(engine)
    builder = engine._spawn_ship(0, ShipType.BUILDER, spawn)
    # Place a P1 scout adjacent to the builder so the build is in fresh view.
    scout_spawn = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        w = (spawn[0] + dx, spawn[1] + dy)
        if (
            engine.map.in_bounds(w)
            and engine.map.is_water(w)
            and not engine.map.is_occupied(w)
        ):
            scout_spawn = w
            break
    assert scout_spawn is not None
    engine._spawn_ship(1, ShipType.DESTROYER, scout_spawn)
    engine._refresh_sightings()
    assert any(s.ship_id == builder.id and s.fresh for s in engine.known_enemy_ships(1))

    engine.step(BuildPortAction(builder.id, target))
    assert not any(s.ship_id == builder.id for s in engine.known_enemy_ships(1))
