"""Smoke tests for gameEngine.enumerate_legal(player_id)."""
from pathlib import Path

import pytest

from battleboats.core.actions import (
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MoveAction,
)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType


MAP_PATH = str(
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)

CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def _find_water_adjacent_to(engine, predicate):
    """Return (target, adjacent_water_tile) for the first land tile matching predicate."""
    m = engine.map
    for x in range(m.width):
        for y in range(m.height):
            target = (x, y)
            if not m.is_land(target) or not predicate(target):
                continue
            for dx, dy in CARDINAL:
                w = (x + dx, y + dy)
                if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
                    return target, w
    return None, None


def _adjacent_water(engine, pos):
    m = engine.map
    for dx, dy in CARDINAL:
        w = (pos[0] + dx, pos[1] + dy)
        if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
            return w
    return None


def test_fresh_game_only_build_ship_and_end_turn(engine):
    legal = engine.enumerate_legal(0)
    types = {type(a) for a in legal}
    assert types == {BuildShipAction, EndTurnAction}
    assert sum(isinstance(a, EndTurnAction) for a in legal) == 1


def test_non_current_player_returns_empty(engine):
    assert engine.enumerate_legal(1) == []


def test_terminal_returns_empty(engine):
    engine.winner = 0
    assert engine.enumerate_legal(0) == []


def test_end_turn_swaps_current_player(engine):
    engine.step(EndTurnAction())
    assert engine.current_player == 1
    assert engine.enumerate_legal(0) == []
    assert len(engine.enumerate_legal(1)) > 0


def test_end_turn_is_always_present_for_current_player(engine):
    assert any(isinstance(a, EndTurnAction) for a in engine.enumerate_legal(0))


def test_build_ship_filtered_by_affordability(engine):
    engine.players[0].cash = 99  # below cheapest cost (100)
    legal = engine.enumerate_legal(0)
    assert not any(isinstance(a, BuildShipAction) for a in legal)


def test_move_emerges_after_build_and_respects_speed_cap(engine):
    build = next(
        a
        for a in engine.enumerate_legal(0)
        if isinstance(a, BuildShipAction) and a.ship_type == ShipType.DESTROYER
    )
    engine.step(build)
    sid = next(iter(engine.players[0].owned_ship_ids))
    ship = engine.ships[sid]

    def moves_for_ship():
        return [
            a
            for a in engine.enumerate_legal(0)
            if isinstance(a, MoveAction) and a.ship_id == sid
        ]

    assert len(moves_for_ship()) >= 1
    for _ in range(ship.stats.speed):
        engine.step(moves_for_ship()[0])
    assert ship.tiles_moved_this_turn == ship.stats.speed
    assert moves_for_ship() == []


def test_builder_emits_build_port_for_adjacent_empty_land(engine):
    target, spawn = _find_water_adjacent_to(
        engine, lambda pos: not engine.map.is_port(pos)
    )
    assert target is not None, "map should contain empty-land tile adjacent to water"
    engine._spawn_ship(0, ShipType.BUILDER, spawn)
    legal = engine.enumerate_legal(0)
    assert any(
        isinstance(a, BuildPortAction) and a.target == target for a in legal
    )


def test_builder_does_not_emit_build_port_for_existing_port(engine):
    own_port = engine.players[0].home_port
    spawn = _adjacent_water(engine, own_port)
    assert spawn is not None, "P0 home port should have an adjacent water tile"
    engine._spawn_ship(0, ShipType.BUILDER, spawn)
    legal = engine.enumerate_legal(0)
    assert not any(
        isinstance(a, BuildPortAction) and a.target == own_port for a in legal
    )


def test_landing_emits_capture_for_adjacent_enemy_port(engine):
    target_port = None
    spawn = None
    for port in engine.players[1].owned_port_positions:
        spawn = _adjacent_water(engine, port)
        if spawn is not None:
            target_port = port
            break
    assert spawn is not None, "expected at least one P1 port with water adjacency"
    engine._spawn_ship(0, ShipType.LANDING, spawn)
    legal = engine.enumerate_legal(0)
    assert any(
        isinstance(a, CapturePortAction) and a.target == target_port for a in legal
    )


def test_landing_does_not_emit_capture_for_own_port(engine):
    own_port = engine.players[0].home_port
    spawn = _adjacent_water(engine, own_port)
    assert spawn is not None
    engine._spawn_ship(0, ShipType.LANDING, spawn)
    legal = engine.enumerate_legal(0)
    assert not any(
        isinstance(a, CapturePortAction) and a.target == own_port for a in legal
    )


def test_non_landing_ship_does_not_emit_capture(engine):
    enemy_port = None
    spawn = None
    for port in engine.players[1].owned_port_positions:
        spawn = _adjacent_water(engine, port)
        if spawn is not None:
            enemy_port = port
            break
    assert spawn is not None
    engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    legal = engine.enumerate_legal(0)
    assert not any(isinstance(a, CapturePortAction) for a in legal)
