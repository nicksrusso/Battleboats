"""Smoke tests for fog-of-war: visible_enemy_ships, known_enemy_ships, _refresh_sightings."""
from pathlib import Path
from typing import List, Tuple

import pytest

from battleboats.core.actions import AttackAction
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType


MAP_PATH = str(
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def _find_water_line(engine, length: int, port_buffer: int = 17) -> List[Tuple[int, int]]:
    """Find a water line whose every tile is at least `port_buffer` manhattan
    distance from any port. The buffer keeps these ship-on-ship visibility
    tests free of port-as-scout interference (max port-scout range is
    PORT_SCOUTING * max_ship_visibility = 4 * 4 = 16; buffer 17 is safe).
    """
    m = engine.map
    ports = list(m.port_positions)
    for y in range(m.height):
        for x in range(m.width - length + 1):
            line = [(x + i, y) for i in range(length)]
            if not all(m.is_water(p) and not m.is_occupied(p) for p in line):
                continue
            if any(
                abs(tile[0] - port[0]) + abs(tile[1] - port[1]) < port_buffer
                for tile in line
                for port in ports
            ):
                continue
            return line
    return []


def _teleport(engine, ship, dest):
    """Test-only helper: move a ship to an arbitrary water tile, bypassing turn budget."""
    engine.map.relocate_ship(ship.id, ship.position, dest)
    ship.position = dest


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def test_no_sightings_on_fresh_reset(engine):
    assert engine.visible_enemy_ships(0) == []
    assert engine.visible_enemy_ships(1) == []
    assert engine.known_enemy_ships(0) == []
    assert engine.known_enemy_ships(1) == []


def test_adjacent_enemy_becomes_visible(engine):
    line = _find_water_line(engine, 2)
    engine._spawn_ship(0, ShipType.DESTROYER, line[0])  # scouting 3
    target = engine._spawn_ship(1, ShipType.BUILDER, line[1])  # visibility 3
    engine._refresh_sightings()
    visible = engine.visible_enemy_ships(0)
    assert [s.id for s in visible] == [target.id]
    known = engine.known_enemy_ships(0)
    assert len(known) == 1
    assert known[0].fresh and known[0].position == line[1]


def test_out_of_range_enemy_not_seen(engine):
    """Battleship scouting=2 vs Submarine visibility=1 → detection 2. Place 3 apart."""
    line = _find_water_line(engine, 4)
    engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    engine._spawn_ship(1, ShipType.SUBMARINE, line[3])
    engine._refresh_sightings()
    assert engine.visible_enemy_ships(0) == []
    assert engine.known_enemy_ships(0) == []


def test_asymmetric_detection(engine):
    """Sub (s=1, v=1) vs Battleship (s=2, v=4) at distance 3:
    Sub-side sees Battleship (1*4=4 >= 3); Battleship does NOT see Sub (2*1=2 < 3).
    """
    line = _find_water_line(engine, 4)
    sub = engine._spawn_ship(0, ShipType.SUBMARINE, line[0])
    battleship = engine._spawn_ship(1, ShipType.BATTLESHIP, line[3])
    engine._refresh_sightings()
    assert [s.id for s in engine.visible_enemy_ships(0)] == [battleship.id]
    assert engine.visible_enemy_ships(1) == []
    assert engine.known_enemy_ships(1) == []
    assert [s.ship_id for s in engine.known_enemy_ships(0)] == [battleship.id]
    _ = sub  # silence unused; ship registered via _spawn_ship


def test_sighting_goes_stale_when_scout_leaves(engine):
    """Sighting flips fresh=False but preserves last-known position and turn_seen."""
    line = _find_water_line(engine, 4)
    scout = engine._spawn_ship(0, ShipType.SUBMARINE, line[0])  # scout=1
    target = engine._spawn_ship(1, ShipType.SUBMARINE, line[1])  # vis=1, detection=1
    engine._refresh_sightings()
    initial = next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id)
    assert initial.fresh and initial.position == line[1]
    last_position = initial.position
    last_turn = initial.turn_seen

    _teleport(engine, scout, line[3])  # distance 2 from target — out of detection range
    engine._refresh_sightings()

    after = next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id)
    assert not after.fresh
    assert after.position == last_position
    assert after.turn_seen == last_turn
    assert engine.visible_enemy_ships(0) == []


def test_fresh_sighting_updates_position_on_enemy_move(engine):
    """While visible, sighting always reflects the enemy's current tile."""
    line = _find_water_line(engine, 4)
    engine._spawn_ship(0, ShipType.DESTROYER, line[0])  # detection vs another destroyer = 3*4=12
    target = engine._spawn_ship(1, ShipType.DESTROYER, line[1])
    engine._refresh_sightings()
    assert next(iter(engine.known_enemy_ships(0))).position == line[1]
    _teleport(engine, target, line[2])
    engine._refresh_sightings()
    s = next(iter(engine.known_enemy_ships(0)))
    assert s.fresh and s.position == line[2]


def test_witness_kill_clears_sighting(engine):
    """Killer with a fresh view of the dying ship forgets it on death."""
    line = _find_water_line(engine, 2)
    attacker = engine._spawn_ship(0, ShipType.BATTLESHIP, line[0])
    defender = engine._spawn_ship(1, ShipType.BUILDER, line[1])
    engine._refresh_sightings()
    assert any(s.ship_id == defender.id for s in engine.known_enemy_ships(0))
    engine.step(AttackAction(attacker.id, defender.id))  # builder strength=0 → guaranteed kill
    assert defender.id not in engine.ships
    assert not any(s.ship_id == defender.id for s in engine.known_enemy_ships(0))


def test_stale_sighting_persists_through_unwitnessed_death(engine):
    """Ship dies while NOT in observer's fresh view → observer keeps the stale record.
    Uses sub-vs-sub (detection = 1*1 = 1) so a 2-tile gap reliably hides the target.
    """
    line = _find_water_line(engine, 4)
    scout = engine._spawn_ship(0, ShipType.SUBMARINE, line[0])
    target = engine._spawn_ship(1, ShipType.SUBMARINE, line[1])
    engine._refresh_sightings()
    assert any(s.ship_id == target.id and s.fresh for s in engine.known_enemy_ships(0))
    _teleport(engine, scout, line[3])
    engine._refresh_sightings()
    assert not next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id).fresh

    engine._destroy_ship(target.id)  # observer is out of view; should NOT clear
    surviving_ids = {s.ship_id for s in engine.known_enemy_ships(0)}
    assert target.id in surviving_ids
    assert not next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id).fresh


def test_sightings_persist_across_turns_without_decay(engine):
    """v1 has no time decay: stale sightings stick around indefinitely.
    Uses sub-vs-sub for tight detection (range 1) so the scout cleanly drops view.
    """
    from battleboats.core.actions import EndTurnAction

    line = _find_water_line(engine, 4)
    scout = engine._spawn_ship(0, ShipType.SUBMARINE, line[0])
    target = engine._spawn_ship(1, ShipType.SUBMARINE, line[1])
    engine._refresh_sightings()
    _teleport(engine, scout, line[3])
    engine._refresh_sightings()
    stale_turn = next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id).turn_seen

    # Burn several full rounds.
    for _ in range(6):
        engine.step(EndTurnAction())
    s = next(s for s in engine.known_enemy_ships(0) if s.ship_id == target.id)
    assert not s.fresh
    assert s.turn_seen == stale_turn  # turn_seen NEVER advances while stale
