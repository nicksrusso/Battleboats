"""Tests for port fog-of-war: PortSighting tracking, ports-as-scouts,
and witness-rules for capture.
"""
from pathlib import Path

import pytest

from battleboats.core.actions import CapturePortAction
from battleboats.core.gameEngine import gameEngine
from battleboats.core.port import Port
from battleboats.core.shipyard.ship_type import ShipType


MAP_PATH = str(
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def _adjacent_water(engine, pos):
    m = engine.map
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        w = (pos[0] + dx, pos[1] + dy)
        if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
            return w
    return None


def _first_non_home_port(engine, player_id):
    home = engine.players[player_id].home_port
    for p in engine.players[player_id].owned_port_positions:
        if p != home:
            return p
    return None


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


def test_initial_no_enemy_port_sightings(engine):
    """At reset (no ships, home ports far apart), neither side has port sightings."""
    assert engine.known_enemy_ports(0) == []
    assert engine.known_enemy_ports(1) == []
    assert engine.visible_enemy_ports(0) == []
    assert engine.visible_enemy_ports(1) == []


def test_ship_adjacent_to_enemy_home_port_creates_is_home_sighting(engine):
    """Scout adjacent to enemy home port produces a fresh sighting marked is_home=True."""
    enemy_home = engine.players[1].home_port
    spawn = _adjacent_water(engine, enemy_home)
    assert spawn is not None
    engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine._refresh_sightings()

    sighting = engine.players[0].port_sightings.get(enemy_home)
    assert sighting is not None
    assert sighting.fresh is True
    assert sighting.last_known_owner == 1
    assert sighting.is_home is True
    assert sighting.position == enemy_home


def test_ship_adjacent_to_enemy_non_home_port_marks_is_home_false(engine):
    """Sighting of a non-home enemy port has is_home=False."""
    enemy_non_home = _first_non_home_port(engine, 1)
    assert enemy_non_home is not None
    spawn = _adjacent_water(engine, enemy_non_home)
    assert spawn is not None
    engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine._refresh_sightings()

    sighting = engine.players[0].port_sightings.get(enemy_non_home)
    assert sighting is not None
    assert sighting.fresh is True
    assert sighting.is_home is False
    assert sighting.last_known_owner == 1


def test_ship_too_far_from_enemy_port_no_sighting(engine):
    """A submarine (scouting=1) at P0's home gets no enemy port sightings —
    range vs a port is 1*4=4, and all P1 ports are far from (0,0)."""
    own_home = engine.players[0].home_port
    spawn = _adjacent_water(engine, own_home)
    assert spawn is not None
    engine._spawn_ship(0, ShipType.SUBMARINE, spawn)
    engine._refresh_sightings()
    assert engine.known_enemy_ports(0) == []


def test_port_sighting_becomes_stale_when_only_scout_destroyed(engine):
    """When the only scout in range dies, the sighting is preserved but stale,
    with last_known_owner and is_home frozen at the last fresh observation."""
    enemy_home = engine.players[1].home_port
    spawn = _adjacent_water(engine, enemy_home)
    destroyer = engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine._refresh_sightings()
    assert engine.players[0].port_sightings[enemy_home].fresh is True

    engine._destroy_ship(destroyer.id)
    engine._refresh_sightings()

    sighting = engine.players[0].port_sightings.get(enemy_home)
    assert sighting is not None  # preserved
    assert sighting.fresh is False  # stale
    assert sighting.last_known_owner == 1
    assert sighting.is_home is True


def test_friendly_port_scouts_enemy_ship(engine):
    """A friendly port can spot enemy ships in range even with no friendly ships."""
    # Place a P1 destroyer (visibility=3) adjacent to P0's home port.
    own_home = engine.players[0].home_port
    spawn = _adjacent_water(engine, own_home)
    assert spawn is not None
    enemy_destroyer = engine._spawn_ship(1, ShipType.DESTROYER, spawn)
    engine._refresh_sightings()

    # P0 has no ships of their own — only the home port. The port-scout
    # (PORT_SCOUTING=4 * destroyer.visibility=3 = 12) covers the adjacent tile.
    visible = engine.visible_enemy_ships(0)
    assert any(s.id == enemy_destroyer.id for s in visible)


def test_port_to_port_detection(engine):
    """Friendly port spots enemy port within PORT_SCOUTING * PORT_VISIBILITY (16)."""
    # Find any P1 port and a land tile within 16 manhattan distance of it
    # that isn't already a port. Plant a P0 port there directly (bypass build
    # mechanics) and verify P0 then has a sighting of the P1 port.
    target_port = None
    plant_pos = None
    for p1_port in engine.players[1].owned_port_positions:
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                if abs(dx) + abs(dy) > 15 or (dx == 0 and dy == 0):
                    continue
                cand = (p1_port[0] + dx, p1_port[1] + dy)
                if not engine.map.in_bounds(cand):
                    continue
                if not engine.map.is_land(cand) or engine.map.is_port(cand):
                    continue
                target_port = p1_port
                plant_pos = cand
                break
            if plant_pos:
                break
        if plant_pos:
            break
    if plant_pos is None:
        pytest.skip("no land tile within 16 of any P1 port for port-to-port test")

    engine.map.add_port(plant_pos, 0)
    engine.players[0].owned_port_positions.add(plant_pos)
    engine.ports[plant_pos] = Port(position=plant_pos, owner=0, is_home=False, stockpile=0)
    engine._refresh_sightings()

    sighting = engine.players[0].port_sightings.get(target_port)
    assert sighting is not None
    assert sighting.fresh is True
    assert sighting.last_known_owner == 1


def test_captured_port_removed_from_captor_port_sightings(engine):
    """After capture, the captor's port_sightings drops the now-owned entry."""
    target_port = None
    spawn = None
    for port in engine.players[1].owned_port_positions:
        s = _adjacent_water(engine, port)
        if s is not None:
            target_port = port
            spawn = s
            break
    assert target_port is not None
    landing = engine._spawn_ship(0, ShipType.LANDING, spawn)

    engine._refresh_sightings()
    assert target_port in engine.players[0].port_sightings  # sighted as enemy

    engine.step(CapturePortAction(landing.id, target_port))
    assert target_port not in engine.players[0].port_sightings  # now own, dropped


def test_clone_preserves_port_sightings(engine):
    """clone() deepcopies port_sightings — mutating clone doesn't affect original."""
    enemy_home = engine.players[1].home_port
    spawn = _adjacent_water(engine, enemy_home)
    engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine._refresh_sightings()
    assert enemy_home in engine.players[0].port_sightings

    e2 = engine.clone()
    assert enemy_home in e2.players[0].port_sightings
    del e2.players[0].port_sightings[enemy_home]
    assert enemy_home in engine.players[0].port_sightings


def test_witnessed_capture_updates_other_observer(engine):
    """If P1 has a scout watching their port at the moment P0 captures it,
    P1's next-refresh sighting flips last_known_owner to P0 (the captor)."""
    # Find a P1 port with water adjacency for a landing ship spawn.
    target_port = None
    spawn = None
    for port in engine.players[1].owned_port_positions:
        s = _adjacent_water(engine, port)
        if s is not None:
            target_port = port
            spawn = s
            break
    assert target_port is not None

    # P0 landing ship adjacent to the port.
    landing = engine._spawn_ship(0, ShipType.LANDING, spawn)
    # P1 scout also adjacent (different adjacency) so P1 will witness the capture.
    p1_scout_pos = None
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cand = (target_port[0] + dx, target_port[1] + dy)
        if cand == spawn:
            continue
        if (
            engine.map.in_bounds(cand)
            and engine.map.is_water(cand)
            and not engine.map.is_occupied(cand)
        ):
            p1_scout_pos = cand
            break
    if p1_scout_pos is None:
        pytest.skip("port has only one water adjacency; cannot place witness scout")
    engine._spawn_ship(1, ShipType.DESTROYER, p1_scout_pos)
    engine._refresh_sightings()
    # P1 doesn't get a port_sighting on a port they own.
    assert target_port not in engine.players[1].port_sightings

    engine.step(CapturePortAction(landing.id, target_port))
    # After capture, P1 no longer owns the port; refresh ran inside step().
    # P1's destroyer is still adjacent; they witness the capture.
    p1_sighting = engine.players[1].port_sightings.get(target_port)
    assert p1_sighting is not None
    assert p1_sighting.fresh is True
    assert p1_sighting.last_known_owner == 0
