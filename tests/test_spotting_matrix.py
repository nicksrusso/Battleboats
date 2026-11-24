"""Spotting matrix (config/visibilityMatrix.csv): loading, asymmetry, engine use."""
import shutil
from pathlib import Path

import pytest

from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard import ship_data
from battleboats.core.shipyard.ship_data import (
    PORT,
    SPOT_KEYS,
    SPOT_MATRIX,
    _load_spot_matrix,
    spot_distance,
)
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import SPOTS_AT_FEATURE, SPOTTED_AT_FEATURE

CONFIG = Path(ship_data.__file__).resolve().parents[1] / "config"
MAP_PATH = str(CONFIG / "map.json")


def test_matrix_loaded_for_every_pair() -> None:
    for a in SPOT_KEYS:
        for b in SPOT_KEYS:
            assert spot_distance(a, b) >= 0
    assert set(SPOT_MATRIX) == set(SPOT_KEYS)
    # Nick's warship block, read straight from the CSV.
    assert spot_distance(ShipType.CARRIER, ShipType.CARRIER) == 6
    assert spot_distance(ShipType.CARRIER, ShipType.SUBMARINE) == 2
    assert spot_distance(ShipType.BATTLESHIP, ShipType.SUBMARINE) == 1


def test_matrix_is_asymmetric() -> None:
    assert spot_distance(ShipType.DESTROYER, ShipType.SUBMARINE) == 3
    assert spot_distance(ShipType.SUBMARINE, ShipType.DESTROYER) == 2


def test_stats_no_longer_carry_coefficients() -> None:
    stats = ship_data.BASE_STATS[ShipType.DESTROYER]
    assert not hasattr(stats, "visibility")
    assert not hasattr(stats, "scouting")


def _water_line(engine, length: int, port_buffer: int = 17):
    m = engine.map
    ports = list(m.port_positions)
    for y in range(m.height):
        for x in range(m.width - length + 1):
            line = [(x + i, y) for i in range(length)]
            if not all(m.is_water(p) and not m.is_occupied(p) for p in line):
                continue
            if any(abs(t[0] - p[0]) + abs(t[1] - p[1]) < port_buffer for t in line for p in ports):
                continue
            return line
    return []


@pytest.mark.parametrize("distance", [1, 2, 3, 4])
def test_engine_detection_matches_matrix(distance: int) -> None:
    engine = gameEngine(MAP_PATH)
    engine.reset(seed=0)
    line = _water_line(engine, distance + 1)
    assert line, "no clear water line on map.json"
    dd = engine._spawn_ship(0, ShipType.DESTROYER, line[0])
    sub = engine._spawn_ship(1, ShipType.SUBMARINE, line[distance])
    engine._refresh_sightings()
    dd_sees = [s.id for s in engine.visible_enemy_ships(0)] == [sub.id]
    sub_sees = [s.id for s in engine.visible_enemy_ships(1)] == [dd.id]
    assert dd_sees == (distance <= spot_distance(ShipType.DESTROYER, ShipType.SUBMARINE))
    assert sub_sees == (distance <= spot_distance(ShipType.SUBMARINE, ShipType.DESTROYER))
    assert engine._detection_distance(dd, sub) == 3
    assert engine._detection_distance(sub, dd) == 2


def test_engine_detection_routes_every_ship_pair_through_matrix() -> None:
    engine = gameEngine(MAP_PATH)
    engine.reset(seed=0)
    line = _water_line(engine, 2)
    for a in ShipType:
        for b in ShipType:
            x = engine._spawn_ship(0, a, line[0])
            y = engine._spawn_ship(1, b, line[1])
            assert engine._detection_distance(x, y) == spot_distance(a, b)
            engine._destroy_ship(x.id)
            engine._destroy_ship(y.id)
    home = engine.players[0].home_port
    port = engine.ports[home]
    ship = engine._spawn_ship(1, ShipType.SUBMARINE, line[0])
    assert engine._detection_distance(port, ship) == spot_distance(PORT, ShipType.SUBMARINE)
    assert engine._detection_distance(ship, port) == spot_distance(ShipType.SUBMARINE, PORT)


def test_missing_pair_errors(tmp_path: Path) -> None:
    rows = (CONFIG / "visibilityMatrix.csv").read_text().splitlines()
    # Drop the Builder column value from the Carrier row.
    header = rows[0].split(",")
    col = header.index("Builder")
    for i, row in enumerate(rows):
        cells = row.split(",")
        if cells[0] == "Carrier":
            cells[col] = ""
            rows[i] = ",".join(cells)
    bad = tmp_path / "visibilityMatrix.csv"
    bad.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="Carrier->Builder"):
        _load_spot_matrix(bad)


def test_missing_row_errors(tmp_path: Path) -> None:
    rows = [r for r in (CONFIG / "visibilityMatrix.csv").read_text().splitlines()
            if not r.startswith("Merchant,")]
    bad = tmp_path / "visibilityMatrix.csv"
    bad.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="Merchant->"):
        _load_spot_matrix(bad)


def test_observation_spotting_features() -> None:
    for t in ShipType:
        assert 0.0 < SPOTTED_AT_FEATURE[t] <= 1.0
        assert 0.0 < SPOTS_AT_FEATURE[t] <= 1.0
    # The submarine is the hardest hull to see.
    assert SPOTTED_AT_FEATURE[ShipType.SUBMARINE] == min(SPOTTED_AT_FEATURE.values())
