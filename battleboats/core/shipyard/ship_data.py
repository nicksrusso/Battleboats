"""Ship base stats and combat modifiers, loaded from CSV at import time.

Source of truth: ./core/config/shipTypes.csv, ./core/config/attackModifiers.csv
and ./core/config/visibilityMatrix.csv. Edit those files (not this module) to
retune balance.
"""
import csv
from pathlib import Path
from typing import Dict, Union

from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.shipyard.ship_stats import ShipStats

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _normalize(name: str) -> ShipType:
    return ShipType(name.strip().title())


def _load_base_stats() -> Dict[ShipType, ShipStats]:
    table: Dict[ShipType, ShipStats] = {}
    with open(_CONFIG_DIR / "shipTypes.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _normalize(row["Ship Type"])
            table[t] = ShipStats(
                speed=int(row["speed"]),
                cost=int(row["Cost"]),
                strength=float(row["Strength"]),
                attack_range=int(row["attack range"]),
            )
    missing = set(ShipType) - table.keys()
    if missing:
        raise ValueError(f"shipTypes.csv is missing rows for: {sorted(t.name for t in missing)}")
    return table


def _load_attack_modifiers() -> Dict[ShipType, Dict[ShipType, float]]:
    table: Dict[ShipType, Dict[ShipType, float]] = {}
    with open(_CONFIG_DIR / "attackModifiers.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        defenders = [_normalize(c) for c in header[1:] if c.strip()]
        for row in reader:
            if not row or not row[0].strip():
                continue
            attacker = _normalize(row[0])
            table[attacker] = {
                d: float(v) for d, v in zip(defenders, row[1:]) if v.strip()
            }
    return table


# Spotter/target key for a port in the spotting matrix. Ports are not ships, so
# they get their own row and column in visibilityMatrix.csv.
PORT = "Port"
SpotKey = Union[ShipType, str]
SPOT_KEYS: tuple = tuple(ShipType) + (PORT,)


def _spot_key(name: str) -> SpotKey:
    name = name.strip()
    if name.lower() == PORT.lower():
        return PORT
    return _normalize(name)


def _load_spot_matrix(path: Path = None) -> Dict[SpotKey, Dict[SpotKey, float]]:
    """Row spots column at this Manhattan distance (tiles). Asymmetric.

    Every (spotter, target) pair over all ShipTypes plus ``PORT`` must be
    present; a missing row, column, or blank cell raises ``ValueError``.
    """
    path = Path(path) if path is not None else _CONFIG_DIR / "visibilityMatrix.csv"
    table: Dict[SpotKey, Dict[SpotKey, float]] = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        targets = [_spot_key(c) for c in header[1:] if c.strip()]
        for row in reader:
            if not row or not row[0].strip():
                continue
            spotter = _spot_key(row[0])
            cells = {}
            for t, v in zip(targets, row[1:]):
                if v.strip():
                    d = float(v)
                    if d < 0:
                        raise ValueError(f"{path.name}: negative distance {spotter}->{t}")
                    cells[t] = d
            table[spotter] = cells
    missing = [
        f"{getattr(a, 'value', a)}->{getattr(b, 'value', b)}"
        for a in SPOT_KEYS
        for b in SPOT_KEYS
        if b not in table.get(a, {})
    ]
    if missing:
        raise ValueError(f"{path.name} is missing spotting pairs: {missing}")
    return table


BASE_STATS: Dict[ShipType, ShipStats] = _load_base_stats()
ATTACK_MODIFIERS: Dict[ShipType, Dict[ShipType, float]] = _load_attack_modifiers()
SPOT_MATRIX: Dict[SpotKey, Dict[SpotKey, float]] = _load_spot_matrix()


def spot_distance(spotter: SpotKey, target: SpotKey) -> float:
    """Max Manhattan distance at which ``spotter`` sees ``target``.

    Keys are ShipType members or ``PORT``. Not symmetric: a Destroyer sees a
    Submarine farther out than the Submarine sees the Destroyer.
    """
    return SPOT_MATRIX[spotter][target]


def attack_modifier(attacker: ShipType, defender: ShipType) -> float:
    """Multiplier on attacker's strength against this defender. Missing entries → 1.0."""
    return ATTACK_MODIFIERS.get(attacker, {}).get(defender, 1.0)
