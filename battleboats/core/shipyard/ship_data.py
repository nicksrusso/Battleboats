"""Ship base stats and combat modifiers, loaded from CSV at import time.

Source of truth: ./core/config/shipTypes.csv and ./core/config/attackModifiers.csv.
Edit those files (not this module) to retune balance.
"""
import csv
from pathlib import Path
from typing import Dict

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
                visibility=float(row["visibility coef"]),
                scouting=float(row["scouting coef"]),
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


BASE_STATS: Dict[ShipType, ShipStats] = _load_base_stats()
ATTACK_MODIFIERS: Dict[ShipType, Dict[ShipType, float]] = _load_attack_modifiers()


def attack_modifier(attacker: ShipType, defender: ShipType) -> float:
    """Multiplier on attacker's strength against this defender. Missing entries → 1.0."""
    return ATTACK_MODIFIERS.get(attacker, {}).get(defender, 1.0)
