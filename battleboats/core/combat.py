"""Public combat helpers shared by agents and the play UI.

These are game-rule calculations (hit chance, material value, combat vs
non-combat types), not a state evaluator. They used to live next to the
retired hand-coded heuristic; they belong here so that module can stay gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

from battleboats.core.shipyard.ship_data import BASE_STATS, attack_modifier
from battleboats.core.shipyard.ship_type import ShipType

if TYPE_CHECKING:
    from battleboats.core.shipyard.ship import Ship

# Type bonuses added to strength*(1+range) in ship_value.
MAT_LANDING_BONUS = 500.0
MAT_BUILDER_BONUS = 100.0
MAT_MERCHANT_BONUS = 300.0
MAT_PORT_VALUE = 400.0

# Combat hulls: anything with a positive attack range and strength.
# Derived from BASE_STATS so a new armed type is included automatically.
COMBAT_TYPES: FrozenSet[ShipType] = frozenset(
    t for t in ShipType if t in BASE_STATS and BASE_STATS[t].attack_range > 0 and BASE_STATS[t].strength > 0
)


def pk(attacker: "Ship", defender: "Ship", kill_curve_k: float) -> float:
    """Probability that `attacker` destroys `defender` in one attack.

    Mirrors gameEngine._resolve_attack's formula without consuming RNG.
    Returns 1.0 when defender.strength <= 0 (engine treats a strength-0
    defender as always-destroyed).
    """
    if defender.stats.strength <= 0:
        return 1.0
    x = attacker.stats.strength * attack_modifier(attacker=attacker.type, defender=defender.type) / defender.stats.strength
    return x**kill_curve_k / (1 + x**kill_curve_k)


def ship_value(ship: "Ship") -> float:
    """Material value of one ship: strength*(1+range) plus type bonus."""
    base = ship.stats.strength * (1 + ship.stats.attack_range)
    if ship.type is ShipType.LANDING:
        base += MAT_LANDING_BONUS
    elif ship.type is ShipType.BUILDER:
        base += MAT_BUILDER_BONUS
    elif ship.type is ShipType.MERCHANT:
        base += MAT_MERCHANT_BONUS
    return base
