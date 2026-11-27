"""Combat helpers moved out of the retired heuristic module."""

from __future__ import annotations

from battleboats.core.combat import COMBAT_TYPES, MAT_LANDING_BONUS, pk, ship_value
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType


class _StandIn:
    def __init__(self, ship_type: ShipType) -> None:
        self.type = ship_type
        self.stats = BASE_STATS[ship_type]


def test_combat_types_are_armed_hulls() -> None:
    assert ShipType.DESTROYER in COMBAT_TYPES
    assert ShipType.BATTLESHIP in COMBAT_TYPES
    assert ShipType.LANDING not in COMBAT_TYPES
    assert ShipType.MERCHANT not in COMBAT_TYPES
    assert ShipType.BUILDER not in COMBAT_TYPES


def test_pk_is_one_when_defender_strength_is_zero() -> None:
    class _DeadStats:
        strength = 0

    class _Dead:
        type = ShipType.LANDING
        stats = _DeadStats()

    assert pk(_StandIn(ShipType.DESTROYER), _Dead(), 2.0) == 1.0


def test_pk_is_in_unit_interval() -> None:
    att = _StandIn(ShipType.DESTROYER)
    dfn = _StandIn(ShipType.CRUISER)
    prob = pk(att, dfn, 2.0)
    assert 0.0 < prob < 1.0


def test_ship_value_adds_type_bonus() -> None:
    landing = _StandIn(ShipType.LANDING)
    expected = landing.stats.strength * (1 + landing.stats.attack_range) + MAT_LANDING_BONUS
    assert ship_value(landing) == expected
