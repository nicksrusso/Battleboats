from dataclasses import dataclass
from typing import Tuple
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.shipyard.ship_stats import ShipStats


@dataclass(slots=True)
class Ship:
    """Per-instance ship record. Stats are frozen at build time from owner's tech."""

    id: int
    type: ShipType
    stats: ShipStats
    owner: int
    position: Tuple[int, int]
    tiles_moved_this_turn: int = 0
    has_attacked: bool = False
    cargo: int = 0  # merchant load; persists across turns until unloaded.
    # Bought this turn: no actions at all (move, attack, load, unload, capture,
    # port build) until its owner's next turn. Set only by a BuildShipAction;
    # ships placed by a scenario or fleet start are not bought.
    bought_this_turn: bool = False
    built_turn: int = -1  # game turn of the BuildShipAction; -1 if never bought

    def reset_turn_flags(self) -> None:
        self.tiles_moved_this_turn = 0
        self.has_attacked = False
        self.bought_this_turn = False
