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

    def reset_turn_flags(self) -> None:
        self.tiles_moved_this_turn = 0
        self.has_attacked = False
