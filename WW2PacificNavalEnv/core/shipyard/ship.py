from dataclasses import dataclass
from typing import Tuple, Optional
from .ship_type import ShipType
from .ship_stats import ShipStats


class Ship:
    """Individual ship instance. Stats are frozen copy from player's tech at build time."""

    def __init__(
        self,
        myType: ShipType,
        stats: ShipStats,
        owner: int,  # 0 or 1
        position: Tuple[int, int],
        id: Optional[int] = None,
    ):
        self.type = myType
        self.stats = stats
        self.owner = owner
        self.position = position
        self.id = id
