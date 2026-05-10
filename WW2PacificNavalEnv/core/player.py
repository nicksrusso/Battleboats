from typing import Dict, List, Tuple
from .shipyard.ship import Ship
from .shipyard.ship_type import ShipType
from .shipyard.ship_stats import ShipStats


class Player:
    """Per-player state: ships, ports, economy, tech."""

    def __init__(self, id: int, home_port: Tuple[int, int]):
        self.id = id
        self.home_port = home_port
        self.ports: List[Tuple[int, int]] = [home_port]
        self.ships: List[Ship] = []
        self.cash: int = 0
        self.raw_materials_at_port: Dict[Tuple[int, int], int] = {}
        self.tech: Dict[ShipType, ShipStats] = {}

    def owns_port(self, position: Tuple[int, int]) -> bool:
        return position in self.ports

    def add_ship(self, ship: Ship) -> None:
        pass

    def remove_ship(self, ship_id: int) -> None:
        pass

    def capture_port(self, position: Tuple[int, int]) -> None:
        pass

    def lose_port(self, position: Tuple[int, int]) -> None:
        pass

    def collect_port_income(self) -> None:
        """Add 25 raw materials to each owned port (called at turn start)."""
        pass

    def deliver_to_home(self, amount: int) -> None:
        """Convert raw materials delivered by a transport into cash."""
        pass

    def can_afford(self, cost: int) -> bool:
        return self.cash >= cost

    def spend(self, amount: int) -> None:
        pass
