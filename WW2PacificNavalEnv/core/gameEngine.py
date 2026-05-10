from typing import List, Optional, Tuple
from .player import Player
from .map.Map import Map
from .shipyard.ship import Ship
from .shipyard.ship_type import ShipType


class gameEngine:
    """Owns full game state. Sequential 2-player turns."""

    def __init__(self, json_path: str):
        self.map_json_path = json_path
        self.map: Optional[Map] = None
        self.players: List[Player] = []
        self.current_player: int = 0
        self.turn: int = 0
        self.winner: Optional[int] = None
        self._next_ship_id: int = 0

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Load map, build players, place starting units."""
        pass

    # ------------------------------------------------------------------ turn
    def step(self, action) -> None:
        """Apply one action for the current player. Action schema TBD."""
        pass

    def end_turn(self) -> None:
        """Finalize current player's turn, swap to the other player."""
        pass

    # ------------------------------------------------------------------ actions
    def move_ship(self, ship_id: int, destination: Tuple[int, int]) -> None:
        pass

    def attack(self, attacker_id: int, target_id: int) -> bool:
        """Resolve attack; return True if defender destroyed."""
        pass

    def build_ship(self, port: Tuple[int, int], ship_type: ShipType) -> Optional[Ship]:
        pass

    def research(self, ship_type: ShipType) -> None:
        pass

    def capture_port(self, landing_ship_id: int) -> None:
        pass

    def build_port(self, builder_ship_id: int) -> None:
        pass

    # ------------------------------------------------------------------ queries
    def visible_ships(self, player_id: int) -> List[Ship]:
        """Enemy ships within detection range of any of player's ships/ports."""
        pass

    def is_terminal(self) -> bool:
        return self.winner is not None

    def get_state(self) -> dict:
        """Full ground-truth state (env layer is responsible for masking)."""
        pass
