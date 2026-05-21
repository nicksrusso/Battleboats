import json
import numpy as np
from typing import Iterable, List, Tuple


WATER: int = 0
LAND: int = 1

NO_SHIP: int = -1
NO_OWNER: int = -1


class Map:
    """Spatial world model. Owns terrain, port ownership, and ship occupancy.

    Holds three numpy grids so spatial queries are O(1) array lookups instead
    of O(N) ship scans. No rules or turn logic live here.
    """

    def __init__(self) -> None:
        self.width: int = 0
        self.height: int = 0
        self.terrain: np.ndarray = np.zeros((0, 0), dtype=np.uint8)
        self.port_owner: np.ndarray = np.zeros((0, 0), dtype=np.int8)
        self.ship_at: np.ndarray = np.zeros((0, 0), dtype=np.int32)
        self.port_positions: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------ load
    def load(self, json_path: str) -> None:
        with open(json_path) as f:
            data = json.load(f)
        self.width = data["width"]
        self.height = data["height"]
        self.terrain = np.full((self.width, self.height), WATER, dtype=np.uint8)
        for x, y in data["land"]:
            self.terrain[x, y] = LAND
        self.port_owner = np.full((self.width, self.height), NO_OWNER, dtype=np.int8)
        self.port_positions = []
        for player_str, positions in data["ports"].items():
            p = int(player_str)
            for x, y in positions:
                self.port_owner[x, y] = p
                self.port_positions.append((x, y))
        self.ship_at = np.full((self.width, self.height), NO_SHIP, dtype=np.int32)

    # ------------------------------------------------------------------ queries
    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_water(self, pos: Tuple[int, int]) -> bool:
        return self.terrain[pos] == WATER

    def is_land(self, pos: Tuple[int, int]) -> bool:
        return self.terrain[pos] == LAND

    def is_port(self, pos: Tuple[int, int]) -> bool:
        return self.port_owner[pos] != NO_OWNER

    def ship_id_at(self, pos: Tuple[int, int]) -> int:
        return int(self.ship_at[pos])

    def is_occupied(self, pos: Tuple[int, int]) -> bool:
        return self.ship_at[pos] != NO_SHIP

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def tiles_within(self, center: Tuple[int, int], r: int) -> Iterable[Tuple[int, int]]:
        """Yield every in-bounds tile within manhattan distance r of center.

        Used for movement reach, attack range, and detection sweeps. Generator
        so callers can short-circuit without allocating a list.
        """
        cx, cy = center
        xmin = max(0, cx - r)
        xmax = min(self.width - 1, cx + r)
        for x in range(xmin, xmax + 1):
            dy = r - abs(x - cx)
            ymin = max(0, cy - dy)
            ymax = min(self.height - 1, cy + dy)
            for y in range(ymin, ymax + 1):
                yield (x, y)

    # ------------------------------------------------------------------ mutations
    def place_ship(self, ship_id: int, pos: Tuple[int, int]) -> None:
        self.ship_at[pos] = ship_id

    def remove_ship(self, pos: Tuple[int, int]) -> None:
        self.ship_at[pos] = NO_SHIP

    def relocate_ship(self, ship_id: int, src: Tuple[int, int], dst: Tuple[int, int]) -> None:
        self.ship_at[src] = NO_SHIP
        self.ship_at[dst] = ship_id

    def set_port_owner(self, pos: Tuple[int, int], player: int) -> None:
        self.port_owner[pos] = player

    def add_port(self, pos: Tuple[int, int], player: int) -> None:
        self.terrain[pos] = LAND
        self.port_owner[pos] = player
        self.port_positions.append(pos)
