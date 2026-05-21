from typing import Dict, Set, Tuple

from battleboats.core.sighting import Sighting, PortSighting


class Player:
    """Per-side bookkeeping: cash, owned units, enemy memory.

    Pure data container. All world mutation goes through gameEngine — Player
    never touches the map or other players. Port state (ownership, stockpile,
    is_home) lives on `gameEngine.ports`, indexed by position.

    `sightings` tracks enemy ships keyed by ship_id. `port_sightings` tracks
    enemy ports keyed by position. Both are updated by
    `gameEngine._refresh_sightings()` and obey identical fog-of-war semantics:
    fresh while a friendly scout is in range, stale after.
    """

    def __init__(self, id: int, home_port: Tuple[int, int]) -> None:
        self.id = id
        self.home_port = home_port
        self.cash: int = 0
        self.owned_ship_ids: Set[int] = set()
        self.owned_port_positions: Set[Tuple[int, int]] = {home_port}
        self.sightings: Dict[int, Sighting] = {}
        self.port_sightings: Dict[Tuple[int, int], PortSighting] = {}
