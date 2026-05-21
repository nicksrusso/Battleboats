from dataclasses import dataclass
from typing import Tuple

from battleboats.core.shipyard.ship_type import ShipType


@dataclass(slots=True)
class Sighting:
    """A player's record of an enemy ship's last-known state.

    `fresh=True` means the ship is currently within visibility range of one
    of the owning player's ships. `fresh=False` means the player has lost the
    view; `position` and `turn_seen` are frozen at the last fresh observation.
    Sightings persist until overwritten by a newer observation or cleared by
    a witnessed kill (see gameEngine._destroy_ship).
    """

    ship_id: int
    type: ShipType
    position: Tuple[int, int]
    turn_seen: int
    fresh: bool


@dataclass(slots=True)
class PortSighting:
    """A player's record of an enemy port's last-known state.

    `fresh=True` means at least one friendly ship or friendly port is
    currently within sighting range of this port. `fresh=False` means the
    player has lost direct view; `last_known_owner` is frozen at the last
    fresh observation, so a port that flips hands while we're not watching
    will still appear under its previous owner in our record until rescouted.
    Once spotted, the `is_home` flag is revealed for as long as the sighting
    persists.
    """

    position: Tuple[int, int]
    last_known_owner: int
    is_home: bool
    turn_seen: int
    fresh: bool
