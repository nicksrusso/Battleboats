from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Port:
    """A port at a fixed position with current owner and material stockpile.

    Home ports convert production directly to cash; non-home ports accumulate
    raw materials that merchants must ferry to a home port. The `is_home` flag
    is set at game start and does not change on capture — a captured former
    home port remains marked as such (the game ends on home-port capture, so
    this is mainly a property for the env layer to expose in observations).
    """

    position: Tuple[int, int]
    owner: int
    is_home: bool = False
    stockpile: int = 0
