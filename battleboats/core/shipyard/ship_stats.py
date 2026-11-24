from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShipStats:
    """Immutable base stats snapshot at build time (updated only by player tech).

    Detection is not a per-ship stat: see ``ship_data.spot_distance``, backed by
    ``config/visibilityMatrix.csv``.
    """

    speed: int
    cost: int
    strength: float
    attack_range: int
