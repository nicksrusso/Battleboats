from dataclasses import dataclass


@dataclass(frozen=True)
class ShipStats:
    """Immutable base stats snapshot at build time (updated only by player tech)."""

    speed: int
    cost: int
    strength: float
    attack_range: int
    detect_range: int
