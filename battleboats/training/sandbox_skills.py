"""Fixed one-hot slots for the shared sandbox policy.

The index is the skill input. It does not depend on curriculum order, so a
checkpoint trained on merchant still means the same slot when combat starts.
New drills are appended so those slots stay put. The sandbox warm-up itself
trains with ``skill_count=0`` and does not read the index.
"""

from __future__ import annotations

from battleboats.core.shipyard.ship_type import ShipType

# One typed-build drill per buildable hull. Order is the curriculum order.
# Costs collide (100 and 250), so this list cannot be inferred from price.
_BUILD_HULLS = (
    "merchant",
    "landing",
    "destroyer",
    "submarine",
    "cruiser",
    "battleship",
    "carrier",
    "builder",
)
_known_hulls = {ship.name.lower() for ship in ShipType}
if set(_BUILD_HULLS) != _known_hulls or len(_BUILD_HULLS) != len(set(_BUILD_HULLS)):
    raise RuntimeError(
        f"build drills {_BUILD_HULLS} do not match ShipType {sorted(_known_hulls)}"
    )

SKILL_NAMES = (
    "merchant",
    "builder",
    "landing",
    "combat_destroyer",
    "combat_submarine",
    "combat_cruiser",
    "combat_battleship",
    "combat_carrier",
    "scouting",
    *(f"build_{name}" for name in _BUILD_HULLS),
)

SKILL_INDEX = {name: i for i, name in enumerate(SKILL_NAMES)}


def skill_index(name: str) -> int:
    if name not in SKILL_INDEX:
        raise ValueError(f"unknown sandbox {name!r}; expected one of {SKILL_NAMES}")
    return SKILL_INDEX[name]
