"""Fixed one-hot slots for the shared sandbox policy.

The index is the skill input. It does not depend on curriculum order, so a
checkpoint trained on merchant still means the same slot when combat starts.
"""

from __future__ import annotations

SKILL_NAMES = (
    "merchant",
    "builder",
    "landing",
    "combat_destroyer",
    "combat_submarine",
    "combat_cruiser",
    "combat_battleship",
    "combat_carrier",
)

SKILL_INDEX = {name: i for i, name in enumerate(SKILL_NAMES)}


def skill_index(name: str) -> int:
    if name not in SKILL_INDEX:
        raise ValueError(f"unknown sandbox {name!r}; expected one of {SKILL_NAMES}")
    return SKILL_INDEX[name]
