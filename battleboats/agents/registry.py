"""Name -> scripted agent, for eval / harvest scripts and the play UI.

``make_agent(kind)`` returns an object with ``act(engine, pid, rng)``.
Stateless agents wrap their functions; ``fog_commander`` gets a fresh
``FogCommander`` (per-seat memory) for each game.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional

from battleboats.agents.fog_commander import DOCTRINES, FogCommander, FogCommanderConfig
from battleboats.agents.random_agent import random_action
from battleboats.agents.scripted import scripted_action
from battleboats.agents.tactical import tactical_scripted_action


class _FnAgent:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def act(self, engine, pid: int, rng: random.Random):
        return self._fn(engine, pid, rng)

    def doctrine_for(self, pid: int):
        return None


AGENT_KINDS: Dict[str, Callable[[], object]] = {
    "fog_commander": FogCommander,
    "scripted": lambda: _FnAgent(scripted_action),
    "tactical": lambda: _FnAgent(tactical_scripted_action),
    "random": lambda: _FnAgent(random_action),
}


def make_agent(kind: str, *, doctrine: Optional[str] = None):
    """Build one agent. ``doctrine`` forces a Fog Commander doctrine (ignored for other kinds)."""
    if kind not in AGENT_KINDS:
        raise ValueError(f"unknown agent kind {kind!r}; choose from {sorted(AGENT_KINDS)}")
    if kind == "fog_commander":
        if doctrine is None:
            return FogCommander()
        if doctrine not in DOCTRINES:
            raise ValueError(f"unknown doctrine {doctrine!r}; choose from {list(DOCTRINES)}")
        return FogCommander(FogCommanderConfig(doctrine=doctrine))
    if doctrine is not None:
        raise ValueError(f"doctrine= is only valid for fog_commander, not {kind!r}")
    return AGENT_KINDS[kind]()
