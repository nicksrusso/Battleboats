"""Name -> scripted agent, for eval / harvest scripts and the play UI.

``make_agent(kind)`` returns an object with ``act(engine, pid, rng)``.
Stateless agents wrap their functions; ``scripted_agent`` gets a fresh
``ScriptedAgent`` (per-seat memory) for each game.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional

from battleboats.agents.random_agent import random_action
from battleboats.agents.scripted import scripted_action
from battleboats.agents.scripted_agent import DOCTRINES, ScriptedAgent, ScriptedAgentConfig
from battleboats.agents.tactical import tactical_scripted_action

# Stored in older harvest metadata and still accepted by configs, CLIs, and the play UI.
LEGACY_AGENT_KIND = "fog_commander"


class _FnAgent:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def act(self, engine, pid: int, rng: random.Random):
        return self._fn(engine, pid, rng)

    def doctrine_for(self, pid: int):
        return None


AGENT_KINDS: Dict[str, Callable[[], object]] = {
    "scripted_agent": ScriptedAgent,
    "scripted": lambda: _FnAgent(scripted_action),
    "tactical": lambda: _FnAgent(tactical_scripted_action),
    "random": lambda: _FnAgent(random_action),
}


def normalize_agent_kind(kind: str) -> str:
    """Map the deprecated ``fog_commander`` name onto ``scripted_agent``."""
    if kind == LEGACY_AGENT_KIND:
        return "scripted_agent"
    return kind


def is_scripted_agent_kind(kind: str) -> bool:
    return normalize_agent_kind(kind) == "scripted_agent"


def agent_kind_choices() -> tuple:
    return tuple(sorted((*AGENT_KINDS, LEGACY_AGENT_KIND)))


def make_agent(kind: str, *, doctrine: Optional[str] = None):
    """Build one agent. ``doctrine`` forces a scripted agent doctrine (ignored for other kinds)."""
    kind = normalize_agent_kind(kind)
    if kind not in AGENT_KINDS:
        raise ValueError(f"unknown agent kind {kind!r}; choose from {sorted(AGENT_KINDS)}")
    if kind == "scripted_agent":
        if doctrine is None:
            return ScriptedAgent()
        if doctrine not in DOCTRINES:
            raise ValueError(f"unknown doctrine {doctrine!r}; choose from {list(DOCTRINES)}")
        return ScriptedAgent(ScriptedAgentConfig(doctrine=doctrine))
    if doctrine is not None:
        raise ValueError(f"doctrine= is only valid for scripted_agent, not {kind!r}")
    return AGENT_KINDS[kind]()
