"""Deprecated import path for the scripted agent.

Configs, CLIs, and stored harvest metadata may still say ``fog_commander``.
New code should import :mod:`battleboats.agents.scripted_agent`.
"""

from __future__ import annotations

import warnings

from battleboats.agents.scripted_agent import *  # noqa: F401,F403
from battleboats.agents.scripted_agent import (
    ScriptedAgent as FogCommander,
    ScriptedAgentConfig as FogCommanderConfig,
    scripted_agent_action as fog_commander_action,
    scripted_agent_on as fog_commander_agent,
)

warnings.warn(
    "battleboats.agents.fog_commander is deprecated; "
    "import battleboats.agents.scripted_agent instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FogCommander",
    "FogCommanderConfig",
    "fog_commander_action",
    "fog_commander_agent",
]
