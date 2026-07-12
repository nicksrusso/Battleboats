"""Full two-player battleboats game as a PettingZoo AECEnv.

Reward: sparse zero-sum, terminal-only (+1 win / -1 loss / 0 draw-truncate).
Termination: enemy home captured. Truncation: max_turns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from battleboats.core.actions import Action
from battleboats.envs.base_aec import DEFAULT_MAX_TURNS, BattleboatsBaseAEC

AGENTS = ("player_0", "player_1")


class BattleboatsAEC(BattleboatsBaseAEC):
    """Two-player full game. Same public API as before the base-class split."""

    metadata = {"name": "battleboats_aec_v0", "is_parallelizable": False}

    def __init__(self, map_json_path: str, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)

    def _possible_agents(self) -> list[str]:
        return list(AGENTS)

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        assert self.engine is not None
        if self.engine.winner is not None:
            winner = self._agent_name(self.engine.winner)
            return {a: (1.0 if a == winner else -1.0) for a in self.agents}
        return {a: 0.0 for a in self.agents}

    def _is_terminated(self) -> bool:
        return self.engine is not None and self.engine.winner is not None

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns
