"""Shared PettingZoo AEC base for battleboats envs.

Owns engine lifecycle, agent bookkeeping, observations, and legal-action
priming. Subclasses specialize scenario setup, reward, and termination via
hooks — not by reimplementing step/reset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gymnasium.spaces import Space
from pettingzoo import AECEnv

from battleboats.core.actions import Action
from battleboats.core.gameEngine import gameEngine
from battleboats.envs import observation

DEFAULT_MAX_TURNS = 500


class BattleboatsBaseAEC(AECEnv):
    """Parent AEC: engine + obs/masks plumbing; reward/done via hooks."""

    metadata = {"name": "battleboats_base_aec_v0", "is_parallelizable": False}

    def __init__(self, map_json_path: str, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Store config; engine is created in reset()."""
        self.map_json_path = map_json_path
        self.max_turns = max_turns
        self.possible_agents = self._possible_agents()
        self.engine: Optional[gameEngine] = None

    # ------------------------------------------------------------------ hooks
    def _possible_agents(self) -> List[str]:
        """AEC agent ids exposed to PettingZoo. Override for 1p skills."""
        return ["player_0", "player_1"]

    def _configure_scenario(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> None:
        """Mutate engine after default load (spawn ships, stockpiles, etc.)."""
        return None

    def _capture_reward_state(self) -> Dict[str, Any]:
        """Snapshot used by _reward_after_step. Default: empty."""
        return {}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        """Per-agent rewards for this transition. Default: all zero."""
        return {a: 0.0 for a in self.agents}

    def _is_terminated(self) -> bool:
        """True if the episode ended successfully / by game rule."""
        return False

    def _is_truncated(self) -> bool:
        """True if the episode hit a time/step budget."""
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        """Optional side effects after the learner action (e.g. auto-opponent)."""
        return None

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str) -> Space:
        """Nominal; real obs is a variable-length token dict."""
        raise NotImplementedError

    def action_space(self, agent: str) -> Space:
        """Nominal; actions pass through as engine.Action objects."""
        raise NotImplementedError

    # --------------------------------------------------------------- lifecycle
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new episode (PettingZoo AEC API).

        options may include ``scenario`` (full-game reset_from_scenario dict).
        Subclasses add skill fields via _configure_scenario.
        """
        self._load_engine(seed=seed, options=options)
        self._configure_scenario(seed=seed, options=options)

        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._sync_agent_selection()
        self._prime_legal_actions()

    def step(self, action: Optional[Action]) -> None:
        """Apply action for agent_selection; settle rewards/dones; advance seat."""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action=action)
            return

        acting = self.agent_selection
        pre = self._capture_reward_state()
        assert self.engine is not None
        self.engine.step(action=action)
        self._after_engine_step(acting, action)
        self._settle_rewards_and_dones(acting, action, pre)

        if not all(self.terminations.values()) and not all(self.truncations.values()):
            self._sync_agent_selection()
            self._prime_legal_actions()

    def observe(self, agent: str) -> Dict[str, Any]:
        """Fog-filtered observation dict for ``agent``."""
        assert self.engine is not None
        return observation.build_observation(self.engine, self._player_id(agent))

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    # ----------------------------------------------------------------- helpers
    def _player_id(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _agent_name(self, player_id: int) -> str:
        return f"player_{player_id}"

    def _load_engine(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> None:
        """Construct engine from scenario option or map_json_path + seed."""
        scenario = (options or {}).get("scenario")
        if scenario:
            self.engine = gameEngine(map_json_path=scenario["map_path"])
            self.engine.reset_from_scenario(scenario)
        else:
            self.engine = gameEngine(map_json_path=self.map_json_path)
            self.engine.reset(seed=seed)

    def _sync_agent_selection(self) -> None:
        """Point agent_selection at engine.current_player when that seat is live."""
        assert self.engine is not None
        name = self._agent_name(self.engine.current_player)
        if name in self.agents:
            self.agent_selection = name
        else:
            # Skill envs may expose only the learner; stay on the sole agent.
            self.agent_selection = self.agents[0]

    def _prime_legal_actions(self) -> None:
        """Write enumerate_legal for the seat about to act."""
        assert self.engine is not None
        pid = self._player_id(self.agent_selection)
        # If engine seat != exposed agent (shouldn't after auto-opp), still
        # list legal moves for the AEC agent so policies don't see empty lists.
        if self.engine.current_player != pid and pid in (
            self._player_id(a) for a in self.agents
        ):
            pid = self.engine.current_player
        self.infos[self.agent_selection]["legal_actions"] = self.engine.enumerate_legal(
            pid
        )

    def _settle_rewards_and_dones(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> None:
        """Zero per-step rewards, apply hook rewards, set term/trunc flags."""
        self.rewards = {a: 0.0 for a in self.agents}
        for a, r in self._reward_after_step(agent, action, pre).items():
            if a in self.rewards:
                self.rewards[a] = float(r)

        if self._is_terminated():
            for a in self.agents:
                self.terminations[a] = True
        elif self._is_truncated():
            for a in self.agents:
                self.truncations[a] = True

        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
