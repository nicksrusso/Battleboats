"""PettingZoo AECEnv wrapper around gameEngine.

Action contract — Approach 1 pass-through:
    env.step(action) accepts an engine.Action dataclass directly. The
    factoring of the action into (asset, verb, target) lives in the policy
    network's heads, not in the env. action_space is nominal.

Observation contract:
    See `observation.build_observation()` — dict of (entity_tokens, globals).
    Legal actions are delivered via self.infos[agent]["legal_actions"] per
    PettingZoo convention.

Reward contract:
    Sparse zero-sum, terminal-only. Winner: +1, Loser: -1, otherwise 0.
    Potential-based shaping (held in reserve per project plan) would be
    added inside _settle_terminal_rewards if/when training stalls.

Termination vs truncation:
    terminated: engine.winner is not None (capture of enemy home port).
    truncated:  turn count exceeds self.max_turns (avoid infinite games).
"""

from typing import Any, Dict, Optional

from gymnasium.spaces import Space
from pettingzoo import AECEnv

from battleboats.core.actions import Action
from battleboats.core.gameEngine import gameEngine
from battleboats.envs import observation

AGENTS = ("player_0", "player_1")
DEFAULT_MAX_TURNS = 500


class BattleboatsAEC(AECEnv):
    """Two-player turn-based naval game wrapped as a PettingZoo AECEnv.

    Lifecycle (PettingZoo convention):
        env = BattleboatsAEC(map_json_path)
        env.reset(seed=0)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                action = None
            else:
                action = my_policy(obs, info["legal_actions"])
            env.step(action)
    """

    metadata = {"name": "battleboats_aec_v0", "is_parallelizable": False}

    def __init__(self, map_json_path: str, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Store env config; defer state initialization to reset()."""
        self.map_json_path = map_json_path
        self.max_turns = max_turns
        self.possible_agents = list(AGENTS)

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str) -> Space:
        """Return a nominal observation space.

        Real observation is a dict with a variable-length token tensor that
        does not map cleanly to a single gym Space primitive.
        """
        raise NotImplementedError

    def action_space(self, agent: str) -> Space:
        """Return a nominal action space; actions pass through as engine.Action objects."""
        raise NotImplementedError

    # --------------------------------------------------------------- lifecycle
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Start a new game.

        Re-instantiates and seeds the engine, repopulates per-agent
        bookkeeping dicts, and primes legal_actions for the opening seat.
        """
        self.engine = gameEngine(map_json_path=self.map_json_path)
        self.engine.reset(seed=seed)
        self.agents = list(AGENTS)
        self.agent_selection = self._agent_name(self.engine.current_player)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.infos[self.agent_selection]["legal_actions"] = self.engine.enumerate_legal(self.engine.current_player)

    def step(self, action: Optional[Action]) -> None:
        """Apply `action` for self.agent_selection.

        Delegates to the engine, settles terminal rewards if a game-end
        condition was triggered, mirrors agent_selection from
        engine.current_player, then refreshes legal_actions for the new seat.
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action=action)
            return

        self.engine.step(action=action)
        self._settle_terminal_rewards()

        self.agent_selection = self._agent_name(self.engine.current_player)
        self.infos[self.agent_selection]["legal_actions"] = self.engine.enumerate_legal(self.engine.current_player)

    def observe(self, agent: str) -> Dict[str, Any]:
        """Build the fog-of-war-filtered obs dict for `agent`.

        Delegates to observation.build_observation(self.engine, player_id).
        """
        return observation.build_observation(self.engine, self._player_id(agent))

    def render(self) -> None:
        """Optional human-readable rendering. Skip for now; pygame UI later."""
        pass

    def close(self) -> None:
        """No persistent resources to release."""
        pass

    # ----------------------------------------------------------------- helpers
    def _player_id(self, agent: str) -> int:
        """`"player_0"` → 0, `"player_1"` → 1."""
        return int(agent.split("_")[1])

    def _agent_name(self, player_id: int) -> str:
        """0 → `"player_0"`, 1 → `"player_1"`."""
        return f"player_{player_id}"

    def _settle_terminal_rewards(self) -> None:
        """Reset per-step rewards and apply terminal effects.

        On engine winner: write +1/-1 into rewards and flip terminations.
        On turn-limit: flip truncations. Always folds rewards into
        _cumulative_rewards. Future home for potential-based shaping.
        """
        self.rewards = {a: 0.0 for a in self.agents}
        if self.engine.winner is not None:
            winner = self._agent_name(self.engine.winner)
            for a in self.agents:
                self.rewards[a] = 1.0 if a == winner else -1.0
                self.terminations[a] = True
        elif self.engine.turn >= self.max_turns:
            for a in self.agents:
                self.truncations[a] = True

        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]
