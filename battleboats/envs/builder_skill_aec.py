"""Builder skill sandbox (PettingZoo AEC).

One learning agent. A builder ship must reach the empty land tile recorded as
``build_target`` on the map JSON and issue BuildPortAction. The builder is
consumed and the tile becomes a friendly port.

Reward: +build_reward on that port appearing. Optional approach_coef pays for
each Manhattan step closer to the target. Opponent seat is auto-EndTurn'd.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, BuildPortAction, EndTurnAction, MoveAction
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.base_aec import BattleboatsBaseAEC
from battleboats.envs.skill_maps import read_build_target

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BUILDER_VERBS: Tuple[type, ...] = (MoveAction, BuildPortAction, EndTurnAction)


class BuilderSkillAEC(BattleboatsBaseAEC):
    """1p builder skill: move a builder next to empty land and build a port."""

    metadata = {"name": "battleboats_builder_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 40,
        learner_id: int = 0,
        build_reward: float = 1.0,
        approach_coef: float = 0.0,
        restrict_verbs: bool = True,
        build_target: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.learner_id = learner_id
        self.build_reward = build_reward
        self.approach_coef = float(approach_coef)
        self.restrict_verbs = restrict_verbs
        recorded = read_build_target(map_json_path)
        self.build_target = tuple(build_target) if build_target is not None else recorded
        self._built = False
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def built(self) -> bool:
        return self._built

    def _configure_scenario(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> None:
        assert self.engine is not None
        learner = self.engine.players[self.learner_id]
        if self.build_target is None:
            self.build_target = self._infer_target()
        if self.build_target is None:
            raise RuntimeError("BuilderSkillAEC: map has no empty land tile to build on.")
        spawn = self._find_adjacent_water(learner.home_port)
        if spawn is None:
            raise RuntimeError("BuilderSkillAEC: no water next to the learner home.")
        self.engine._spawn_ship(self.learner_id, ShipType.BUILDER, spawn)
        self._built = False
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        if self.restrict_verbs:
            self._install_verb_filter()

    def _infer_target(self) -> Optional[Tuple[int, int]]:
        assert self.engine is not None
        m = self.engine.map
        for x in range(m.width):
            for y in range(m.height):
                pos = (x, y)
                if m.is_land(pos) and not m.is_port(pos):
                    return pos
        return None

    def _install_verb_filter(self) -> None:
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, _BUILDER_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _builder_dist(self) -> Optional[int]:
        assert self.engine is not None
        if self.build_target is None:
            return None
        best: Optional[int] = None
        for sid in self.engine.players[self.learner_id].owned_ship_ids:
            ship = self.engine.ships.get(sid)
            if ship is None or ship.type is not ShipType.BUILDER:
                continue
            d = self.engine.map.manhattan(ship.position, self.build_target)
            best = d if best is None else min(best, d)
        return best

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {"dist": self._builder_dist()}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        learner = self._agent_name(self.learner_id)
        if not self._built and self.engine is not None and self.build_target is not None:
            port = self.engine.ports.get(self.build_target)
            if port is not None and port.owner == self.learner_id:
                self._built = True
                rewards[learner] = float(self.build_reward)
        if self.approach_coef and pre.get("dist") is not None:
            post = self._builder_dist()
            if post is not None:
                rewards[learner] = rewards.get(learner, 0.0) + self.approach_coef * (
                    pre["dist"] - post
                )
        return rewards

    def _is_terminated(self) -> bool:
        return self._built

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not self._built
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1

    def _find_adjacent_water(self, port: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        assert self.engine is not None
        m = self.engine.map
        for dx, dy in _CARDINALS:
            pos = (port[0] + dx, port[1] + dy)
            if not m.in_bounds(pos):
                continue
            if m.terrain[pos] != WATER:
                continue
            if m.is_occupied(pos):
                continue
            return pos
        return None
