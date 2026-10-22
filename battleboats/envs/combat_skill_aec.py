"""Combat skill sandbox (PettingZoo AEC).

One learning agent. The opponent seat is auto-EndTurn'd. The learner starts
with a destroyer and must sink a strength-0 merchant (a hit is a kill). The
target is injected into sightings so the skill is "close and shoot", not a
blind search.

Reward: +kill_reward when the target ship is removed. Optional approach_coef
pays for each Manhattan step closer to the target. Episode ends on the kill
and truncates at max_turns.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, AttackAction, EndTurnAction, MoveAction
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.sighting import Sighting
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_COMBAT_VERBS: Tuple[type, ...] = (MoveAction, AttackAction, EndTurnAction)


class CombatSkillAEC(BattleboatsBaseAEC):
    """1p combat skill: move a destroyer into range and sink the target."""

    metadata = {"name": "battleboats_combat_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 40,
        learner_id: int = 0,
        kill_reward: float = 1.0,
        approach_coef: float = 0.0,
        restrict_verbs: bool = True,
        randomize_target: bool = False,
        reveal_target: bool = True,
    ) -> None:
        self.learner_id = learner_id
        self.kill_reward = kill_reward
        self.approach_coef = float(approach_coef)
        self.restrict_verbs = restrict_verbs
        self.randomize_target = randomize_target
        self.reveal_target = reveal_target
        self._killed = False
        self._target_id: Optional[int] = None
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        if self.randomize_target:
            self._init_random_target()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def _enemy_id(self) -> int:
        return 1 - self.learner_id

    def _init_random_target(self) -> None:
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        self._learner_home = tuple(base["ports"][str(self.learner_id)][0])
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="combat_rand_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.randomize_target and not (options or {}).get("scenario"):
            self._write_random_layout(seed)
        super().reset(seed=seed, options=options)

    def _write_random_layout(self, seed: Optional[int]) -> None:
        w, h = self._board
        lh = self._learner_home
        rng = random.Random(seed)
        exclude = {lh}
        for dx, dy in _CARDINALS:
            exclude.add((lh[0] + dx, lh[1] + dy))
        candidates = [(x, y) for x in range(w) for y in range(h) if (x, y) not in exclude]
        if not candidates:
            raise RuntimeError("CombatSkillAEC: board too small to place a target.")
        target = rng.choice(candidates)
        layout = {
            "width": w,
            "height": h,
            "seed": 0,
            "land": [list(lh), list(target)],
            "ports": {
                str(self.learner_id): [list(lh)],
                str(self._enemy_id): [list(target)],
            },
        }
        with open(self._tmp_map_path, "w") as f:
            json.dump(layout, f)
        self.map_json_path = self._tmp_map_path

    def close(self) -> None:
        if self._tmp_map_path and os.path.exists(self._tmp_map_path):
            try:
                os.remove(self._tmp_map_path)
            except OSError:
                pass

    def _configure_scenario(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> None:
        assert self.engine is not None
        learner = self.engine.players[self.learner_id]
        enemy = self.engine.players[self._enemy_id]
        own = self._find_adjacent_water(learner.home_port)
        tgt = self._find_adjacent_water(enemy.home_port)
        if own is None or tgt is None or own == tgt:
            raise RuntimeError(
                "CombatSkillAEC: need distinct water tiles next to both homes."
            )
        hunter = self.engine._spawn_ship(self.learner_id, ShipType.DESTROYER, own)
        target = self.engine._spawn_ship(self._enemy_id, ShipType.MERCHANT, tgt)
        self._target_id = target.id
        self._killed = False
        self._hunter_id = hunter.id
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._reveal_target()
        if self.restrict_verbs:
            self._install_verb_filter()

    def _reveal_target(self) -> None:
        if not self.reveal_target or self.engine is None or self._target_id is None:
            return
        ship = self.engine.ships.get(self._target_id)
        if ship is None:
            return
        self.engine.players[self.learner_id].sightings[ship.id] = Sighting(
            ship_id=ship.id,
            type=ship.type,
            position=ship.position,
            turn_seen=self.engine.turn,
            fresh=True,
        )

    def _install_verb_filter(self) -> None:
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, _COMBAT_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _target_dist(self) -> Optional[int]:
        assert self.engine is not None
        if self._target_id is None:
            return None
        target = self.engine.ships.get(self._target_id)
        hunter = self.engine.ships.get(getattr(self, "_hunter_id", None))
        if target is None or hunter is None:
            return None
        return self.engine.map.manhattan(hunter.position, target.position)

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {"dist": self._target_dist()}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        learner = self._agent_name(self.learner_id)
        if (
            not self._killed
            and self.engine is not None
            and self._target_id is not None
            and self._target_id not in self.engine.ships
        ):
            self._killed = True
            rewards[learner] = float(self.kill_reward)
        if self.approach_coef and pre.get("dist") is not None:
            post = self._target_dist()
            if post is not None:
                rewards[learner] = rewards.get(learner, 0.0) + self.approach_coef * (
                    pre["dist"] - post
                )
        return rewards

    def _is_terminated(self) -> bool:
        return self._killed

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not self._killed
            and self._target_id in self.engine.ships
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1
        self._reveal_target()

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
