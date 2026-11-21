"""Move-ship skill sandbox (PettingZoo AEC).

One learning agent. A single ship must sail until it is Manhattan-adjacent to
a friendly waypoint port (the port itself is land, so the ship cannot occupy
it). The waypoint is a friendly port, so it is always in the observation.

Reward: +arrive_reward on the step the ship first becomes adjacent, plus an
optional potential-based approach bonus (difference in Manhattan distance).
The approach term does not change the optimal policy. Success is ``reached``,
not the shaped return. Opponent seat is auto-EndTurn'd. Verbs are move and
endturn, same observation and action heads as the full game.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, EndTurnAction, MoveAction
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MOVE_VERBS: Tuple[type, ...] = (MoveAction, EndTurnAction)


class MoveSkillAEC(BattleboatsBaseAEC):
    """1p move skill: sail a ship next to a visible friendly waypoint port."""

    metadata = {"name": "battleboats_move_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 20,
        learner_id: int = 0,
        arrive_reward: float = 1.0,
        approach_coef: float = 0.0,
        restrict_verbs: bool = True,
        randomize_waypoint: bool = False,
    ) -> None:
        self.learner_id = learner_id
        self.arrive_reward = arrive_reward
        self.approach_coef = float(approach_coef)
        self.restrict_verbs = restrict_verbs
        self.randomize_waypoint = randomize_waypoint
        self._reached = False
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        if self.randomize_waypoint:
            self._init_random_waypoint()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def reached(self) -> bool:
        return self._reached

    def _init_random_waypoint(self) -> None:
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="move_rand_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.randomize_waypoint and not (options or {}).get("scenario"):
            self._write_random_layout(seed)
        super().reset(seed=seed, options=options)

    def _write_random_layout(self, seed: Optional[int]) -> None:
        """Fixed home, random waypoint port, fixed enemy home.

        The waypoint is not the home and not a neighbor of the home, so the
        spawn tile next to home is not already adjacent to the goal.
        """
        w, h = self._board
        home = (0, 0)
        opp = (w - 1, h - 1)
        rng = random.Random(seed)
        exclude = {home, opp}
        for dx, dy in _CARDINALS:
            exclude.add((home[0] + dx, home[1] + dy))
        candidates = [(x, y) for x in range(w) for y in range(h) if (x, y) not in exclude]
        if not candidates:
            raise RuntimeError("MoveSkillAEC: board too small to place a waypoint.")
        waypoint = rng.choice(candidates)
        layout = {
            "width": w,
            "height": h,
            "seed": 0,
            "land": [list(home), list(waypoint), list(opp)],
            "ports": {
                str(self.learner_id): [list(home), list(waypoint)],
                str(1 - self.learner_id): [list(opp)],
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
        waypoint = self._waypoint()
        spawn = self._spawn_tile(waypoint)
        self.engine._spawn_ship(self.learner_id, ShipType.DESTROYER, spawn)
        self._reached = False
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        if self.restrict_verbs:
            self._install_verb_filter()

    def _waypoint(self) -> Tuple[int, int]:
        assert self.engine is not None
        learner = self.engine.players[self.learner_id]
        others = [p for p in learner.owned_port_positions if p != learner.home_port]
        if not others:
            raise RuntimeError("MoveSkillAEC: map has no friendly waypoint port.")
        return others[0]

    def _spawn_tile(self, waypoint: Tuple[int, int]) -> Tuple[int, int]:
        """Water nearest the home that is not already adjacent to the waypoint."""
        assert self.engine is not None
        m = self.engine.map
        home = self.engine.players[self.learner_id].home_port
        ranked = []
        for x in range(m.width):
            for y in range(m.height):
                pos = (x, y)
                if m.terrain[pos] != WATER or m.is_occupied(pos):
                    continue
                dist = m.manhattan(pos, waypoint)
                if dist <= 1:
                    continue
                ranked.append((m.manhattan(pos, home), dist, pos))
        if not ranked:
            raise RuntimeError("MoveSkillAEC: no water tile that still requires a move.")
        ranked.sort()
        return ranked[0][2]

    def _install_verb_filter(self) -> None:
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, _MOVE_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _dist(self) -> Optional[int]:
        assert self.engine is not None
        waypoint = self._waypoint()
        best: Optional[int] = None
        for sid in self.engine.players[self.learner_id].owned_ship_ids:
            ship = self.engine.ships.get(sid)
            if ship is None:
                continue
            d = self.engine.map.manhattan(ship.position, waypoint)
            best = d if best is None else min(best, d)
        return best

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {"dist": self._dist()}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        learner = self._agent_name(self.learner_id)
        post = self._dist()
        if not self._reached and post is not None and post <= 1:
            self._reached = True
            rewards[learner] = float(self.arrive_reward)
        if self.approach_coef and pre.get("dist") is not None and post is not None:
            rewards[learner] = rewards.get(learner, 0.0) + self.approach_coef * (
                pre["dist"] - post
            )
        return rewards

    def _is_terminated(self) -> bool:
        return self._reached

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not self._reached
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1
