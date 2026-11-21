"""Opening sandbox: empty fleet, buy a landing ship, capture the enemy home.

The other sandboxes hand the learner the ship already built. The full game
does not. This episode starts the way a match starts (250 cash, no ships)
and ends when the enemy home is captured. Verbs are not filtered, so the
legal set is the engine's: at the opening that is build-ship and end-turn,
then move and capture.

Reward: +buy_bonus the step a landing ship appears, a potential-based
approach bonus while that ship exists, and +capture_reward on the capture.
Success is the capture (``captured``), not the shaping.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, EndTurnAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.sighting import PortSighting
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


class OpeningSkillAEC(BattleboatsBaseAEC):
    """1p opening: purchase a landing ship and take the enemy home."""

    metadata = {"name": "battleboats_opening_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 12,
        learner_id: int = 0,
        capture_reward: float = 1.0,
        buy_bonus: float = 0.25,
        approach_coef: float = 0.0,
        randomize_target: bool = False,
        reveal_enemy_home: bool = True,
    ) -> None:
        self.learner_id = learner_id
        self.capture_reward = capture_reward
        self.buy_bonus = buy_bonus
        self.approach_coef = float(approach_coef)
        self.randomize_target = randomize_target
        self.reveal_enemy_home = reveal_enemy_home
        self._captured = False
        self._bought = False
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        if self.randomize_target:
            self._init_random_target()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def _enemy_id(self) -> int:
        return 1 - self.learner_id

    def _init_random_target(self) -> None:
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="opening_rand_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        if self.randomize_target and not (options or {}).get("scenario"):
            self._write_random_layout(seed)
        super().reset(seed=seed, options=options)

    def _write_random_layout(self, seed: Optional[int]) -> None:
        """Same cues as a full-game opening: extra port, empty land, visible enemy home.

        A two-home board teaches "buy a landing", but the full game also has a
        coastline, and the builder prior then buys a builder instead. This
        layout keeps that coastline in the observation.
        """
        w, h = self._board
        home = (0, 0)
        rng = random.Random(seed)
        exclude = {home}
        for dx, dy in _CARDINALS:
            exclude.add((home[0] + dx, home[1] + dy))
        candidates = [(x, y) for x in range(w) for y in range(h) if (x, y) not in exclude]
        if len(candidates) < 3:
            raise RuntimeError("OpeningSkillAEC: board too small for a full-game opening.")
        far = [p for p in candidates if abs(p[0] - home[0]) + abs(p[1] - home[1]) >= 4]
        target = rng.choice(far or candidates)
        rest = [p for p in candidates if p != target]
        extra_port = rng.choice(rest)
        rest.remove(extra_port)
        empty = rng.choice(rest)
        layout = {
            "width": w,
            "height": h,
            "seed": 0,
            "land": [list(home), list(target), list(extra_port), list(empty)],
            "ports": {
                str(self.learner_id): [list(home), list(extra_port)],
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
        self._captured = False
        self._bought = False
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._reveal_enemy_home()

    def _reveal_enemy_home(self) -> None:
        if not self.reveal_enemy_home or self.engine is None:
            return
        home = self.engine.players[self._enemy_id].home_port
        port = self.engine.ports[home]
        if port.owner == self.learner_id:
            return
        self.engine.players[self.learner_id].port_sightings[home] = PortSighting(
            position=home,
            last_known_owner=port.owner,
            is_home=port.is_home,
            turn_seen=self.engine.turn,
            fresh=True,
        )

    def _owns_landing(self) -> bool:
        assert self.engine is not None
        for sid in self.engine.players[self.learner_id].owned_ship_ids:
            ship = self.engine.ships.get(sid)
            if ship is not None and ship.type is ShipType.LANDING:
                return True
        return False

    def _landing_dist(self) -> Optional[int]:
        assert self.engine is not None
        opp = self.engine.players[self._enemy_id].home_port
        best: Optional[int] = None
        for sid in self.engine.players[self.learner_id].owned_ship_ids:
            ship = self.engine.ships.get(sid)
            if ship is None or ship.type is not ShipType.LANDING:
                continue
            d = self.engine.map.manhattan(ship.position, opp)
            best = d if best is None else min(best, d)
        return best

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {"dist": self._landing_dist(), "had_landing": self._owns_landing()}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        learner = self._agent_name(self.learner_id)
        if not self._bought and self._owns_landing():
            self._bought = True
            rewards[learner] = float(self.buy_bonus)
        if (
            not self._captured
            and self.engine is not None
            and self.engine.winner == self.learner_id
        ):
            self._captured = True
            rewards[learner] = rewards.get(learner, 0.0) + float(self.capture_reward)
        if self.approach_coef and pre.get("dist") is not None:
            post = self._landing_dist()
            if post is not None:
                rewards[learner] = rewards.get(learner, 0.0) + self.approach_coef * (
                    pre["dist"] - post
                )
        return rewards

    def _is_terminated(self) -> bool:
        return self._captured

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not self._captured
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1
        self._reveal_enemy_home()
