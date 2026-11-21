"""Landing capture skill sandbox (PettingZoo AEC).

Single learning agent (player_0 by default). Opponent seat is auto-EndTurn'd
so the learner keeps the cycle without a second policy.

Task: navigate one LANDING ship to the enemy home port and capture it. Capturing
an enemy home port sets engine.winner (see gameEngine._do_capture_port), which is
this env's terminal.

Reward (sparse): +capture_reward on the transition that captures the enemy home
port; 0 otherwise. Episode terminates on capture; truncates on max_turns.

Map JSON is supplied by the caller (width/height/ports of your choosing). This
env spawns the landing ship after engine.reset(); by default next to the
learner's own home so it must traverse the board.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import (
    Action,
    CapturePortAction,
    EndTurnAction,
    MoveAction,
)
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.sighting import PortSighting
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Verbs the landing skill is allowed to use. Enumerating anything else (e.g.
# BuildShipAction from the home port) just adds irrelevant options for the policy
# to learn to avoid; restricting keeps the sandbox focused on navigate + capture.
# EndTurnAction is always present in enumerate_legal, so the filtered list is
# never empty.
_LANDING_VERBS: Tuple[type, ...] = (MoveAction, CapturePortAction, EndTurnAction)


class LandingSkillAEC(BattleboatsBaseAEC):
    """1p landing skill: move a landing ship to the enemy home port and capture."""

    metadata = {"name": "battleboats_landing_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 50,
        learner_id: int = 0,
        capture_reward: float = 1.0,
        landing_position: Optional[Tuple[int, int]] = None,
        restrict_verbs: bool = True,
        reveal_enemy_home: bool = True,
        randomize_target: bool = False,
        approach_coef: float = 0.0,
    ) -> None:
        """
        Args:
            map_json_path: map with a learner home port and an enemy home port.
            max_turns: truncation budget (engine turn counter).
            learner_id: engine player id that trains (usually 0).
            capture_reward: reward on capturing the enemy home port.
            landing_position: optional spawn tile; else auto-pick water adjacent
                to the learner's own home so the ship must travel.
            restrict_verbs: if True (default), the learner's legal actions are
                filtered to move/capture/endturn (see _LANDING_VERBS), hiding
                irrelevant options like building ships from the home port.
            reveal_enemy_home: if True (default), the enemy home port is injected
                into the learner's sightings every step so it is always in the
                observation. Turns the skill into relational navigation (move to a
                known target) instead of blind search under fog — see
                _reveal_enemy_home.
            randomize_target: if True, the enemy home port is placed on a random
                tile every reset (learner home fixed, so the spawn is stable).
                Forces the policy to READ the enemy-home token instead of
                memorizing a fixed route — the fix for the fixed-target overfit.
            approach_coef: potential-style bonus per Manhattan step closer to the
                enemy home. 0 (default) leaves the sparse capture reward unchanged.
                Larger boards set a small coef so the ship is not waiting on a
                single +1 at the end of a long walk.
        """
        self.learner_id = learner_id
        self.capture_reward = capture_reward
        self.landing_position = landing_position
        self.restrict_verbs = restrict_verbs
        self.reveal_enemy_home = reveal_enemy_home
        self.randomize_target = randomize_target
        self.approach_coef = float(approach_coef)
        self._captured = False
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        if self.randomize_target:
            self._init_random_target()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def _enemy_id(self) -> int:
        return 1 - self.learner_id

    # ---------------------------------------------------------- target randomize
    def _init_random_target(self) -> None:
        """Cache the board size + fixed learner home; open a reusable temp map."""
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        self._learner_home = tuple(base["ports"][str(self.learner_id)][0])
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="landing_rand_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Regenerate a random-target layout (if enabled) before the engine loads."""
        if self.randomize_target and not (options or {}).get("scenario"):
            self._write_random_layout(seed)
        super().reset(seed=seed, options=options)

    def _write_random_layout(self, seed: Optional[int]) -> None:
        """Write a fresh layout: fixed learner home + a random enemy home tile.

        Learner home and all its cardinal neighbors are excluded from candidate
        enemy-home tiles, so the ship's spawn (auto-picked water next to the
        learner home) stays stable and the target is never adjacent to it. On the
        all-water sandbox boards every remaining tile has water neighbors, so any
        candidate is capturable. Seeded by the reset seed for reproducibility.
        """
        w, h = self._board
        lh = self._learner_home
        rng = random.Random(seed)
        exclude = {lh}
        for dx, dy in _CARDINALS:
            exclude.add((lh[0] + dx, lh[1] + dy))
        candidates = [
            (x, y) for x in range(w) for y in range(h) if (x, y) not in exclude
        ]
        if not candidates:
            raise RuntimeError("LandingSkillAEC: board too small to place a target.")
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
        opts = options or {}
        landing_position = opts.get("landing_position", self.landing_position)
        if landing_position is not None:
            landing_position = tuple(landing_position)

        learner = self.engine.players[self.learner_id]
        if landing_position is None:
            landing_position = self._find_adjacent_water(learner.home_port)
            if landing_position is None:
                raise RuntimeError(
                    "LandingSkillAEC: no free water adjacent to learner home "
                    f"{learner.home_port}; pass landing_position= or fix the map."
                )

        self.engine._spawn_ship(self.learner_id, ShipType.LANDING, landing_position)

        self._captured = False
        # Ensure learner moves first for a clean skill episode.
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._reveal_enemy_home()

        if self.restrict_verbs:
            self._install_verb_filter()

    def _reveal_enemy_home(self) -> None:
        """Force the enemy home port into the learner's sightings (overrides fog).

        ActionMasks/build_observation read enemy ports from player.port_sightings
        (visibility-gated), so without this the target is invisible at spawn and
        the ship can only find it by memorized sweep. Injecting a fresh sighting
        every step makes the target's location always observable, so the policy
        can learn to navigate to it relationally (and thus generalize to moved /
        randomized ports). No-op once the learner owns it (post-capture).
        """
        if not self.reveal_enemy_home or self.engine is None:
            return
        home = self.engine.players[self._enemy_id].home_port
        port = self.engine.ports[home]
        if port.owner == self.learner_id:  # already captured — nothing to reveal
            return
        self.engine.players[self.learner_id].port_sightings[home] = PortSighting(
            position=home,
            last_known_owner=port.owner,
            is_home=port.is_home,
            turn_seen=self.engine.turn,
            fresh=True,
        )

    def _install_verb_filter(self) -> None:
        """Wrap this episode's engine so enumerate_legal only yields _LANDING_VERBS.

        ActionMasks (and the env's legal_actions priming) read enumerate_legal
        directly, so filtering here is the only place that reaches the policy's
        legality masks. Applied to the per-episode engine instance — the engine
        never calls enumerate_legal internally, and reset() builds a fresh engine,
        so this neither perturbs engine mechanics nor compounds across resets.
        """
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, _LANDING_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {"dist": self._landing_dist()}

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

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        learner = self._agent_name(self.learner_id)
        if not self._captured and self.engine is not None:
            if self.engine.winner == self.learner_id:
                self._captured = True
                rewards[learner] = float(self.capture_reward)
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
        """Auto-EndTurn the non-learner seat so AEC only exposes the skill agent."""
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
        # Re-assert the revealed target: engine.step above ran _refresh_sightings,
        # which flips our injected sighting stale (or drops nothing but ages it).
        self._reveal_enemy_home()

    def _find_adjacent_water(
        self, port: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
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
