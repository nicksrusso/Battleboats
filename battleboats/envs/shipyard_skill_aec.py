"""Buy-a-ship skill sandbox (PettingZoo AEC).

The full game starts with cash and an empty fleet. The other sandboxes spawn
the ship already built, so a policy trained only on those never selects
``build_ship``. This sandbox is that missing mechanic.

Each episode is one of four tasks. The task is visible in the ordinary
observation (no extra goal channel, so the weights match the full game):

- merchant: a friendly port is holding a stockpile, cash equals a merchant.
  Success: a merchant exists.
- combat: an enemy merchant is in sight, cash equals a destroyer.
  Success: a ship with attack range > 0 exists.
- landing: the enemy home is revealed and nothing else is on the board,
  cash equals a landing ship. Success: a landing ship exists.
- builder: an empty coastline tile is present, cash equals a builder.
  Success: a builder exists.

Reward is +1 on the step the correct ship appears, and 0 otherwise. Home-port
income is clamped back while the fleet is still empty, so waiting does not
widen the affordable set. A wrong purchase spends the cash and is not clamped
back. Success is ``built_correct``.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, BuildShipAction, EndTurnAction
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.sighting import PortSighting, Sighting
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SHIPYARD_VERBS: Tuple[type, ...] = (BuildShipAction, EndTurnAction)
SHIPYARD_TASKS: Tuple[str, ...] = ("merchant", "combat", "landing", "builder")

_TASK_SHIP = {
    "merchant": ShipType.MERCHANT,
    "landing": ShipType.LANDING,
    "builder": ShipType.BUILDER,
    "combat": ShipType.DESTROYER,
}


class ShipyardSkillAEC(BattleboatsBaseAEC):
    """1p shipyard skill: read the board and buy the ship it calls for."""

    metadata = {"name": "battleboats_shipyard_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 8,
        learner_id: int = 0,
        build_reward: float = 1.0,
        restrict_verbs: bool = True,
        randomize_layout: bool = False,
        task: Optional[str] = None,
    ) -> None:
        if task is not None and task not in SHIPYARD_TASKS:
            raise ValueError(f"unknown shipyard task {task!r}; expected one of {SHIPYARD_TASKS}")
        self.learner_id = learner_id
        self.build_reward = build_reward
        self.restrict_verbs = restrict_verbs
        self.randomize_layout = randomize_layout
        self.fixed_task = task
        self.active_task = task or "merchant"
        self._built_correct = False
        self._task_cash = 0
        self._satellite: Optional[Tuple[int, int]] = None
        self._enemy_ship_id: Optional[int] = None
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        self._init_board()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    @property
    def built_correct(self) -> bool:
        return self._built_correct

    @property
    def _enemy_id(self) -> int:
        return 1 - self.learner_id

    def _init_board(self) -> None:
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="shipyard_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        opts = options or {}
        if not opts.get("scenario"):
            self.active_task = self._pick_task(seed, opts.get("shipyard_task"))
            self._write_layout(seed)
        super().reset(seed=seed, options=options)

    def _pick_task(self, seed: Optional[int], override: Optional[str]) -> str:
        if override is not None:
            if override not in SHIPYARD_TASKS:
                raise ValueError(f"unknown shipyard task {override!r}")
            return override
        if self.fixed_task is not None:
            return self.fixed_task
        return SHIPYARD_TASKS[int(seed or 0) % len(SHIPYARD_TASKS)]

    def _write_layout(self, seed: Optional[int]) -> None:
        w, h = self._board
        home = (0, 0)
        opp = (w - 1, h - 1)
        land = [home, opp]
        ports0 = [home]
        self._satellite = None
        extra = self._extra_tile(seed, home, opp)
        if self.active_task == "merchant":
            self._satellite = extra
            land.append(extra)
            ports0.append(extra)
        elif self.active_task == "builder":
            land.append(extra)
        layout = {
            "width": w,
            "height": h,
            "seed": 0,
            "land": [list(p) for p in land],
            "ports": {
                str(self.learner_id): [list(p) for p in ports0],
                str(self._enemy_id): [list(opp)],
            },
        }
        with open(self._tmp_map_path, "w") as f:
            json.dump(layout, f)
        self.map_json_path = self._tmp_map_path

    def _extra_tile(self, seed: Optional[int], home: Tuple[int, int], opp: Tuple[int, int]) -> Tuple[int, int]:
        """A tile that is neither home nor the enemy home, with a water neighbor."""
        w, h = self._board
        canonical = (w - 1, 0)
        if canonical in (home, opp):
            canonical = (0, h - 1)
        if not self.randomize_layout:
            if canonical in (home, opp):
                raise RuntimeError("ShipyardSkillAEC: board has no tile for the task marker.")
            return canonical
        rng = random.Random((seed or 0) + 17)
        exclude = {home, opp}
        candidates = [(x, y) for x in range(w) for y in range(h) if (x, y) not in exclude]
        if not candidates:
            raise RuntimeError("ShipyardSkillAEC: board too small for a task marker.")
        return rng.choice(candidates)

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
        task = self.active_task
        ship_type = _TASK_SHIP[task]
        self._task_cash = int(BASE_STATS[ship_type].cost)
        self.engine.players[self.learner_id].cash = self._task_cash
        self._enemy_ship_id = None
        self._built_correct = False
        if task == "merchant" and self._satellite is not None:
            self.engine.ports[self._satellite].stockpile = 10_000
        if task == "combat":
            opp_home = self.engine.players[self._enemy_id].home_port
            spawn = self._adjacent_water(opp_home)
            if spawn is None:
                raise RuntimeError("ShipyardSkillAEC: no water next to the enemy home.")
            ship = self.engine._spawn_ship(self._enemy_id, ShipType.MERCHANT, spawn)
            self._enemy_ship_id = ship.id
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._reveal()
        if self.restrict_verbs:
            self._install_verb_filter()

    def _reveal(self) -> None:
        """Keep the task's cue in the learner's sightings after fog refresh."""
        if self.engine is None:
            return
        if self.active_task == "landing":
            home = self.engine.players[self._enemy_id].home_port
            port = self.engine.ports[home]
            self.engine.players[self.learner_id].port_sightings[home] = PortSighting(
                position=home,
                last_known_owner=port.owner,
                is_home=port.is_home,
                turn_seen=self.engine.turn,
                fresh=True,
            )
        if self.active_task == "combat" and self._enemy_ship_id is not None:
            ship = self.engine.ships.get(self._enemy_ship_id)
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
            return [a for a in _raw(player_id) if isinstance(a, _SHIPYARD_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _clamp_opening_cash(self) -> None:
        """Drop home-port income while the fleet is empty.

        A wrong purchase is left spent, so the learner cannot buy every type
        in one episode by waiting for income.
        """
        assert self.engine is not None
        player = self.engine.players[self.learner_id]
        if not player.owned_ship_ids:
            player.cash = self._task_cash

    def _correct_ship_present(self) -> bool:
        assert self.engine is not None
        owned = []
        for sid in self.engine.players[self.learner_id].owned_ship_ids:
            ship = self.engine.ships.get(sid)
            if ship is not None:
                owned.append(ship)
        task = self.active_task
        if task == "combat":
            return any(s.stats.attack_range > 0 for s in owned)
        wanted = _TASK_SHIP[task]
        return any(s.type is wanted for s in owned)

    def _capture_reward_state(self) -> Dict[str, Any]:
        return {}

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        if not self._built_correct and self._correct_ship_present():
            self._built_correct = True
            rewards[self._agent_name(self.learner_id)] = float(self.build_reward)
        return rewards

    def _is_terminated(self) -> bool:
        return self._built_correct

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not self._built_correct
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1
        self._clamp_opening_cash()
        self._reveal()

    def _adjacent_water(self, port: Tuple[int, int]) -> Optional[Tuple[int, int]]:
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
