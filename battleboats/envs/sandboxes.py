"""Sandboxes cut down from the real engine, not a second simulator.

Each one is a 1-player episode on a freshly randomized map. The opponent seat
is auto-end-turned, except combat, where it plays ``scripted_action``. Rewards
are sparse completion rewards (+1, then the episode ends):

- merchant: +1 only after the ship has loaded cargo and then unloaded it at home
- builder: +1 when a ship is bought. Cash starts at 0 and passive home-port
  income is removed, so the gold has to come from a merchant delivery. The
  buy itself is the real ``BuildShipAction``.
- landing: +1 when the landing ship captures the enemy home. No enemy ships.
  Sightings are the engine's real fog.
- combat_<type>: a fixed number of turns against a scripted enemy of the
  same type. The reward is the change in relative combat power, not a kill.
  On small boards that enemy's strength is scaled down (15% at 3×3, full
  strength at 10×10). Sightings are real fog.

``skill_index`` is the one-hot slot the shared policy conditions on.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from battleboats.agents.scripted import scripted_action
from battleboats.core.actions import (
    Action,
    AttackAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import CASH_PER_MATERIAL
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.base_aec import BattleboatsBaseAEC
from battleboats.training.sandbox_skills import SKILL_INDEX

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
Pos = Tuple[int, int]

COMBAT_SHIPS = {
    "destroyer": ShipType.DESTROYER,
    "submarine": ShipType.SUBMARINE,
    "cruiser": ShipType.CRUISER,
    "battleship": ShipType.BATTLESHIP,
    "carrier": ShipType.CARRIER,
}


def combat_reach(ship_type: ShipType) -> int:
    """Tiles at which this ship can both see and hit another of the same type."""
    me = BASE_STATS[ship_type]
    detect = me.scouting * me.visibility
    return max(1, int(min(me.attack_range, detect)))


def enemy_strength_scale(width: int) -> float:
    """0.15 at 3×3, 1.0 at 10×10. Linear in board width."""
    t = (int(width) - 3) / 7.0
    t = min(1.0, max(0.0, t))
    return 0.15 + 0.85 * t


def ship_combat_power(stats: ShipStats) -> float:
    """Power of one live ship. A destroyed ship contributes 0.

    ``strength`` is the engine kill-curve stat. Range and speed scale it.
    No per-type constants: every hull uses the same product.

    power = strength * (1 + attack_range) * speed
    """
    return float(stats.strength) * (1.0 + float(stats.attack_range)) * float(stats.speed)


def side_power(engine, player_id: int) -> float:
    total = 0.0
    for sid in engine.players[player_id].owned_ship_ids:
        ship = engine.ships.get(sid)
        if ship is not None:
            total += ship_combat_power(ship.stats)
    return total


def combat_outcome(my_start: float, enemy_start: float, my_end: float, enemy_end: float) -> float:
    """Change in (own power - enemy power), divided by the starting total.

    Unchanged fleets score 0. Killing the enemy and living is positive.
    Dying while the enemy lives is negative.
    """
    denom = float(my_start) + float(enemy_start)
    if denom <= 0.0:
        return 0.0
    start_edge = float(my_start) - float(enemy_start)
    end_edge = float(my_end) - float(enemy_end)
    return (end_edge - start_edge) / denom


def weaken_strength(stats: ShipStats, scale: float) -> ShipStats:
    """Copy of ``stats`` with strength multiplied by ``scale`` (at least 1)."""
    return ShipStats(
        speed=stats.speed,
        cost=stats.cost,
        strength=max(1.0, float(stats.strength) * float(scale)),
        attack_range=stats.attack_range,
        visibility=stats.visibility,
        scouting=stats.scouting,
    )


def _water_neighbor(land: set, w: int, h: int, pos: Pos) -> bool:
    for dx, dy in _CARDINALS:
        q = (pos[0] + dx, pos[1] + dy)
        if 0 <= q[0] < w and 0 <= q[1] < h and q not in land:
            return True
    return False


def _land_neighbor(pos: Pos, w: int, h: int, banned: set) -> Optional[Pos]:
    for dx, dy in _CARDINALS:
        q = (pos[0] + dx, pos[1] + dy)
        if 0 <= q[0] < w and 0 <= q[1] < h and q not in banned:
            return q
    return None


def combat_layout(rng: random.Random, w: int, h: int, reach: int, close: bool) -> dict:
    """Ship tiles first, then a land neighbor for each home.

    On a trivial board (``close``) the two ships start within attack/detection
    range so a random shot can land. Larger boards may place them anywhere.
    """
    cells = [(x, y) for x in range(w) for y in range(h)]
    rng.shuffle(cells)
    for a in cells:
        for b in cells:
            if a == b:
                continue
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if close and not (1 <= dist <= max(1, reach)):
                continue
            banned = {a, b}
            home = _land_neighbor(a, w, h, banned)
            if home is None:
                continue
            opp = _land_neighbor(b, w, h, banned | {home})
            if opp is None or opp == home:
                continue
            rest = [p for p in cells if p not in {a, b, home, opp}]
            if not rest:
                continue
            return {
                "learner_ship": a,
                "enemy_ship": b,
                "home": home,
                "opp": opp,
                "satellite": rest[0],
            }
    raise RuntimeError(f"no combat layout on {w}x{h} reach={reach} close={close}")


def _neighbors(pos: Pos, w: int, h: int) -> List[Pos]:
    out = []
    for dx, dy in _CARDINALS:
        q = (pos[0] + dx, pos[1] + dy)
        if 0 <= q[0] < w and 0 <= q[1] < h:
            out.append(q)
    return out


def _ports_connected(land: set, w: int, h: int, start: Pos, goal: Pos) -> bool:
    """Some water tile next to ``start`` can reach some water tile next to ``goal``."""
    waters = {(x, y) for x in range(w) for y in range(h) if (x, y) not in land}
    goals = {p for p in _neighbors(goal, w, h) if p in waters}
    stack = [p for p in _neighbors(start, w, h) if p in waters]
    seen = set()
    while stack:
        pos = stack.pop()
        if pos in seen:
            continue
        seen.add(pos)
        if pos in goals:
            return True
        for nxt in _neighbors(pos, w, h):
            if nxt in waters:
                stack.append(nxt)
    return False


def _sample_lands(rng: random.Random, w: int, h: int, n: int) -> List[Pos]:
    """n land tiles that each have a water neighbor."""
    for _ in range(40):
        tiles = [(x, y) for x in range(w) for y in range(h)]
        rng.shuffle(tiles)
        if len(tiles) < n:
            break
        chosen = tiles[:n]
        land = set(chosen)
        if all(_water_neighbor(land, w, h, p) for p in chosen):
            return chosen
    raise RuntimeError(f"no land layout for {w}x{h} with {n} ports")


class _Sandbox(BattleboatsBaseAEC):
    """Shared bookkeeping: one learner, randomized map, optional verb filter."""

    skill_name = ""

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.learner_id = 0
        self.skill_index = SKILL_INDEX[self.skill_name]
        self._rng = random.Random(0)
        self._tmp: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        fd, self._tmp = tempfile.mkstemp(suffix=".json", prefix="sandbox_")
        os.close(fd)

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    def close(self) -> None:
        if self._tmp and os.path.exists(self._tmp):
            try:
                os.remove(self._tmp)
            except OSError:
                pass

    def _write_map(self, land: Sequence[Pos], ports0: Sequence[Pos], ports1: Sequence[Pos]) -> None:
        layout = {
            "width": self.width,
            "height": self.height,
            "seed": 0,
            "land": [list(p) for p in land],
            "ports": {"0": [list(p) for p in ports0], "1": [list(p) for p in ports1]},
        }
        with open(self._tmp, "w") as f:
            json.dump(layout, f)
        self.map_json_path = self._tmp

    def _filter(self, verbs: Tuple[type, ...]) -> None:
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, verbs)]

        self.engine.enumerate_legal = enumerate_legal

    def _spawn_reaching(self, home: Pos, goal: Pos) -> Optional[Pos]:
        assert self.engine is not None
        m = self.engine.map
        w, h = m.width, m.height
        waters = {(x, y) for x in range(w) for y in range(h) if m.is_water((x, y))}
        goals = {p for p in _neighbors(goal, w, h) if p in waters}
        for start in _neighbors(home, w, h):
            if start not in waters or m.is_occupied(start):
                continue
            seen = set()
            stack = [start]
            while stack:
                pos = stack.pop()
                if pos in seen:
                    continue
                seen.add(pos)
                if pos in goals:
                    return start
                for nxt in _neighbors(pos, w, h):
                    if nxt in waters:
                        stack.append(nxt)
        return None

    def _adjacent_water(self, pos: Pos) -> Optional[Pos]:
        assert self.engine is not None
        m = self.engine.map
        for dx, dy in _CARDINALS:
            q = (pos[0] + dx, pos[1] + dy)
            if m.in_bounds(q) and m.terrain[q] == WATER and not m.is_occupied(q):
                return q
        return None

    def _pass_opponent(self, *, scripted: bool, stop) -> None:
        assert self.engine is not None
        guard = 0
        limit = 48 if scripted else 8
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and not stop()
            and self.engine.turn < self.max_turns
            and guard < limit
        ):
            if scripted:
                action = scripted_action(self.engine, self.engine.current_player, self._rng)
            else:
                action = EndTurnAction()
            self.engine.step(action)
            guard += 1

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns


class MerchantSandbox(_Sandbox):
    """Load at a resource port, then unload at home. +1 on that unload."""

    skill_name = "merchant"

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int = 40) -> None:
        self._hauled = False
        self._loaded = False
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def hauled(self) -> bool:
        return self._hauled

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if not (options or {}).get("scenario"):
            rng = random.Random(seed)
            placed = False
            for _ in range(40):
                home, satellite, opp = _sample_lands(rng, self.width, self.height, 3)
                land = {home, satellite, opp}
                if _ports_connected(land, self.width, self.height, home, satellite):
                    self._write_map([home, satellite, opp], [home, satellite], [opp])
                    placed = True
                    break
            if not placed:
                raise RuntimeError("MerchantSandbox: no route from home to the satellite")
        super().reset(seed=seed, options=options)

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._rng = random.Random((seed or 0) + 1)
        self._hauled = False
        self._loaded = False
        learner = self.engine.players[0]
        satellite = next(p for p in learner.owned_port_positions if p != learner.home_port)
        self.engine.ports[satellite].stockpile = 500
        spawn = self._spawn_reaching(learner.home_port, satellite)
        if spawn is None:
            raise RuntimeError("MerchantSandbox: no water next to home")
        self.engine._spawn_ship(0, ShipType.MERCHANT, spawn)
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._filter((MoveAction, MerchantLoadAction, MerchantUnloadAction, EndTurnAction))

    def _capture_reward_state(self) -> Dict:
        assert self.engine is not None
        cargo = 0
        for sid in self.engine.players[0].owned_ship_ids:
            ship = self.engine.ships[sid]
            if ship.type is ShipType.MERCHANT:
                cargo += ship.cargo
        return {"cargo": int(cargo)}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        cargo = 0
        for sid in self.engine.players[0].owned_ship_ids:
            ship = self.engine.ships[sid]
            if ship.type is ShipType.MERCHANT:
                cargo += ship.cargo
        before = int(pre.get("cargo", 0))
        if cargo > before:
            self._loaded = True
        if self._loaded and cargo < before and not self._hauled:
            self._hauled = True
            rewards[self._agent_name(0)] = 1.0
        return rewards

    def _is_terminated(self) -> bool:
        return self._hauled

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._hauled)


class ShipBuildSandbox(_Sandbox):
    """Haul cash home with a merchant, then buy a ship. +1 on the buy only.

    Passive home-port income is removed after every step. Starting cash is 0,
    and the spawned merchant was not bought, so a ``BuildShipAction`` is legal
    only after a delivery.
    """

    skill_name = "builder"

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int = 20) -> None:
        self._built = False
        self._haul_cash = 0
        self._spent = 0
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def built(self) -> bool:
        return self._built

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if not (options or {}).get("scenario"):
            rng = random.Random(seed)
            placed = False
            for _ in range(40):
                home, satellite, opp = _sample_lands(rng, self.width, self.height, 3)
                land = {home, satellite, opp}
                if _ports_connected(land, self.width, self.height, home, satellite):
                    self._write_map([home, satellite, opp], [home, satellite], [opp])
                    placed = True
                    break
            if not placed:
                raise RuntimeError("ShipBuildSandbox: no route from home to the satellite")
        super().reset(seed=seed, options=options)

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._built = False
        self._haul_cash = 0
        self._spent = 0
        learner = self.engine.players[0]
        learner.cash = 0
        satellite = next(p for p in learner.owned_port_positions if p != learner.home_port)
        self.engine.ports[satellite].stockpile = 500
        spawn = self._spawn_reaching(learner.home_port, satellite)
        if spawn is None:
            raise RuntimeError("ShipBuildSandbox: no water next to home")
        self.engine._spawn_ship(0, ShipType.MERCHANT, spawn)
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._filter(
            (MoveAction, MerchantLoadAction, MerchantUnloadAction, BuildShipAction, EndTurnAction)
        )
        self._apply_cash()

    def _apply_cash(self) -> None:
        assert self.engine is not None
        self.engine.players[0].cash = max(0, self._haul_cash - self._spent)

    def _merchant_cargo(self) -> int:
        assert self.engine is not None
        cargo = 0
        for sid in self.engine.players[0].owned_ship_ids:
            ship = self.engine.ships[sid]
            if ship.type is ShipType.MERCHANT:
                cargo += ship.cargo
        return int(cargo)

    def _capture_reward_state(self) -> Dict:
        assert self.engine is not None
        return {
            "cargo": self._merchant_cargo(),
            "ships": set(self.engine.players[0].owned_ship_ids),
        }

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        cargo = self._merchant_cargo()
        before = int(pre.get("cargo", 0))
        if cargo < before:
            self._haul_cash += (before - cargo) * CASH_PER_MATERIAL
        new = set(self.engine.players[0].owned_ship_ids) - set(pre.get("ships", ()))
        if new and not self._built:
            for sid in new:
                self._spent += int(self.engine.ships[sid].stats.cost)
            self._built = True
            rewards[self._agent_name(0)] = 1.0
        self._apply_cash()
        return rewards

    def _is_terminated(self) -> bool:
        return self._built

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._built)


class LandingSandbox(_Sandbox):
    """One landing ship, one enemy home, no enemy ships. +1 on capture."""

    skill_name = "landing"

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int = 30) -> None:
        self._captured = False
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def captured(self) -> bool:
        return self._captured

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if not (options or {}).get("scenario"):
            rng = random.Random(seed)
            placed = False
            for _ in range(40):
                home, opp = _sample_lands(rng, self.width, self.height, 2)
                if _ports_connected({home, opp}, self.width, self.height, home, opp):
                    self._write_map([home, opp], [home], [opp])
                    placed = True
                    break
            if not placed:
                raise RuntimeError("LandingSandbox: no route to the enemy home")
        super().reset(seed=seed, options=options)

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._captured = False
        spawn = self._spawn_reaching(
            self.engine.players[0].home_port, self.engine.players[1].home_port
        )
        if spawn is None:
            raise RuntimeError("LandingSandbox: no water next to home")
        self.engine._spawn_ship(0, ShipType.LANDING, spawn)
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._filter((MoveAction, CapturePortAction, EndTurnAction))

    def _capture_reward_state(self) -> Dict:
        return {}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        if not self._captured and self.engine is not None and self.engine.winner == 0:
            self._captured = True
            rewards[self._agent_name(0)] = 1.0
        return rewards

    def _is_terminated(self) -> bool:
        return self._captured

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._captured)


class CombatSandbox(_Sandbox):
    """One combat ship versus a scripted enemy of the same type.

    The episode always lasts ``max_turns``. The only reward is
    ``combat_outcome`` at that horizon. A kill does not end the episode
    and does not add a bonus.
    """

    def __init__(
        self,
        map_json_path: str,
        *,
        ship: str,
        width: int,
        height: int,
        max_turns: int = 30,
    ) -> None:
        if ship not in COMBAT_SHIPS:
            raise ValueError(f"unknown combat ship {ship!r}")
        self.ship = ship
        self.skill_name = f"combat_{ship}"
        self.ship_type = COMBAT_SHIPS[ship]
        self._killed = False
        self._target_id: Optional[int] = None
        self._scored = False
        self.episode_reward = 0.0
        self._my0 = 0.0
        self._en0 = 0.0
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def killed(self) -> bool:
        return self._killed

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if (options or {}).get("scenario"):
            super().reset(seed=seed, options=options)
            return
        reach = combat_reach(self.ship_type)
        close = self.width <= 4
        last_error: Optional[BaseException] = None
        for attempt in range(20):
            rng = random.Random((seed or 0) + attempt * 17)
            try:
                self._layout = combat_layout(rng, self.width, self.height, reach, close)
                lay = self._layout
                self._write_map([lay["home"], lay["opp"]], [lay["home"]], [lay["opp"]])
                super().reset(seed=seed, options=options)
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"CombatSandbox: {last_error}")

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._rng = random.Random((seed or 0) + 3)
        self._killed = False
        self._scored = False
        self.episode_reward = 0.0
        lay = self._layout
        own = lay["learner_ship"]
        tgt = lay["enemy_ship"]
        m = self.engine.map
        if m.terrain[own] != WATER or m.terrain[tgt] != WATER or own == tgt:
            raise RuntimeError("CombatSandbox: spawn tiles are not water")
        hunter = self.engine._spawn_ship(0, self.ship_type, own)
        target = self.engine._spawn_ship(1, self.ship_type, tgt)
        full = BASE_STATS[self.ship_type].strength
        scale = enemy_strength_scale(self.width)
        target.stats = weaken_strength(BASE_STATS[self.ship_type], scale)
        self._strength_scale = scale
        if target.stats.strength >= full and scale < 1.0:
            raise RuntimeError("CombatSandbox: enemy strength was not reduced")
        # No cash: the scripted policy fights this ship instead of buying more.
        self.engine.players[1].cash = 0
        self._hunter_id = hunter.id
        self._target_id = target.id
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._my0 = side_power(self.engine, 0)
        self._en0 = side_power(self.engine, 1)
        self._filter((MoveAction, AttackAction, EndTurnAction))

    def _capture_reward_state(self) -> Dict:
        return {}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        if self._target_id is not None and self._target_id not in self.engine.ships:
            self._killed = True
        if not self._scored and self.engine.turn >= self.max_turns:
            self.episode_reward = combat_outcome(
                self._my0, self._en0, side_power(self.engine, 0), side_power(self.engine, 1)
            )
            self._scored = True
            rewards[self._agent_name(0)] = self.episode_reward
        return rewards

    def _is_terminated(self) -> bool:
        return False

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=True, stop=lambda: False)
        if self.engine is not None:
            self.engine.players[1].cash = 0


def make_sandbox(skill: str, map_json_path: str, *, width: int, height: int, max_turns: int):
    """Build the sandbox named by ``skill`` (see ``SKILL_NAMES``)."""
    if skill == "merchant":
        return MerchantSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    if skill == "builder":
        return ShipBuildSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    if skill == "landing":
        return LandingSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    if skill.startswith("combat_"):
        return CombatSandbox(
            map_json_path,
            ship=skill.split("_", 1)[1],
            width=width,
            height=height,
            max_turns=max_turns,
        )
    raise ValueError(f"unknown sandbox {skill!r}")


def horizon(skill: str, size: int) -> Tuple[int, int]:
    """(max_turns, max_steps) for a square board of this size.

    Combat is a fixed fight: ``max(8, 3 * size)`` turns (9 on 3×3, 30 on 10×10).
    The reward is paid at that horizon, not when a ship dies.
    """
    if skill == "builder":
        return max(10, size * 4), max(36, size * 12)
    if skill == "merchant":
        return max(8, size * 4), max(24, size * 12)
    if skill.startswith("combat_"):
        turns = max(8, int(size) * 3)
        return turns, max(24, turns * 8)
    return max(8, size * 3), max(16, size * 8)
