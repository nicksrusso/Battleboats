"""Sandboxes cut down from the real engine, not a second simulator.

Each one is a 1-player episode on a freshly randomized map. The opponent seat
is auto-end-turned, except combat, where it plays ``scripted_action``. Rewards
are completion rewards scaled by ``turn_decay`` (game turns, not actions):

- merchant: the decayed reward only after the ship has loaded cargo and then
  unloaded it at home. Loading alone pays 0.
- builder: the decayed reward when a ship is bought. Cash starts at 0 and
  passive home-port income is removed, so the gold has to come from a merchant
  delivery. The buy itself is the real ``BuildShipAction``.
- landing: the decayed reward when the landing ship captures the enemy home.
  No enemy ships. Sightings are the engine's real fog.
- landing_build: no ships and enough cash for one landing ship (250). Every
  hull is buyable by default (``hulls="all"``), so the drill teaches picking
  the landing ship; ``hulls="landing"`` allows only landing-ship builds. The
  decayed reward when a landing ship it built captures the enemy home. A
  wrong hull pays nothing and delays the landing ship until income catches up.
- scouting: random hull vs random hull. The decayed reward on the first fresh
  sighting of the enemy ship, and the episode ends. The enemy holds still and
  starts outside the learner ship's sight.
- combat_<type>: a fixed number of turns against a scripted enemy of the
  same type. The reward is the change in relative combat power, each change
  scaled by the turn it happened, paid once at the horizon. A kill is not a
  win and does not end the episode. On small boards that enemy's strength is
  scaled down (15% at 3×3, full strength at 10×10). Sightings are real fog.
- build_<type>: one drill per buildable hull. Cash starts one home-port
  income tick below that hull's cost, and income is clamped at the cost so
  a dearer hull never becomes legal. The cue is a visible enemy ship of the
  target type (several hulls share a price, so cash cannot name it). Buying
  that type pays more the sooner it happens after it becomes affordable.
  Any other purchase pays 0. The episode ends on the purchase or the horizon.
  The haul-then-buy ``builder`` drill stays: it is a delivery, not a typed buy.

``skill_index`` is the one-hot slot the shared policy conditions on.
The warm-up trains with ``skill_count=0`` and does not read it.
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
from battleboats.core.gameEngine import CASH_PER_MATERIAL, PORT_PRODUCTION
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.sighting import Sighting
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

# Every ShipType has a cost in BASE_STATS and is a legal BuildShipAction target.
BUILD_SHIPS = {ship.name.lower(): ship for ship in ShipType}
HOME_TURN_INCOME = PORT_PRODUCTION * CASH_PER_MATERIAL


def ship_detection_range(scout: ShipType, target: ShipType) -> float:
    """Tiles at which ``scout``'s hull spots ``target``.

    Same product the engine uses for two ships: scouting × visibility.
    Ports are not part of this number. A port's own radius covers a 3×3,
    which would make a search drill see the enemy at spawn.
    """
    me = BASE_STATS[scout]
    them = BASE_STATS[target]
    return float(me.scouting) * float(them.visibility)


def turn_decay(turn: int, horizon_turns: int, floor: float = 0.0) -> float:
    """1 on turn 0, falling linearly to ``floor`` at ``horizon_turns``.

    The clock is the engine turn, which advances when both seats have ended,
    not the number of move actions. A ship that spends its speed oscillating
    has to end the turn, and the reward for finishing later is smaller.
    A completion before the horizon stays strictly above ``floor`` (and above
    0 when ``floor`` is 0). Past the horizon the value stays at ``floor``.
    ``horizon_turns <= 0`` means no decay.
    """
    horizon_turns = int(horizon_turns)
    floor = float(floor)
    if horizon_turns <= 0:
        return 1.0
    turn = int(turn)
    if turn >= horizon_turns:
        return floor
    if turn <= 0:
        return 1.0
    remaining = (horizon_turns - turn) / float(horizon_turns)
    return floor + (1.0 - floor) * remaining


def scouting_find_reward(turn: int, horizon_turns: int, floor: float = 0.0) -> float:
    """Full reward on a sighting at turn 0, then ``turn_decay``."""
    return turn_decay(turn, horizon_turns, floor)


def opening_cash(cost: int) -> int:
    """Cash at the start of a typed-build drill: one income tick short of ``cost``."""
    return max(0, int(cost) - HOME_TURN_INCOME)


def typed_build_reward(delay: int, horizon_turns: int, floor: float = 0.0) -> float:
    """1 when the cued hull is bought on the turn it becomes affordable.

    ``delay`` is how many engine turns passed after that moment. The mandatory
    income wait is not part of the delay. Waiting after the hull is affordable
    pays ``turn_decay``.
    """
    return turn_decay(delay, horizon_turns, floor)


def discounted_combat_reward(
    samples: Sequence[Tuple[int, float]],
    horizon_turns: int,
    floor: float = 0.0,
) -> float:
    """Weight each change in normalized combat edge by the turn it happens.

    ``samples`` is ``(turn, edge)`` beginning with the opening edge. ``edge``
    is ``(own power - enemy power) / starting total``, so the undiscounted
    sum of changes is ``combat_outcome``. An earlier change keeps more of its
    weight. Weights lie in ``[floor, 1]``, and a ship dies at most once, so
    the result stays in ``[-1, 1]``. This is not a win bonus.
    """
    if len(samples) < 2:
        return 0.0
    total = 0.0
    prev = float(samples[0][1])
    for turn, edge in samples[1:]:
        now = float(edge)
        delta = now - prev
        if delta != 0.0:
            total += turn_decay(int(turn), horizon_turns, floor) * delta
        prev = now
    return total


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


def _ships_connected(land: set, w: int, h: int, start: Pos, goal: Pos) -> bool:
    """Water path from ``start`` to ``goal``. Both cells are water."""
    if start in land or goal in land:
        return False
    waters = {(x, y) for x in range(w) for y in range(h) if (x, y) not in land}
    stack = [start]
    seen = set()
    while stack:
        pos = stack.pop()
        if pos in seen:
            continue
        seen.add(pos)
        if pos == goal:
            return True
        for nxt in _neighbors(pos, w, h):
            if nxt in waters and nxt not in seen:
                stack.append(nxt)
    return False


def scouting_layout(rng: random.Random, w: int, h: int, reach: float) -> Optional[dict]:
    """Two ship tiles farther apart than ``reach``, plus a land port each.

    Returns None when this board cannot hide a pair at that detection range.
    The learner ship can walk to the enemy: the water between them connects.
    """
    cells = [(x, y) for x in range(w) for y in range(h)]
    if len(cells) < 4:
        return None
    for _ in range(200):
        learner, enemy, home, opp = rng.sample(cells, 4)
        dist = abs(learner[0] - enemy[0]) + abs(learner[1] - enemy[1])
        if dist <= reach:
            continue
        land = {home, opp}
        if not _ships_connected(land, w, h, learner, enemy):
            continue
        return {
            "learner_ship": learner,
            "enemy_ship": enemy,
            "home": home,
            "opp": opp,
        }
    return None


def typed_build_layout(rng: random.Random, w: int, h: int) -> Optional[dict]:
    """Two homes, a cue tile by the enemy home, and a free spawn by ours.

    The cue tile is water and is not the learner's only build tile.
    """
    cells = [(x, y) for x in range(w) for y in range(h)]
    if len(cells) < 4:
        return None
    for _ in range(80):
        home, opp = rng.sample(cells, 2)
        land = {home, opp}
        home_water = [p for p in _neighbors(home, w, h) if p not in land]
        opp_water = [p for p in _neighbors(opp, w, h) if p not in land]
        if not home_water or not opp_water:
            continue
        cue = None
        for candidate in opp_water:
            if candidate not in home_water or len(home_water) > 1:
                cue = candidate
                break
        if cue is None:
            continue
        return {"home": home, "opp": opp, "cue": cue}
    return None


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
        # 0 in the config means "this episode's max_turns". Floor 0 keeps a
        # completion on the last legal turn strictly above a stall.
        self.decay_horizon = int(max_turns)
        self.decay_floor = 0.0
        fd, self._tmp = tempfile.mkstemp(suffix=".json", prefix="sandbox_")
        os.close(fd)

    def _turn_scale(self, turn: int) -> float:
        return turn_decay(int(turn), self.decay_horizon, self.decay_floor)

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
        return {"cargo": int(cargo), "turn": int(self.engine.turn)}

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
            rewards[self._agent_name(0)] = self._turn_scale(int(pre.get("turn", 0)))
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
            "turn": int(self.engine.turn),
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
            rewards[self._agent_name(0)] = self._turn_scale(int(pre.get("turn", 0)))
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
        assert self.engine is not None
        return {"turn": int(self.engine.turn)}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        if not self._captured and self.engine is not None and self.engine.winner == 0:
            self._captured = True
            rewards[self._agent_name(0)] = self._turn_scale(int(pre.get("turn", 0)))
        return rewards

    def _is_terminated(self) -> bool:
        return self._captured

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._captured)


class LandingBuildSandbox(_Sandbox):
    """Build a landing ship at home, sail it over, capture the enemy home.

    No ships at the start, cash for exactly one landing ship (its cost), real
    home income, no enemy ships. Verbs: build, move, capture, end turn.
    ``hulls="all"`` leaves every hull buyable (it has to pick the landing
    ship); ``hulls="landing"`` filters builds to landing ships only.
    Success is the capture.
    """

    skill_name = "landing_build"
    HULLS = ("all", "landing")

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int = 30, hulls: str = "all") -> None:
        if hulls not in self.HULLS:
            raise ValueError(f"LandingBuildSandbox: hulls must be one of {self.HULLS}, got {hulls!r}")
        self.hulls = hulls
        self._captured = False
        self._built_landing = False
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def built_landing(self) -> bool:
        return self._built_landing

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
                raise RuntimeError("LandingBuildSandbox: no route to the enemy home")
        super().reset(seed=seed, options=options)

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._captured = False
        self._built_landing = False
        if self._spawn_reaching(self.engine.players[0].home_port, self.engine.players[1].home_port) is None:
            raise RuntimeError("LandingBuildSandbox: no water next to home that reaches the enemy home")
        self.engine.players[0].cash = int(BASE_STATS[ShipType.LANDING].cost)
        self.engine.players[1].cash = 0
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._filter((BuildShipAction, MoveAction, CapturePortAction, EndTurnAction))
        if self.hulls == "landing":
            raw = self.engine.enumerate_legal

            def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
                return [
                    a for a in _raw(player_id)
                    if not isinstance(a, BuildShipAction) or a.ship_type == ShipType.LANDING
                ]

            self.engine.enumerate_legal = enumerate_legal

    def _capture_reward_state(self) -> Dict:
        assert self.engine is not None
        return {"turn": int(self.engine.turn), "ships": set(self.engine.players[0].owned_ship_ids)}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        new = set(self.engine.players[0].owned_ship_ids) - set(pre.get("ships", ()))
        if any(self.engine.ships[s].type == ShipType.LANDING for s in new):
            self._built_landing = True
        if not self._captured and self.engine.winner == 0:
            self._captured = True
            rewards[self._agent_name(0)] = self._turn_scale(int(pre.get("turn", 0)))
        return rewards

    def _is_terminated(self) -> bool:
        return self._captured

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._captured)


class CombatSandbox(_Sandbox):
    """One combat ship versus a scripted enemy of the same type.

    The episode always lasts ``max_turns``. The only reward is the
    turn-weighted ``combat_outcome``, paid once at that horizon. An early
    change in relative power keeps more of its weight than the same change
    later. A kill does not end the episode and does not add a win bonus.
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
        self._power_samples: List[Tuple[int, float]] = []
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
        self._power_samples = [(0, self._edge())]
        self._filter((MoveAction, AttackAction, EndTurnAction))

    def _edge(self) -> float:
        """Normalized (own power - enemy power). Opening value is the baseline."""
        denom = float(self._my0) + float(self._en0)
        if denom <= 0.0 or self.engine is None:
            return 0.0
        return (side_power(self.engine, 0) - side_power(self.engine, 1)) / denom

    def _capture_reward_state(self) -> Dict:
        return {}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        if self._target_id is not None and self._target_id not in self.engine.ships:
            self._killed = True
        self._power_samples.append((int(self.engine.turn), self._edge()))
        if not self._scored and self.engine.turn >= self.max_turns:
            self.episode_reward = discounted_combat_reward(
                self._power_samples, self.decay_horizon, self.decay_floor
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


class ScoutingSandbox(_Sandbox):
    """Find one enemy ship under the engine's ship fog. It does not fight back.

    Hulls are random each episode, including merchants and landings. The
    enemy is placed outside the learner ship's detection range and then
    only ends its turn, so the drill is a search rather than a chase. A
    fleeing hull would make slow-versus-fast matchups unwinnable, and the
    success bar would be the wrong gate.

    Home ports stay on the map (the engine boots from them) but are taken
    out of ``owned_port_positions`` before the sighting refresh. A port
    scouts at radius 4 × visibility, which covers every tile of a 3×3, so
    the enemy would be visible at spawn and the search would be free.
    Detection is otherwise the real sighting: fresh, ship scouting times
    target visibility.

    Reward is ``scouting_find_reward`` on the first fresh sighting and 0
    if the horizon arrives first. The episode ends on that sighting.
    Success for the curriculum window is whether it was found, same bar as
    the other goal drills. A reward plateau would also pass a policy that
    never looks.
    """

    skill_name = "scouting"

    def __init__(self, map_json_path: str, *, width: int, height: int, max_turns: int = 40) -> None:
        self._found = False
        self._target_id: Optional[int] = None
        self.episode_reward = 0.0
        self._learner_type = ShipType.DESTROYER
        self._enemy_type = ShipType.DESTROYER
        self._layout: dict = {}
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def found(self) -> bool:
        return self._found

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if (options or {}).get("scenario"):
            super().reset(seed=seed, options=options)
            return
        types = tuple(ShipType)
        last_error: Optional[BaseException] = None
        for attempt in range(60):
            rng = random.Random((seed or 0) + attempt * 17)
            scout = types[rng.randrange(len(types))]
            target = types[rng.randrange(len(types))]
            reach = ship_detection_range(scout, target)
            layout = scouting_layout(rng, self.width, self.height, reach)
            if layout is None:
                continue
            self._learner_type = scout
            self._enemy_type = target
            self._layout = layout
            try:
                self._write_map([layout["home"], layout["opp"]], [layout["home"]], [layout["opp"]])
                super().reset(seed=seed, options=options)
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            f"ScoutingSandbox: no hidden matchup on {self.width}x{self.height}: {last_error}"
        )

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._rng = random.Random((seed or 0) + 5)
        self._found = False
        self.episode_reward = 0.0
        lay = self._layout
        own = lay["learner_ship"]
        tgt = lay["enemy_ship"]
        m = self.engine.map
        if m.terrain[own] != WATER or m.terrain[tgt] != WATER or own == tgt:
            raise RuntimeError("ScoutingSandbox: spawn tiles are not water")
        hunter = self.engine._spawn_ship(0, self._learner_type, own)
        target = self.engine._spawn_ship(1, self._enemy_type, tgt)
        self._hunter_id = hunter.id
        self._target_id = target.id
        # Ports still exist as land. Dropping them from the owned set keeps
        # their radius-4 scan from revealing the whole small board.
        self.engine.players[0].owned_port_positions.clear()
        self.engine.players[1].owned_port_positions.clear()
        self.engine.players[0].cash = 0
        self.engine.players[1].cash = 0
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        if self._sees_enemy():
            raise RuntimeError("ScoutingSandbox: enemy started inside sight")
        self._filter((MoveAction, EndTurnAction))

    def _sees_enemy(self) -> bool:
        assert self.engine is not None and self._target_id is not None
        sight = self.engine.players[0].sightings.get(self._target_id)
        return sight is not None and bool(sight.fresh)

    def _capture_reward_state(self) -> Dict:
        return {}

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        if not self._found and self._sees_enemy():
            self._found = True
            self.episode_reward = scouting_find_reward(
                self.engine.turn, self.decay_horizon, self.decay_floor
            )
            rewards[self._agent_name(0)] = self.episode_reward
        return rewards

    def _is_terminated(self) -> bool:
        return self._found

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._found)


class TypedBuildSandbox(_Sandbox):
    """Wait for income, then buy the hull the visible enemy ship names.

    Merchant, destroyer, and submarine all cost 100. Carrier, landing, and
    builder all cost 250. A cash ceiling cannot tell those apart, and the
    warm-up has no skill embedding. The cue is an enemy ship token of the
    target type, the same type one-hot the full game already shows, kept
    fresh so a large board's fog does not hide it. Home-port income is the
    real tick (``HOME_TURN_INCOME``) and is then clamped to the target cost,
    so the learner must EndTurn at least once and a dearer hull stays illegal.
    Cheaper hulls can be legal earlier. Buying one pays 0 and ends the episode.
    Buying the cued hull pays ``typed_build_reward``.
    """

    def __init__(
        self,
        map_json_path: str,
        *,
        ship: str,
        width: int,
        height: int,
        max_turns: int = 12,
    ) -> None:
        if ship not in BUILD_SHIPS:
            raise ValueError(f"unknown build ship {ship!r}; expected one of {tuple(BUILD_SHIPS)}")
        self.ship = ship
        self.skill_name = f"build_{ship}"
        self.ship_type = BUILD_SHIPS[ship]
        self._cost = int(BASE_STATS[self.ship_type].cost)
        self._built_target = False
        self._built_wrong = False
        self._affordable_turn: Optional[int] = None
        self._cue_id: Optional[int] = None
        self.episode_reward = 0.0
        self._layout: dict = {}
        super().__init__(map_json_path, width=width, height=height, max_turns=max_turns)

    @property
    def built_target(self) -> bool:
        return self._built_target

    def reset(self, seed: Optional[int] = None, options=None) -> None:
        if (options or {}).get("scenario"):
            super().reset(seed=seed, options=options)
            return
        last_error: Optional[BaseException] = None
        for attempt in range(40):
            rng = random.Random((seed or 0) + attempt * 17)
            layout = typed_build_layout(rng, self.width, self.height)
            if layout is None:
                continue
            self._layout = layout
            try:
                self._write_map([layout["home"], layout["opp"]], [layout["home"]], [layout["opp"]])
                super().reset(seed=seed, options=options)
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"TypedBuildSandbox: no build layout on {self.width}x{self.height}: {last_error}")

    def _configure_scenario(self, seed, options) -> None:
        assert self.engine is not None
        self._built_target = False
        self._built_wrong = False
        self._affordable_turn = None
        self.episode_reward = 0.0
        lay = self._layout
        cue = lay["cue"]
        m = self.engine.map
        if m.terrain[cue] != WATER:
            raise RuntimeError("TypedBuildSandbox: cue tile is not water")
        ship = self.engine._spawn_ship(1, self.ship_type, cue)
        self._cue_id = ship.id
        if self._adjacent_water(self.engine.players[0].home_port) is None:
            raise RuntimeError("TypedBuildSandbox: no free water next to home")
        self.engine.players[0].cash = opening_cash(self._cost)
        self.engine.players[1].cash = 0
        self.engine.current_player = 0
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()
        self._show_cue()
        self._filter((BuildShipAction, EndTurnAction))

    def _show_cue(self) -> None:
        """Keep the target type in the learner's tokens on every board size."""
        assert self.engine is not None and self._cue_id is not None
        ship = self.engine.ships.get(self._cue_id)
        if ship is None:
            return
        self.engine.players[0].sightings[ship.id] = Sighting(
            ship_id=ship.id,
            type=ship.type,
            position=ship.position,
            turn_seen=self.engine.turn,
            fresh=True,
        )

    def _cap_cash(self) -> None:
        """Real income, stopped at the target cost so a dearer hull stays illegal."""
        assert self.engine is not None
        if self._built_target or self._built_wrong:
            return
        player = self.engine.players[0]
        if player.cash > self._cost:
            player.cash = self._cost

    def _capture_reward_state(self) -> Dict:
        assert self.engine is not None
        turn = int(self.engine.turn)
        if self._affordable_turn is None and self.engine.players[0].cash >= self._cost:
            self._affordable_turn = turn
        return {
            "ships": set(self.engine.players[0].owned_ship_ids),
            "turn": turn,
        }

    def _reward_after_step(self, agent, action, pre) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        assert self.engine is not None
        new = set(self.engine.players[0].owned_ship_ids) - set(pre.get("ships", ()))
        if not new or self._built_target or self._built_wrong:
            return rewards
        bought = self.engine.ships[next(iter(new))].type
        if bought is self.ship_type and self._affordable_turn is not None:
            delay = int(pre.get("turn", 0)) - int(self._affordable_turn)
            self.episode_reward = typed_build_reward(
                delay, self.decay_horizon, self.decay_floor
            )
            self._built_target = True
            rewards[self._agent_name(0)] = self.episode_reward
        else:
            self._built_wrong = True
            self.episode_reward = 0.0
        return rewards

    def _is_terminated(self) -> bool:
        return self._built_target or self._built_wrong

    def _after_engine_step(self, agent, action) -> None:
        self._pass_opponent(scripted=False, stop=lambda: self._built_target or self._built_wrong)
        self._cap_cash()
        self._show_cue()


def make_sandbox(
    skill: str,
    map_json_path: str,
    *,
    width: int,
    height: int,
    max_turns: int,
    decay_horizon: int = 0,
    decay_floor: float = 0.0,
    landing_build_hulls: str = "all",
):
    """Build the sandbox named by ``skill`` (see ``SKILL_NAMES``).

    ``decay_horizon`` 0 uses this episode's ``max_turns``. The floor is the
    multiplier once that many turns have passed.
    """
    if skill == "merchant":
        env = MerchantSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    elif skill == "builder":
        env = ShipBuildSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    elif skill == "landing":
        env = LandingSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    elif skill == "landing_build":
        env = LandingBuildSandbox(
            map_json_path, width=width, height=height, max_turns=max_turns, hulls=landing_build_hulls
        )
    elif skill == "scouting":
        env = ScoutingSandbox(map_json_path, width=width, height=height, max_turns=max_turns)
    elif skill.startswith("combat_"):
        env = CombatSandbox(
            map_json_path,
            ship=skill.split("_", 1)[1],
            width=width,
            height=height,
            max_turns=max_turns,
        )
    elif skill.startswith("build_"):
        env = TypedBuildSandbox(
            map_json_path,
            ship=skill.split("_", 1)[1],
            width=width,
            height=height,
            max_turns=max_turns,
        )
    else:
        raise ValueError(f"unknown sandbox {skill!r}")
    env.decay_floor = float(decay_floor)
    if int(decay_horizon) > 0:
        env.decay_horizon = int(decay_horizon)
    return env


def horizon(skill: str, size: int) -> Tuple[int, int]:
    """(max_turns, max_steps) for a square board of this size.

    Combat is a fixed fight: ``max(8, 3 * size)`` turns (9 on 3×3, 30 on 10×10).
    The reward is paid at that horizon, not when a ship dies.
    Scouting is a search: ``max(8, 4 * size)`` turns (12 on 3×3, 40 on 10×10),
    long enough for a speed-3 hull to walk a board it can only see adjacent to.
    """
    if skill == "builder":
        return max(10, size * 4), max(36, size * 12)
    if skill == "merchant":
        return max(8, size * 4), max(24, size * 12)
    if skill == "landing_build":
        # The landing drill's 3 turns per board side, plus a few for the build
        # (and for income to refill after a wrong hull).
        turns = max(10, int(size) * 3 + 3)
        return turns, max(24, turns * 8)
    if skill == "scouting":
        turns = max(8, int(size) * 4)
        return turns, max(32, turns * 8)
    if skill.startswith("combat_"):
        turns = max(8, int(size) * 3)
        return turns, max(24, turns * 8)
    if skill.startswith("build_"):
        # One income tick makes the hull affordable. Extra turns are the
        # wait the speed reward penalizes. ``builder`` (no underscore) is
        # the haul-then-buy drill and is not this branch.
        turns = max(6, int(size) * 2)
        return turns, max(12, turns * 2)
    return max(8, size * 3), max(16, size * 8)
