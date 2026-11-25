"""Fog Commander: a fog-limited scripted agent.

Implements the design in ``docs/gemini_carbon_scripted_agent_pseudocode.md``
(Gemini 4 Carbon), with the fixes listed under "Deviations" below.

Information contract
--------------------
The agent reads only what its seat may know:

* its own ships, ports, cash, stockpiles and the turn number;
* ``player.sightings`` / ``player.port_sightings`` (fresh and stale records);
* the static terrain mask (``engine.map.terrain``). Land never changes during
  a game (builders may only found ports on tiles that are already land, and
  map generators list every port tile as land), so the mask carries no hidden
  state. The play UI shows the human the same full terrain;
* the public rule tables (``BASE_STATS``, spot matrix, attack modifiers,
  ``engine.kill_curve_k``);
* ``engine.enumerate_legal(pid)``, reduced to fog-safe facts (which moves,
  attacks, captures, loads, unloads, port builds and (port, type) buys are
  legal). The engine's canonical ``spawn_position`` is ignored, because it is
  the free tile closest to the *true* enemy home (a fog leak). The agent picks
  its own spawn tile from what it can see (every tile next to an own port is
  always visible: all spot ranges are >= 1).

It never reads the opponent's ships, ports, cash, stockpiles, the engine RNG,
absolute cross-owner ship-id order, or the coastline token list from
``observation.py`` (that list omits *unsighted* enemy ports too, so its holes
leak where hidden ports are).

Every choice uses deterministic keys (positions, types, own ship ids); all
randomness comes from the ``rng`` the caller passes, so a game is
reproducible per seed.

Deviations from the pseudocode (see the PR / docs for reasons)
--------------------------------------------------------------
1. Water is ``terrain != LAND`` from the static mask. The pseudocode built
   water by flood-filling everything not in the observed coastline list, so
   unsighted ports (absent from that list) became "water". Unknown tiles are
   never assumed to be water here.
2. Spawn tiles are chosen by the agent (see above), not taken from the legal
   action.
3. The enemy-home search prior uses only static coastal land tiles that no
   own unit has yet covered at Port spot range. The "enemy half" and x-mirror
   terms are soft weights; the y-flip term was dropped (no generator
   guarantees it).
4. Paths use cached BFS distance fields over static water plus a one-step
   threat penalty, instead of a fresh threat-weighted Dijkstra per move
   (keeps decisions in the low-millisecond range on CPU).
5. Threat reach uses Manhattan ``speed + effective range`` rather than an
   exact water BFS of the enemy's move.
"""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from battleboats.core.actions import (
    Action,
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import MERCHANT_CAPACITY, PORT_PRODUCTION
from battleboats.core.map.Map import LAND
from battleboats.core.shipyard.ship_data import BASE_STATS, PORT, attack_modifier, spot_distance
from battleboats.core.shipyard.ship_type import ShipType

Pos = Tuple[int, int]
INF = float("inf")
_BIG = 10**9

CARRIER = ShipType.CARRIER
BATTLESHIP = ShipType.BATTLESHIP
CRUISER = ShipType.CRUISER
DESTROYER = ShipType.DESTROYER
SUBMARINE = ShipType.SUBMARINE
MERCHANT = ShipType.MERCHANT
LANDING = ShipType.LANDING
BUILDER = ShipType.BUILDER

ALL_TYPES: Tuple[ShipType, ...] = tuple(t for t in ShipType if t in BASE_STATS)
COMBAT_TYPES: FrozenSet[ShipType] = frozenset(
    t for t in ALL_TYPES if BASE_STATS[t].attack_range > 0 and BASE_STATS[t].strength > 0
)
NON_COMBAT_TYPES: FrozenSet[ShipType] = frozenset(ALL_TYPES) - COMBAT_TYPES
_TYPE_ORDER = {t: i for i, t in enumerate(ALL_TYPES)}


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def effective_range(att: ShipType, dfn: ShipType) -> int:
    """Shots need distance <= attack_range AND <= the attacker's spot range for the target."""
    if att not in COMBAT_TYPES:
        return 0
    return int(min(BASE_STATS[att].attack_range, spot_distance(att, dfn)))


def kill_probability(att: ShipType, dfn: ShipType, k: float = 2.0) -> float:
    if att not in COMBAT_TYPES:
        return 0.0
    sd = BASE_STATS[dfn].strength
    if sd <= 0:
        return 1.0
    x = BASE_STATS[att].strength * attack_modifier(att, dfn) / sd
    xk = x**k
    return xk / (1.0 + xk)


# ---------------------------------------------------------------- doctrines
@dataclass(frozen=True)
class Doctrine:
    name: str
    weights: Tuple[Tuple[ShipType, float], ...]
    aggression: float
    builder_mode: str  # ECO_HOME | FORWARD | ADAPTIVE
    target_merchants: int
    buy_temperature: float

    def weight(self, t: ShipType) -> float:
        return dict(self.weights)[t]


def _w(**kw) -> Tuple[Tuple[ShipType, float], ...]:
    return tuple((ShipType(k), float(v)) for k, v in kw.items())


DOCTRINES: Dict[str, Doctrine] = {
    d.name: d
    for d in (
        Doctrine("RAIDER_BLITZ", _w(Destroyer=1.5, Cruiser=1.4, Landing=1.3, Merchant=1.1, Submarine=1.0,
                                    Carrier=0.8, Battleship=0.8, Builder=0.6), 1.15, "FORWARD", 2, 0.45),
        Doctrine("ECONOMIC_BOOM", _w(Merchant=1.6, Builder=1.4, Carrier=1.3, Battleship=1.3, Landing=1.1,
                                     Destroyer=1.0, Cruiser=0.9, Submarine=0.9), 0.90, "ECO_HOME", 3, 0.45),
        Doctrine("WOLFPACK_STEALTH", _w(Submarine=1.6, Destroyer=1.4, Merchant=1.2, Landing=1.2, Cruiser=1.0,
                                        Carrier=0.9, Builder=0.8, Battleship=0.7), 1.05, "FORWARD", 2, 0.45),
        Doctrine("CAPITAL_ARMADA", _w(Carrier=1.5, Battleship=1.5, Merchant=1.3, Destroyer=1.2, Landing=1.1,
                                      Cruiser=0.9, Submarine=0.8, Builder=0.8), 0.95, "ECO_HOME", 3, 0.45),
        Doctrine("ISLAND_HOPPER", _w(Builder=1.4, Landing=1.4, Cruiser=1.3, Destroyer=1.3, Merchant=1.2,
                                     Submarine=0.9, Carrier=0.9, Battleship=0.9), 1.05, "FORWARD", 2, 0.50),
        Doctrine("BALANCED_ADAPTIVE", _w(Carrier=1.1, Battleship=1.1, Cruiser=1.1, Destroyer=1.1, Submarine=1.1,
                                         Landing=1.1, Merchant=1.1, Builder=1.0), 1.00, "ADAPTIVE", 2, 0.50),
    )
}
DOCTRINE_NAMES: Tuple[str, ...] = tuple(DOCTRINES)


def all_doctrine_pairs() -> Tuple[Tuple[str, str], ...]:
    """All 36 ordered pairs (including same-doctrine), seat 0 x seat 1."""
    return tuple((a, b) for a in DOCTRINE_NAMES for b in DOCTRINE_NAMES)


@dataclass
class FogCommanderConfig:
    max_steps_per_turn: int = 250
    stale_horizon: int = 8  # stale sightings older than this are ignored
    late_game_turn: int = 55  # all-in posture (pseudocode's absolute turn)
    landing_ramp_turn: int = 25
    checkmate_turn: int = 20
    third_landing_turn: int = 45
    use_map_prior: bool = True  # soft "enemy half" / x-mirror weights for the home search
    doctrine: Optional[str] = None  # force one doctrine (tests / ablations)


# ---------------------------------------------------------------- fogged view
@dataclass(frozen=True)
class OwnShip:
    id: int
    type: ShipType
    pos: Pos
    cargo: int
    has_attacked: bool
    bought: bool
    remaining: int


@dataclass(frozen=True)
class OwnPort:
    pos: Pos
    is_home: bool
    stockpile: int
    locked: bool


@dataclass(frozen=True)
class EnemyShip:
    id: int
    type: ShipType
    pos: Pos
    fresh: bool
    turn_seen: int


@dataclass(frozen=True)
class EnemyPort:
    pos: Pos
    is_home: bool
    fresh: bool
    turn_seen: int


@dataclass
class LegalIndex:
    """Fog-safe summary of ``enumerate_legal``; canonical spawn tiles dropped."""

    moves: Dict[int, Tuple[Pos, ...]]
    attacks: Set[Tuple[int, int]]
    captures: Tuple[Tuple[int, Pos], ...]
    unloads: Tuple[Tuple[int, Pos], ...]
    loads: Tuple[Tuple[int, Pos], ...]
    port_builds: Tuple[Tuple[int, Pos], ...]
    builds: Dict[Pos, FrozenSet[ShipType]]

    @classmethod
    def from_actions(cls, actions: Sequence[Action]) -> "LegalIndex":
        moves: Dict[int, List[Pos]] = {}
        attacks: Set[Tuple[int, int]] = set()
        caps, unl, lds, pbs = [], [], [], []
        builds: Dict[Pos, Set[ShipType]] = {}
        for a in actions:
            if isinstance(a, MoveAction):
                moves.setdefault(a.ship_id, []).append(a.destination)
            elif isinstance(a, AttackAction):
                attacks.add((a.attacker_id, a.target_id))
            elif isinstance(a, CapturePortAction):
                caps.append((a.landing_ship_id, a.target))
            elif isinstance(a, MerchantUnloadAction):
                unl.append((a.merchant_id, a.port))
            elif isinstance(a, MerchantLoadAction):
                lds.append((a.merchant_id, a.port))
            elif isinstance(a, BuildPortAction):
                pbs.append((a.builder_ship_id, a.target))
            elif isinstance(a, BuildShipAction):
                builds.setdefault(a.port, set()).add(a.ship_type)
        return cls(
            moves={k: tuple(sorted(v)) for k, v in moves.items()},
            attacks=attacks,
            captures=tuple(sorted(caps)),
            unloads=tuple(sorted(unl)),
            loads=tuple(sorted(lds)),
            port_builds=tuple(sorted(pbs)),
            builds={k: frozenset(v) for k, v in builds.items()},
        )


@dataclass
class FogView:
    pid: int
    width: int
    height: int
    turn: int
    cash: int
    kill_k: float
    ships: Tuple[OwnShip, ...]
    ports: Tuple[OwnPort, ...]
    home: OwnPort
    enemy_ships: Tuple[EnemyShip, ...]
    enemy_ports: Tuple[EnemyPort, ...]
    legal: LegalIndex


def build_fog_view(engine, pid: int, legal: Optional[Sequence[Action]] = None) -> FogView:
    """The only place the agent reads the engine (besides the static terrain mask)."""
    player = engine.players[pid]
    if legal is None:
        legal = engine.enumerate_legal(pid)
    ships = []
    for sid in sorted(player.owned_ship_ids):
        s = engine.ships[sid]
        rem = 0 if s.bought_this_turn else max(0, s.stats.speed - s.tiles_moved_this_turn)
        ships.append(OwnShip(sid, s.type, tuple(s.position), int(s.cargo), bool(s.has_attacked),
                             bool(s.bought_this_turn), rem))
    ports = []
    for pos in sorted(player.owned_port_positions):
        p = engine.ports[pos]
        ports.append(OwnPort(tuple(pos), bool(p.is_home), int(p.stockpile), bool(p.locked_this_turn)))
    home = next((p for p in ports if p.is_home), None)
    if home is None:  # cannot happen in a live game (losing home ends it); keep a safe fallback
        home = OwnPort(tuple(player.home_port), True, 0, True)
    enemy_ships = tuple(
        EnemyShip(s.ship_id, s.type, tuple(s.position), bool(s.fresh), int(s.turn_seen))
        for _, s in sorted(player.sightings.items())
    )
    own_pos = set(player.owned_port_positions)
    enemy_ports = tuple(
        EnemyPort(tuple(pos), bool(ps.is_home), bool(ps.fresh), int(ps.turn_seen))
        for pos, ps in sorted(player.port_sightings.items())
        if pos not in own_pos
    )
    return FogView(
        pid=pid,
        width=engine.map.width,
        height=engine.map.height,
        turn=int(engine.turn),
        cash=int(player.cash),
        kill_k=float(getattr(engine, "kill_curve_k", 2.0)),
        ships=tuple(ships),
        ports=tuple(ports),
        home=home,
        enemy_ships=enemy_ships,
        enemy_ports=enemy_ports,
        legal=LegalIndex.from_actions(legal),
    )


# ---------------------------------------------------------------- seat memory
@dataclass
class SeatMemory:
    pid: int
    width: int
    height: int
    land: FrozenSet[Pos]
    water: FrozenSet[Pos]
    coast: Tuple[Pos, ...]  # land tiles with >= 1 water neighbour (static)
    doctrine: Doctrine
    jitter: Dict[ShipType, float]
    planned_buy: Optional[ShipType] = None
    built_counts: Counter = field(default_factory=Counter)
    ports_built: int = 0
    builder_assignments: Dict[int, Pos] = field(default_factory=dict)
    pruned: Dict[int, int] = field(default_factory=dict)  # ship_id -> turn_seen proven gone
    checked_coast: Set[Pos] = field(default_factory=set)  # coast tiles ever covered at Port range
    turn_key: Optional[int] = None
    steps_in_turn: int = 0
    visited: Dict[int, Set[Pos]] = field(default_factory=dict)
    sidestepped: Set[int] = field(default_factory=set)
    field_cache: "OrderedDict[FrozenSet[Pos], List[int]]" = field(default_factory=OrderedDict)

    def __deepcopy__(self, memo):
        import copy

        new = SeatMemory.__new__(SeatMemory)
        for k, v in self.__dict__.items():
            if k in ("land", "water", "coast", "doctrine", "field_cache"):
                setattr(new, k, v if k != "field_cache" else OrderedDict(v))
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new


def _neighbors(p: Pos, w: int, h: int) -> Iterable[Pos]:
    x, y = p
    if x > 0:
        yield (x - 1, y)
    if x + 1 < w:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y + 1 < h:
        yield (x, y + 1)


def init_memory(engine, pid: int, rng: random.Random, config: FogCommanderConfig) -> SeatMemory:
    w, h = engine.map.width, engine.map.height
    terrain = engine.map.terrain
    land = frozenset((x, y) for x in range(w) for y in range(h) if int(terrain[x, y]) == LAND)
    water = frozenset((x, y) for x in range(w) for y in range(h) if (x, y) not in land)
    coast = tuple(sorted(p for p in land if any(n in water for n in _neighbors(p, w, h))))
    if config.doctrine is not None:
        doctrine = DOCTRINES[config.doctrine]
    else:
        doctrine = DOCTRINES[DOCTRINE_NAMES[rng.randrange(len(DOCTRINE_NAMES))]]
    jitter = {t: rng.uniform(0.85, 1.15) for t in ALL_TYPES}
    return SeatMemory(pid=pid, width=w, height=h, land=land, water=water, coast=coast,
                      doctrine=doctrine, jitter=jitter)


# ---------------------------------------------------------------- the agent
class FogCommander:
    """Stateful per-seat agent. ``act(engine, pid, rng)`` returns one action."""

    def __init__(self, config: Optional[FogCommanderConfig] = None) -> None:
        self.config = config or FogCommanderConfig()
        self.memory: Dict[int, SeatMemory] = {}

    def doctrine_for(self, pid: int) -> Optional[str]:
        m = self.memory.get(pid)
        return m.doctrine.name if m else None

    def act(self, engine, pid: int, rng: random.Random) -> Action:
        legal = engine.enumerate_legal(pid)
        if not legal:
            raise RuntimeError(f"fog_commander: no legal actions for player {pid}")
        mem = self.memory.get(pid)
        if mem is None:
            mem = init_memory(engine, pid, rng, self.config)
            self.memory[pid] = mem
        view = build_fog_view(engine, pid, legal)
        return decide(view, mem, rng, self.config)


def decide(view: FogView, mem: SeatMemory, rng: random.Random, config: FogCommanderConfig) -> Action:
    if mem.turn_key != view.turn:
        mem.turn_key = view.turn
        mem.steps_in_turn = 0
        mem.visited = {}
        mem.sidestepped = set()
    mem.steps_in_turn += 1
    for s in view.ships:
        mem.visited.setdefault(s.id, set()).add(s.pos)
    if mem.steps_in_turn > config.max_steps_per_turn:
        return EndTurnAction()
    d = _Decision(view, mem, rng, config)
    act = d.run()
    return act if act is not None else EndTurnAction()


class _Decision:
    """All work for one atomic decision. Caches live only for this call."""

    def __init__(self, view: FogView, mem: SeatMemory, rng: random.Random, config: FogCommanderConfig):
        self.v = view
        self.m = mem
        self.rng = rng
        self.cfg = config
        self.W, self.H = view.width, view.height
        self.k = view.kill_k
        self._threat: Dict[Tuple[Pos, ShipType], float] = {}
        self.own_pos = {s.pos for s in view.ships}
        self.own_port_set = {p.pos for p in view.ports}
        self._build_world()

    # ------------------------------------------------------------ world model
    def _build_world(self) -> None:
        v, m = self.v, self.m
        fresh, stale = [], []
        for e in v.enemy_ships:
            if e.fresh:
                fresh.append(e)
                continue
            if m.pruned.get(e.id) == e.turn_seen:
                continue
            if self._covered(e.pos, e.type):
                m.pruned[e.id] = e.turn_seen  # we look at that tile now: it is gone
                continue
            if v.turn - e.turn_seen <= self.cfg.stale_horizon:
                stale.append(e)
        self.fresh = fresh
        self.stale = stale
        self.fresh_by_pos = {e.pos: e for e in fresh}
        self.occupied = self.own_pos | set(self.fresh_by_pos)
        self.enemy_home = next((p for p in v.enemy_ports if p.is_home), None)
        self.satellites = [p for p in v.enemy_ports if not p.is_home]
        known_ports = self.own_port_set | {p.pos for p in v.enemy_ports}
        self.known_ports = known_ports
        # Home-search prior over static coast tiles never covered at Port range.
        self.prior: Dict[Pos, float] = {}
        if self.enemy_home is None:
            hx = v.home.pos[0]
            for c in m.coast:
                if c in m.checked_coast or c in known_ports:
                    continue
                if self._covered(c, PORT):
                    m.checked_coast.add(c)
                    continue
                wgt = 1.0
                if self.cfg.use_map_prior:
                    same_side = (c[0] < self.W / 2.0) == (hx < self.W / 2.0)
                    wgt *= 0.3 if same_side else 3.0
                    if manhattan(c, v.home.pos) < self.W // 2:
                        wgt *= 0.5
                    wgt *= 1.0 + 1.0 / (1.0 + abs(c[0] - (self.W - 1 - hx)))
                self.prior[c] = wgt
        self.home_threat = any(
            e.type == LANDING and manhattan(e.pos, v.home.pos) <= 5 for e in fresh
        )
        self.reserved_eh_adj: Set[Pos] = set()
        if self.enemy_home is not None:
            eh = self.enemy_home.pos
            if any(s.type == LANDING and manhattan(s.pos, eh) <= 6 for s in v.ships):
                self.reserved_eh_adj = set(self.water_adj(eh))

    def _covered(self, target: Pos, ttype) -> bool:
        for s in self.v.ships:
            if manhattan(s.pos, target) <= spot_distance(s.type, ttype):
                return True
        for p in self.v.ports:
            if manhattan(p.pos, target) <= spot_distance(PORT, ttype):
                return True
        return False

    def water_adj(self, tile: Pos) -> List[Pos]:
        return [n for n in _neighbors(tile, self.W, self.H) if n in self.m.water]

    def ring(self, center: Pos, rmax: int, rmin: int = 0) -> Set[Pos]:
        out = set()
        cx, cy = center
        water = self.m.water
        for x in range(max(0, cx - rmax), min(self.W, cx + rmax + 1)):
            rem = rmax - abs(x - cx)
            for y in range(max(0, cy - rem), min(self.H, cy + rem + 1)):
                if (x, y) in water and abs(x - cx) + abs(y - cy) >= rmin:
                    out.add((x, y))
        return out

    def threat(self, tile: Pos, hull: ShipType) -> float:
        key = (tile, hull)
        t = self._threat.get(key)
        if t is not None:
            return t
        surv = 1.0
        for e in self.fresh:
            if e.type not in COMBAT_TYPES:
                continue
            er = effective_range(e.type, hull)
            if er <= 0:
                continue
            if manhattan(e.pos, tile) <= BASE_STATS[e.type].speed + er:
                surv *= 1.0 - kill_probability(e.type, hull, self.k)
        for e in self.stale:
            if e.type not in COMBAT_TYPES:
                continue
            age = max(1, self.v.turn - e.turn_seen)
            if age > 3 or effective_range(e.type, hull) <= 0:
                continue
            if manhattan(e.pos, tile) <= BASE_STATS[e.type].speed + 2:
                surv *= 1.0 - (0.35**age) * kill_probability(e.type, hull, self.k)
        t = 1.0 - surv
        self._threat[key] = t
        return t

    def sub_danger(self, tile: Pos) -> bool:
        for e in self.fresh:
            if e.type == DESTROYER and manhattan(e.pos, tile) <= BASE_STATS[DESTROYER].speed + 3:
                return True
            if e.type == CARRIER and manhattan(e.pos, tile) <= 3:
                return True
        for p in self.v.enemy_ports:
            if manhattan(p.pos, tile) <= 1:
                return True
        return False

    # ------------------------------------------------------------ pathing
    def dist_field(self, goals: FrozenSet[Pos]) -> List[int]:
        cache = self.m.field_cache
        f = cache.get(goals)
        if f is not None:
            cache.move_to_end(goals)
            return f
        W, H = self.W, self.H
        water = self.m.water
        f = [_BIG] * (W * H)
        q = deque()
        for g in sorted(goals):
            if g in water and f[g[0] * H + g[1]] == _BIG:
                f[g[0] * H + g[1]] = 0
                q.append(g)
        while q:
            cur = q.popleft()
            d = f[cur[0] * H + cur[1]] + 1
            for n in _neighbors(cur, W, H):
                i = n[0] * H + n[1]
                if f[i] == _BIG and n in water:
                    f[i] = d
                    q.append(n)
        cache[goals] = f
        if len(cache) > 512:
            cache.popitem(last=False)
        return f

    def reach(self, ship: OwnShip, steps: Optional[int] = None) -> Dict[Pos, int]:
        """Exact water steps within the ship's remaining moves, around visible blockers."""
        steps = ship.remaining if steps is None else steps
        out = {ship.pos: 0}
        q = deque([ship.pos])
        while q:
            cur = q.popleft()
            d = out[cur]
            if d >= steps:
                continue
            for n in _neighbors(cur, self.W, self.H):
                if n in out or n not in self.m.water or n in self.occupied:
                    continue
                out[n] = d + 1
                q.append(n)
        return out

    def move_toward(self, ship: OwnShip, goals: Set[Pos], risk: float, *, extra_blocked: Set[Pos] = frozenset(),
                    penalty: Optional[Dict[Pos, float]] = None, strict: bool = True) -> Optional[MoveAction]:
        if not goals or ship.remaining <= 0:
            return None
        if strict and ship.pos in goals:
            return None
        visited = self.m.visited.get(ship.id, set())
        dests = [d for d in self.v.legal.moves.get(ship.id, ()) if d not in visited and d not in extra_blocked]
        if not dests:
            return None
        f = self.dist_field(frozenset(goals))
        H = self.H
        cur_d = f[ship.pos[0] * H + ship.pos[1]]
        pen = penalty or {}
        cur_cost = cur_d + risk * self.threat(ship.pos, ship.type) + pen.get(ship.pos, 0.0)
        scored = []
        for d in dests:
            fd = f[d[0] * H + d[1]]
            if fd >= _BIG:
                continue
            if strict and fd >= cur_d:
                continue
            cost = fd + risk * self.threat(d, ship.type) + pen.get(d, 0.0)
            if strict and cur_d < _BIG and cost >= cur_cost:
                continue
            scored.append((cost, d))
        if not scored:
            return None
        best = min(c for c, _ in scored)
        ties = sorted(d for c, d in scored if c <= best + 0.15)
        return MoveAction(ship.id, ties[self.rng.randrange(len(ties))] if len(ties) > 1 else ties[0])

    # ------------------------------------------------------------ top level
    def run(self) -> Optional[Action]:
        for phase in (
            self.capture,
            self.merchant_transfer,
            self.cash_delivery,
            self.combat,
            self.post_attack,
            self.unclog,
            self.builder,
            self.merchant_move,
            self.landing_move,
            self.fleet_move,
            self.purchase,
        ):
            a = phase()
            if a is not None:
                return a
        return None

    # Phase 1
    def capture(self) -> Optional[Action]:
        caps = self.v.legal.captures
        if not caps:
            return None
        if self.enemy_home is not None:
            for sid, tgt in caps:
                if tgt == self.enemy_home.pos:
                    return CapturePortAction(sid, tgt)
        sid, tgt = caps[0]
        return CapturePortAction(sid, tgt)

    # Phase 2
    def merchant_transfer(self) -> Optional[Action]:
        L = self.v.legal
        if L.unloads:
            mid, port = L.unloads[0]
            return MerchantUnloadAction(mid, port)
        if L.loads:
            stock = {p.pos: p.stockpile for p in self.v.ports}
            mid, port = sorted(L.loads, key=lambda a: (-stock.get(a[1], 0), a))[0]
            return MerchantLoadAction(mid, port)
        return None

    # Phase 3
    def cash_delivery(self) -> Optional[Action]:
        home_adj = set(self.water_adj(self.v.home.pos))
        for s in self.v.ships:
            if s.type != MERCHANT or s.cargo <= 0 or s.remaining <= 0 or s.pos in home_adj:
                continue
            r = self.reach(s)
            if any(g in r for g in home_adj):
                mv = self.move_toward(s, home_adj, 8.0)
                if mv is not None:
                    return mv
        return None

    # Phase 4
    def target_value(self, e: EnemyShip) -> float:
        val = float(BASE_STATS[e.type].cost)
        if e.type == LANDING:
            val = 550.0
            dh = manhattan(e.pos, self.v.home.pos)
            if dh <= 9:
                val = 2200.0 + 250.0 * (10 - dh)
        elif e.type == BUILDER:
            val = 420.0
        elif e.type == MERCHANT:
            val = 280.0
        elif e.type == SUBMARINE:
            val += 130.0
        if self.enemy_home is not None and manhattan(e.pos, self.enemy_home.pos) <= 5:
            val *= 1.35
        if any(s.type == LANDING and manhattan(e.pos, s.pos) <= 6 for s in self.v.ships):
            val *= 1.45
        return val

    def combat(self) -> Optional[Action]:
        if not self.fresh:
            return None
        aggr = self.m.doctrine.aggression
        legal_att = self.v.legal.attacks
        cands = []
        for s in self.v.ships:
            if s.type not in COMBAT_TYPES or s.has_attacked or s.bought:
                continue
            r = self.reach(s)
            my_cost = float(BASE_STATS[s.type].cost)
            my_str = float(BASE_STATS[s.type].strength)
            for e in self.fresh:
                er = effective_range(s.type, e.type)
                if er <= 0:
                    continue
                pk = kill_probability(s.type, e.type, self.k)
                gain = pk * self.target_value(e)
                overkill = 0.30 * my_str if BASE_STATS[e.type].strength <= 0 else 0.0
                best, best_tile = -INF, None
                for tile in sorted(r):
                    steps = r[tile]
                    dist = manhattan(tile, e.pos)
                    if dist > er:
                        continue
                    if steps == 0 and (s.id, e.id) not in legal_att:
                        continue
                    adj_risk = self.threat(tile, s.type) * (1.0 - 0.65 * pk)
                    kite = 0.65 if s.remaining - steps >= 2 else 1.0
                    exposure = my_cost * adj_risk * kite
                    if s.type == SUBMARINE and self.sub_danger(tile):
                        exposure += 75.0
                    net = gain * aggr - exposure - overkill + 2.5 * dist - 2.0 * steps
                    if net > best:
                        best, best_tile = net, (tile, steps)
                emergency = e.type == LANDING and manhattan(e.pos, self.v.home.pos) <= 8
                if best_tile is not None and (best > -15.0 or emergency):
                    cands.append((best, best_tile[1], s.id, e.id, best_tile[0], s))
        if not cands:
            return None
        immediate = [c for c in cands if c[1] == 0]
        if immediate:
            immediate.sort(key=lambda c: (-c[0], c[2], c[4]))
            return AttackAction(immediate[0][2], immediate[0][3])
        cands.sort(key=lambda c: (-c[0], c[2], c[4]))
        for c in cands:
            mv = self.move_toward(c[5], {c[4]}, 1.5)
            if mv is not None:
                return mv
        return None

    # Phase 5
    def post_attack(self) -> Optional[Action]:
        v = self.v
        home_adj = set(self.water_adj(v.home.pos))
        if self.home_threat:
            open_adj = home_adj - self.occupied
            if open_adj:
                blockers = [
                    s for s in v.ships
                    if s.remaining > 0 and s.pos not in home_adj
                    and (s.has_attacked or s.type in COMBAT_TYPES or (s.type == MERCHANT and s.cargo == 0))
                ]
                blockers.sort(key=lambda s: (manhattan(s.pos, v.home.pos), s.id))
                for s in blockers:
                    mv = self.move_toward(s, open_adj, 0.5)
                    if mv is not None:
                        return mv
        if self.reserved_eh_adj:
            eh = self.enemy_home.pos
            r23 = self.ring(eh, 3, 2)
            for s in v.ships:
                if s.type != LANDING and s.pos in self.reserved_eh_adj and s.remaining > 0:
                    mv = self.move_toward(s, r23, 2.0)
                    if mv is not None:
                        return mv
        hold = home_adj if self.home_threat else set()
        for s in v.ships:
            if s.type not in COMBAT_TYPES or not s.has_attacked or s.remaining <= 0 or s.pos in hold:
                continue
            sub = s.type == SUBMARINE
            cur = self.threat(s.pos, s.type) + (0.4 if sub and self.sub_danger(s.pos) else 0.0)
            if cur <= 0.05:
                continue
            visited = self.m.visited.get(s.id, set())
            best, best_d = -cur, None
            for d in v.legal.moves.get(s.id, ()):
                if d in visited or d in self.reserved_eh_adj:
                    continue
                nt = self.threat(d, s.type) + (0.4 if sub and self.sub_danger(d) else 0.0)
                if -nt > best + 0.02:
                    best, best_d = -nt, d
            if best_d is not None:
                return MoveAction(s.id, best_d)
        return None

    # Phase 6
    def unclog(self) -> Optional[Action]:
        v = self.v
        if v.cash < 100 or self.home_threat:
            return None
        ship_at = {s.pos: s for s in v.ships}
        for p in v.ports:
            if p.locked:
                continue
            adj = self.water_adj(p.pos)
            if any(t not in self.occupied for t in adj):
                continue
            for t in adj:
                b = ship_at.get(t)
                if b is None or b.remaining <= 0 or b.id in self.m.sidestepped:
                    continue
                outs = [d for d in v.legal.moves.get(b.id, ()) if d not in adj and self.threat(d, b.type) <= 0.35]
                if outs:
                    self.m.sidestepped.add(b.id)
                    return MoveAction(b.id, outs[self.rng.randrange(len(outs))])
        return None

    # Phase 7
    def coast_score(self, c: Pos, mode: str) -> float:
        v = self.v
        adj = self.water_adj(c)
        if not adj or c in self.known_ports:
            return -INF
        if min((manhattan(c, pp) for pp in self.known_ports), default=99) <= 2:
            return -INF
        local = max(self.threat(w, BUILDER) for w in adj)
        if local > 0.25:
            return -INF
        openness = 12.0 * min(3, len(adj))
        dh = manhattan(c, v.home.pos)
        if mode == "ECO_HOME" or (mode == "ADAPTIVE" and len(v.ports) <= 3):
            ds = 45.0 - 5.0 * abs(dh - 4.5)
        else:
            ds = 40.0 - 2.2 * abs(c[0] - self.W / 2.0)
            if self.enemy_home is not None:
                ds += 20.0 - 2.0 * abs(manhattan(c, self.enemy_home.pos) - 11.0)
        return openness + ds - 40.0 * local

    def builder(self) -> Optional[Action]:
        v, m = self.v, self.m
        builders = [s for s in v.ships if s.type == BUILDER]
        if not builders:
            return None
        mode = m.doctrine.builder_mode
        if v.legal.port_builds:
            scored = sorted(
                ((self.coast_score(t, mode), bid, t) for bid, t in v.legal.port_builds),
                key=lambda x: (-x[0], x[1], x[2]),
            )
            sc, bid, tgt = scored[0]
            if sc > 5.0 or m.builder_assignments.get(bid) == tgt:
                m.ports_built += 1
                m.builder_assignments.pop(bid, None)
                return BuildPortAction(bid, tgt)
        live = {b.id for b in builders}
        for bid in [k for k in m.builder_assignments if k not in live]:
            del m.builder_assignments[bid]
        claimed = set(m.builder_assignments.values())
        for b in builders:
            if b.remaining <= 0:
                continue
            tgt = m.builder_assignments.get(b.id)
            if tgt is None or self.coast_score(tgt, mode) <= -100.0:
                claimed.discard(tgt)
                best, best_c = -INF, None
                for c in m.coast:
                    if c in claimed:
                        continue
                    sc = self.coast_score(c, mode)
                    if sc == -INF:
                        continue
                    sc -= 1.2 * manhattan(b.pos, c)
                    if sc > best:
                        best, best_c = sc, c
                if best_c is None:
                    m.builder_assignments.pop(b.id, None)
                    continue
                m.builder_assignments[b.id] = best_c
                claimed.add(best_c)
                tgt = best_c
            mv = self.move_toward(b, set(self.water_adj(tgt)), 9.0)
            if mv is not None:
                return mv
        return None

    def any_viable_coast(self) -> bool:
        mode = self.m.doctrine.builder_mode
        return any(self.coast_score(c, mode) > -INF for c in self.m.coast)

    # Phase 8a
    def merchant_move(self) -> Optional[Action]:
        v = self.v
        merchants = [s for s in v.ships if s.type == MERCHANT and s.remaining > 0]
        if not merchants:
            return None
        home_adj = set(self.water_adj(v.home.pos))
        f_home = self.dist_field(frozenset(home_adj))
        H = self.H
        sats = [p for p in v.ports if not p.is_home]
        stock = {p.pos: p.stockpile for p in sats}
        merchants.sort(key=lambda s: (-s.cargo, s.id))
        for mch in merchants:
            if self.threat(mch.pos, MERCHANT) > 0.35 and mch.pos not in home_adj:
                mv = self.move_toward(mch, home_adj, 12.0)
                if mv is not None:
                    return mv
            best_rate, best_goals, best_sat = -INF, None, None
            if mch.cargo > 0:
                dh = f_home[mch.pos[0] * H + mch.pos[1]]
                dh = 30 if dh >= _BIG else dh
                turns = max(0.5, math.ceil(dh / 3.0))
                urgency = 1.35 if (v.cash < 250 <= v.cash + 2 * mch.cargo) else 1.0
                best_rate = urgency * 2.0 * mch.cargo / turns
                best_goals = home_adj
            if mch.cargo < MERCHANT_CAPACITY:
                for sat in sats:
                    adj = self.water_adj(sat.pos)
                    if not adj:
                        continue
                    if min(self.threat(w, MERCHANT) for w in adj) > 0.45:
                        continue
                    t_to = math.ceil(min(manhattan(mch.pos, w) for w in adj) / 3.0)
                    dsh = min(f_home[w[0] * H + w[1]] for w in adj)
                    dsh = 30 if dsh >= _BIG else dsh
                    t_back = max(1.0, math.ceil(dsh / 3.0))
                    proj = stock[sat.pos] + PORT_PRODUCTION * t_to
                    gain = min(MERCHANT_CAPACITY - mch.cargo, proj)
                    if gain <= 0:
                        continue
                    rate = 2.0 * (mch.cargo + gain) / max(1.0, t_to + t_back)
                    if rate > best_rate:
                        best_rate, best_goals = rate, set(adj)
                        best_sat = (sat.pos, min(MERCHANT_CAPACITY - mch.cargo, stock[sat.pos]))
            if best_sat is not None:
                stock[best_sat[0]] = max(0, stock[best_sat[0]] - best_sat[1])
            if best_goals is not None and mch.pos not in best_goals:
                mv = self.move_toward(mch, best_goals, 9.0)
                if mv is not None:
                    return mv
        return None

    # Phase 8b
    def landing_move(self) -> Optional[Action]:
        v = self.v
        landings = [s for s in v.ships if s.type == LANDING and s.remaining > 0]
        if not landings:
            return None
        combat = [s for s in v.ships if s.type in COMBAT_TYPES]
        late = v.turn >= self.cfg.late_game_turn
        if self.enemy_home is not None:
            eh = self.enemy_home.pos
            eh_adj = set(self.water_adj(eh))
            landings.sort(key=lambda s: (manhattan(s.pos, eh), s.id))
            for idx, lnd in enumerate(landings):
                d_eh = manhattan(lnd.pos, eh)
                r = self.reach(lnd)
                if any(g in r for g in eh_adj):
                    mv = self.move_toward(lnd, eh_adj, 0.0)
                    if mv is not None:
                        return mv
                goals = eh_adj
                if self.satellites:
                    if idx >= 1:
                        near = min(self.satellites, key=lambda p: (manhattan(lnd.pos, p.pos), p.pos))
                        if manhattan(lnd.pos, near.pos) < d_eh and self.water_adj(near.pos):
                            goals = set(self.water_adj(near.pos))
                    else:
                        for sat in sorted(self.satellites, key=lambda p: p.pos):
                            if manhattan(lnd.pos, sat.pos) + manhattan(sat.pos, eh) <= d_eh + 2:
                                adj = self.water_adj(sat.pos)
                                if adj:
                                    goals = set(adj)
                                    break
                escort_close = any(manhattan(c.pos, lnd.pos) <= 4 for c in combat)
                escort_catching = any(4 < manhattan(c.pos, lnd.pos) <= 8 for c in combat)
                defenders = any(manhattan(e.pos, eh) <= 6 for e in self.fresh)
                known_clear = self.enemy_home.fresh and not defenders and d_eh <= 6
                if (7 <= d_eh <= 10 and not escort_close and escort_catching and not known_clear and not late
                        and self.threat(lnd.pos, LANDING) <= 0.10):
                    continue  # wave sync: wait for escorts outside the home's radar
                risk = 2.5 if (escort_close or late) else 7.0
                mv = self.move_toward(lnd, goals, risk)
                if mv is not None:
                    return mv
        else:
            for lnd in landings:
                escort_close = any(manhattan(c.pos, lnd.pos) <= 4 for c in combat)
                if self.satellites and escort_close:
                    sat = min(self.satellites, key=lambda p: (manhattan(lnd.pos, p.pos), p.pos))
                    goals = set(self.water_adj(sat.pos))
                else:
                    mid = self.W // 2
                    goals = {t for t in self.m.water if abs(t[0] - mid) <= 3}
                    safe = {t for t in goals if self.threat(t, LANDING) <= 0.10}
                    goals = safe or goals
                mv = self.move_toward(lnd, goals, 8.0)
                if mv is not None:
                    return mv
        return None

    # Phase 8c
    def scout_goals(self, s: OwnShip) -> Set[Pos]:
        v = self.v
        if self.enemy_home is None and self.prior:
            rp = int(spot_distance(s.type, PORT))
            prior = sorted(self.prior.items())
            scored = []
            for wt in self.m.water:
                if (wt[0] + wt[1]) % 2:
                    continue
                gain = 0.0
                for c, pw in prior:
                    if abs(wt[0] - c[0]) + abs(wt[1] - c[1]) <= rp:
                        gain += pw
                if gain <= 0.0:
                    continue
                if self.threat(wt, s.type) > 0.45:
                    continue
                scored.append((gain - 0.8 * manhattan(s.pos, wt), wt))
            if scored:
                best = max(val for val, _ in scored)
                return {p for val, p in scored if val >= best - 0.2 * abs(best)}
        if self.enemy_home is None:
            hx, hy = v.home.pos
            return self.ring((self.W - 1 - hx, hy), 6)
        eh = self.enemy_home.pos
        if not self.enemy_home.fresh or v.turn - self.enemy_home.turn_seen >= 3:
            return self.ring(eh, 7, 4)
        if self.satellites:
            sat = max(self.satellites, key=lambda p: (v.turn - p.turn_seen, p.pos))
            return self.ring(sat.pos, 5)
        return set()

    def forward_port(self) -> OwnPort:
        v = self.v
        if self.enemy_home is not None:
            ref = self.enemy_home.pos
        else:
            ref = (self.W - 1 - v.home.pos[0], v.home.pos[1])
        return min(v.ports, key=lambda p: (manhattan(p.pos, ref), p.pos))

    def fleet_move(self) -> Optional[Action]:
        v = self.v
        movable = [s for s in v.ships if s.type in COMBAT_TYPES and s.remaining > 0 and not s.has_attacked]
        if not movable:
            return None
        reserved = self.reserved_eh_adj
        assigned: Set[int] = set()
        threats = [
            e for e in self.fresh
            if manhattan(e.pos, v.home.pos) <= 10
            or any(not p.is_home and manhattan(e.pos, p.pos) <= 5 for p in v.ports)
        ]
        if threats:
            threats.sort(key=lambda e: (0 if e.type == LANDING else 1, manhattan(e.pos, v.home.pos), e.pos))
            te = threats[0]
            ics = sorted(movable, key=lambda s: (-kill_probability(s.type, te.type, self.k),
                                                 manhattan(s.pos, te.pos), s.id))
            for ic in ics[:2]:
                er = max(1, effective_range(ic.type, te.type))
                mv = self.move_toward(ic, self.ring(te.pos, er), 1.0)
                if mv is not None:
                    return mv
                assigned.add(ic.id)
        need_scout = self.enemy_home is None or (
            not self.enemy_home.fresh and v.turn - self.enemy_home.turn_seen >= 4
        )
        if need_scout:
            pool = [s for s in movable if s.id not in assigned]
            if pool:
                pool.sort(key=lambda s: (-spot_distance(s.type, PORT), -BASE_STATS[s.type].speed, s.id))
                sc = pool[0]
                goals = self.scout_goals(sc) - reserved
                if goals:
                    mv = self.move_toward(sc, goals, 3.0, extra_blocked=reserved)
                    if mv is not None:
                        return mv
                assigned.add(sc.id)
        own_power = sum(BASE_STATS[s.type].strength for s in v.ships if s.type in COMBAT_TYPES)
        seen_power = sum(BASE_STATS[e.type].strength for e in self.fresh if e.type in COMBAT_TYPES)
        advantage = own_power >= 0.9 * seen_power / self.m.doctrine.aggression or v.turn >= self.cfg.late_game_turn
        landings = [s for s in v.ships if s.type == LANDING]
        escorts = 0
        for s in movable:
            if s.id in assigned:
                continue
            if s.type == SUBMARINE:
                prey = [e for e in self.fresh + self.stale
                        if e.type in (BATTLESHIP, CARRIER, CRUISER, MERCHANT, LANDING)]
                if prey:
                    pr = min(prey, key=lambda e: (manhattan(s.pos, e.pos), e.pos))
                    er = max(1, effective_range(SUBMARINE, pr.type))
                    rmin = 2 if er >= 2 and pr.type in (BATTLESHIP, CRUISER, CARRIER) else 1
                    goals = self.ring(pr.pos, er, rmin) - reserved
                    pen = {t: 6.0 for t in goals | set(v.legal.moves.get(s.id, ())) if self.sub_danger(t)}
                    mv = self.move_toward(s, goals, 2.0, extra_blocked=reserved, penalty=pen)
                    if mv is not None:
                        return mv
            if landings and escorts < 2 * len(landings):
                vip = min(landings, key=lambda l: (manhattan(s.pos, l.pos), l.id))
                escorts += 1
                if self.enemy_home is not None:
                    eh = self.enemy_home.pos
                    ahead = {t for t in self.ring(vip.pos, 3)
                             if manhattan(t, eh) <= manhattan(vip.pos, eh) and manhattan(t, eh) >= 2} - reserved
                else:
                    ahead = self.ring(vip.pos, 2)
                if ahead:
                    mv = self.move_toward(s, ahead, 1.8, extra_blocked=reserved)
                    if mv is not None:
                        return mv
            if advantage:
                targets = self.fresh if self.fresh else self.stale
                if targets:
                    bt = max(targets, key=lambda e: (
                        kill_probability(s.type, e.type, self.k) * self.target_value(e) - 6.0 * manhattan(s.pos, e.pos),
                        tuple(-c for c in e.pos)))
                    er = max(1, effective_range(s.type, bt.type))
                    rmin = 5 if (s.type == CARRIER and er >= 6) else 1
                    goals = self.ring(bt.pos, er, rmin) - reserved
                elif self.enemy_home is not None:
                    goals = self.ring(self.enemy_home.pos, 4, 2) - reserved
                else:
                    goals = self.scout_goals(s)
            else:
                goals = self.ring(self.forward_port().pos, 3)
            if goals:
                mv = self.move_toward(s, goals, 2.2, extra_blocked=reserved)
                if mv is not None:
                    return mv
        return None

    # Phase 9
    def hull_scores(self) -> Dict[ShipType, float]:
        v, m = self.v, self.m
        own = Counter(s.type for s in v.ships)
        n_combat = sum(own[t] for t in COMBAT_TYPES)
        sats = [p for p in v.ports if not p.is_home]
        total = max(1, sum(m.built_counts.values()))
        sc: Dict[ShipType, float] = {}
        for t in ALL_TYPES:
            share = m.built_counts[t] / float(total)
            sc[t] = math.log(m.doctrine.weight(t) * m.jitter[t]) + 0.45 * max(0.0, 0.125 - share) / 0.125
        desired = min(len(sats), m.doctrine.target_merchants)
        if not sats or own[MERCHANT] >= len(sats) + 1:
            sc[MERCHANT] -= 4.0
        elif own[MERCHANT] == 0:
            sc[MERCHANT] += 1.6
        elif own[MERCHANT] < desired:
            sc[MERCHANT] += 0.6
        if self.enemy_home is None:
            sc[DESTROYER] += 1.1
            sc[CRUISER] += 0.5
            sc[CARRIER] += 0.4
            if v.turn == 0:
                sc[LANDING] -= 1.2
        ec = Counter(e.type for e in self.fresh + self.stale)
        if ec[SUBMARINE]:
            sc[DESTROYER] += 1.2 * min(2, ec[SUBMARINE])
        if ec[BATTLESHIP] + ec[CARRIER]:
            sc[SUBMARINE] += 0.9 * min(2, ec[BATTLESHIP] + ec[CARRIER])
            sc[CARRIER] += 0.4
        if ec[DESTROYER]:
            sc[CRUISER] += 0.8 * min(2, ec[DESTROYER])
            sc[BATTLESHIP] += 0.6
        if ec[CRUISER]:
            sc[BATTLESHIP] += 0.8 * min(2, ec[CRUISER])
            sc[SUBMARINE] += 0.4
        if ec[LANDING] + ec[MERCHANT]:
            sc[CARRIER] += 0.6
            sc[DESTROYER] += 0.4
        if own[BUILDER] >= 1 or m.ports_built >= 3 or not self.any_viable_coast():
            sc[BUILDER] -= 4.5
        elif v.turn < 2 and m.doctrine.name != "ECONOMIC_BOOM":
            sc[BUILDER] -= 1.0
        elif v.cash >= 250 and n_combat >= 1:
            sc[BUILDER] += 0.3
        max_land = 3 if v.turn >= self.cfg.third_landing_turn else 2
        if own[LANDING] >= max_land:
            sc[LANDING] -= 3.5
        else:
            if n_combat >= 2 and own[LANDING] == 0:
                sc[LANDING] += 1.3
            if self.enemy_home is not None and own[LANDING] == 0:
                sc[LANDING] += 1.2
            if v.turn >= self.cfg.landing_ramp_turn and own[LANDING] == 0:
                sc[LANDING] += 1.5
        return sc

    def sample_hull(self, scores: Dict[ShipType, float]) -> ShipType:
        temp = max(0.10, self.m.doctrine.buy_temperature)
        mx = max(scores.values())
        ws = [math.exp((scores[t] - mx) / temp) for t in ALL_TYPES]
        r = self.rng.random() * sum(ws)
        acc = 0.0
        for t, w in zip(ALL_TYPES, ws):
            acc += w
            if r <= acc:
                return t
        return ALL_TYPES[-1]

    def spawn_tiles(self, port: Pos) -> List[Pos]:
        return [t for t in self.water_adj(port) if t not in self.occupied]

    def build_for(self, t: ShipType) -> Optional[BuildShipAction]:
        v = self.v
        options = []  # (port, spawn)
        for port in sorted(v.legal.builds):
            if t not in v.legal.builds[port]:
                continue
            for sp in self.spawn_tiles(port):
                options.append((port, sp))
        if not options:
            return None
        port_by = {p.pos: p for p in v.ports}
        home = v.home.pos
        if t in COMBAT_TYPES and any(manhattan(e.pos, home) <= 8 for e in self.fresh):
            key = lambda o: (manhattan(o[0], home), -min((manhattan(o[1], e.pos) for e in self.fresh), default=0), o)
        elif t == MERCHANT:
            def key(o):
                p = port_by[o[0]]
                bonus = 0.0 if p.is_home else 2.0 * p.stockpile + 30.0
                return (-(bonus - 200.0 * self.threat(o[1], MERCHANT)), o)
        elif t == BUILDER and self.m.doctrine.builder_mode == "ECO_HOME":
            key = lambda o: (manhattan(o[0], home), 300.0 * self.threat(o[1], BUILDER), o)
        else:
            ref = self.enemy_home.pos if self.enemy_home is not None else (self.W - 1 - home[0], home[1])
            pen = 300.0 if t in NON_COMBAT_TYPES else 40.0

            def key(o):
                return (manhattan(o[1], ref) + pen * self.threat(o[1], t), o)
        port, spawn = min(options, key=key)
        return BuildShipAction(port, spawn, t)

    def _bought(self, act: Optional[BuildShipAction]) -> Optional[BuildShipAction]:
        if act is not None:
            self.m.built_counts[act.ship_type] += 1
        return act

    def purchase(self) -> Optional[Action]:
        v, m = self.v, self.m
        if not v.legal.builds:
            return None
        affordable = set().union(*v.legal.builds.values())
        own = Counter(s.type for s in v.ships)
        close = [e for e in self.fresh if manhattan(e.pos, v.home.pos) <= 8]
        if close:
            defenders = [s for s in v.ships if s.type in COMBAT_TYPES and manhattan(s.pos, v.home.pos) <= 7]
            if len(defenders) < len(close) + 1:
                aff = [t for t in ALL_TYPES if t in COMBAT_TYPES and t in affordable]
                if aff:
                    te = min(close, key=lambda e: (manhattan(e.pos, v.home.pos), e.pos))
                    best = max(aff, key=lambda t: (kill_probability(t, te.type, self.k),
                                                   BASE_STATS[t].speed, -_TYPE_ORDER[t]))
                    act = self._bought(self.build_for(best))
                    if act is not None:
                        return act
        if (LANDING in affordable and own[LANDING] == 0 and self.enemy_home is not None
                and (sum(own[t] for t in COMBAT_TYPES) >= 2 or v.turn >= self.cfg.checkmate_turn)):
            act = self._bought(self.build_for(LANDING))
            if act is not None:
                if m.planned_buy == LANDING:
                    m.planned_buy = None
                return act
        if m.planned_buy is None:
            m.planned_buy = self.sample_hull(self.hull_scores())
        stocked = any(not p.is_home and p.stockpile >= 25 for p in v.ports)
        if m.planned_buy not in affordable and own[MERCHANT] == 0 and stocked and MERCHANT in affordable:
            act = self._bought(self.build_for(MERCHANT))
            if act is not None:
                return act
        if m.planned_buy in affordable:
            chosen = m.planned_buy
            act = self._bought(self.build_for(chosen))
            if act is not None:
                m.planned_buy = None
                return act
        return None


# ---------------------------------------------------------------- helpers
def engine_accepts(engine, pid: int, action: Action, legal: Optional[Sequence[Action]] = None) -> bool:
    """True if ``engine.step(action)`` would take effect for ``pid`` now.

    ``enumerate_legal`` lists one canonical spawn per (port, type); any other
    free water tile next to the port is equally valid to the engine.
    """
    if legal is None:
        legal = engine.enumerate_legal(pid)
    if action in legal:
        return True
    if not isinstance(action, BuildShipAction):
        return False
    if not any(isinstance(a, BuildShipAction) and a.port == action.port and a.ship_type == action.ship_type
               for a in legal):
        return False
    sp = action.spawn_position
    mp = engine.map
    return (mp.in_bounds(sp) and mp.manhattan(sp, action.port) == 1 and mp.is_water(sp)
            and not mp.is_occupied(sp))


_STATE_ATTR = "_fog_commander_state"


def fog_commander_action(engine, player_id: int, rng: random.Random,
                         config: Optional[FogCommanderConfig] = None) -> Action:
    """Functional entry point used by the play UI and scripts.

    Memory lives on the engine, keyed by ``id(engine.players[0])`` so a reset
    (which rebuilds the players) starts fresh memory. ``engine.clone()`` does
    not copy it.
    """
    marker = id(engine.players[0]) if engine.players else None
    state = getattr(engine, _STATE_ATTR, None)
    if state is None or state[0] != marker:
        state = (marker, FogCommander(config))
        setattr(engine, _STATE_ATTR, state)
    return state[1].act(engine, player_id, rng)


def fog_commander_agent(engine) -> Optional[FogCommander]:
    """The FogCommander instance attached to ``engine`` by ``fog_commander_action``, if any."""
    state = getattr(engine, _STATE_ATTR, None)
    if state is None or not engine.players or state[0] != id(engine.players[0]):
        return None
    return state[1]


def canonical_build_action(engine, pid: int, action: Action) -> Action:
    """Map a BuildShipAction to the legal one with the same (port, type).

    For label encoders whose action space has one buy per (port, type).
    Any other action is returned unchanged.
    """
    if not isinstance(action, BuildShipAction):
        return action
    for a in engine.enumerate_legal(pid):
        if isinstance(a, BuildShipAction) and a.port == action.port and a.ship_type == action.ship_type:
            return a
    return action
