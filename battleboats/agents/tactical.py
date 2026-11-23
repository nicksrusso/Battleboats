"""Noisy full-information tactical scripted agent.

Plugs in wherever ``scripted_action(engine, pid, rng)`` does (evaluate,
``collect_trajectory``'s ``opponent_action``, head-to-heads). This module
is a new opponent; the greedy sandbox baseline in ``scripted.py`` is
unchanged.

The agent is limited to the same fog of war as the trained policy.
Enemy ships and ports come only from ``engine.known_enemy_ships`` and
``engine.known_enemy_ports`` (the same fresh + stale sighting records
that ``build_observation`` tokenizes). Before each decision it calls
``engine._refresh_sightings`` so those records match current vision.
It never reads ``engine.ships``, ``engine.ports``, or the opponent's
``home_port`` for hidden enemies. Map terrain and friendly state are
fully known, as they are in the policy observation.

Design
------
Buys
    Per-game, per-player build queue. Each entry is a ship type sampled
    uniformly from every buildable type (equal probability, no cost
    weighting), using the caller's seeded ``rng``. Each turn, if gold
    covers the cost of the queue head and a legal unlocked port can
    spawn it, buy it at a uniformly chosen such port, pop the head, and
    repeat while still affordable. Otherwise save — do not buy a cheaper
    or different type. A head that can never be built on this map (type
    not in the buildable set) is skipped and resampled; a head that is
    only unaffordable or blocked this turn is held.

Merchants
    The BAT-7 / merchant-sandbox loop: load at an owned non-home port
    with stockpile, sail to home, unload, repeat. If a combat ship
    threatens them, they flee first.

Builders
    Move to a water tile adjacent to the empty (non-port) land tile
    closest to our home port, then build there. Reachable-by-water
    targets are preferred; Manhattan-to-home is the primary key.

Landings
    Target the nearest last-known / currently known capturable port
    (any sighted enemy port, home or satellite, and any sighted
    neutral) measured by water-path distance from the landing ship.
    A sighted enemy home is not preferred over a closer satellite.
    If no enemy or neutral port has been seen, explore toward the
    far side of the map from our home (the enemy home is not known
    at kickoff unless a scout or friendly port has already sighted
    it — the policy observation starts empty too). Always flee from
    hostile combat ships that we can see (or recently saw);
    survival outranks capturing. A landing ship never issues
    ``AttackAction``.

Combat ships (and other non-landing hulls' fight-or-run)
    Goal-directed moves with softmax noise. Local power is the
    distance-weighted combat strength of friendlies vs *known* enemies
    within radius R. Only sighted enemies count: fresh sightings at
    full weight, stale last-known positions decaying linearly to 0
    over ``memory_turns`` (default 4). The engine has no HP, so power
    is ``stats.strength`` from the public type table; strength-0 hulls
    contribute 0. The spatial weight is the old heuristic linear
    kernel ``max(0, 1 - d/R)``, cut off at R (or the acting ship's
    attack range, whichever is larger). Engage — attack and close —
    if ``ours > theirs * margin``; otherwise retreat away from the
    known enemy center of mass and toward friendlies or home. Attack
    targeting reuses ``pk`` and material value from
    ``battleboats.agents.heuristics`` on the sighted type, not the
    live hidden ship.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from battleboats.agents.heuristics import COMBAT_TYPES, _ship_value, pk
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
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType

Pos = Tuple[int, int]
_CARDINALS: Tuple[Pos, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INF = 10**9


@dataclass(frozen=True, slots=True)
class TacticalConfig:
    """Tunable fight-or-run and movement-noise knobs."""

    radius: int = 6
    margin: float = 1.2
    temperature: float = 0.5
    memory_turns: int = 4


DEFAULT_CONFIG = TacticalConfig()


def tactical_scripted_action(
    engine,
    player_id: int,
    rng: random.Random,
    cfg: TacticalConfig = DEFAULT_CONFIG,
) -> Action:
    """Return one legal action. Always legal when ``enumerate_legal`` is non-empty."""
    # Same refresh the engine runs after every step / reset, so
    # known_enemy_* matches what the policy observation would tokenize.
    engine._refresh_sightings()
    legal = engine.enumerate_legal(player_id)
    if not legal:
        raise RuntimeError(
            f"tactical_scripted_action: no legal actions for player {player_id}"
        )
    return _choose(engine, player_id, rng, cfg, legal)


def local_combat_power(
    engine,
    player_id: int,
    origin: Pos,
    radius: int,
) -> Tuple[float, float]:
    """Distance-weighted combat strength of friendlies vs enemies around ``origin``.

    ``power = stats.strength`` (no HP in this engine). A ship at distance
    ``d`` contributes ``power * max(0, 1 - d/radius)`` when ``d <= radius``.
    Returns ``(ours, theirs)``, including the ship at ``origin`` if any.
    """
    if radius <= 0:
        return 0.0, 0.0
    ours = 0.0
    theirs = 0.0
    man = engine.map.manhattan
    for ship in engine.ships.values():
        d = man(ship.position, origin)
        if d > radius:
            continue
        weight = 1.0 - d / radius
        power = float(ship.stats.strength) * weight
        if ship.owner == player_id:
            ours += power
        else:
            theirs += power
    return ours, theirs


def should_engage(
    ours: float, theirs: float, margin: float = DEFAULT_CONFIG.margin
) -> bool:
    """Attack / close when local friendly power exceeds enemy power by ``margin``."""
    return ours > theirs * margin


@dataclass(frozen=True, slots=True)
class _KnownEnemy:
    """A sighted enemy ship, with public type stats and a memory weight."""

    ship_id: int
    type: ShipType
    position: Pos
    stats: object
    weight: float


def _sighting_weight(engine, sighting, cfg: TacticalConfig) -> float:
    """1.0 if fresh, else linear decay of last-seen position over memory_turns."""
    if sighting.fresh:
        return 1.0
    if cfg.memory_turns <= 0:
        return 0.0
    age = engine.turn - sighting.turn_seen
    return max(0.0, 1.0 - age / cfg.memory_turns)


def _known_enemies(engine, player_id: int, cfg: TacticalConfig) -> List[_KnownEnemy]:
    """Enemy ships the policy can see: known_enemy_ships with memory weights."""
    out: List[_KnownEnemy] = []
    for sighting in engine.known_enemy_ships(player_id):
        weight = _sighting_weight(engine, sighting, cfg)
        if weight <= 0:
            continue
        out.append(
            _KnownEnemy(
                ship_id=sighting.ship_id,
                type=sighting.type,
                position=sighting.position,
                stats=BASE_STATS[sighting.type],
                weight=weight,
            )
        )
    return out


def _known_enemy_home(engine, player_id: int) -> Optional[Pos]:
    """Last-known enemy home, or None if no is_home port has been sighted."""
    homes = [s for s in engine.known_enemy_ports(player_id) if s.is_home]
    if not homes:
        return None
    homes.sort(key=lambda s: (s.fresh, s.turn_seen, s.position), reverse=True)
    return homes[0].position


def _explore_goal(engine, player_id: int) -> List[Pos]:
    """Far-side water from our home, used when the enemy home is unknown."""
    home = engine.players[player_id].home_port
    far = (engine.map.width - 1 - home[0], engine.map.height - 1 - home[1])
    if engine.map.in_bounds(far) and engine.map.is_water(far):
        return [far]
    tiles = _water_neighbors(engine, far)
    return tiles if tiles else [far]


def _local_combat_power_known(
    engine,
    player_id: int,
    origin: Pos,
    radius: int,
    cfg: TacticalConfig,
) -> Tuple[float, float]:
    """Like ``local_combat_power``, but enemies come only from sightings."""
    if radius <= 0:
        return 0.0, 0.0
    ours = 0.0
    theirs = 0.0
    man = engine.map.manhattan
    for ship in engine.ships.values():
        if ship.owner != player_id:
            continue
        d = man(ship.position, origin)
        if d > radius:
            continue
        ours += float(ship.stats.strength) * (1.0 - d / radius)
    for enemy in _known_enemies(engine, player_id, cfg):
        d = man(enemy.position, origin)
        if d > radius:
            continue
        theirs += float(enemy.stats.strength) * (1.0 - d / radius) * enemy.weight
    return ours, theirs


class _TypeShip:
    """Public type-stats stand-in so pk / _ship_value never touch a live enemy."""

    __slots__ = ("type", "stats")

    def __init__(self, ship_type: ShipType) -> None:
        self.type = ship_type
        self.stats = BASE_STATS[ship_type]


def _defender_type(engine, player_id: int, target_id: int) -> Optional[ShipType]:
    for sighting in engine.known_enemy_ships(player_id):
        if sighting.ship_id == target_id:
            return sighting.type
    return None


# Types the rules allow to be bought. A map may drop some; the queue
# skips a head that is not in the engine's current buildable set.
BUILDABLE_TYPES: Tuple[ShipType, ...] = tuple(ShipType)
_QUEUE_FILL = 32


def buildable_types(engine=None) -> Tuple[ShipType, ...]:
    """Ship types that can ever be built in this game (not just this turn)."""
    types = tuple(t for t in BUILDABLE_TYPES if t in BASE_STATS)
    extra = (
        getattr(engine, "_tactical_buildable_types", None)
        if engine is not None
        else None
    )
    if extra is not None:
        allowed = set(extra)
        types = tuple(t for t in types if t in allowed)
    return types


def sample_build_queue(
    rng: random.Random,
    n: int,
    types: Optional[Sequence[ShipType]] = None,
) -> List[ShipType]:
    """``n`` types drawn independently and uniformly. Used to fill a queue."""
    pool = tuple(types) if types is not None else buildable_types()
    if not pool or n <= 0:
        return []
    return [rng.choice(pool) for _ in range(n)]


def build_queue_for(engine, player_id: int) -> deque:
    """Per-game, per-player queue stored on the engine.

    Reset is detected via ``id(engine.players[0])``, which changes both
    when a new ``gameEngine`` is constructed and when ``engine.reset()``
    rebuilds the player list.
    """
    marker = id(engine.players[0])
    state = getattr(engine, "_tactical_buy_state", None)
    if state is None or state["marker"] != marker:
        state = {"marker": marker, "queues": {0: deque(), 1: deque()}}
        engine._tactical_buy_state = state
    return state["queues"][player_id]


def _discard_unbuildable(queue: deque, types: Sequence[ShipType]) -> None:
    allowed = set(types)
    while queue and queue[0] not in allowed:
        queue.popleft()


def _ensure_queue_head(
    queue: deque,
    rng: random.Random,
    types: Sequence[ShipType],
) -> Optional[ShipType]:
    _discard_unbuildable(queue, types)
    if not queue:
        queue.extend(sample_build_queue(rng, _QUEUE_FILL, types))
        _discard_unbuildable(queue, types)
    return queue[0] if queue else None


def choose_build_action(
    engine,
    player_id: int,
    options: Sequence[Action],
    rng: random.Random,
    *,
    queue: Optional[deque] = None,
    types: Optional[Sequence[ShipType]] = None,
) -> Optional[Action]:
    """Buy the queue head if it is legal and affordable this turn; else save.

    ``options`` should be the current legal actions (or just the builds).
    The head is popped only when a buy is actually returned. A uniformly
    random legal port is chosen when more than one can spawn the head.
    """
    builds = [a for a in options if isinstance(a, BuildShipAction)]
    buildable = tuple(types) if types is not None else buildable_types(engine)
    q = queue if queue is not None else build_queue_for(engine, player_id)
    head = _ensure_queue_head(q, rng, buildable)
    if head is None:
        return None
    matching = [a for a in builds if a.ship_type is head]
    if not matching:
        return None
    chosen = rng.choice(matching)
    q.popleft()
    if not q:
        q.extend(sample_build_queue(rng, _QUEUE_FILL, buildable))
    return chosen


def choose_builder_target(
    engine, player_id: int, builder_pos: Optional[Pos] = None
) -> Optional[Pos]:
    """Empty land closest to our home port, preferring water-reachable tiles.

    Tie-break: water distance from the builder (if given), then Manhattan
    from the builder, then coordinates.
    """
    home = engine.players[player_id].home_port
    empties = _empty_land_tiles(engine)
    if not empties:
        return None
    approachable: List[Tuple[int, int, int, Pos]] = []
    reachable_from: Optional[Dict[Pos, int]] = None
    if builder_pos is not None and engine.map.is_water(builder_pos):
        reachable_from = _water_dist_map(engine, [builder_pos])
    for tile in empties:
        approaches = _water_neighbors(engine, tile)
        if not approaches:
            continue
        d_home = engine.map.manhattan(tile, home)
        if reachable_from is not None:
            d_ship = min(
                (reachable_from.get(p, _INF) for p in approaches), default=_INF
            )
        else:
            d_ship = (
                engine.map.manhattan(tile, builder_pos)
                if builder_pos is not None
                else 0
            )
        approachable.append((d_home, d_ship, tile[0] + tile[1], tile))
    if not approachable:
        return None
    reachable = [c for c in approachable if c[1] < _INF]
    pool = reachable if reachable else approachable
    pool.sort()
    return pool[0][3]


def landing_is_threatened(
    engine, player_id: int, ship, cfg: TacticalConfig = DEFAULT_CONFIG
) -> bool:
    """True if a hostile combat ship can reach or shoot this ship soon."""
    return bool(_threats(engine, player_id, ship.position, cfg))


def _choose(
    engine,
    player_id: int,
    rng: random.Random,
    cfg: TacticalConfig,
    legal: Sequence[Action],
) -> Action:
    by_cls: Dict[type, List[Action]] = {}
    for action in legal:
        by_cls.setdefault(type(action), []).append(action)

    captures = by_cls.get(CapturePortAction, [])
    if captures:
        capture = _choose_capture(engine, player_id, captures, cfg)
        if capture is not None:
            return capture

    attacks = by_cls.get(AttackAction, [])
    if attacks:
        free = _guaranteed_kills(engine, player_id, attacks)
        if free:
            return rng.choice(free)

    unloads = by_cls.get(MerchantUnloadAction, [])
    if unloads:
        safe = [
            a
            for a in unloads
            if not _merchant_action_threatened(engine, player_id, a, cfg)
        ]
        if safe:
            return rng.choice(safe)

    loads = by_cls.get(MerchantLoadAction, [])
    if loads:
        safe = [
            a
            for a in loads
            if not _merchant_action_threatened(engine, player_id, a, cfg)
        ]
        if safe:
            return rng.choice(safe)

    builds_port = by_cls.get(BuildPortAction, [])
    if builds_port:
        built = _choose_build_port(engine, player_id, builds_port, cfg)
        if built is not None:
            return built

    if attacks:
        engaging = [
            a for a in attacks if _attack_is_engaging(engine, player_id, a, cfg)
        ]
        if engaging:
            return _best_attack(engine, player_id, engaging)

    moves = by_cls.get(MoveAction, [])
    move = _choose_move(engine, player_id, moves, cfg, rng)
    if move is not None:
        return move

    buy = choose_build_action(engine, player_id, by_cls.get(BuildShipAction, []), rng)
    if buy is not None:
        return buy

    ends = by_cls.get(EndTurnAction, [])
    if ends:
        return ends[0]
    return legal[-1]


def _choose_capture(
    engine, player_id: int, captures: Sequence[Action], cfg: TacticalConfig
) -> Optional[Action]:
    """Take a capture unless the landing is threatened — survival comes first.

    Any capturable port is fair game; a sighted enemy home is not
    preferred over a satellite or a neutral.
    """
    safe: List[Action] = []
    for action in captures:
        if not isinstance(action, CapturePortAction):
            continue
        ship = engine.ships.get(action.landing_ship_id)
        if ship is None or ship.type is not ShipType.LANDING:
            continue
        if landing_is_threatened(engine, player_id, ship, cfg):
            continue
        safe.append(action)
    if not safe:
        return None
    return safe[0]


def _merchant_action_threatened(
    engine, player_id: int, action, cfg: TacticalConfig
) -> bool:
    ship = engine.ships.get(action.merchant_id)
    if ship is None:
        return False
    return bool(_threats(engine, player_id, ship.position, cfg))


def _choose_build_port(
    engine, player_id: int, options: Sequence[Action], cfg: TacticalConfig
) -> Optional[Action]:
    for action in options:
        if not isinstance(action, BuildPortAction):
            continue
        ship = engine.ships.get(action.builder_ship_id)
        if ship is None:
            continue
        if _threats(engine, player_id, ship.position, cfg):
            continue
        target = choose_builder_target(engine, player_id, ship.position)
        if target is not None and action.target == target:
            return action
    return None


def _guaranteed_kills(
    engine, player_id: int, attacks: Sequence[Action]
) -> List[Action]:
    """Attacks by non-landing ships that always destroy a strength-0 target."""
    out: List[Action] = []
    for action in attacks:
        if not isinstance(action, AttackAction):
            continue
        attacker = engine.ships.get(action.attacker_id)
        if attacker is None:
            continue
        if attacker.owner != player_id or attacker.type is ShipType.LANDING:
            continue
        def_type = _defender_type(engine, player_id, action.target_id)
        if def_type is None:
            continue
        if BASE_STATS[def_type].strength <= 0:
            out.append(action)
    return out


def _attack_is_engaging(
    engine, player_id: int, action: Action, cfg: TacticalConfig
) -> bool:
    if not isinstance(action, AttackAction):
        return False
    attacker = engine.ships.get(action.attacker_id)
    if attacker is None or attacker.type is ShipType.LANDING:
        return False
    radius = max(cfg.radius, int(attacker.stats.attack_range))
    ours, theirs = _local_combat_power_known(
        engine, player_id, attacker.position, radius, cfg
    )
    return should_engage(ours, theirs, cfg.margin)


def _best_attack(engine, player_id: int, attacks: Sequence[Action]) -> Action:
    def score(action: Action) -> float:
        if not isinstance(action, AttackAction):
            return 0.0
        attacker = engine.ships.get(action.attacker_id)
        def_type = _defender_type(engine, player_id, action.target_id)
        if attacker is None or def_type is None:
            return 0.0
        defender = _TypeShip(def_type)
        return pk(attacker, defender, engine.kill_curve_k) * _ship_value(defender)

    return max(attacks, key=score)


def _choose_move(
    engine,
    player_id: int,
    moves: Sequence[Action],
    cfg: TacticalConfig,
    rng: random.Random,
) -> Optional[Action]:
    if not moves:
        return None
    by_ship: Dict[int, List[MoveAction]] = {}
    for action in moves:
        if isinstance(action, MoveAction):
            by_ship.setdefault(action.ship_id, []).append(action)
    if not by_ship:
        return None

    ranked: List[Tuple[int, int, List[MoveAction]]] = []
    for sid, options in by_ship.items():
        ship = engine.ships.get(sid)
        if ship is None:
            continue
        urgency = _move_urgency(engine, player_id, ship, cfg)
        ranked.append((urgency, sid, options))
    ranked.sort(reverse=True)
    for urgency, _sid, options in ranked:
        if urgency < 0:
            continue
        chosen = _softmax_move(engine, player_id, options, cfg, rng)
        if chosen is not None:
            return chosen
    return None


def _move_urgency(engine, player_id: int, ship, cfg: TacticalConfig) -> int:
    threatened = bool(_threats(engine, player_id, ship.position, cfg))
    if ship.type is ShipType.LANDING:
        return 100 if threatened else 60
    if ship.type is ShipType.MERCHANT:
        return 80 if threatened else 50
    if ship.type is ShipType.BUILDER:
        return 75 if threatened else 40
    if threatened:
        ours, theirs = _local_combat_power_known(
            engine,
            player_id,
            ship.position,
            max(cfg.radius, int(ship.stats.attack_range)),
            cfg,
        )
        return 70 if not should_engage(ours, theirs, cfg.margin) else 35
    return 30


def _softmax_move(
    engine,
    player_id: int,
    options: Sequence[MoveAction],
    cfg: TacticalConfig,
    rng: random.Random,
) -> Optional[Action]:
    ship = engine.ships.get(options[0].ship_id)
    if ship is None:
        return None
    scores = [
        _move_score(engine, player_id, ship, action.destination, cfg)
        for action in options
    ]
    if ship.type is not ShipType.LANDING:
        # Idle ships with no improving or fleeing move yield the turn to buys.
        if max(scores) < 0 and not _should_flee(engine, player_id, ship, cfg):
            return None
    return _softmax_choice(list(options), scores, rng, cfg.temperature)


def _should_flee(engine, player_id: int, ship, cfg: TacticalConfig) -> bool:
    if ship.type is ShipType.LANDING:
        return landing_is_threatened(engine, player_id, ship, cfg)
    threats = _threats(engine, player_id, ship.position, cfg)
    if not threats:
        return False
    if ship.type in (ShipType.MERCHANT, ShipType.BUILDER):
        return True
    radius = max(cfg.radius, int(ship.stats.attack_range))
    ours, theirs = _local_combat_power_known(
        engine, player_id, ship.position, radius, cfg
    )
    return not should_engage(ours, theirs, cfg.margin)


def _move_score(engine, player_id: int, ship, dest: Pos, cfg: TacticalConfig) -> float:
    if _should_flee(engine, player_id, ship, cfg):
        return _flee_score(engine, player_id, ship.position, dest, cfg)
    goal = _goal_tiles(engine, player_id, ship, cfg)
    if not goal:
        # No objective: bias toward a known enemy home, else explore
        # the far side of the map from our home.
        target = _known_enemy_home(engine, player_id)
        if target is None:
            explore = _explore_goal(engine, player_id)
            target = explore[0]
        man = engine.map.manhattan
        return float(man(ship.position, target) - man(dest, target))
    dist = _water_dist_map(engine, goal)
    before = dist.get(ship.position, _INF)
    after = dist.get(dest, _INF)
    return float(before - after)


def _flee_score(
    engine, player_id: int, src: Pos, dest: Pos, cfg: TacticalConfig
) -> float:
    threats = _threats(engine, player_id, src, cfg)
    man = engine.map.manhattan
    if threats:
        sw = sum(
            (float(s.stats.strength) or 1.0) * getattr(s, "weight", 1.0)
            for s in threats
        )
        cx = (
            sum(
                (float(s.stats.strength) or 1.0)
                * getattr(s, "weight", 1.0)
                * s.position[0]
                for s in threats
            )
            / sw
        )
        cy = (
            sum(
                (float(s.stats.strength) or 1.0)
                * getattr(s, "weight", 1.0)
                * s.position[1]
                for s in threats
            )
            / sw
        )
        ecom = (cx, cy)
    else:
        ecom = (
            _known_enemy_home(engine, player_id) or engine.players[player_id].home_port
        )
    score = _float_man(dest, ecom) - _float_man(src, ecom)
    friends = [
        s
        for s in engine.ships.values()
        if s.owner == player_id and s.position != src and s.type in COMBAT_TYPES
    ]
    if friends:
        fx = sum(s.position[0] for s in friends) / len(friends)
        fy = sum(s.position[1] for s in friends) / len(friends)
        score += 0.5 * (_float_man(src, (fx, fy)) - _float_man(dest, (fx, fy)))
    home = engine.players[player_id].home_port
    score += 0.3 * (man(src, home) - man(dest, home))
    return score


def _goal_tiles(engine, player_id: int, ship, cfg: TacticalConfig) -> List[Pos]:
    if ship.type is ShipType.LANDING:
        return _landing_goal_tiles(engine, player_id, ship.position)
    if ship.type is ShipType.MERCHANT:
        port = _merchant_goal_port(engine, player_id, ship)
        return _water_neighbors(engine, port) if port is not None else []
    if ship.type is ShipType.BUILDER:
        target = choose_builder_target(engine, player_id, ship.position)
        return _water_neighbors(engine, target) if target is not None else []
    # Combat: close on the nearest *known* enemy, else a known home, else explore.
    foes = [e.position for e in _known_enemies(engine, player_id, cfg)]
    if foes:
        nearest = min(foes, key=lambda p: engine.map.manhattan(ship.position, p))
        return [nearest]
    known_home = _known_enemy_home(engine, player_id)
    if known_home is not None:
        return _water_neighbors(engine, known_home)
    return _explore_goal(engine, player_id)


def _path_dist_to_port(engine, origin: Pos, port: Pos) -> int:
    """Water-path length from ``origin`` to a water tile adjacent to ``port``."""
    neighbors = _water_neighbors(engine, port)
    if not neighbors:
        return _INF
    if origin in neighbors:
        return 0
    if not engine.map.is_water(origin):
        return _INF
    dist = _water_dist_map(engine, [origin])
    return min((dist.get(n, _INF) for n in neighbors), default=_INF)


def _landing_goal_tiles(engine, player_id: int, origin: Pos) -> List[Pos]:
    """Approach tiles of the nearest *known* capturable port by water path.

    Any sighted enemy port (home or not) and any sighted neutral counts.
    Falls back to Manhattan if no water path exists, and to far-side
    exploration if no capturable port has been seen.
    """
    ports = _known_capturable_ports(engine, player_id)
    if not ports:
        return _explore_goal(engine, player_id)

    def nearest(key) -> Pos:
        return min(ports, key=lambda p: (key(p), p))

    reachable = [p for p in ports if _path_dist_to_port(engine, origin, p) < _INF]
    best = (
        nearest(lambda p: _path_dist_to_port(engine, origin, p))
        if reachable
        else nearest(lambda p: engine.map.manhattan(origin, p))
    )
    tiles = _water_neighbors(engine, best)
    return tiles if tiles else _explore_goal(engine, player_id)


def _known_capturable_ports(engine, player_id: int) -> List[Pos]:
    """Positions of enemy/neutral ports this player has sighted (fresh or stale)."""
    return [s.position for s in engine.known_enemy_ports(player_id)]


def _merchant_goal_port(engine, player_id: int, ship) -> Optional[Pos]:
    me = engine.players[player_id]
    if ship.cargo > 0:
        return me.home_port
    stocked = [
        p
        for p in me.owned_port_positions
        if p != me.home_port and engine.ports[p].stockpile > 0
    ]
    if stocked:
        return min(stocked, key=lambda p: engine.map.manhattan(ship.position, p))
    satellites = [p for p in me.owned_port_positions if p != me.home_port]
    if satellites:
        return min(satellites, key=lambda p: engine.map.manhattan(ship.position, p))
    return None


def _threats(
    engine, player_id: int, origin: Pos, cfg: TacticalConfig
) -> List[_KnownEnemy]:
    """Known hostile combat ships within max(radius, their attack range) of origin.

    Uses last-known positions and memory weights from official sightings.
    """
    man = engine.map.manhattan
    out: List[_KnownEnemy] = []
    for enemy in _known_enemies(engine, player_id, cfg):
        if enemy.type not in COMBAT_TYPES or enemy.stats.attack_range <= 0:
            continue
        reach = max(cfg.radius, int(enemy.stats.attack_range))
        if man(enemy.position, origin) <= reach:
            out.append(enemy)
    return out


def _empty_land_tiles(engine) -> List[Pos]:
    m = engine.map
    return [
        (x, y)
        for x in range(m.width)
        for y in range(m.height)
        if m.is_land((x, y)) and not m.is_port((x, y))
    ]


def _water_neighbors(engine, tile: Optional[Pos]) -> List[Pos]:
    if tile is None:
        return []
    out: List[Pos] = []
    m = engine.map
    for dx, dy in _CARDINALS:
        q = (tile[0] + dx, tile[1] + dy)
        if m.in_bounds(q) and m.is_water(q):
            out.append(q)
    return out


def _water_dist_map(engine, origins: Iterable[Pos]) -> Dict[Pos, int]:
    m = engine.map
    dist: Dict[Pos, int] = {}
    q: deque = deque()
    for p in origins:
        if not m.in_bounds(p) or not m.is_water(p) or p in dist:
            continue
        dist[p] = 0
        q.append(p)
    while q:
        p = q.popleft()
        d = dist[p] + 1
        for dx, dy in _CARDINALS:
            n = (p[0] + dx, p[1] + dy)
            if n in dist or not m.in_bounds(n) or not m.is_water(n):
                continue
            dist[n] = d
            q.append(n)
    return dist


def _float_man(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _softmax_choice(
    actions: Sequence[Action],
    scores: Sequence[float],
    rng: random.Random,
    temperature: float,
) -> Action:
    if len(actions) == 1:
        return actions[0]
    if temperature <= 0:
        best = max(scores)
        tied = [a for a, s in zip(actions, scores) if s == best]
        return rng.choice(tied)
    peak = max(scores)
    weights = [math.exp((s - peak) / temperature) for s in scores]
    total = sum(weights)
    if total <= 0 or any(not math.isfinite(w) for w in weights):
        best = max(scores)
        tied = [a for a, s in zip(actions, scores) if s == best]
        return rng.choice(tied)
    return rng.choices(list(actions), weights=weights, k=1)[0]
