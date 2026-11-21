"""Text view of one seat's decision, format version 1.

The prompt is built only from what that seat can know: own ships and ports
are read from the engine, enemy ships and ports only from sightings (fresh
or stale). Terrain is public. Enemy cargo, enemy gold, and unspotted units
are omitted. Distances use the engine's rules: 4-directional water moves,
Manhattan attack range, and detection = scouting * visibility.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

from battleboats.core.actions import (Action, AttackAction, BuildPortAction,
                                      BuildShipAction, CapturePortAction,
                                      EndTurnAction, MerchantLoadAction,
                                      MerchantUnloadAction, MoveAction)
from battleboats.core.gameEngine import MERCHANT_CAPACITY, gameEngine
from battleboats.core.shipyard.ship import Ship
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType

FORMAT_VERSION = 1

Pos = Tuple[int, int]
_CARDINAL: Tuple[Pos, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# One character per ship. Builder is U so it does not collide with Battleship.
_LETTER = {
    ShipType.CARRIER: "C",
    ShipType.BATTLESHIP: "B",
    ShipType.CRUISER: "R",
    ShipType.DESTROYER: "D",
    ShipType.SUBMARINE: "S",
    ShipType.MERCHANT: "M",
    ShipType.LANDING: "L",
    ShipType.BUILDER: "U",
}

_LEGEND = (
    "legend: .=water #=land H/P=my home/port h/p=sighted enemy home/port "
    "upper=my ship lower=enemy Ccarrier Bbattleship Rcruiser Ddestroyer "
    "Ssub Mmerchant Llanding Ubuilder. Row 0 is the top (high y); x increases right."
)


def _whole(value: float) -> int | float:
    """CSV stats are floats; whole numbers read better as ints in the prompt."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def ship_id(sid: int) -> str:
    return f"s{sid}"


def port_id(pos: Pos) -> str:
    return f"p{pos[0]}_{pos[1]}"


def build_view(
    engine: gameEngine,
    player_id: int,
    legal: List[Action],
    *,
    horizon: int = 4,
    max_edges: int = 20,
) -> dict:
    """Structured, fog-filtered view. ``render_prompt`` prints this dict."""
    player = engine.players[player_id]
    entities = _entities(engine, player_id)
    facts = _facts(engine, player_id)
    edges = build_edges(engine, player_id, horizon=horizon, max_edges=max_edges)
    return {
        "format_version": FORMAT_VERSION,
        "seat": f"player_{player_id}",
        "seat_id": player_id,
        "turn": engine.turn,
        "gold": player.cash,
        "board": [engine.map.width, engine.map.height],
        "entities": entities,
        "grid": render_grid(engine, player_id),
        "facts": facts,
        "edges": edges,
        "horizon": horizon,
        "max_edges": max_edges,
        "moves": [describe_action(action, engine, player_id) for action in legal],
        "legal": [action_record(action) for action in legal],
    }


def render_prompt(view: dict) -> str:
    """Exact user message for one decision. Compact on purpose (slow CPU inference)."""
    lines = [
        f"FORMAT {view['format_version']}",
        (f"You are {view['seat']}. Reply with a MOVES number and one short sentence."),
        (
            f"board {view['board'][0]}x{view['board'][1]} turn {view['turn']} "
            f"gold {view['gold']}. hp=strength (ships die in one hit). "
            "turns 0=this turn. Hidden: enemy cargo, enemy gold, unspotted units."
        ),
        "ENTITIES",
    ]
    for ent in view["entities"]:
        lines.append(_compact(ent))
    lines.append("GRID")
    lines.extend(view["grid"])
    lines.append(_LEGEND)
    lines.append("FACTS")
    for fact in view["facts"]:
        lines.append(_compact(fact))
    lines.append(f"EDGES horizon={view['horizon']}")
    lines.append("threats assume the enemy still has a full move and has not attacked.")
    for edge in view["edges"]:
        lines.append(_compact(edge))
    lines.append("MOVES")
    for i, label in enumerate(view["moves"]):
        lines.append(f"{i} {label}")
    lines.append("Reply with the move number and one short sentence.")
    return "\n".join(lines)


def render_grid(engine: gameEngine, player_id: int) -> List[str]:
    """One character per cell. Unsighted ports stay land. High y is the top row."""
    mp = engine.map
    player = engine.players[player_id]
    cells = [["." for _ in range(mp.width)] for _ in range(mp.height)]
    for x in range(mp.width):
        for y in range(mp.height):
            if mp.is_land((x, y)):
                cells[y][x] = "#"
    for pos in player.owned_port_positions:
        port = engine.ports[pos]
        cells[pos[1]][pos[0]] = "H" if port.is_home else "P"
    for sight in player.port_sightings.values():
        x, y = sight.position
        cells[y][x] = "h" if sight.is_home else "p"
    stale = [s for s in player.sightings.values() if not s.fresh]
    fresh = [s for s in player.sightings.values() if s.fresh]
    for sight in stale + fresh:
        x, y = sight.position
        if mp.in_bounds((x, y)):
            cells[y][x] = _LETTER[sight.type].lower()
    for sid in player.owned_ship_ids:
        ship = engine.ships[sid]
        x, y = ship.position
        cells[y][x] = _LETTER[ship.type]
    return ["".join(cells[y]) for y in range(mp.height - 1, -1, -1)]


def build_edges(
    engine: gameEngine,
    player_id: int,
    *,
    horizon: int = 4,
    max_edges: int = 20,
) -> List[dict]:
    """Interactions within ``horizon`` turns, nearest first, capped.

    Kinds: can_attack, can_load, can_unload, can_capture, threat.
    A threat is an enemy who can attack one of our ships. Enemy movement
    state is not observed, so threats assume a full move and no attack yet.
    Paths walk water only and treat own ships and fresh sightings as blocked.
    Stale contacts use their last-known tile and do not block.
    """
    player = engine.players[player_id]
    own = [engine.ships[sid] for sid in sorted(player.owned_ship_ids)]
    sightings = sorted(player.sightings.values(), key=lambda s: s.ship_id)
    edges: List[dict] = []

    for ship in own:
        remaining = _remaining(ship)
        blocked = _known_blocked(engine, player_id, ship.position)
        for sight in sightings:
            stats = BASE_STATS[sight.type]
            turns = _turns_to_fire(
                engine,
                ship.position,
                ship.stats,
                remaining,
                act_now=not ship.has_attacked,
                target_pos=sight.position,
                target_vis=stats.visibility,
                blocked=blocked,
            )
            if turns is None or turns > horizon:
                continue
            edges.append(
                {
                    "from": ship_id(ship.id),
                    "to": ship_id(sight.ship_id),
                    "kind": "can_attack",
                    "turns": turns,
                }
            )
        if ship.type is ShipType.MERCHANT and ship.cargo < MERCHANT_CAPACITY:
            for pos in sorted(player.owned_port_positions):
                port = engine.ports[pos]
                if port.is_home or pos == player.home_port or port.stockpile <= 0:
                    continue
                turns = _turns_to_adjacent(
                    engine, ship.position, remaining, ship.stats.speed, pos, blocked
                )
                if turns is None or turns > horizon:
                    continue
                edges.append(
                    {
                        "from": ship_id(ship.id),
                        "to": port_id(pos),
                        "kind": "can_load",
                        "turns": turns,
                    }
                )
        if ship.type is ShipType.MERCHANT and ship.cargo > 0:
            turns = _turns_to_adjacent(
                engine,
                ship.position,
                remaining,
                ship.stats.speed,
                player.home_port,
                blocked,
            )
            if turns is not None and turns <= horizon:
                edges.append(
                    {
                        "from": ship_id(ship.id),
                        "to": port_id(player.home_port),
                        "kind": "can_unload",
                        "turns": turns,
                    }
                )
        if ship.type is ShipType.LANDING:
            for sight in sorted(
                player.port_sightings.values(), key=lambda s: s.position
            ):
                turns = _turns_to_adjacent(
                    engine,
                    ship.position,
                    remaining,
                    ship.stats.speed,
                    sight.position,
                    blocked,
                )
                if turns is None or turns > horizon:
                    continue
                edges.append(
                    {
                        "from": ship_id(ship.id),
                        "to": port_id(sight.position),
                        "kind": "can_capture",
                        "turns": turns,
                    }
                )

    for sight in sightings:
        stats = BASE_STATS[sight.type]
        if stats.attack_range <= 0 or stats.speed <= 0:
            continue
        blocked = _known_blocked(engine, player_id, sight.position)
        for ship in own:
            turns = _turns_to_fire(
                engine,
                sight.position,
                stats,
                stats.speed,
                act_now=True,
                target_pos=ship.position,
                target_vis=ship.stats.visibility,
                blocked=blocked,
            )
            if turns is None or turns > horizon:
                continue
            edges.append(
                {
                    "from": ship_id(sight.ship_id),
                    "to": ship_id(ship.id),
                    "kind": "threat",
                    "turns": turns,
                }
            )

    edges.sort(key=lambda e: (e["turns"], e["kind"], e["from"], e["to"]))
    if max_edges >= 0:
        edges = edges[:max_edges]
    return edges


def describe_action(action: Action, engine: gameEngine, player_id: int) -> str:
    """One short menu line. Does not read hidden enemy positions."""
    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        who = _own_label(ship, action.ship_id)
        return f"Move {who} {_xy(ship.position if ship else None)}->{_xy(action.destination)}"
    if isinstance(action, AttackAction):
        ship = engine.ships.get(action.attacker_id)
        sight = engine.players[player_id].sightings.get(action.target_id)
        who = _own_label(ship, action.attacker_id)
        if sight is None:
            tgt = f"#{action.target_id}"
            extra = ""
        else:
            tgt = f"{_LETTER[sight.type].lower()}#{sight.ship_id}"
            origin = ship.position if ship else sight.position
            extra = f" d={_manhattan(origin, sight.position)}"
        return f"Attack {who} -> {tgt}{extra}"
    if isinstance(action, BuildShipAction):
        cost = BASE_STATS[action.ship_type].cost
        return f"Build {action.ship_type.value} port={_xy(action.port)} cost={cost}"
    if isinstance(action, BuildPortAction):
        ship = engine.ships.get(action.builder_ship_id)
        who = _own_label(ship, action.builder_ship_id)
        return f"BuildPort {who} at {_xy(action.target)}"
    if isinstance(action, CapturePortAction):
        ship = engine.ships.get(action.landing_ship_id)
        who = _own_label(ship, action.landing_ship_id)
        return f"Capture {who} port={_xy(action.target)}"
    if isinstance(action, MerchantLoadAction):
        ship = engine.ships.get(action.merchant_id)
        port = engine.ports.get(action.port)
        cargo = ship.cargo if ship else 0
        stock = port.stockpile if port else 0
        return (
            f"Load M#{action.merchant_id} port={_xy(action.port)} "
            f"cargo={cargo} stock={stock}"
        )
    if isinstance(action, MerchantUnloadAction):
        ship = engine.ships.get(action.merchant_id)
        cargo = ship.cargo if ship else 0
        return f"Unload M#{action.merchant_id} port={_xy(action.port)} cargo={cargo}"
    if isinstance(action, EndTurnAction):
        return "End"
    return type(action).__name__


def action_record(action: Action) -> dict:
    """JSON-ready action, parallel to the menu order."""
    if isinstance(action, MoveAction):
        return {"type": "move", "ship": action.ship_id, "to": list(action.destination)}
    if isinstance(action, AttackAction):
        return {
            "type": "attack",
            "ship": action.attacker_id,
            "target": action.target_id,
        }
    if isinstance(action, BuildShipAction):
        return {
            "type": "build_ship",
            "port": list(action.port),
            "spawn": list(action.spawn_position),
            "ship_type": action.ship_type.value,
        }
    if isinstance(action, BuildPortAction):
        return {
            "type": "build_port",
            "ship": action.builder_ship_id,
            "at": list(action.target),
        }
    if isinstance(action, CapturePortAction):
        return {
            "type": "capture",
            "ship": action.landing_ship_id,
            "port": list(action.target),
        }
    if isinstance(action, MerchantLoadAction):
        return {"type": "load", "ship": action.merchant_id, "port": list(action.port)}
    if isinstance(action, MerchantUnloadAction):
        return {"type": "unload", "ship": action.merchant_id, "port": list(action.port)}
    if isinstance(action, EndTurnAction):
        return {"type": "end"}
    return {"type": type(action).__name__}


def _entities(engine: gameEngine, player_id: int) -> List[dict]:
    player = engine.players[player_id]
    out: List[dict] = []
    for sid in sorted(player.owned_ship_ids):
        ship = engine.ships[sid]
        out.append(
            {
                "id": ship_id(ship.id),
                "owner": "me",
                "type": ship.type.value,
                "pos": list(ship.position),
                "hp": _whole(ship.stats.strength),
                "cargo": ship.cargo,
                "speed": ship.stats.speed,
                "range": ship.stats.attack_range,
                "moved": ship.tiles_moved_this_turn,
                "attacked": ship.has_attacked,
            }
        )
    for pos in sorted(player.owned_port_positions):
        port = engine.ports[pos]
        out.append(
            {
                "id": port_id(pos),
                "kind": "port",
                "owner": "me",
                "pos": list(pos),
                "home": port.is_home or pos == player.home_port,
                "stock": port.stockpile,
            }
        )
    for sight in sorted(player.sightings.values(), key=lambda s: s.ship_id):
        stats = BASE_STATS[sight.type]
        out.append(
            {
                "id": ship_id(sight.ship_id),
                "owner": "enemy",
                "type": sight.type.value,
                "pos": list(sight.position),
                "hp": _whole(stats.strength),
                "speed": stats.speed,
                "range": stats.attack_range,
                "fresh": sight.fresh,
                "seen": sight.turn_seen,
            }
        )
    for sight in sorted(player.port_sightings.values(), key=lambda s: s.position):
        out.append(
            {
                "id": port_id(sight.position),
                "kind": "port",
                "owner": "enemy",
                "pos": list(sight.position),
                "home": sight.is_home,
                "fresh": sight.fresh,
                "seen": sight.turn_seen,
            }
        )
    out.append({"kind": "gold", "owner": "me", "amount": player.cash})
    out.append({"kind": "turn", "n": engine.turn})
    return out


def _facts(engine: gameEngine, player_id: int) -> List[dict]:
    player = engine.players[player_id]
    sightings = list(player.sightings.values())
    stocked = [
        pos
        for pos in player.owned_port_positions
        if pos != player.home_port and engine.ports[pos].stockpile > 0
    ]
    facts = []
    for sid in sorted(player.owned_ship_ids):
        ship = engine.ships[sid]
        remaining = _remaining(ship)
        blocked = _known_blocked(engine, player_id, ship.position)
        home_dist = _bfs_to_adjacent(engine, ship.position, player.home_port, blocked)
        cargo_dist = _nearest_adjacent(engine, ship.position, stocked, blocked)
        in_range = []
        for sight in sightings:
            stats = BASE_STATS[sight.type]
            if _in_range(ship.stats, stats.visibility, ship.position, sight.position):
                in_range.append(ship_id(sight.ship_id))
        in_range.sort()
        facts.append(
            {
                "id": ship_id(ship.id),
                "home_turns": _turns_to_act(
                    home_dist, remaining, ship.stats.speed, True
                ),
                "cargo_turns": _turns_to_act(
                    cargo_dist, remaining, ship.stats.speed, True
                ),
                "in_range": in_range,
            }
        )
    return facts


def _turns_to_fire(
    engine: gameEngine,
    start: Pos,
    stats: ShipStats,
    remaining: int,
    *,
    act_now: bool,
    target_pos: Pos,
    target_vis: float,
    blocked: Set[Pos],
) -> Optional[int]:
    if stats.attack_range <= 0:
        return None
    goals = _firing_tiles(engine, stats, target_vis, target_pos, blocked, start)
    dist = _bfs_dist(engine, start, goals, blocked)
    return _turns_to_act(dist, remaining, stats.speed, act_now)


def _turns_to_adjacent(
    engine: gameEngine,
    start: Pos,
    remaining: int,
    speed: int,
    target: Pos,
    blocked: Set[Pos],
) -> Optional[int]:
    dist = _bfs_to_adjacent(engine, start, target, blocked)
    return _turns_to_act(dist, remaining, speed, True)


def _firing_tiles(
    engine: gameEngine,
    stats: ShipStats,
    target_vis: float,
    target_pos: Pos,
    blocked: Set[Pos],
    start: Pos,
) -> Set[Pos]:
    detection = stats.scouting * target_vis
    goals: Set[Pos] = set()
    for pos in engine.map.tiles_within(target_pos, stats.attack_range):
        if engine.map.manhattan(pos, target_pos) > detection:
            continue
        if pos == start or (engine.map.is_water(pos) and pos not in blocked):
            goals.add(pos)
    return goals


def _bfs_to_adjacent(
    engine: gameEngine, start: Pos, target: Pos, blocked: Set[Pos]
) -> Optional[int]:
    return _bfs_dist(
        engine, start, _adjacent_water(engine, target, blocked, start), blocked
    )


def _nearest_adjacent(
    engine: gameEngine, start: Pos, targets: Iterable[Pos], blocked: Set[Pos]
) -> Optional[int]:
    best: Optional[int] = None
    for target in targets:
        dist = _bfs_to_adjacent(engine, start, target, blocked)
        if dist is None:
            continue
        if best is None or dist < best:
            best = dist
    return best


def _adjacent_water(
    engine: gameEngine, target: Pos, blocked: Set[Pos], start: Pos
) -> Set[Pos]:
    goals: Set[Pos] = set()
    for dx, dy in _CARDINAL:
        nxt = (target[0] + dx, target[1] + dy)
        if not engine.map.in_bounds(nxt) or not engine.map.is_water(nxt):
            continue
        if nxt == start or nxt not in blocked:
            goals.add(nxt)
    return goals


def _bfs_dist(
    engine: gameEngine, start: Pos, goals: Set[Pos], blocked: Set[Pos]
) -> Optional[int]:
    """Water-only 4-connected distance. ``start`` may be a goal (distance 0)."""
    if not goals:
        return None
    if start in goals:
        return 0
    seen = {start}
    queue: deque[Tuple[Pos, int]] = deque([(start, 0)])
    mp = engine.map
    while queue:
        pos, dist = queue.popleft()
        for dx, dy in _CARDINAL:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or nxt in blocked:
                continue
            if not mp.in_bounds(nxt) or not mp.is_water(nxt):
                continue
            seen.add(nxt)
            nd = dist + 1
            if nxt in goals:
                return nd
            queue.append((nxt, nd))
    return None


def _turns_to_act(
    dist: Optional[int], remaining: int, speed: int, act_now: bool
) -> Optional[int]:
    """Turns until the ship can be on the goal tile and act.

    0 means this turn (it can still spend ``remaining`` movement, then act).
    Later turns get a full ``speed`` after EndTurn. ``act_now`` is false when
    this ship has already attacked.
    """
    if dist is None:
        return None
    remaining = max(0, remaining)
    if dist <= remaining:
        return 0 if act_now else 1
    if speed <= 0:
        return None
    left = dist - remaining
    return (left + speed - 1) // speed


def _known_blocked(engine: gameEngine, player_id: int, mover_pos: Pos) -> Set[Pos]:
    """Tiles we know a ship occupies, excluding the mover's own tile.

    Fresh sightings are really there. Stale sightings are not blocked: the
    ship may have left, and blocking on a guess would invent a rule the
    engine does not apply. Hidden ships are not blocked either.
    """
    player = engine.players[player_id]
    blocked: Set[Pos] = set()
    for sid in player.owned_ship_ids:
        pos = engine.ships[sid].position
        if pos != mover_pos:
            blocked.add(pos)
    for sight in player.sightings.values():
        if sight.fresh and sight.position != mover_pos:
            blocked.add(sight.position)
    return blocked


def _in_range(stats: ShipStats, target_vis: float, origin: Pos, target: Pos) -> bool:
    if stats.attack_range <= 0:
        return False
    dist = _manhattan(origin, target)
    if dist > stats.attack_range:
        return False
    return dist <= stats.scouting * target_vis


def _remaining(ship: Ship) -> int:
    return max(0, ship.stats.speed - ship.tiles_moved_this_turn)


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _own_label(ship: Optional[Ship], sid: int) -> str:
    if ship is None:
        return f"#{sid}"
    return f"{_LETTER[ship.type]}#{ship.id}"


def _xy(pos: Optional[Pos]) -> str:
    if pos is None:
        return "?"
    return f"({pos[0]},{pos[1]})"


def _compact(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"))
