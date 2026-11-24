"""Text view of one seat's decision.

The board text is the same fog-filtered set the RL encoder sees: friendly
ships and ports, enemy ship and port sightings, and coastline land tokens
(``battleboats.envs.observation.build_entity_refs``). Values are the raw
token fields, not the normalized floats. The legal MOVES menu is unchanged.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from battleboats.core.actions import (Action, AttackAction, BuildPortAction,
                                      BuildShipAction, CapturePortAction,
                                      EndTurnAction, MerchantLoadAction,
                                      MerchantUnloadAction, MoveAction)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship import Ship
from battleboats.core.shipyard.ship_data import BASE_STATS, PORT, spot_distance
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import build_entity_refs

FORMAT_VERSION = 4

# Byte-stable prefix of every prompt. llama-server cache_prompt reuses it.
# Nothing in here may depend on the seat, the board, or the turn.
_STATIC_HEAD = (
    "FORMAT 4\n"
    "Reply with one MOVES number then one short sentence.\n"
    "One line per ship and per port. Land is tokens, one row y: x or x-x. "
    "board WxH is stated once. Positions are cells.\n"
    "Ship: id owner type (x,y) speed strength range cost "
    "cargo moved attacked fresh stale.\n"
    "Port: id owner (x,y) home stock cash fresh stale. "
    "range=attack_range.\n"
    "Always on a ship: speed strength range cost. "
    "Omitted when zero: cargo, moved, attacked, home, stock, cash. "
    "A fresh sighting omits fresh and stale.\n"
    "Builds: B@(x,y): n Type cost | n Type cost.\n"
)

Pos = Tuple[int, int]

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

_SHIP_OPTIONAL = ("cargo", "moved", "attacked", "fresh", "stale")
_PORT_OPTIONAL = ("home", "stock", "cash", "fresh", "stale")


def _whole(value: float) -> int | float:
    """CSV stats are floats; whole numbers read better as ints in the prompt."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _spot_table() -> str:
    """Spotting matrix from config/visibilityMatrix.csv, one row per spotter.

    Static (no seat, board, or turn), so it stays in the cached prefix.
    """
    keys = list(ShipType) + [PORT]
    letter = lambda k: "P" if k == PORT else _LETTER[k]
    lines = [
        "Spotting: the row unit sees the column unit within N tiles (Manhattan). "
        "Not symmetric. Letters: " + " ".join(
            f"{letter(k)}={'Port' if k == PORT else k.value}" for k in keys
        ) + ".",
        "   " + " ".join(f"{letter(k):>2}" for k in keys),
    ]
    for a in keys:
        lines.append(
            f"{letter(a)}: " + " ".join(f"{_whole(spot_distance(a, b))!s:>2}" for b in keys)
        )
    return "\n".join(lines) + "\n"


# Byte-stable prefix of every prompt (see _STATIC_HEAD).
STATIC_PREFIX = _STATIC_HEAD + _spot_table()


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
    """Fog-filtered token view. ``render_prompt`` prints this dict.

    ``horizon`` and ``max_edges`` are accepted so existing callers keep
    working. Path and threat facts are not part of the RL tokens, so they
    are not part of the prompt.
    """
    _ = horizon, max_edges
    player = engine.players[player_id]
    ships, ports, land = _token_entities(engine, player_id)
    return {
        "format_version": FORMAT_VERSION,
        "seat": f"player_{player_id}",
        "seat_id": player_id,
        "turn": engine.turn,
        "gold": player.cash,
        "board": [engine.map.width, engine.map.height],
        "ships": ships,
        "ports": ports,
        "land": land,
        "moves": [describe_action(action, engine, player_id) for action in legal],
        "legal": [action_record(action) for action in legal],
    }


def render_prompt(view: dict) -> str:
    """Exact user message. ``STATIC_PREFIX`` is identical on every call."""
    width, height = view["board"]
    lines = [
        STATIC_PREFIX.rstrip("\n"),
        (
            f"seat {view['seat']} board {width}x{height} "
            f"turn {view['turn']} gold {view['gold']}"
        ),
        "SHIPS",
    ]
    for ent in view["ships"]:
        lines.append(_ship_line(ent))
    lines.append("PORTS")
    for ent in view["ports"]:
        lines.append(_port_line(ent))
    lines.append("LAND")
    lines.extend(_land_lines(view["land"]))
    lines.append("MOVES")
    lines.extend(menu_lines(view))
    return "\n".join(lines)


def menu_lines(view: dict) -> List[str]:
    """Numbered menu. Builds at one port share a line; other actions stay one each.

    Numbers are indexes into ``view["legal"]``. Types the seat cannot afford
    are absent, because they are absent from the legal list.
    """
    legal = view["legal"]
    moves = view["moves"]
    lines: List[str] = []
    i = 0
    n = len(legal)
    while i < n:
        if legal[i].get("type") == "build_ship":
            port = tuple(legal[i]["port"])
            bits = []
            while (
                i < n
                and legal[i].get("type") == "build_ship"
                and tuple(legal[i]["port"]) == port
            ):
                cost = BASE_STATS[ShipType(legal[i]["ship_type"])].cost
                bits.append(f"{i} {legal[i]['ship_type']} {cost}")
                i += 1
            lines.append(f"B@({port[0]},{port[1]}): " + " | ".join(bits))
        else:
            lines.append(f"{i} {moves[i]}")
            i += 1
    return lines


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


def _token_entities(engine: gameEngine, player_id: int):
    """Ships, ports, and coastline cells in RL token order."""
    ships: List[dict] = []
    ports: List[dict] = []
    land: List[List[int]] = []
    ship_sights = {s.ship_id: s for s in engine.known_enemy_ships(player_id)}
    port_sights = {s.position: s for s in engine.known_enemy_ports(player_id)}
    for kind, ident in build_entity_refs(engine, player_id):
        if kind == "friendly_ship":
            ships.append(_friendly_ship(engine.ships[ident]))
        elif kind == "enemy_ship":
            ships.append(_enemy_ship(ship_sights[ident], engine.turn))
        elif kind == "friendly_port":
            ports.append(_friendly_port(engine.ports[ident], engine))
        elif kind == "enemy_port":
            ports.append(_enemy_port(port_sights[ident], engine.turn))
        elif kind == "coastline":
            land.append([int(ident[0]), int(ident[1])])
    return ships, ports, land


def _stat_fields(stats: ShipStats) -> dict:
    return {
        "speed": _whole(stats.speed),
        "strength": _whole(stats.strength),
        "range": _whole(stats.attack_range),
        "cost": _whole(stats.cost),
    }


def _friendly_ship(ship: Ship) -> dict:
    ent = {
        "id": ship_id(ship.id),
        "owner": "me",
        "type": ship.type.value,
        "pos": list(ship.position),
        **_stat_fields(ship.stats),
    }
    if ship.cargo:
        ent["cargo"] = ship.cargo
    if ship.tiles_moved_this_turn:
        ent["moved"] = ship.tiles_moved_this_turn
    if ship.has_attacked:
        ent["attacked"] = 1
    return ent


def _enemy_ship(sight, turn: int) -> dict:
    ent = {
        "id": ship_id(sight.ship_id),
        "owner": "enemy",
        "type": sight.type.value,
        "pos": list(sight.position),
        **_stat_fields(BASE_STATS[sight.type]),
    }
    if not sight.fresh:
        ent["fresh"] = 0
        ent["stale"] = turn - sight.turn_seen
    return ent


def _friendly_port(port, engine: gameEngine) -> dict:
    ent = {
        "id": port_id(port.position),
        "owner": "me",
        "pos": list(port.position),
    }
    if port.is_home:
        ent["home"] = 1
        if engine.players[port.owner].cash:
            ent["cash"] = engine.players[port.owner].cash
    if port.stockpile:
        ent["stock"] = port.stockpile
    return ent


def _enemy_port(sight, turn: int) -> dict:
    ent = {
        "id": port_id(sight.position),
        "owner": "enemy",
        "pos": list(sight.position),
    }
    if sight.is_home:
        ent["home"] = 1
    if not sight.fresh:
        ent["fresh"] = 0
        ent["stale"] = turn - sight.turn_seen
    return ent


def _ship_line(ent: dict) -> str:
    parts = [
        ent["id"],
        ent["owner"],
        ent["type"],
        _xy(tuple(ent["pos"])),
        f"speed={ent['speed']}",
        f"strength={ent['strength']}",
        f"range={ent['range']}",
        f"cost={ent['cost']}",
    ]
    for key in _SHIP_OPTIONAL:
        if key in ent:
            parts.append(f"{key}={ent[key]}")
    return " ".join(parts)


def _port_line(ent: dict) -> str:
    parts = [ent["id"], ent["owner"], _xy(tuple(ent["pos"]))]
    for key in _PORT_OPTIONAL:
        if key in ent:
            parts.append(f"{key}={ent[key]}")
    return " ".join(parts)


def _land_lines(positions: Iterable) -> List[str]:
    """Horizontal runs of coastline tokens. Same cells, fewer characters."""
    by_y: dict[int, List[int]] = {}
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        by_y.setdefault(y, []).append(x)
    lines: List[str] = []
    for y in sorted(by_y):
        xs = sorted(set(by_y[y]))
        spans: List[str] = []
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
                continue
            spans.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = x
        spans.append(str(start) if start == prev else f"{start}-{prev}")
        lines.append(f"{y}: {','.join(spans)}")
    return lines


def _own_label(ship: Optional[Ship], sid: int) -> str:
    if ship is None:
        return f"#{sid}"
    return f"{_LETTER[ship.type]}#{ship.id}"


def _xy(pos: Optional[Pos]) -> str:
    if pos is None:
        return "?"
    return f"({pos[0]},{pos[1]})"


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
