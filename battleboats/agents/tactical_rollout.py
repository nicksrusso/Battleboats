"""Short tactical rollouts used at god-mode MCTS leaves.

This is a policy, not a state evaluator: it never scores a position. Landing
ships head for the nearest enemy home (else nearest enemy port) and capture
when legal. Combat ships attack threatening landings when in range, otherwise
chase the landing closest to our home. Every other unit ends the turn.

Rollouts are 20–40 engine steps by default. A terminal during the rollout is
scored exactly ±1 / 0 by the caller; otherwise the value net is applied at
the rollout end.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from battleboats.agents.rust_mcts_engine import (
    current_player_of,
    decode_action,
    encode_action,
    uses_compact_actions,
)
from battleboats.core.combat import COMBAT_TYPES
from battleboats.core.shipyard.ship_type import ShipType

END_TURN = ["e"]


def _pos(value: Any) -> Tuple[int, int]:
    return (int(value[0]), int(value[1]))


def _as_type(value: Any) -> ShipType:
    return value if isinstance(value, ShipType) else ShipType(value)


def _norm_ship(row: Any) -> dict:
    if isinstance(row, dict):
        typ = _as_type(row["type"])
        return {
            "id": int(row["id"]),
            "type": typ,
            "owner": int(row["owner"]),
            "position": _pos(row["position"]),
            "has_attacked": bool(row.get("has_attacked", False)),
        }
    typ = getattr(row, "type", None)
    if typ is None:
        typ = row.ship_type
    return {
        "id": int(row.id),
        "type": _as_type(typ),
        "owner": int(row.owner),
        "position": _pos(row.position),
        "has_attacked": bool(getattr(row, "has_attacked", False)),
    }


def _norm_port(row: Any, position: Optional[Any] = None) -> dict:
    if isinstance(row, dict):
        pos = row.get("position", position)
        return {
            "position": _pos(pos),
            "owner": int(row["owner"]),
            "is_home": bool(row.get("is_home", False)),
        }
    pos = getattr(row, "position", position)
    return {
        "position": _pos(pos),
        "owner": int(row.owner),
        "is_home": bool(getattr(row, "is_home", False)),
    }


def iter_ships(engine: Any) -> List[dict]:
    lister = getattr(engine, "list_ships", None)
    if callable(lister):
        return [_norm_ship(row) for row in lister()]
    ships = getattr(engine, "ships", None)
    if callable(ships):
        return [_norm_ship(row) for row in ships()]
    if ships is None:
        return []
    if isinstance(ships, dict):
        return [_norm_ship(row) for row in ships.values()]
    return [_norm_ship(row) for row in ships]


def iter_ports(engine: Any) -> List[dict]:
    lister = getattr(engine, "list_ports", None)
    if callable(lister):
        return [_norm_port(row) for row in lister()]
    ports = getattr(engine, "ports", None)
    if callable(ports):
        return [_norm_port(row) for row in ports()]
    if ports is None:
        return []
    if isinstance(ports, dict):
        return [_norm_port(row, position=pos) for pos, row in ports.items()]
    return [_norm_port(row) for row in ports]


def legal_encoded(engine: Any, player_id: int) -> List[list]:
    acts = engine.enumerate_legal(player_id)
    if not acts:
        return []
    first = acts[0]
    if isinstance(first, (list, tuple)):
        return [list(a) for a in acts]
    return [encode_action(a) for a in acts]


def step_encoded(engine: Any, action: Sequence[Any]) -> None:
    rec = list(action)
    if uses_compact_actions(engine):
        engine.step(rec)
        return
    engine.step(decode_action(rec))


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _enemy_targets(ports: Iterable[dict], me: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    homes: List[Tuple[int, int]] = []
    other: List[Tuple[int, int]] = []
    for port in ports:
        if int(port["owner"]) == me:
            continue
        if port["is_home"]:
            homes.append(port["position"])
        else:
            other.append(port["position"])
    return homes, other


def _nearest(origin: Tuple[int, int], goals: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not goals:
        return None
    return min(goals, key=lambda g: (_manhattan(origin, g), g))


def _our_home(ports: Iterable[dict], me: int) -> Optional[Tuple[int, int]]:
    for port in ports:
        if int(port["owner"]) == me and port["is_home"]:
            return port["position"]
    return None


def _best_approach(
    moves: Sequence[list],
    ship_pos: dict,
    goal: Tuple[int, int],
) -> Optional[list]:
    """Pick a legal move that strictly reduces Manhattan distance to ``goal``."""
    best: Optional[list] = None
    best_key = None
    for rec in moves:
        dest = (int(rec[2]), int(rec[3]))
        new_d = _manhattan(dest, goal)
        old_d = _manhattan(ship_pos[int(rec[1])], goal)
        if new_d >= old_d:
            continue
        key = (new_d, int(rec[1]), dest)
        if best_key is None or key < best_key:
            best = rec
            best_key = key
    return best


def tactical_action(engine: Any, player_id: Optional[int] = None) -> list:
    """One compact action for the current seat under the leaf rollout policy."""
    me = int(player_id) if player_id is not None else current_player_of(engine)
    legal = legal_encoded(engine, me)
    if not legal:
        return list(END_TURN)
    if legal == [END_TURN] or (len(legal) == 1 and legal[0] == END_TURN):
        return list(END_TURN)

    ships = iter_ships(engine)
    ports = iter_ports(engine)
    by_id = {s["id"]: s for s in ships}
    pos = {s["id"]: s["position"] for s in ships}
    own = {s["id"]: s for s in ships if s["owner"] == me}
    enemy_landings = [s for s in ships if s["owner"] != me and s["type"] is ShipType.LANDING]
    homes, other_enemy_ports = _enemy_targets(ports, me)
    capture_goals = homes + other_enemy_ports
    our_home = _our_home(ports, me)

    captures_home: List[list] = []
    captures_other: List[list] = []
    attacks: List[list] = []
    landing_moves: List[list] = []
    combat_moves: List[list] = []
    landing_ids = {sid for sid, s in own.items() if s["type"] is ShipType.LANDING}
    combat_ids = {sid for sid, s in own.items() if s["type"] in COMBAT_TYPES}

    for rec in legal:
        tag = rec[0]
        if tag == "c":
            target = (int(rec[2]), int(rec[3]))
            if target in homes:
                captures_home.append(rec)
            elif target in other_enemy_ports or target in capture_goals:
                captures_other.append(rec)
        elif tag == "a":
            attacker = int(rec[1])
            target_id = int(rec[2])
            if attacker in combat_ids and any(s["id"] == target_id for s in enemy_landings):
                attacks.append(rec)
        elif tag == "m":
            sid = int(rec[1])
            if sid in landing_ids:
                landing_moves.append(rec)
            elif sid in combat_ids:
                combat_moves.append(rec)

    def _capture_key(rec: list) -> tuple:
        return (int(rec[1]), int(rec[2]), int(rec[3]))

    if captures_home:
        return min(captures_home, key=_capture_key)
    if captures_other:
        return min(captures_other, key=_capture_key)

    if attacks:
        def _attack_key(rec: list) -> tuple:
            target = by_id.get(int(rec[2]))
            if target is None or our_home is None:
                threat = 0
            else:
                threat = _manhattan(target["position"], our_home)
            return (threat, int(rec[1]), int(rec[2]))

        return min(attacks, key=_attack_key)

    if landing_moves:
        # Each landing heads for the nearest enemy home, else nearest enemy port.
        best: Optional[list] = None
        best_key = None
        for rec in landing_moves:
            sid = int(rec[1])
            here = pos[sid]
            goal = _nearest(here, homes) or _nearest(here, other_enemy_ports)
            if goal is None:
                continue
            dest = (int(rec[2]), int(rec[3]))
            new_d = _manhattan(dest, goal)
            old_d = _manhattan(here, goal)
            if new_d >= old_d:
                continue
            key = (new_d, sid, dest)
            if best_key is None or key < best_key:
                best = rec
                best_key = key
        if best is not None:
            return best

    if combat_moves and enemy_landings:
        def _threat(landing: dict) -> tuple:
            if our_home is None:
                return (0, landing["id"])
            return (_manhattan(landing["position"], our_home), landing["id"])

        threat_order = sorted(enemy_landings, key=_threat)
        for landing in threat_order:
            goal = landing["position"]
            chosen = _best_approach(combat_moves, pos, goal)
            if chosen is not None:
                return chosen

    return list(END_TURN)


def apply_tactical_rollout(engine: Any, steps: int) -> int:
    """Apply up to ``steps`` tactical actions. Returns the number of steps taken.

    Mutates ``engine``. Stops early on a terminal position.
    """
    taken = 0
    budget = max(0, int(steps))
    while taken < budget and not engine.is_terminal():
        step_encoded(engine, tactical_action(engine))
        taken += 1
    return taken
