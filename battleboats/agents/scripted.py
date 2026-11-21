"""Full-information scripted baseline.

Priority: capture, attack, unload, load, build a port, buy the ship the fleet
is missing, then a move that shortens the relevant Manhattan distance. If
nothing improves, end the turn.

This opponent reads ``engine`` directly (both fleets, both homes). It is a
yardstick, not a fog-of-war agent. The play UI labels it as full information.
Swap the body later if a fog-limited script is wanted; callers only need
``scripted_action(engine, pid, rng)``.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

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
from battleboats.core.shipyard.ship_type import ShipType

_BUY_ORDER = (
    ShipType.MERCHANT,
    ShipType.LANDING,
    ShipType.DESTROYER,
    ShipType.SUBMARINE,
    ShipType.CRUISER,
)


def scripted_action(engine, player_id: int, rng: random.Random) -> Action:
    """Return one legal action. Always legal when ``enumerate_legal`` is non-empty."""
    legal = engine.enumerate_legal(player_id)
    if not legal:
        raise RuntimeError(f"scripted_action: no legal actions for player {player_id}")

    def pick(cls) -> List[Action]:
        return [a for a in legal if isinstance(a, cls)]

    for cls in (
        CapturePortAction,
        AttackAction,
        MerchantUnloadAction,
        MerchantLoadAction,
        BuildPortAction,
    ):
        options = pick(cls)
        if options:
            return rng.choice(options)

    build = _choose_build(engine, player_id, pick(BuildShipAction))
    if build is not None:
        return build

    move = _choose_move(engine, player_id, pick(MoveAction))
    if move is not None:
        return move

    ends = pick(EndTurnAction)
    if ends:
        return ends[0]
    return legal[-1]


def _counts(engine, player_id: int) -> dict:
    counts = {t: 0 for t in ShipType}
    for sid in engine.players[player_id].owned_ship_ids:
        ship = engine.ships.get(sid)
        if ship is not None:
            counts[ship.type] += 1
    return counts


def _choose_build(engine, player_id: int, options: Sequence[Action]) -> Optional[Action]:
    if not options:
        return None
    counts = _counts(engine, player_id)
    combat = sum(counts[t] for t in (
        ShipType.CARRIER, ShipType.BATTLESHIP, ShipType.CRUISER,
        ShipType.DESTROYER, ShipType.SUBMARINE,
    ))
    wanted = None
    if counts[ShipType.MERCHANT] < 1:
        wanted = ShipType.MERCHANT
    elif counts[ShipType.LANDING] < 1:
        wanted = ShipType.LANDING
    elif combat < 2:
        wanted = ShipType.DESTROYER
    else:
        return None
    matching = [a for a in options if isinstance(a, BuildShipAction) and a.ship_type is wanted]
    if matching:
        return matching[0]
    for ship_type in _BUY_ORDER:
        matching = [a for a in options if isinstance(a, BuildShipAction) and a.ship_type is ship_type]
        if matching:
            return matching[0]
    return None


def _objective_distance(engine, player_id: int, ship, pos) -> Optional[int]:
    """Manhattan to the tile this ship should be closing on, or None if it has no goal."""
    me = engine.players[player_id]
    opp = engine.players[1 - player_id]
    man = engine.map.manhattan
    if ship.type is ShipType.LANDING:
        return man(pos, opp.home_port)
    if ship.type is ShipType.MERCHANT:
        if ship.cargo > 0:
            return man(pos, me.home_port)
        ports = [p for p in me.owned_port_positions if p != me.home_port]
        if not ports:
            return None
        return min(man(pos, p) for p in ports)
    if ship.type is ShipType.BUILDER:
        empties = []
        m = engine.map
        for x in range(m.width):
            for y in range(m.height):
                tile = (x, y)
                if m.is_land(tile) and not m.is_port(tile):
                    empties.append(tile)
        if not empties:
            return None
        return min(man(pos, t) for t in empties)
    if ship.stats.attack_range > 0:
        foes = [engine.ships[s].position for s in opp.owned_ship_ids if s in engine.ships]
        if foes:
            return min(man(pos, p) for p in foes)
        return man(pos, opp.home_port)
    return None


def _choose_move(engine, player_id: int, options: Sequence[Action]) -> Optional[Action]:
    """Move that strictly reduces some ship's objective distance. Else None (end the turn)."""
    best = None
    best_gain = 0
    for action in options:
        if not isinstance(action, MoveAction):
            continue
        ship = engine.ships.get(action.ship_id)
        if ship is None:
            continue
        before = _objective_distance(engine, player_id, ship, ship.position)
        after = _objective_distance(engine, player_id, ship, action.destination)
        if before is None or after is None:
            continue
        gain = before - after
        if gain > best_gain:
            best = action
            best_gain = gain
    return best
