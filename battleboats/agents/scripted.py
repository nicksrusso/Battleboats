"""Full-information scripted baseline.

Priority: capture, attack, unload, load, build a port, buy the next ship in a
per-seat randomized build queue, then a move that shortens the relevant
Manhattan distance. If nothing improves, end the turn.

The queue is a shuffled cycle of every buyable ship type, stored on the
engine and refilled when it runs out. The head is bought only when it is
legal and affordable this turn; otherwise the agent saves cash and does
not skip ahead to a cheaper type. Draws use the caller's RNG so a game
is deterministic given the seed.

This opponent reads ``engine`` directly (both fleets, both homes). It is a
yardstick, not a fog-of-war agent. The play UI labels it as full information.
Swap the body later if a fog-limited script is wanted; callers only need
``scripted_action(engine, pid, rng)``.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional, Sequence

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

# Every type the rules allow to be bought. A cycle of this tuple is shuffled
# each time a seat's queue is empty, so long-run buy rates stay nearly equal.
BUYABLE_TYPES: tuple = tuple(t for t in ShipType if t in BASE_STATS)


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

    build = _choose_build(engine, player_id, pick(BuildShipAction), rng)
    if build is not None:
        return build

    move = _choose_move(engine, player_id, pick(MoveAction))
    if move is not None:
        return move

    ends = pick(EndTurnAction)
    if ends:
        return ends[0]
    return legal[-1]


def sample_build_cycle(rng: random.Random) -> List[ShipType]:
    """One shuffled permutation of every buyable type."""
    cycle = list(BUYABLE_TYPES)
    rng.shuffle(cycle)
    return cycle


def build_queue_for(engine, player_id: int, rng: Optional[random.Random] = None) -> Deque[ShipType]:
    """Per-game, per-player buy queue stored on the engine.

    Reset is detected via ``id(engine.players[0])``, which changes both
    when a new ``gameEngine`` is constructed and when ``engine.reset()``
    rebuilds the player list. An empty queue is filled from ``rng`` when
    one is given; tests may also seed or replace the deque themselves.
    """
    marker = id(engine.players[0])
    state = getattr(engine, "_scripted_buy_state", None)
    if state is None or state["marker"] != marker:
        state = {"marker": marker, "queues": {0: deque(), 1: deque()}}
        engine._scripted_buy_state = state
    queue = state["queues"][player_id]
    if not queue and rng is not None:
        queue.extend(sample_build_cycle(rng))
    return queue


def _choose_build(
    engine,
    player_id: int,
    options: Sequence[Action],
    rng: random.Random,
) -> Optional[Action]:
    if not options:
        return None
    queue = build_queue_for(engine, player_id, rng)
    if not queue:
        return None
    wanted = queue[0]
    matching = [a for a in options if isinstance(a, BuildShipAction) and a.ship_type is wanted]
    if not matching:
        return None
    queue.popleft()
    if not queue:
        queue.extend(sample_build_cycle(rng))
    return matching[0]


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
