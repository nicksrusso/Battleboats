"""Two engine rules (Nick, 2026-11-23):

(a) A ship bought this turn takes no actions that turn (move, attack, load,
    unload, capture, port build). It acts from its owner's next turn. Ships
    placed by a scenario / fleet start / sandbox (_spawn_ship) are not bought.
(b) A port captured or built this turn can't build ships until its owner's
    next turn.
"""

import json
from collections import deque

import pytest

from battleboats.core.actions import (
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.map_pool import earliest_capture_turn


def _engine(tmp_path, w, h, ports, land=None, name="m.json"):
    land = land if land is not None else [p for ps in ports.values() for p in ps]
    path = tmp_path / name
    path.write_text(json.dumps({"width": w, "height": h, "seed": 0, "land": land, "ports": ports}))
    e = gameEngine(str(path))
    e.reset(seed=0)
    return e


def _ship_actions(engine, pid, sid):
    out = []
    for a in engine.enumerate_legal(pid):
        for f in ("ship_id", "attacker_id", "landing_ship_id", "merchant_id", "builder_ship_id"):
            if getattr(a, f, None) == sid:
                out.append(a)
    return out


def _end_round(engine):
    engine.step(EndTurnAction())
    engine.step(EndTurnAction())


def _build(engine, port, hull):
    a = next(a for a in engine.enumerate_legal(engine.current_player)
             if isinstance(a, BuildShipAction) and a.port == port and a.ship_type == hull)
    before = set(engine.ships)
    engine.step(a)
    (sid,) = set(engine.ships) - before
    return engine.ships[sid]


@pytest.mark.parametrize("hull", list(ShipType))
def test_bought_ship_has_no_actions_until_next_turn(tmp_path, hull):
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2]]})
    e.players[0].cash = 1000
    ship = _build(e, (1, 1), hull)
    assert ship.bought_this_turn and ship.built_turn == 0
    # Obs features read "spent": moved full speed, attacked.
    assert ship.tiles_moved_this_turn == ship.stats.speed and ship.has_attacked
    assert _ship_actions(e, 0, ship.id) == []
    masks = ActionMasks(e, 0)
    assert all(a not in masks._engine.enumerate_legal(0) for a in _ship_actions(e, 0, ship.id))
    # Direct steps are no-ops.
    x, y = ship.position
    for dest in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        e.step(MoveAction(ship.id, dest))
    assert ship.position == (x, y)
    _end_round(e)
    assert not ship.bought_this_turn and ship.tiles_moved_this_turn == 0 and not ship.has_attacked
    assert any(isinstance(a, MoveAction) for a in _ship_actions(e, 0, ship.id))


def test_bought_ship_cannot_attack_capture_load_unload_or_build_port(tmp_path):
    # P0 home (1,1), satellite (3,1) with stock, empty land (1,3). P1 home (5,1); P1 ship at (5,2).
    e = _engine(tmp_path, 8, 5, {"0": [[1, 1], [3, 1]], "1": [[5, 1]]}, land=[[1, 1], [3, 1], [5, 1], [1, 3]])
    e.players[0].cash = 5000
    e.ports[(3, 1)].stockpile = 300
    enemy = e._spawn_ship(1, ShipType.MERCHANT, (5, 2))
    # Spawn tiles are canonical (closest to the enemy home); force spots via direct steps.
    for hull, port, spawn in ((ShipType.LANDING, (3, 1), (4, 1)), (ShipType.MERCHANT, (3, 1), (3, 2)),
                              (ShipType.DESTROYER, (1, 1), (2, 1)), (ShipType.BUILDER, (1, 1), (1, 2))):
        e.step(BuildShipAction(port, spawn, hull))
    by_type = {s.type: s for s in e.ships.values() if s.owner == 0}
    land, merch, dd, bld = (by_type[t] for t in (ShipType.LANDING, ShipType.MERCHANT, ShipType.DESTROYER, ShipType.BUILDER))
    merch.cargo = 50
    e.step(CapturePortAction(land.id, (5, 1)))
    e.step(MerchantLoadAction(merch.id, (3, 1)))
    e.step(AttackAction(dd.id, enemy.id))
    e.step(BuildPortAction(bld.id, (1, 3)))
    assert e.winner is None and int(e.map.port_owner[(5, 1)]) == 1
    assert merch.cargo == 50 and e.ports[(3, 1)].stockpile == 300
    assert enemy.id in e.ships and not e.map.is_port((1, 3))
    for s in (land, merch, dd, bld):
        assert _ship_actions(e, 0, s.id) == []
    # Next own turn everything opens up again.
    _end_round(e)
    acts = {type(a) for s in (land, merch, dd, bld) for a in _ship_actions(e, 0, s.id)}
    assert {CapturePortAction, MerchantLoadAction, BuildPortAction, MoveAction} <= acts


def test_spawned_fleet_ships_are_not_bought(tmp_path):
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2]]})
    ship = e._spawn_ship(0, ShipType.LANDING, (2, 1))
    assert not ship.bought_this_turn
    assert any(isinstance(a, MoveAction) for a in _ship_actions(e, 0, ship.id))


def test_fleet_start_ships_act_on_turn_0(tmp_path):
    import random

    from battleboats.training.fleet_starts import FleetStarts, apply_fleet_start

    e = _engine(tmp_path, 16, 8, {"0": [[1, 3], [5, 6]], "1": [[14, 4], [10, 1]]})
    fleet = apply_fleet_start(e, FleetStarts(prob=1.0, budget=1200, landing_prob=1.0, radius=6), random.Random(0))
    assert fleet is not None
    mine = [s for s in e.ships.values() if s.owner == 0]
    assert mine and not any(s.bought_this_turn for s in mine)
    assert any(isinstance(a, MoveAction) and e.ships[a.ship_id].owner == 0 for a in e.enumerate_legal(0))


def test_captured_port_cannot_build_until_next_turn(tmp_path):
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2], [4, 1]]})
    landing = e._spawn_ship(0, ShipType.LANDING, (4, 2))
    e.step(CapturePortAction(landing.id, (4, 1)))
    assert int(e.map.port_owner[(4, 1)]) == 0 and e.ports[(4, 1)].locked_this_turn
    assert not any(isinstance(a, BuildShipAction) and a.port == (4, 1) for a in e.enumerate_legal(0))
    cash = e.players[0].cash
    e.step(BuildShipAction((4, 1), (5, 1), ShipType.LANDING))  # direct step is a no-op
    assert e.players[0].cash == cash and e.map.is_water((5, 1)) and not e.map.is_occupied((5, 1))
    _end_round(e)
    assert any(isinstance(a, BuildShipAction) and a.port == (4, 1) for a in e.enumerate_legal(0))


def test_recaptured_port_locks_for_the_new_owner(tmp_path):
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2], [4, 1]]})
    l0 = e._spawn_ship(0, ShipType.LANDING, (4, 2))
    l1 = e._spawn_ship(1, ShipType.LANDING, (4, 0))
    e.step(CapturePortAction(l0.id, (4, 1)))
    e.step(EndTurnAction())
    assert not e.ports[(4, 1)].locked_this_turn  # P0's turn ended
    e.step(CapturePortAction(l1.id, (4, 1)))
    assert int(e.map.port_owner[(4, 1)]) == 1 and e.ports[(4, 1)].locked_this_turn
    assert not any(isinstance(a, BuildShipAction) and a.port == (4, 1) for a in e.enumerate_legal(1))


def test_built_port_cannot_build_until_next_turn(tmp_path):
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2]]}, land=[[1, 1], [8, 2], [4, 1]])
    b = e._spawn_ship(0, ShipType.BUILDER, (4, 2))
    e.step(BuildPortAction(b.id, (4, 1)))
    assert e.ports[(4, 1)].locked_this_turn
    cash = e.players[0].cash
    e.step(BuildShipAction((4, 1), (5, 1), ShipType.DESTROYER))
    assert e.players[0].cash == cash
    _end_round(e)
    assert any(isinstance(a, BuildShipAction) and a.port == (4, 1) for a in e.enumerate_legal(0))


def _all_turn0_sequences_win(e, depth):
    """DFS over every legal P0 action sequence within one turn (no EndTurn).
    True if any of them sets winner=0."""
    if e.winner == 0:
        return True
    if depth == 0:
        return False
    for a in e.enumerate_legal(0):
        if isinstance(a, EndTurnAction):
            continue
        c = e.clone()
        c.step(a)
        if _all_turn0_sequences_win(c, depth - 1):
            return True
    return False


def test_forward_start_chain_cannot_win_on_turn_0(tmp_path):
    """Last report's chain: a forward landing captures the enemy satellite, buys a
    landing there that spawns next to the enemy home, and captures it, all on
    turn 0. Now: the captured port can't build, and a bought ship can't act."""
    # 8x3: P1 home (6,1), P1 satellite (4,1) whose spawn (5,1) touches P1's home.
    e = _engine(tmp_path, 8, 3, {"0": [[0, 1]], "1": [[6, 1], [4, 1]]})
    landing = e._spawn_ship(0, ShipType.LANDING, (3, 1))  # fleet-start ship, not bought
    e.players[0].cash = 250
    e.step(CapturePortAction(landing.id, (4, 1)))
    assert int(e.map.port_owner[(4, 1)]) == 0
    assert not any(isinstance(a, BuildShipAction) and a.port == (4, 1) for a in e.enumerate_legal(0))
    # Exhaustive: no sequence of up to 4 more turn-0 actions wins. The landing
    # still has 3 moves but needs 4 to reach a tile next to (6,1) around (4,1).
    assert not _all_turn0_sequences_win(e, depth=4)
    # Old chain written out explicitly: each step is a no-op now.
    e.step(BuildShipAction((4, 1), (5, 1), ShipType.LANDING))
    e.step(CapturePortAction(max(e.ships), (6, 1)))
    assert e.winner is None


def _bfs_path(e, start, goals):
    prev = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        if p in goals:
            path = []
            while p != start:
                path.append(p)
                p = prev[p]
            return path[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (p[0] + dx, p[1] + dy)
            if e.map.in_bounds(n) and e.map.is_water(n) and not e.map.is_occupied(n) and n not in prev:
                prev[n] = p
                q.append(n)
    return None


@pytest.mark.parametrize("width", [5, 7, 8, 10, 11, 13, 14])
def test_landing_rush_lands_one_turn_later(tmp_path, width):
    """Scripted fastest rush on an open strip: P0 buys a landing on turn 0 and
    walks the shortest water path; P1 passes. The capture turn equals
    earliest_capture_turn(moves) = old rule + 1."""
    e = _engine(tmp_path, width, 3, {"0": [[0, 1]], "1": [[width - 1, 1]]}, name=f"strip{width}.json")
    landing = _build(e, (0, 1), ShipType.LANDING)
    goals = {(width - 2, 0), (width - 2, 1), (width - 1, 0), (width - 1, 2), (width - 2, 2)}
    goals = {g for g in goals if e.map.manhattan(g, (width - 1, 1)) == 1}
    moves = len(_bfs_path(e, landing.position, goals))
    old_turn = 0 if moves <= 3 else (moves - 1) // 3
    for _ in range(50):
        if e.current_player == 0:
            cap = [a for a in e.enumerate_legal(0) if isinstance(a, CapturePortAction) and a.target == (width - 1, 1)]
            if cap:
                e.step(cap[0])
                break
            step = _bfs_path(e, landing.position, goals)
            if step and landing.tiles_moved_this_turn < landing.stats.speed and not landing.bought_this_turn:
                e.step(MoveAction(landing.id, step[0]))
                continue
        e.step(EndTurnAction())
    assert e.winner == 0
    assert e.turn == earliest_capture_turn(moves) == old_turn + 1
