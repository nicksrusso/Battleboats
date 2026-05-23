"""Heuristic monotonicity probe.

Drives a hand-crafted "obviously good" trajectory through the engine and
prints `heuristic_eval` (decomposed by term) at every step. Each step in
a good trajectory should make H go UP, or at minimum not down. Any "DIP"
flagged below is a sign that the heuristic's terms disagree with your
stated intent — a bug in the heuristic, not the trajectory.

Movement is by direct map relocation (no turn-flag bookkeeping needed);
load/unload go through real engine actions so the conversion physics is
exercised end-to-end.

Run:
    poetry run python scripts/heuristic_monotonicity.py
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

from battleboats.agents.heuristics import (
    COMBAT_K_TURNS,
    COMBAT_TYPES,
    HOME_K_PER_DIAGONAL,
    MAT_PORT_VALUE,
    SCALE_COMBAT,
    SCALE_ECON,
    SCALE_HOME,
    SCALE_MAT,
    W_COMBAT,
    W_ECON,
    W_HOME,
    W_MAT,
    _combat_balance,
    _home_threat,
    _merchant_logistics_value,
    _ship_value,
)
from battleboats.core.actions import MerchantLoadAction, MerchantUnloadAction
from battleboats.core.gameEngine import MERCHANT_CAPACITY, gameEngine
from battleboats.core.shipyard.ship_type import ShipType

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"


# --------------------------------------------------------------- decomposition
def decompose(engine: gameEngine, me: int) -> dict:
    """Recompute each heuristic term independently for visibility."""
    opp_player = engine.players[1 - me]
    my_player = engine.players[me]
    opp_ships = [engine.ships[s] for s in opp_player.owned_ship_ids]
    my_ships = [engine.ships[s] for s in my_player.owned_ship_ids]
    opp_combat = [s for s in opp_ships if s.type in COMBAT_TYPES]
    my_combat = [s for s in my_ships if s.type in COMBAT_TYPES]

    char = (engine.map.width + engine.map.height) / HOME_K_PER_DIAGONAL
    home_raw = _home_threat(opp_player.home_port, my_ships, engine.map.manhattan, char) - _home_threat(
        my_player.home_port, opp_ships, engine.map.manhattan, char
    )
    combat_raw = _combat_balance(my_combat, opp_combat, engine.kill_curve_k, engine.map.manhattan, COMBAT_K_TURNS)

    opp_val = sum(_ship_value(s) for s in opp_ships) + MAT_PORT_VALUE * len(opp_player.owned_port_positions)
    my_val = sum(_ship_value(s) for s in my_ships) + MAT_PORT_VALUE * len(my_player.owned_port_positions)
    mat_raw = my_val - opp_val

    map_diag = engine.map.width + engine.map.height
    econ_raw = (
        my_player.cash
        + sum(engine.ports[pos].stockpile for pos in my_player.owned_port_positions)
        + sum(s.cargo for s in my_ships if s.type is ShipType.MERCHANT)
        + _merchant_logistics_value(
            my_ships=my_ships,
            owned_port_positions=my_player.owned_port_positions,
            ports=engine.ports,
            home_port=my_player.home_port,
            manhattan=engine.map.manhattan,
            map_diag=map_diag,
        )
    )

    T_HOME = math.tanh(home_raw / SCALE_HOME)
    T_COMBAT = math.tanh(combat_raw / SCALE_COMBAT)
    T_MAT = math.tanh(mat_raw / SCALE_MAT)
    T_ECON = math.tanh(econ_raw / SCALE_ECON)
    H = (W_HOME * T_HOME + W_COMBAT * T_COMBAT + W_MAT * T_MAT + W_ECON * T_ECON) / (
        W_HOME + W_COMBAT + W_MAT + W_ECON
    )
    return {
        "T_HOME": T_HOME, "T_COMBAT": T_COMBAT, "T_MAT": T_MAT, "T_ECON": T_ECON,
        "home_raw": home_raw, "combat_raw": combat_raw, "mat_raw": mat_raw, "econ_raw": econ_raw,
        "H": H,
    }


def print_row(label: str, terms: dict, prev_H: Optional[float]) -> None:
    H = terms["H"]
    delta = H - prev_H if prev_H is not None else 0.0
    marker = "  <-- DIP" if (prev_H is not None and delta < -1e-9) else ""
    print(
        f"  {label:<32} H={H:+.5f} dH={delta:+.5f}  "
        f"home={terms['T_HOME']:+.3f} combat={terms['T_COMBAT']:+.3f} "
        f"mat={terms['T_MAT']:+.3f} econ={terms['T_ECON']:+.3f}  "
        f"econ_raw={terms['econ_raw']:+8.1f}{marker}"
    )


# ---------------------------------------------------------------- map helpers
def bfs_path(src: Tuple[int, int], dst: Tuple[int, int], engine: gameEngine) -> Optional[List[Tuple[int, int]]]:
    """Shortest water path from src ending at a tile adjacent to dst."""
    if engine.map.manhattan(src, dst) == 1:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + d[0], cur[1] + d[1])
            if n in parent or not engine.map.in_bounds(n) or not engine.map.is_water(n):
                continue
            parent[n] = cur
            if engine.map.manhattan(n, dst) == 1:
                path = [n]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            q.append(n)
    return None


def teleport(engine: gameEngine, ship_id: int, pos: Tuple[int, int]) -> None:
    ship = engine.ships[ship_id]
    engine.map.relocate_ship(ship_id, ship.position, pos)
    ship.position = pos


def water_neighbor(engine: gameEngine, pos: Tuple[int, int]) -> Tuple[int, int]:
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c = (pos[0] + d[0], pos[1] + d[1])
        if engine.map.in_bounds(c) and engine.map.is_water(c) and not engine.map.is_occupied(c):
            return c
    raise RuntimeError(f"No water neighbor for {pos}")


# ---------------------------------------------------------------- trajectory
def trajectory_merchant_loop(engine: gameEngine, me: int) -> None:
    """Empty merchant -> load at non-home port -> unload at home."""
    print("=== Trajectory: merchant logistics loop ===")
    my_player = engine.players[me]
    home = my_player.home_port

    non_home = [p for p in my_player.owned_port_positions if p != home]
    if not non_home:
        print("  player has no non-home port; aborting")
        return
    load_port = min(non_home, key=lambda p: engine.map.manhattan(p, home))
    engine.ports[load_port].stockpile = MERCHANT_CAPACITY  # guarantee something to load
    print(f"  home={home}  load_port={load_port}  manhattan={engine.map.manhattan(home, load_port)}")

    spawn = water_neighbor(engine, home)
    merchant = engine._spawn_ship(me, ShipType.MERCHANT, spawn)
    engine._refresh_sightings()

    path_out = bfs_path(spawn, load_port, engine)
    if path_out is None:
        print(f"  no water path from spawn {spawn} to load_port {load_port}; aborting")
        return

    prev_H: Optional[float] = None
    terms = decompose(engine, me)
    print_row(f"start: empty @ {spawn}", terms, prev_H)
    prev_H = terms["H"]

    # Outbound walk (skip path[0] — already there).
    for pos in path_out[1:]:
        teleport(engine, merchant.id, pos)
        terms = decompose(engine, me)
        d = engine.map.manhattan(pos, load_port)
        print_row(f"outbound: d(load)={d}", terms, prev_H)
        prev_H = terms["H"]

    # Real load via engine.
    engine.step(MerchantLoadAction(merchant_id=merchant.id, port=load_port))
    terms = decompose(engine, me)
    print_row(f"LOAD: cargo={merchant.cargo}", terms, prev_H)
    prev_H = terms["H"]

    # Inbound walk — destination is the spawn tile (1 away from home).
    path_in = bfs_path(merchant.position, home, engine)
    if path_in is None:
        print(f"  no water path back to home from {merchant.position}; aborting")
        return
    for pos in path_in[1:]:
        teleport(engine, merchant.id, pos)
        terms = decompose(engine, me)
        d = engine.map.manhattan(pos, home)
        print_row(f"inbound: d(home)={d}", terms, prev_H)
        prev_H = terms["H"]

    # Real unload via engine.
    cash_before = engine.players[me].cash
    engine.step(MerchantUnloadAction(merchant_id=merchant.id, port=home))
    terms = decompose(engine, me)
    cash_after = engine.players[me].cash
    print_row(f"UNLOAD: cash {cash_before}->{cash_after}", terms, prev_H)


def main() -> None:
    engine = gameEngine(MAP_JSON)
    engine.reset(seed=0)
    trajectory_merchant_loop(engine, me=0)


if __name__ == "__main__":
    main()
