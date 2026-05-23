"""Interactive heuristic debugger: human (player 0) vs random (player 1).

For every human action, enumerate all legal actions, evaluate the heuristic
(decomposed by term) on the *resulting* state of each, sort highest-first,
write the ranked table to a markdown file (overwritten each action), render
the current board, and prompt for the action id to apply. Random opponent
just samples uniformly.

Run:
    poetry run python scripts/play_human_vs_random.py
    poetry run python scripts/play_human_vs_random.py --seed 7 --out /tmp/h.md
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Optional, Tuple

from battleboats.agents.debug_plot import plot_state
from battleboats.agents.heuristics import (
    COMBAT_K_TURNS,
    COMBAT_TYPES,
    HOME_K_PER_DIAGONAL,
    MAT_PORT_VALUE,
    SCALE_COMBAT,
    SCALE_ECON,
    SCALE_HOME,
    SCALE_MAT,
    W_CARGO,
    W_COMBAT,
    W_ECON,
    W_HOME,
    W_MAT,
    _combat_balance,
    _home_threat,
    _merchant_logistics_value,
    _ship_value,
    pk,
)
from battleboats.agents.random_agent import random_action
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
from battleboats.core.gameEngine import MERCHANT_CAPACITY, gameEngine
from battleboats.core.shipyard.ship_type import ShipType

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"
DEFAULT_OUT = "/home/nick/Desktop/repos/Battleboats/temp.md"


# --------------------------------------------------------------- decomposition
def decompose(engine: gameEngine, me: int) -> dict:
    if engine.is_terminal():
        h = 1.0 if engine.winner == me else -1.0
        return {"T_HOME": h, "T_COMBAT": h, "T_MAT": h, "T_ECON": h, "H": h, "terminal": True}
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
        + W_CARGO * sum(s.cargo for s in my_ships if s.type is ShipType.MERCHANT)
        + _merchant_logistics_value(
            my_ships=my_ships,
            owned_port_positions=my_player.owned_port_positions,
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
    return {"T_HOME": T_HOME, "T_COMBAT": T_COMBAT, "T_MAT": T_MAT, "T_ECON": T_ECON, "H": H, "terminal": False}


# ---------------------------------------------------------------- action label
def _action_parts(action: Action, engine: gameEngine, me: int) -> Tuple[str, str, str, str]:
    """Return (type, ship, target, extra) — short strings for table columns."""
    if isinstance(action, MoveAction):
        s = engine.ships.get(action.ship_id)
        ship = f"{s.type.value}#{s.id}" if s else "?"
        target = f"{s.position}->{action.destination}" if s else str(action.destination)
        extra = ""
        if s:
            opp_ids = engine.players[1 - me].owned_ship_ids
            if opp_ids:
                d_enemy = min(engine.map.manhattan(action.destination, engine.ships[o].position) for o in opp_ids)
                extra = f"d_enemy={d_enemy}"
            d_home = engine.map.manhattan(action.destination, engine.players[1 - me].home_port)
            extra = (extra + " " if extra else "") + f"d_oppHome={d_home}"
        return "Move", ship, target, extra
    if isinstance(action, AttackAction):
        a = engine.ships.get(action.attacker_id)
        t = engine.ships.get(action.target_id)
        ship = f"{a.type.value}#{a.id}" if a else "?"
        target = f"{t.type.value}#{t.id}" if t else "?"
        extra = f"P(kill)={pk(a, t, engine.kill_curve_k):.2f}" if (a and t) else ""
        return "Attack", ship, target, extra
    if isinstance(action, BuildShipAction):
        from battleboats.core.shipyard.ship_data import BASE_STATS
        cost = BASE_STATS[action.ship_type].cost
        return "BuildShip", action.ship_type.value, f"port={action.port}", f"cost={cost}"
    if isinstance(action, BuildPortAction):
        s = engine.ships.get(action.builder_ship_id)
        return "BuildPort", f"{s.type.value}#{s.id}" if s else "?", f"at {action.target}", ""
    if isinstance(action, CapturePortAction):
        s = engine.ships.get(action.landing_ship_id)
        return "Capture", f"{s.type.value}#{s.id}" if s else "?", f"port={action.target}", ""
    if isinstance(action, MerchantLoadAction):
        s = engine.ships.get(action.merchant_id)
        port = engine.ports.get(action.port)
        extra = f"stock={port.stockpile} cargo={s.cargo}" if (s and port) else ""
        return "Load", f"Merchant#{action.merchant_id}", f"port={action.port}", extra
    if isinstance(action, MerchantUnloadAction):
        s = engine.ships.get(action.merchant_id)
        extra = f"cargo={s.cargo}" if s else ""
        return "Unload", f"Merchant#{action.merchant_id}", f"port={action.port}", extra
    if isinstance(action, EndTurnAction):
        return "EndTurn", "-", "-", ""
    return type(action).__name__, "-", "-", ""


# ---------------------------------------------------------------- ranking
def _action_merchant_cargo(action: Action, engine_after: gameEngine) -> Optional[float]:
    """Post-step cargo fraction for the merchant involved in this action, or None."""
    mid: Optional[int] = None
    if isinstance(action, MoveAction):
        mid = action.ship_id
    elif isinstance(action, MerchantLoadAction):
        mid = action.merchant_id
    elif isinstance(action, MerchantUnloadAction):
        mid = action.merchant_id
    if mid is None:
        return None
    ship = engine_after.ships.get(mid)
    if ship is None or ship.type is not ShipType.MERCHANT:
        return None
    return ship.cargo / MERCHANT_CAPACITY


def rank_actions(engine: gameEngine, me: int):
    legal = engine.enumerate_legal(me)
    baseline = decompose(engine, me)
    rows = []
    for action in legal:
        clone = engine.clone()
        clone.step(action)
        terms = decompose(clone, me)
        cash_after = clone.players[me].cash
        cargo_after = _action_merchant_cargo(action, clone)
        rows.append((action, terms, cash_after, cargo_after))
    rows.sort(key=lambda r: r[1]["H"], reverse=True)
    return rows, baseline


def write_table_md(rows, baseline: dict, engine: gameEngine, me: int, step: int, out_path: str, top: int) -> None:
    me_player = engine.players[me]
    opp_player = engine.players[1 - me]
    my_merchants = [engine.ships[s] for s in me_player.owned_ship_ids if engine.ships[s].type is ShipType.MERCHANT]
    if my_merchants:
        fleet_cargo = sum(m.cargo for m in my_merchants) / (MERCHANT_CAPACITY * len(my_merchants))
    else:
        fleet_cargo = 0.0

    lines = []
    lines.append(f"# Action table — step {step}, turn {engine.turn}")
    lines.append("")
    lines.append(
        f"**Baseline** (current state, no action applied): "
        f"H = `{baseline['H']:+.5f}`  —  "
        f"home=`{baseline['T_HOME']:+.3f}` combat=`{baseline['T_COMBAT']:+.3f}` "
        f"mat=`{baseline['T_MAT']:+.3f}` econ=`{baseline['T_ECON']:+.3f}`"
    )
    lines.append("")
    lines.append(f"- Cash: me=`{me_player.cash}`, opp=`{opp_player.cash}`")
    lines.append(f"- Ships: me=`{len(me_player.owned_ship_ids)}`, opp=`{len(opp_player.owned_ship_ids)}`")
    lines.append(
        f"- Ports: me=`{len(me_player.owned_port_positions)}`, opp=`{len(opp_player.owned_port_positions)}`"
    )
    lines.append(f"- Fleet cargo: `{fleet_cargo:.0%}` across `{len(my_merchants)}` merchant(s)")
    lines.append("")
    lines.append(
        "| # | type | ship | target | H | dH | T_HOME | T_COMBAT | T_MAT | T_ECON | cash | cargo | info |"
    )
    lines.append(
        "|---:|------|------|--------|---:|---:|---:|---:|---:|---:|---:|---:|------|"
    )

    n = len(rows)
    shown = rows if (top <= 0 or top >= n) else rows[:top]
    for i, (action, t, cash_after, cargo_after) in enumerate(shown):
        atype, ship, target, extra = _action_parts(action, engine, me)
        dh = t["H"] - baseline["H"]
        cargo_str = f"{cargo_after:.0%}" if cargo_after is not None else "—"
        # Markdown cells: escape pipe; tuples/parens are fine.
        target_md = str(target).replace("|", "\\|")
        info_md = str(extra).replace("|", "\\|")
        lines.append(
            f"| {i} | {atype} | {ship} | {target_md} "
            f"| {t['H']:+.5f} | {dh:+.5f} "
            f"| {t['T_HOME']:+.3f} | {t['T_COMBAT']:+.3f} | {t['T_MAT']:+.3f} | {t['T_ECON']:+.3f} "
            f"| {cash_after} | {cargo_str} | {info_md} |"
        )

    if top > 0 and n > top:
        lines.append("")
        lines.append(f"_{n - top} more actions hidden — pass `--top 0` to see all._")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def prompt_choice(rows) -> int:
    n = len(rows)
    while True:
        raw = input(f"  pick action id [0-{n - 1}] (q to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        try:
            i = int(raw)
        except ValueError:
            print("  not a number, try again")
            continue
        if 0 <= i < n:
            return i
        print("  out of range, try again")


# ---------------------------------------------------------------- main loop
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=0, help="Max rows in table (0 = show all).")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to markdown table (overwritten each action).")
    args = parser.parse_args()

    engine = gameEngine(MAP_JSON)
    engine.reset(seed=args.seed)
    me = 0
    rng = random.Random(args.seed)

    step = 0
    while not engine.is_terminal() and engine.turn < args.max_turns:
        if engine.current_player == me:
            rows, baseline = rank_actions(engine, me)
            write_table_md(rows, baseline, engine, me, step, args.out, args.top)
            print(f"  step={step} turn={engine.turn}  {len(rows)} actions ranked -> {args.out}")
            plot_state(engine, None, "human", me, step=step, value=baseline["H"])
            choice = prompt_choice(rows)
            action = rows[choice][0]
            atype, ship, target, _ = _action_parts(action, engine, me)
            print(f"  -> applying: {atype} {ship} {target}")
            engine.step(action)
        else:
            action = random_action(engine, engine.current_player, rng)
            atype, ship, target, _ = _action_parts(action, engine, 1 - me)
            print(f"  [random] {atype} {ship} {target}")
            engine.step(action)
        step += 1

    print()
    if engine.is_terminal():
        print(f"  GAME OVER — winner: player_{engine.winner}")
    else:
        print(f"  max_turns ({args.max_turns}) reached, no winner")


if __name__ == "__main__":
    main()
