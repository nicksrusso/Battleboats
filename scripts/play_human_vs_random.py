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
import random
from typing import Optional, Tuple

from battleboats.agents.debug_plot import plot_state
from battleboats.agents.heuristics import (
    FEATURE_KEYS,
    decompose as _h_decompose,
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
# Short column labels for the markdown table — must cover every key in
# heuristics.FEATURE_KEYS, in display order.
FEATURE_LABEL = {
    "material_diff": "mat",
    "home_pressure_diff": "hp",
    "combat_balance": "comb",
    "econ_value_self": "econ",
    "merchant_count_value_self": "mrchN",
    "has_landing_self": "hasL",
    "landing_pressure_self": "lndP",
}


def decompose(engine: gameEngine, me: int) -> dict:
    """Thin wrapper around `heuristics.decompose` adapted for this script.

    Returns a dict with:
        - H              total heuristic value
        - terminal       bool
        - phi            features dict (raw feature values)
        - contrib        contributions dict (post-weight, pre-tanh)
    """
    H, phi, contrib = _h_decompose(engine, me)
    return {"H": H, "terminal": engine.is_terminal(), "phi": phi, "contrib": contrib}


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
    baseline_contrib = " ".join(
        f"{FEATURE_LABEL[k]}=`{baseline['contrib'][k]:+.3f}`" for k in FEATURE_KEYS
    )
    lines.append(
        f"**Baseline** (current state, no action applied): "
        f"H = `{baseline['H']:+.5f}`  —  {baseline_contrib}"
    )
    lines.append("")
    lines.append(f"- Cash: me=`{me_player.cash}`, opp=`{opp_player.cash}`")
    lines.append(f"- Ships: me=`{len(me_player.owned_ship_ids)}`, opp=`{len(opp_player.owned_ship_ids)}`")
    lines.append(
        f"- Ports: me=`{len(me_player.owned_port_positions)}`, opp=`{len(opp_player.owned_port_positions)}`"
    )
    lines.append(f"- Fleet cargo: `{fleet_cargo:.0%}` across `{len(my_merchants)}` merchant(s)")
    lines.append("")

    # Build dynamic table header from feature keys (one contribution column each).
    feat_headers = " | ".join(FEATURE_LABEL[k] for k in FEATURE_KEYS)
    feat_sep = " | ".join(["---:"] * len(FEATURE_KEYS))
    lines.append(
        f"| # | type | ship | target | H | dH | {feat_headers} | cash | cargo | info |"
    )
    lines.append(
        f"|---:|------|------|--------|---:|---:| {feat_sep} |---:|---:|------|"
    )

    n = len(rows)
    shown = rows if (top <= 0 or top >= n) else rows[:top]
    for i, (action, t, cash_after, cargo_after) in enumerate(shown):
        atype, ship, target, extra = _action_parts(action, engine, me)
        dh = t["H"] - baseline["H"]
        cargo_str = f"{cargo_after:.0%}" if cargo_after is not None else "—"
        target_md = str(target).replace("|", "\\|")
        info_md = str(extra).replace("|", "\\|")
        feat_cells = " | ".join(f"{t['contrib'][k]:+.3f}" for k in FEATURE_KEYS)
        lines.append(
            f"| {i} | {atype} | {ship} | {target_md} "
            f"| {t['H']:+.5f} | {dh:+.5f} "
            f"| {feat_cells} "
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
