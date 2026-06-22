"""Score the legal moves at a game state by 1-ply heuristic lookahead.

Reconstructs a *steppable* engine from a harvest row — the entity tokens give
ship/port positions, types, owners, cargo, turn-state, and stockpile; the logged
`cash` (added to the harvest after this analysis was requested) supplies the one
piece the tokens lack, which is what makes build-affordability — and therefore
the legal move set — reconstructable.

For each legal move of the to-move player: clone -> step(move) -> decompose() to
get the resulting state's per-feature contributions (w*phi) and total value. The
top-N are plotted as stacked bars: each segment is one feature's contribution,
the stack's extent is the heuristic score the search maximizes (final value =
tanh(sum), annotated). This exposes *which features make the heuristic prefer a
move* — e.g. combat shuffling out-scoring advancing the Landing.

    poetry run python scripts/move_scores.py --selftest         # validate reconstruction
    poetry run python scripts/move_scores.py runs/harvests/<run>/game_5.jsonl --step 8

NOTE: needs a harvest logged WITH cash (runs produced after the logging edit).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import List

import numpy as np

from battleboats.agents.debug_plot import _describe_action
from battleboats.agents.heuristics import DEFAULT_INTERCEPT, DEFAULT_WEIGHTS, decompose, heuristic_eval
from battleboats.core.actions import AttackAction
from battleboats.core.gameEngine import MERCHANT_CAPACITY, gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import (
    OWNER_OFFSET,
    PORT_STATE_OFFSET,
    POSITION_OFFSET,
    SHIP_STATE_OFFSET,
    STOCKPILE_SCALE,
    TOKEN_TYPE_ONEHOT_DIM,
    build_entity_tokens,
)

SHIP_TYPES = list(ShipType)
PORT_TYPE_INDEX = len(SHIP_TYPES)  # 8


# ---------------------------------------------------------------- reconstruction
def engine_from_tokens(
    map_path: str,
    seed,
    tokens_p0: List[List[float]],
    tokens_p1: List[List[float]],
    cash: List[int],
    turn: int,
    current_player: int,
) -> gameEngine:
    """Rebuild a steppable engine from a logged step's two perspectives + cash.

    Uses reset() for the static map + port scaffolding, then overwrites ships,
    port ownership/stockpile, player cash, turn, and current player from the
    tokens. _refresh_sightings() rebuilds fog from positions (so attack legality
    is correct). Full enough that enumerate_legal()/clone()/step() behave as in
    the real game.
    """
    eng = gameEngine(map_json_path=map_path)
    eng.reset(seed=seed)
    eng.turn = int(turn)
    eng.current_player = int(current_player)
    eng.winner = None

    W, H = eng.map.width, eng.map.height
    for p in eng.players:
        p.cash = 0
        p.owned_ship_ids = set()
        p.owned_port_positions = set()
    eng.players[0].cash, eng.players[1].cash = int(cash[0]), int(cash[1])

    for pid, toks in ((0, tokens_p0), (1, tokens_p1)):
        for t in toks:
            if t[OWNER_OFFSET] < 0.5:        # only this perspective's own entities
                continue
            ti = int(np.argmax(t[:TOKEN_TYPE_ONEHOT_DIM]))
            pos = (int(round(t[POSITION_OFFSET] * W)), int(round(t[POSITION_OFFSET + 1] * H)))
            if ti == PORT_TYPE_INDEX:        # port (ownership can differ from map after captures)
                port = eng.ports.get(pos)
                if port is None:
                    continue
                port.owner = pid
                port.is_home = t[PORT_STATE_OFFSET + 1] > 0.5
                port.stockpile = int(round(t[PORT_STATE_OFFSET] * STOCKPILE_SCALE))
                eng.players[pid].owned_port_positions.add(pos)
                eng.map.port_owner[pos] = pid  # enumerate_legal reads map.port_owner
            elif ti < PORT_TYPE_INDEX:       # ship
                ship = eng._spawn_ship(pid, SHIP_TYPES[ti], pos)
                ship.cargo = int(round(t[SHIP_STATE_OFFSET] * MERCHANT_CAPACITY))
                ship.has_attacked = t[SHIP_STATE_OFFSET + 1] > 0.5
                ship.tiles_moved_this_turn = int(round(t[SHIP_STATE_OFFSET + 2] * ship.stats.speed))

    eng._refresh_sightings()
    return eng


# --------------------------------------------------------------------- scoring
def score_moves(engine: gameEngine, pid: int, top_n: int = 10, samples: int = 25):
    """Return [(action, value, contributions_dict), ...] sorted best-first.

    Attacks resolve stochastically (the kill curve), so a single clone/step is a
    coin-flip. For AttackActions we average over `samples` clones, each given a
    distinct rng so the kill outcomes vary — the bar then shows the move's
    EXPECTED contributions, not one lucky/unlucky sample. Deterministic moves use
    one clone. value = tanh(Σ avg-contributions + intercept) so the stacked bar
    stays self-consistent (its net == the score the value squashes).
    """
    scored = []
    for mv in engine.enumerate_legal(player_id=pid):
        n = samples if isinstance(mv, AttackAction) else 1
        csum: dict = defaultdict(float)
        term_H, n_term = None, 0
        for s in range(n):
            sim = engine.clone()
            if n > 1:
                sim.rng = np.random.default_rng(s)  # vary the kill outcome across samples
            sim.step(mv)
            _H, _phi, contribs = decompose(sim, pid)
            if contribs:
                for k, v in contribs.items():
                    csum[k] += v
            else:  # terminal (e.g. capture -> win); deterministic moves only
                n_term += 1
                term_H = _H
        if n_term == n:
            scored.append((mv, float(term_H), {}))
        else:
            avg = {k: v / n for k, v in csum.items()}
            value = math.tanh(sum(avg.values()) + DEFAULT_INTERCEPT)
            scored.append((mv, float(value), avg))
    scored.sort(key=lambda r: r[1], reverse=True)
    return scored[:top_n]


# ----------------------------------------------------------------------- plot
def plot_move_scores(ax, engine: gameEngine, scored, title: str = "") -> None:
    import matplotlib.pyplot as plt
    feats = list(DEFAULT_WEIGHTS.keys())
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % 10) for i, f in enumerate(feats)}

    labels = []
    for i, (mv, H, contribs) in enumerate(scored):
        if not contribs:  # terminal move (capture -> win/loss): no breakdown, plot the ±1 outcome
            ax.bar(i, H, color="0.3", edgecolor="black", label="terminal (win/loss)" if i == 0 else None)
            ax.text(i, H, f"{H:+.2f}", ha="center", va="bottom" if H >= 0 else "top", fontsize=8, fontweight="bold")
            labels.append(_describe_action(mv, engine))
            continue
        # Signed two-sided stack: positives up from 0, negatives down. The intercept
        # is a constant segment so the parts sum to the full pre-tanh score. The black
        # tick marks the NET (= the heuristic score); the bounded value tanh(net) is
        # annotated above.
        segs = [("(intercept)", float(DEFAULT_INTERCEPT), "0.6")]
        segs += [(f, float(contribs.get(f, 0.0)), colors[f]) for f in feats]
        pos = neg = 0.0
        for name, c, col in segs:
            if c == 0.0:
                continue
            bottom = pos if c >= 0 else neg
            ax.bar(i, c, bottom=bottom, color=col, edgecolor="white", linewidth=0.4,
                   label=name.replace("_", " ") if i == 0 else None)
            if c >= 0:
                pos += c
            else:
                neg += c
        net = pos + neg
        ax.plot([i - 0.42, i + 0.42], [net, net], color="black", lw=2.2, zorder=6)
        ax.text(i, pos + 0.004, f"{H:+.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        labels.append(_describe_action(mv, engine))

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(scored)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("feature contributions (pre-tanh)\nblack tick = net score · value = tanh(net), annotated")
    ax.set_title(title or "Top legal moves by 1-ply heuristic value", fontsize=11, fontweight="bold")
    # cash drives build affordability (which build moves are even legal), so surface it
    pid = engine.current_player
    opp = 1 - pid
    ax.text(0.99, 0.98,
            f"cash — player_{pid} (acting): {engine.players[pid].cash}\n"
            f"cash — player_{opp}: {engine.players[opp].cash}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.85))
    # de-dupe legend entries
    handles, labs = ax.get_legend_handles_labels()
    seen = dict(zip(labs, handles))
    ax.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="feature")


# ----------------------------------------------------------------- self-test
def _selftest() -> None:
    """Round-trip: real state -> tokens+cash -> reconstruct -> compare legal moves + eval."""
    import json
    from battleboats.envs.action_masks import ActionMasks
    from battleboats.agents.godmode_mcts import godmode_mcts_action
    import random

    scn = json.load(open("runs/scenarios/scenarios_1000.json"))[0]
    truth = gameEngine(map_json_path=scn["map_path"])
    truth.reset_from_scenario(scn)
    rng = random.Random(0)
    # advance a few real plies to reach a non-trivial mid-state
    for _ in range(6):
        if truth.is_terminal():
            break
        pid = truth.current_player
        mv = godmode_mcts_action(truth, pid, rng, iterations=30)
        truth.step(mv)
    pid = truth.current_player
    t0 = build_entity_tokens(truth, 0).tolist()
    t1 = build_entity_tokens(truth, 1).tolist()
    cash = [truth.players[0].cash, truth.players[1].cash]

    recon = engine_from_tokens(scn["map_path"], scn["seed"], t0, t1, cash, truth.turn, pid)

    def keys(e):
        am = ActionMasks(e, pid)
        return sorted(tuple(am.factor(m)) for m in e.enumerate_legal(player_id=pid))

    kt, kr = keys(truth), keys(recon)
    ev_t, ev_r = heuristic_eval(truth, pid), heuristic_eval(recon, pid)
    print(f"current_player={pid}  turn={truth.turn}")
    print(f"legal moves: truth={len(kt)}  recon={len(kr)}  identical={kt == kr}")
    if kt != kr:
        print(f"  only-in-truth: {[k for k in kt if k not in set(kr)][:8]}")
        print(f"  only-in-recon: {[k for k in kr if k not in set(kt)][:8]}")
    print(f"heuristic_eval: truth={ev_t:+.6f}  recon={ev_r:+.6f}  diff={abs(ev_t - ev_r):.2e}")
    print("ROUND-TRIP OK" if kt == kr and abs(ev_t - ev_r) < 1e-6 else "ROUND-TRIP MISMATCH")


def engine_for_step(game_file, step, scenarios_file=None, run_config=None):
    """Reconstruct the steppable engine + acting player for a logged step.

    Shared by this CLI and replay_game's --moves panel. Raises if the row lacks
    cash (harvest predates the cash-logging edit).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import json
    from replay_game import _load_rows, _steps_by_index, _find_run_config, _resolve

    game_file = Path(game_file)
    rows, _footer = _load_rows(game_file)
    steps = _steps_by_index(rows)
    rec = next((s for s in steps if s["step"] == step), None)
    if rec is None:
        raise SystemExit(f"step {step} not found (range {steps[0]['step']}..{steps[-1]['step']})")
    persp = rec["persp"]
    any_row = next(iter(persp.values()))
    if "cash" not in any_row:
        raise SystemExit("This shard has no `cash` field — re-harvest with the cash-logging edit.")
    cfg = _find_run_config(game_file, run_config)
    sfile = scenarios_file or cfg.get("scenarios_file", "runs/scenarios/scenarios_1000.json")
    scen = json.loads(Path(_resolve(str(sfile))).read_text())
    sc = scen[any_row["game_idx"] % len(scen)]
    pid = rec["acting_pid"] if rec["acting_pid"] is not None else 0
    eng = engine_from_tokens(_resolve(sc["map_path"]), sc.get("seed"),
                             persp[0]["tokens"], persp[1]["tokens"], any_row["cash"],
                             any_row["turn"], pid)
    return eng, pid, any_row["turn"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("game_file", nargs="?", help="harvest shard (needs cash-logged run)")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--samples", type=int, default=25, help="rng samples to average per stochastic (attack) move")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--scenarios-file", default=None)
    p.add_argument("--out", default="docs/move_scores.png")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    elif args.game_file:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eng, pid, turn = engine_for_step(args.game_file, args.step, args.scenarios_file)
        scored = score_moves(eng, pid, top_n=args.top_n, samples=args.samples)
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(scored) + 4), 6.5))
        fig.subplots_adjust(right=0.8, bottom=0.34)
        plot_move_scores(ax, eng, scored, title=f"Top legal moves — player_{pid}, turn {turn} (step {args.step})")
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"saved {args.out}  (top move: {_describe_action(scored[0][0], eng)}  value={scored[0][1]:+.3f})")
    else:
        p.error("provide a game_file or --selftest")
