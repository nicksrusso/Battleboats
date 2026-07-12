"""Replay a harvested game and visualize it with agents.debug_plot.

Renders the RECORDED state at each step directly from the saved entity tokens —
no engine re-stepping, no MCTS, no RNG. This is faithful and cannot desync:
every harvest step stores the full token set for BOTH perspectives, so

    perspective-0 friendly ships  -> all player-0 ships
    perspective-1 friendly ships  -> all player-1 ships
    friendly ports (either side)  -> current port ownership (reflects captures)

reconstruct the complete board. Static terrain comes from a one-time
deterministic reset (no combat involved). The just-chosen action is decoded
from the acting perspective's own tokens and highlighted.

(An earlier version replayed the saved actions through the engine; that desynced
on any game with combat because stochastic attacks don't reproduce move-for-move.)

Two modes:

  step  (default)  interactive — one move per <enter>; the moving token is
                   highlighted. Type 'q' then <enter> to quit early.
  gif              render the whole game end-to-end to an animated .gif.

    poetry run python scripts/replay_game.py runs/harvests/<run>/game_5.jsonl
    poetry run python scripts/replay_game.py runs/harvests/<run>/game_5.jsonl --mode gif --out game_5.gif

Run from the repo root (scenario map paths are repo-relative).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Token field offsets (see battleboats/envs/observation.py).
TYPE_SLICE = slice(0, 10)  # one-hot: 0-7 ship types, 8 port, 9 coastline
POS_X, POS_Y = 10, 11  # x/width, y/height
IS_FRIENDLY = 12  # owner one-hot (relative to the perspective)
IS_HOME = 24  # port_state[1]


def _load_rows(game_file: Path):
    rows, footer = [], None
    with open(game_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_type") == "game_footer":
                footer = r
            else:
                rows.append(r)
    return rows, footer


def _steps_by_index(rows):
    """Group rows into per-step records: {perspective: row} plus the acting move."""
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], {})[r["perspective"]] = r
    out = []
    for step in sorted(by_step):
        persp = by_step[step]
        acting = next((p for p, row in persp.items() if row.get("action") is not None), None)
        out.append({"step": step, "persp": persp, "acting_pid": acting})
    return out


def _find_run_config(game_file: Path, override):
    if override is not None:
        return json.loads(Path(override).read_text())
    cand = game_file.parent / "_run_config.json"
    if cand.exists():
        return json.loads(cand.read_text())
    metas = list(game_file.parent.glob("*_meta.json"))
    if metas:
        return json.loads(metas[0].read_text()).get("config", {})
    return {}


def _resolve(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.exists() else REPO_ROOT / path_str)


def plot_mcts_root(ax, root_stats, title, top_n: int = 12) -> None:
    """Horizontal bar chart of the MCTS root's children = the search's policy.

    `root_stats` is a harvest row's `mcts_root`: a list of
    {"action": <label>, "visits": int, "q": float}. Children sharing an action
    LABEL are aggregated (e.g. the 8 move directions of one ship, which
    `_describe_action` collapses to 'Move(Cruiser)'): visits summed, Q
    visit-weighted-averaged. Bar length = visits = how search apportioned its
    budget — and the argmax IS the chosen move. Combat actions (Attack/Capture)
    are drawn in red so their (typically near-zero) visit share is unmissable.
    """

    def is_combat(a: str) -> bool:
        return ("Attack" in a) or ("Capture" in a)

    agg: dict = {}
    for s in root_stats:
        e = agg.setdefault(s["action"], {"visits": 0, "qw": 0.0})
        e["visits"] += s["visits"]
        e["qw"] += s["q"] * s["visits"]
    items = sorted(
        ({"action": a, "visits": e["visits"], "q": (e["qw"] / e["visits"] if e["visits"] else 0.0)} for a, e in agg.items()),
        key=lambda d: d["visits"],
    )
    total = sum(e["visits"] for e in agg.values())
    combat = sum(e["visits"] for a, e in agg.items() if is_combat(a))
    # The CHOSEN move is the single highest-visit child (NOT the largest
    # aggregated bar — moves fragment across directions, so a summed bar can top
    # the chart without any one child being the pick). Mark it so the two don't
    # get conflated.
    chosen = max(root_stats, key=lambda s: s["visits"])["action"] if root_stats else None
    items = items[-top_n:]  # largest-visit children land at the top of a barh

    y = list(range(len(items)))
    visits = [d["visits"] for d in items]
    colors = ["#d62728" if is_combat(d["action"]) else "#4c72b0" for d in items]
    edges = ["black" if d["action"] == chosen else "none" for d in items]
    lws = [2.0 if d["action"] == chosen else 0.0 for d in items]
    ax.barh(y, visits, color=colors, edgecolor=edges, linewidth=lws)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d['action']} ←chosen" if d["action"] == chosen else d["action"] for d in items], fontsize=9)
    for yi, d in zip(y, items):
        ax.text(d["visits"], yi, f"  N={d['visits']}  Q={d['q']:+.2f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, (max(visits) * 1.28) if visits else 1)
    ax.set_xlabel("root visits per action  (summed over directions/targets)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    share = (100 * combat / total) if total else 0.0
    ax.text(
        0.98,
        0.02,
        f"combat visits: {combat}/{total}  ({share:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#d62728" if combat == 0 else "0.3",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.85),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game_file", type=Path, help="Path to a harvest shard, e.g. runs/harvests/<run>/game_5.jsonl")
    parser.add_argument("--mode", choices=["step", "gif"], default="step")
    parser.add_argument("--out", type=Path, default=None, help="Gif output path (default: alongside the shard).")
    parser.add_argument("--ms", type=int, default=600, help="Gif frame duration in ms (default: 600).")
    parser.add_argument("--scenarios-file", type=Path, default=None, help="Override scenarios file (else from run config).")
    parser.add_argument("--run-config", type=Path, default=None, help="Override path to _run_config.json.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth move (handy for long truncated games).")
    parser.add_argument(
        "--moves",
        action="store_true",
        help="Also show a stacked-bar panel of the to-move player's top-10 legal moves by "
        "1-ply heuristic value (needs a cash-logged harvest).",
    )
    parser.add_argument(
        "--mcts",
        action="store_true",
        help="Also show a horizontal-bar panel of the MCTS root's children = the search's "
        "policy (visits + Q per action, combat actions in red). Needs a harvest run with "
        "--debug-mcts (the `mcts_root` field).",
    )
    args = parser.parse_args()

    if args.mode == "gif":
        os.environ["MPLBACKEND"] = "Agg"
        import warnings

        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

    import numpy as np
    from PIL import Image
    from types import SimpleNamespace

    from battleboats.agents import debug_plot
    from battleboats.agents.debug_plot import plot_state
    from battleboats.envs.action_masks import DIRECTIONS
    from battleboats.envs.battleboats_aec import BattleboatsAEC
    from battleboats.core.shipyard.ship_type import ShipType
    from battleboats.training.policy import VERB_TO_IDX

    IDX_TO_VERB = {v: k for k, v in VERB_TO_IDX.items()}
    SHIP_TYPES = list(ShipType)

    rows, footer = _load_rows(args.game_file)
    if not rows:
        sys.exit(f"No data rows in {args.game_file}")

    cfg = _find_run_config(args.game_file, args.run_config)
    scenarios_file = args.scenarios_file or Path(cfg.get("scenarios_file", "runs/scenarios/scenarios_1000.json"))
    game_idx = rows[0]["game_idx"]
    mcts_player_id = rows[0]["mcts_player_id"]
    winner = footer.get("winner") if footer else None
    outcome = "truncated" if winner is None else f"player_{winner} won"

    scenarios = json.loads(Path(_resolve(str(scenarios_file))).read_text())
    scenario = scenarios[game_idx % len(scenarios)]
    map_path = _resolve(scenario["map_path"])

    # Static terrain only (reset has no combat -> safe + deterministic).
    base = BattleboatsAEC(map_json_path=map_path, max_turns=cfg.get("max_turns", 400))
    base.reset(seed=scenario.get("seed"), options={"scenario": scenario})
    game_map = base.engine.map
    W, H = game_map.width, game_map.height

    steps = _steps_by_index(rows)
    print(
        f"Replaying game_idx={game_idx} mcts_player={mcts_player_id} | "
        f"{len(steps)} moves | {outcome} | map={Path(map_path).name}"
    )

    def decode_pos(tok):
        return (int(round(tok[POS_X] * W)), int(round(tok[POS_Y] * H)))

    def reconstruct(persp):
        """Full state from both perspectives' friendly tokens."""
        ships, ports, sid = {}, {}, 0
        for pid in (0, 1):
            row = persp.get(pid)
            if row is None:
                continue
            for tok in row["tokens"]:
                ti = int(np.argmax(tok[TYPE_SLICE]))
                if tok[IS_FRIENDLY] < 0.5:  # only this perspective's own entities
                    continue
                pos = decode_pos(tok)
                if ti < 8:  # ship
                    ships[sid] = SimpleNamespace(position=pos, type=SHIP_TYPES[ti], owner=pid)
                    sid += 1
                elif ti == 8:  # port
                    ports[pos] = SimpleNamespace(position=pos, owner=pid, is_home=tok[IS_HOME] > 0.5)
        return ships, ports

    def decode_action(rec):
        """(description, overlay) from the acting perspective's own tokens."""
        pid = rec["acting_pid"]
        if pid is None:
            return "None", None
        row = rec["persp"][pid]
        a, v, t = row["action"]
        toks = row["tokens"]
        if a >= len(toks):
            return f"verb={v}", None
        verb = IDX_TO_VERB.get(v, str(v))
        origin = decode_pos(toks[a])
        if verb == "move":
            dx, dy = DIRECTIONS[t]
            dest = (origin[0] + dx, origin[1] + dy)
            return f"Move -> {dest}", ("move", origin, dest)
        if verb == "attack" and t < len(toks):
            tpos = decode_pos(toks[t])
            return f"Attack -> {tpos}", ("attack", origin, tpos)
        if verb == "build_ship":
            return f"BuildShip({SHIP_TYPES[t].value}) at {origin}", ("ring", origin)
        if verb == "build_port":
            dx, dy = DIRECTIONS[t]
            tile = (origin[0] + dx, origin[1] + dy)
            return f"BuildPort at {tile}", ("ring", tile)
        if verb in ("capture", "load", "unload"):
            return f"{verb.capitalize()} near {origin}", ("ring", origin)
        return verb, None

    def draw_overlay(ax, overlay):
        if overlay is None:
            return
        HL, HLA = "#ff00ff", "#ff8800"
        if overlay[0] == "move":
            _, o, d = overlay
            ax.annotate("", xy=d, xytext=o, arrowprops=dict(arrowstyle="->", color=HL, lw=2.5, mutation_scale=20))
            ax.scatter([o[0]], [o[1]], s=300, facecolors="none", edgecolors=HL, linewidths=2.5, zorder=5)
        elif overlay[0] == "attack":
            _, o, tp = overlay
            ax.plot([o[0], tp[0]], [o[1], tp[1]], color=HLA, ls="--", lw=2, zorder=5)
            ax.scatter([tp[0]], [tp[1]], s=350, marker="X", c=HLA, edgecolors="white", linewidths=1.5, zorder=6)
        elif overlay[0] == "ring":
            _, tile = overlay
            ax.scatter([tile[0]], [tile[1]], s=400, facecolors="none", edgecolors=HL, linewidths=2.5, zorder=5)

    # Optional move-score panel — its own figure (a SEPARATE window in step mode,
    # composited under the board for gif). replay_game owns both; move_scores does
    # the reconstruction + scoring + bar rendering.
    moves_on = args.moves
    if moves_on:
        import matplotlib.pyplot as plt
        from move_scores import engine_from_tokens, score_moves, plot_move_scores

        sample = next(iter(steps[0]["persp"].values()))
        if "cash" not in sample:
            print("  --moves: shard has no `cash` field (pre-edit harvest) — disabling move panel.")
            moves_on = False
        else:
            if args.mode == "step":
                plt.ion()  # so both figures are live, independently-windowed
            move_fig, move_ax = plt.subplots(figsize=(11.5, 5.2))
            move_fig.subplots_adjust(right=0.78, bottom=0.36)
            if args.mode == "step":
                try:
                    move_fig.canvas.manager.set_window_title("Battleboats — move scores")
                except Exception:  # noqa: BLE001 — window-title support is backend-specific
                    pass
                move_fig.show()

    # Optional MCTS-root panel — its own figure, same lifecycle as the move panel.
    # Needs a --debug-mcts harvest (the `mcts_root` field on each acting row).
    mcts_on = args.mcts
    if mcts_on:
        import matplotlib.pyplot as plt

        has_root = any(st["acting_pid"] is not None and st["persp"].get(st["acting_pid"], {}).get("mcts_root") for st in steps)
        if not has_root:
            print("  --mcts: shard has no `mcts_root` field (harvest without --debug-mcts) — disabling MCTS panel.")
            mcts_on = False
        else:
            if args.mode == "step":
                plt.ion()
            mcts_fig, mcts_ax = plt.subplots(figsize=(8.5, 5.2))
            mcts_fig.subplots_adjust(left=0.30, right=0.95, bottom=0.12, top=0.9)
            if args.mode == "step":
                try:
                    mcts_fig.canvas.manager.set_window_title("Battleboats — MCTS root")
                except Exception:  # noqa: BLE001 — window-title support is backend-specific
                    pass
                mcts_fig.show()

    def _vstack_many(images):
        w = max(im.width for im in images)
        canvas = Image.new("RGB", (w, sum(im.height for im in images)), "white")
        y = 0
        for im in images:
            canvas.paste(im, ((w - im.width) // 2, y))
            y += im.height
        return canvas

    frames = []
    selected = steps[:: max(1, args.stride)]
    for i, rec in enumerate(selected):
        ships, ports = reconstruct(rec["persp"])
        pid = rec["acting_pid"]
        value_row = rec["persp"].get(pid) if pid is not None else None
        value = value_row.get("mcts_root_value") if value_row else None
        turn = (value_row or next(iter(rec["persp"].values())))["turn"]
        desc, overlay = decode_action(rec)

        engine = SimpleNamespace(map=game_map, ports=ports, ships=ships, turn=turn)
        plot_state(engine, action=None, actor=f"player_{pid}", mcts_player_id=mcts_player_id, step=rec["step"])

        if args.mode == "step" and i == 0:
            try:
                debug_plot._FIG.canvas.manager.set_window_title("Battleboats — board")
            except Exception:  # noqa: BLE001 — window-title support is backend-specific
                pass

        ax = debug_plot._AX
        parts = [f"step={rec['step']}", f"turn={turn}", f"actor=player_{pid}", f"action={desc}"]
        if value is not None:
            parts.append(f"value={value:+.4f}")
        parts.append(f"move {i + 1}/{len(selected)}  ({outcome})")
        ax.set_title("  ".join(parts), fontsize=10)
        draw_overlay(ax, overlay)

        fig = debug_plot._FIG
        fig.canvas.draw()
        board_img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")

        if moves_on and pid is not None:
            move_ax.clear()
            cash = rec["persp"][pid].get("cash")
            try:
                eng = engine_from_tokens(
                    map_path, scenario.get("seed"), rec["persp"][0]["tokens"], rec["persp"][1]["tokens"], cash, turn, pid
                )
                scored = score_moves(eng, pid, top_n=12)
                plot_move_scores(move_ax, eng, scored, title=f"top legal moves — player_{pid}, turn {turn}")
            except Exception as e:  # noqa: BLE001 — keep stepping if one state fails to reconstruct
                move_ax.text(0.5, 0.5, f"move-score unavailable:\n{e}", ha="center", va="center", transform=move_ax.transAxes)
            move_fig.canvas.draw()

        if mcts_on and pid is not None:
            mcts_ax.clear()
            root_stats = rec["persp"][pid].get("mcts_root")
            if root_stats:
                plot_mcts_root(mcts_ax, root_stats, title=f"MCTS root — player_{pid}, turn {turn}")
            else:
                mcts_ax.text(0.5, 0.5, "no mcts_root for this step", ha="center", va="center", transform=mcts_ax.transAxes)
            mcts_fig.canvas.draw()

        if args.mode == "gif":
            panels = []
            if moves_on and pid is not None:
                panels.append(Image.fromarray(np.asarray(move_fig.canvas.buffer_rgba())).convert("RGB"))
            if mcts_on and pid is not None:
                panels.append(Image.fromarray(np.asarray(mcts_fig.canvas.buffer_rgba())).convert("RGB"))
            frames.append(board_img if not panels else _vstack_many([board_img] + panels))
        else:
            import matplotlib.pyplot as plt

            plt.pause(0.001)
            if input(f"[{i + 1}/{len(selected)}] player_{pid}: {desc}  — [enter] next / q quit ").strip().lower() == "q":
                break

    if args.mode == "gif":
        out = args.out or args.game_file.with_suffix(".gif")
        if not frames:
            sys.exit("No frames captured.")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=args.ms, loop=0, disposal=2)
        print(f"saved {out}  ({len(frames)} frames, {args.ms}ms each)")


if __name__ == "__main__":
    main()
