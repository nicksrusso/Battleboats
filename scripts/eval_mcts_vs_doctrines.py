"""God-mode MCTS (value-net leaf) vs each scripted-agent doctrine.

Search is god-mode UCT on the Rust engine with batched value-net leaves and
a short tactical rollout. The opponent is the fog-limited Rust scripted
agent, forced to one doctrine.

Maps come from the same 48x24 harvest pool (`resolve_map`). The turn cap is
that map's harvest recommended_max_turns (typically 150-230). Capped games
are recorded as draws; a material (+ cash/stockpile) tiebreak is also stored.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# scripts/ is not a package; allow importing the sibling bench wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from types import SimpleNamespace

from battleboats.agents.godmode_mcts import (
    DEFAULT_LEAF_BATCH,
    ROLLOUT_STEP_BUDGET,
    godmode_mcts_search,
    load_mcts_value_net,
)
from battleboats.agents.rust_mcts_engine import RustMctsEngine
from battleboats.agents.scripted_agent import DOCTRINE_NAMES
from battleboats.core.combat import MAT_PORT_VALUE, ship_value
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.ui.maps import resolve_map

DEFAULT_ITERATIONS = 400
DEFAULT_MAP_SPEC = "48x24"
MAX_ACTIONS_PER_TURN = 400  # harvest / rust_play forced-end threshold
OUTPUT_DIR = Path("/home/nick/Desktop/repos/Battleboats/runs/mcts_vs_doctrines_rust_batched")

# Display labels only; code and logs use the repo's DOCTRINE_NAMES.
DOCTRINE_LABELS = {
    "RAIDER_BLITZ": "Raider Blitz",
    "ECONOMIC_BOOM": "Economic Boom",
    "WOLFPACK_STEALTH": "Wolfpack Stealth",
    "CAPITAL_ARMADA": "Capital Armada",
    "ISLAND_HOPPER": "Island Hopper",
    "BALANCED_ADAPTIVE": "Balanced Adaptive",
}

# Per-seat win rates in the 72k mixed-doctrine harvest (Fog Commander self-play).
HARVEST_SEAT_WINRATE = {
    "RAIDER_BLITZ": 0.403,
    "WOLFPACK_STEALTH": 0.398,
    "ECONOMIC_BOOM": 0.384,
    "BALANCED_ADAPTIVE": 0.372,
    "CAPITAL_ARMADA": 0.371,
    "ISLAND_HOPPER": 0.355,
}


def _as_type(value) -> ShipType:
    return value if isinstance(value, ShipType) else ShipType(value)


def _ship_value_row(row) -> float:
    typ = _as_type(row["type"] if isinstance(row, dict) else row.type)
    return ship_value(SimpleNamespace(type=typ, stats=BASE_STATS[typ]))


def _is_end_turn(action) -> bool:
    return list(action) == ["e"]


def _material_parts(engine, me: int) -> Tuple[float, float]:
    """(ship+port material, cash+stockpile) from `me`'s perspective."""
    ships = engine.list_ships()
    ports = engine.list_ports()
    players = engine.list_players()
    ship_mat = sum(_ship_value_row(s) for s in ships if int(s["owner"]) == me)
    own_ports = [p for p in ports if int(p["owner"]) == me]
    port_mat = MAT_PORT_VALUE * len(own_ports)
    cash = float(next(p["cash"] for p in players if int(p["id"]) == me))
    stockpile = float(sum(p["stockpile"] for p in own_ports))
    return ship_mat + port_mat, cash + stockpile


def material_scores(engine) -> Dict[str, Any]:
    m0, e0 = _material_parts(engine, 0)
    m1, e1 = _material_parts(engine, 1)
    if m0 != m1:
        winner: Optional[int] = 0 if m0 > m1 else 1
    elif e0 != e1:
        winner = 0 if e0 > e1 else 1
    else:
        winner = None
    return {
        "material": {0: m0, 1: m1},
        "econ": {0: e0, 1: e1},
        "material_winner": winner,
    }


def play_one_game(job: Dict[str, Any]) -> Dict[str, Any]:
    """Play one MCTS vs scripted-agent game. Designed to run in a worker."""
    os.nice(19)
    doctrine = str(job["doctrine"])
    mcts_seat = int(job["mcts_seat"])
    fc_seat = 1 - mcts_seat
    seed = int(job["seed"])
    iterations = int(job["iterations"])
    map_spec = str(job["map_spec"])
    # Unique dir per game so workers do not clobber the shared
    # mapConfig_<preset>.yaml that resolve_map writes in map_dir.
    map_dir = Path(job["map_dir"]) / f"game_{int(job['game_idx'])}_seed_{seed}"
    map_dir.mkdir(parents=True, exist_ok=True)

    rm = resolve_map(map_spec, seed=seed, map_dir=map_dir)
    max_turns = int(job.get("max_turns") or rm.recommended_max_turns)
    max_steps = int(job.get("max_steps") or (8000 * max_turns // 80))

    import battleboats_core

    engine = RustMctsEngine.create(str(rm.path), seed=seed)
    value_net = load_mcts_value_net(job["checkpoint"], job.get("device") or "auto")
    scripted = battleboats_core.ScriptedAgent(doctrine)
    mcts_rng = random.Random(seed)
    # Same seeding pattern harvest uses for a seat's scripted-agent RNG.
    fc_rng = random.Random(seed * 2 + (fc_seat + 1))
    leaf_batch = int(job.get("leaf_batch") or DEFAULT_LEAF_BATCH)
    rollout_steps = int(job.get("rollout_steps") if job.get("rollout_steps") is not None else ROLLOUT_STEP_BUDGET)

    mcts_times: List[float] = []
    per_turn = {0: 0, 1: 0}
    last_turn = -1
    forced_end = {0: 0, 1: 0}
    steps = 0
    t0 = time.perf_counter()
    reason = "ok"

    while (not engine.is_terminal()) and engine.turn < max_turns and steps < max_steps:
        turn = int(engine.turn)
        if turn != last_turn:
            per_turn = {0: 0, 1: 0}
            last_turn = turn
        pid = int(engine.current_player)
        if pid == mcts_seat:
            dt0 = time.perf_counter()
            action, root_value = godmode_mcts_search(
                engine,
                pid,
                mcts_rng,
                value_net,
                iterations=iterations,
                leaf_batch=leaf_batch,
                rollout_steps=rollout_steps,
            )
            mcts_times.append(time.perf_counter() - dt0)
            root_value = float(root_value)
        else:
            action = scripted.act(engine.raw, pid, fc_rng)
            root_value = None
        per_turn[pid] += 1
        if per_turn[pid] > MAX_ACTIONS_PER_TURN and not _is_end_turn(action):
            action = ["e"]
            forced_end[pid] += 1
        engine.step(action)
        steps += 1

    if engine.is_terminal():
        reason = "home_capture"
        official = int(engine.winner) if engine.winner is not None else None
    elif engine.turn >= max_turns:
        reason = "turn_cap"
        official = None
    else:
        reason = "step_cap"
        official = None

    mats = material_scores(engine)
    if official is None:
        outcome = "draw"
        mcts_result = "D"
    elif official == mcts_seat:
        outcome = "mcts_win"
        mcts_result = "W"
    else:
        outcome = "mcts_loss"
        mcts_result = "L"

    tb_winner = mats["material_winner"]
    if official is not None:
        tiebreak_result = mcts_result
    elif tb_winner is None:
        tiebreak_result = "D"
    elif tb_winner == mcts_seat:
        tiebreak_result = "W"
    else:
        tiebreak_result = "L"

    n = len(mcts_times)
    wall = time.perf_counter() - t0
    record = {
        "game_idx": int(job["game_idx"]),
        "doctrine": doctrine,
        "mcts_seat": mcts_seat,
        "fc_seat": fc_seat,
        "seed": seed,
        "iterations": iterations,
        "map_spec": map_spec,
        "map_path": str(rm.path),
        "map_width": int(rm.width),
        "map_height": int(rm.height),
        "max_turns": max_turns,
        "max_steps": max_steps,
        "turns": int(engine.turn),
        "steps": steps,
        "mcts_decisions": n,
        "winner": official,
        "reason": reason,
        "outcome": outcome,
        "mcts_result": mcts_result,
        "material": mats["material"],
        "econ": mats["econ"],
        "material_winner": tb_winner,
        "tiebreak_result": tiebreak_result,
        "forced_end": forced_end,
        "wall_time_s": wall,
        "mcts_ms_per_decision": (1000.0 * sum(mcts_times) / n) if n else 0.0,
        "mcts_decision_sum_s": sum(mcts_times),
        "mcts_median_decision_s": (statistics.median(mcts_times) if mcts_times else 0.0),
        "status": "ok",
    }
    return record


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return play_one_game(job)
    except Exception as exc:  # noqa: BLE001 — worker must return, not die
        import traceback

        return {
            "game_idx": int(job.get("game_idx", -1)),
            "doctrine": job.get("doctrine"),
            "mcts_seat": job.get("mcts_seat"),
            "seed": job.get("seed"),
            "status": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }


def build_jobs(
    *,
    games_per_seat: int,
    base_seed: int,
    iterations: int,
    map_spec: str,
    map_dir: Path,
    doctrines: List[str],
    max_turns: Optional[int],
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    game_idx = 0
    for d_i, doctrine in enumerate(doctrines):
        for seat in (0, 1):
            for k in range(games_per_seat):
                # Distinct seeds per (doctrine, seat, replicate).
                seed = int(base_seed) + d_i * 100 + seat * 10 + k
                jobs.append(
                    {
                        "game_idx": game_idx,
                        "doctrine": doctrine,
                        "mcts_seat": seat,
                        "seed": seed,
                        "iterations": iterations,
                        "map_spec": map_spec,
                        "map_dir": str(map_dir),
                        "max_turns": max_turns,
                    }
                )
                game_idx += 1
    return jobs


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    by: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOCTRINE_NAMES}
    for r in ok:
        by.setdefault(r["doctrine"], []).append(r)

    def pack(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        w = sum(1 for r in rows if r["mcts_result"] == "W")
        l = sum(1 for r in rows if r["mcts_result"] == "L")
        d = sum(1 for r in rows if r["mcts_result"] == "D")
        n = len(rows)
        tb_w = sum(1 for r in rows if r["tiebreak_result"] == "W")
        tb_l = sum(1 for r in rows if r["tiebreak_result"] == "L")
        tb_d = sum(1 for r in rows if r["tiebreak_result"] == "D")
        seat0 = [r for r in rows if r["mcts_seat"] == 0]
        seat1 = [r for r in rows if r["mcts_seat"] == 1]

        def wr(rs: List[Dict[str, Any]], key: str = "mcts_result") -> Optional[float]:
            return (sum(1 for r in rs if r[key] == "W") / len(rs)) if rs else None

        caps = sum(1 for r in rows if r["reason"] in {"turn_cap", "step_cap"})
        mcts_dec = sum(int(r["mcts_decisions"]) for r in rows)
        mcts_time = sum(float(r["mcts_decision_sum_s"]) for r in rows)
        return {
            "games": n,
            "w": w,
            "l": l,
            "d": d,
            "win_rate": (w / n) if n else None,
            "tiebreak_w": tb_w,
            "tiebreak_l": tb_l,
            "tiebreak_d": tb_d,
            "tiebreak_win_rate": (tb_w / n) if n else None,
            "avg_turns": (statistics.fmean(r["turns"] for r in rows) if rows else None),
            "capped": caps,
            "seat0": {
                "games": len(seat0),
                "w": sum(1 for r in seat0 if r["mcts_result"] == "W"),
                "l": sum(1 for r in seat0 if r["mcts_result"] == "L"),
                "d": sum(1 for r in seat0 if r["mcts_result"] == "D"),
                "win_rate": wr(seat0),
            },
            "seat1": {
                "games": len(seat1),
                "w": sum(1 for r in seat1 if r["mcts_result"] == "W"),
                "l": sum(1 for r in seat1 if r["mcts_result"] == "L"),
                "d": sum(1 for r in seat1 if r["mcts_result"] == "D"),
                "win_rate": wr(seat1),
            },
            "mcts_decisions": mcts_dec,
            "mcts_ms_per_decision": (1000.0 * mcts_time / mcts_dec) if mcts_dec else None,
            "wall_sum_s": sum(float(r["wall_time_s"]) for r in rows),
        }

    per_doctrine = {d: pack(by[d]) for d in DOCTRINE_NAMES if by.get(d)}
    overall = pack(ok)
    return {
        "ok": len(ok),
        "crashed": len(records) - len(ok),
        "per_doctrine": per_doctrine,
        "overall": overall,
    }


def _print_table(summary: Dict[str, Any]) -> None:
    print()
    print(
        f"{'Doctrine':<22} {'W-L-D':>9} {'WR':>7} {'TB-WR':>7} "
        f"{'avg turns':>10} {'caps':>5} {'seat0 W-L-D':>13} {'seat1 W-L-D':>13}",
        flush=True,
    )
    for d in DOCTRINE_NAMES:
        s = summary["per_doctrine"].get(d)
        if not s:
            continue
        label = DOCTRINE_LABELS.get(d, d)
        s0, s1 = s["seat0"], s["seat1"]
        wr = f"{100 * s['win_rate']:.1f}%" if s["win_rate"] is not None else "  n/a"
        tb = f"{100 * s['tiebreak_win_rate']:.1f}%" if s["tiebreak_win_rate"] is not None else "  n/a"
        print(
            f"{label:<22} {s['w']}-{s['l']}-{s['d']:>1} {wr:>7} {tb:>7} "
            f"{s['avg_turns'] or 0:10.1f} {s['capped']:5d} "
            f"{s0['w']}-{s0['l']}-{s0['d']:>1} {s1['w']}-{s1['l']}-{s1['d']:>1}",
            flush=True,
        )
    o = summary["overall"]
    if o["games"]:
        print(
            f"{'OVERALL':<22} {o['w']}-{o['l']}-{o['d']:>1} "
            f"{100 * o['win_rate']:.1f}% {100 * o['tiebreak_win_rate']:.1f}% "
            f"{o['avg_turns']:10.1f} {o['capped']:5d}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS (value-net leaf) vs scripted-agent doctrines.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; each game gets a distinct offset.")
    parser.add_argument("--map", default=DEFAULT_MAP_SPEC)
    parser.add_argument("--games-per-seat", type=int, default=5, help="Games per doctrine per MCTS seat (10 total per doctrine at 5).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=None, help="Override harvest recommended cap.")
    parser.add_argument("--pilot", type=int, default=0, help="If >0, run only this many games and exit.")
    parser.add_argument("--doctrines", type=str, default="", help="Optional comma-separated doctrine names.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Value-net checkpoint for MCTS leaves.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, cuda:N, or auto.")
    parser.add_argument("--leaf-batch", type=int, default=DEFAULT_LEAF_BATCH, help="Virtual-loss leaf batch size.")
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEP_BUDGET, help="Tactical rollout steps before the value net.")
    parser.add_argument(
        "--probes",
        action="store_true",
        help="Run only the three fast-loss seeds from the heuristic eval: "
        "Raider Blitz 13/14 and Balanced Adaptive 510 (MCTS seat 1).",
    )
    args = parser.parse_args()

    if args.doctrines:
        doctrines = [x.strip().upper() for x in args.doctrines.split(",") if x.strip()]
        unknown = [d for d in doctrines if d not in DOCTRINE_NAMES]
        if unknown:
            raise SystemExit(f"unknown doctrine(s) {unknown}; choose from {list(DOCTRINE_NAMES)}")
    else:
        doctrines = list(DOCTRINE_NAMES)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        games_per_seat=args.games_per_seat,
        base_seed=args.seed,
        iterations=args.iterations,
        map_spec=args.map,
        map_dir=map_dir,
        doctrines=doctrines,
        max_turns=args.max_turns,
    )
    for job in jobs:
        job["checkpoint"] = str(args.checkpoint)
        job["device"] = args.device
        job["leaf_batch"] = int(args.leaf_batch)
        job["rollout_steps"] = int(args.rollout_steps)
    if args.probes:
        wanted = {( "RAIDER_BLITZ", 1, 13), ("RAIDER_BLITZ", 1, 14), ("BALANCED_ADAPTIVE", 1, 510)}
        jobs = [j for j in jobs if (j["doctrine"], j["mcts_seat"], j["seed"]) in wanted]
        if len(jobs) != 3:
            raise SystemExit(f"--probes expected 3 jobs, found {len(jobs)}: {jobs}")
    if args.pilot > 0:
        jobs = jobs[: args.pilot]

    workers = max(1, min(int(args.workers), len(jobs)))
    print(
        f"MCTS vs scripted-agent doctrines  games={len(jobs)}  "
        f"doctrines={len(doctrines)}  games_per_seat={args.games_per_seat}  "
        f"iters={args.iterations}  batch={args.leaf_batch}  rollout={args.rollout_steps}  "
        f"workers={workers}  map={args.map}  "
        f"max_turns={'harvest' if args.max_turns is None else args.max_turns}",
        flush=True,
    )
    print(
        "Caveat: MCTS is god-mode (perfect information). The scripted agent "
        "plays under the same fog rules as harvest.",
        flush=True,
    )

    jsonl_path = out_dir / ("pilot.jsonl" if args.pilot else "games.jsonl")
    # Fresh file for a new run; append as games complete.
    if jsonl_path.exists():
        jsonl_path.unlink()

    records: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            status = rec.get("status")
            if status != "ok":
                print(
                    f"  CRASH game={rec.get('game_idx')} {rec.get('doctrine')} "
                    f"seat={rec.get('mcts_seat')} seed={rec.get('seed')}: {rec.get('error')}",
                    flush=True,
                )
                continue
            print(
                f"  [{len(records):3d}/{len(jobs)}] {rec['doctrine']:<20} "
                f"mcts_seat={rec['mcts_seat']} seed={rec['seed']:<5} "
                f"{rec['mcts_result']}  {rec['reason']:<13} "
                f"turns={rec['turns']:<4} dec={rec['mcts_decisions']:<4} "
                f"{rec['wall_time_s']:.1f}s  {rec['mcts_ms_per_decision']:.1f} ms/dec",
                flush=True,
            )

    wall = time.perf_counter() - t0
    summary = _summarize(records)
    payload = {
        "settings": {
            "iterations": args.iterations,
            "base_seed": args.seed,
            "map": args.map,
            "games_per_seat": args.games_per_seat,
            "workers": workers,
            "max_turns": args.max_turns,
            "pilot": args.pilot or None,
            "doctrines": doctrines,
            "mcts": "godmode_mcts, rust engine, tactical rollout then batched value-net leaf",
            "opponent": "scripted_agent (fog, forced doctrine)",
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "leaf_batch": args.leaf_batch,
            "rollout_steps": args.rollout_steps,
            "engine": "battleboats_core.Game via RustMctsEngine (no heuristic hydration)",
            "cap_policy": "turn/step cap = draw; material+econ tiebreak stored separately",
        },
        "harvest_seat_winrate_baseline": HARVEST_SEAT_WINRATE,
        "elapsed_s": wall,
        "summary": summary,
        "games": records,
    }
    out_json = out_dir / ("pilot_summary.json" if args.pilot else "summary.json")
    out_json.write_text(json.dumps(payload, indent=2))
    _print_table(summary)
    o = summary["overall"]
    print(f"\nWall time: {wall:.1f}s  workers={workers}", flush=True)
    if o.get("mcts_ms_per_decision") is not None:
        print(f"MCTS ms/decision: {o['mcts_ms_per_decision']:.1f}", flush=True)
    if args.pilot and o["games"]:
        per = wall / max(1, o["games"])
        est_60 = per * 60 / max(1, workers) * workers / min(workers, 60)  # wall if same workers
        # Parallel estimate: (60 / workers) * mean per-game wall, using observed wall/games * workers as mean game time.
        mean_game = wall * workers / o["games"]
        est_full = mean_game * 60 / workers
        est_reduced = mean_game * 36 / workers
        print(
            f"Pilot estimate: mean game ~{mean_game:.1f}s; "
            f"60 games / {workers} workers ~{est_full/60:.1f} min; "
            f"36 games ~{est_reduced/60:.1f} min.",
            flush=True,
        )
        payload["pilot_estimate"] = {
            "mean_game_s": mean_game,
            "est_60_games_s": est_full,
            "est_36_games_s": est_reduced,
        }
        out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
