"""Evaluate the scripted agent against itself, the scripted baseline, and random.

    PYTHONPATH=. python scripts/eval_scripted_agent.py --maps 32x16 48x24 \
        --games 100 --workers 12 --output runs/eval_scripted_agent/summary.json

Each matchup alternates seats (even game index: scripted_agent is seat 0).
Every game uses a fresh preset map (``resolve_map(preset, seed=...)``, same
generator as the play UI) and the UI's recommended turn cap. Per-seat RNGs
are ``Random(seed*2+1)`` / ``Random(seed*2+2)``, as in the other eval scripts.

Reports: decisive rate, wins by agent and seat, average game length, buy
mix per agent, doctrine win rates, ms/decision for scripted_agent, crashes,
actions the engine would reject ("illegal"), and forced end turns.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from battleboats.agents.registry import (  # noqa: E402
    agent_kind_choices,
    make_agent,
    normalize_agent_kind,
)
from battleboats.agents.scripted_agent import engine_accepts  # noqa: E402
from battleboats.core.actions import BuildShipAction, EndTurnAction  # noqa: E402
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.ui.maps import resolve_map  # noqa: E402

MAX_ACTIONS_PER_TURN = 400  # harness guard; the agent caps itself at 250


def play_game(job: Dict[str, Any]) -> Dict[str, Any]:
    seed = job["seed"]
    kinds = {0: job["kinds"][0], 1: job["kinds"][1]}
    out: Dict[str, Any] = {**job, "winner": None, "turns": 0, "steps": 0, "crash": None,
                           "illegal": {0: 0, 1: 0}, "forced_end": {0: 0, 1: 0},
                           "buys": {0: {}, 1: {}}, "doctrine": {0: None, 1: None},
                           "fc_ms": [], "max_turns": None}
    try:
        with tempfile.TemporaryDirectory() as td:
            rm = resolve_map(job["map"], seed=seed, map_dir=Path(td))
            engine = gameEngine(str(rm.path))
            engine.reset(seed=seed)
            max_turns = int(rm.recommended_max_turns)
        out["max_turns"] = max_turns
        max_steps = 8000 * max_turns // 80
        rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
        agents = {pid: make_agent(kinds[pid]) for pid in (0, 1)}
        buys = {0: Counter(), 1: Counter()}
        per_turn = Counter()
        fc_ms: List[float] = []
        steps = 0
        while engine.winner is None and engine.turn < max_turns and steps < max_steps:
            pid = engine.current_player
            legal = engine.enumerate_legal(pid)
            t0 = time.perf_counter()
            action = agents[pid].act(engine, pid, rngs[pid])
            dt = (time.perf_counter() - t0) * 1000.0
            if normalize_agent_kind(kinds[pid]) == "scripted_agent":
                fc_ms.append(dt)
            key = (pid, engine.turn)
            per_turn[key] += 1
            if not engine_accepts(engine, pid, action, legal):
                out["illegal"][pid] += 1
                action = EndTurnAction()
            elif per_turn[key] > MAX_ACTIONS_PER_TURN and not isinstance(action, EndTurnAction):
                out["forced_end"][pid] += 1
                action = EndTurnAction()
            if isinstance(action, BuildShipAction):
                buys[pid][action.ship_type.value] += 1
            engine.step(action)
            steps += 1
        out["winner"] = engine.winner
        out["turns"] = int(engine.turn)
        out["steps"] = steps
        out["buys"] = {pid: dict(buys[pid]) for pid in (0, 1)}
        out["doctrine"] = {pid: getattr(agents[pid], "doctrine_for", lambda _p: None)(pid) for pid in (0, 1)}
        out["fc_ms"] = fc_ms
    except Exception as exc:  # report, never hide
        out["crash"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    return out


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def summarize(games: List[Dict[str, Any]], focal: str = "scripted_agent") -> Dict[str, Any]:
    n = len(games)
    ok = [g for g in games if g["crash"] is None]
    decisive = [g for g in ok if g["winner"] is not None]
    by_seat = Counter()
    by_kind = Counter()
    focal_seat_wins = Counter()
    focal_seat_games = Counter()
    for g in ok:
        k = {int(p): v for p, v in enumerate(g["kinds"])}
        if g["winner"] is not None:
            by_seat[g["winner"]] += 1
            by_kind[k[g["winner"]]] += 1
        for pid in (0, 1):
            if k[pid] == focal:
                focal_seat_games[pid] += 1
                if g["winner"] == pid:
                    focal_seat_wins[pid] += 1
    buys: Dict[str, Counter] = defaultdict(Counter)
    doctrine = defaultdict(lambda: Counter())
    ms: List[float] = []
    for g in ok:
        for pid in (0, 1):
            kind = g["kinds"][pid]
            buys[kind].update(g["buys"][pid] if pid in g["buys"] else g["buys"][str(pid)])
            if kind == focal:
                d = g["doctrine"][pid]
                res = "win" if g["winner"] == pid else ("draw" if g["winner"] is None else "loss")
                doctrine[d][res] += 1
                doctrine[d]["games"] += 1
        ms.extend(g["fc_ms"])
    buy_mix = {}
    for kind, c in buys.items():
        tot = sum(c.values())
        buy_mix[kind] = {"total": tot, "pct": {t: _pct(v, tot) for t, v in sorted(c.items())}}
    ms_sorted = sorted(ms)

    def q(p):
        return round(ms_sorted[min(len(ms_sorted) - 1, int(p * len(ms_sorted)))], 3) if ms_sorted else None

    return {
        "games": n,
        "crashes": n - len(ok),
        "decisive": len(decisive),
        "decisive_pct": _pct(len(decisive), len(ok)),
        "wins_by_seat": {str(s): by_seat[s] for s in (0, 1)},
        "wins_by_kind": dict(by_kind),
        "focal_win_by_seat": {str(s): f"{focal_seat_wins[s]}/{focal_seat_games[s]}" for s in (0, 1)},
        "avg_turns": round(statistics.mean(g["turns"] for g in ok), 1) if ok else None,
        "avg_turns_decisive": round(statistics.mean(g["turns"] for g in decisive), 1) if decisive else None,
        "max_turns_cap": sorted({g["max_turns"] for g in ok}),
        "illegal": sum(g["illegal"][0] + g["illegal"][1] for g in ok),
        "forced_end_turns": sum(g["forced_end"][0] + g["forced_end"][1] for g in ok),
        "buy_mix": buy_mix,
        "doctrine": {d: dict(c) for d, c in sorted(doctrine.items(), key=lambda kv: str(kv[0]))},
        "fc_decisions": len(ms),
        "fc_ms_mean": round(statistics.mean(ms), 3) if ms else None,
        "fc_ms_p50": q(0.5),
        "fc_ms_p95": q(0.95),
        "fc_ms_max": round(ms_sorted[-1], 2) if ms_sorted else None,
        "crash_samples": [g["crash"] for g in games if g["crash"]][:3],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=["32x16"])
    ap.add_argument("--games", type=int, default=100, help="games per matchup per map")
    ap.add_argument("--opponents", nargs="+", default=["scripted_agent", "scripted", "random"],
                    choices=agent_kind_choices(),
                    help="fog_commander is a deprecated alias for scripted_agent")
    ap.add_argument("--focal", default="scripted_agent", choices=agent_kind_choices())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    args.focal = normalize_agent_kind(args.focal)
    args.opponents = [normalize_agent_kind(o) for o in args.opponents]

    jobs = []
    for mi, m in enumerate(args.maps):
        for oi, opp in enumerate(args.opponents):
            for i in range(args.games):
                seed = args.seed + 1_000_000 * mi + 10_000 * oi + i
                kinds = (args.focal, opp) if i % 2 == 0 else (opp, args.focal)
                jobs.append({"map": m, "opponent": opp, "index": i, "seed": seed, "kinds": kinds})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(play_game, jobs, chunksize=1))
    wall = time.time() - t0
    report: Dict[str, Any] = {"wall_seconds": round(wall, 1), "args": {k: str(v) for k, v in vars(args).items()},
                              "matchups": {}}
    for m in args.maps:
        for opp in args.opponents:
            gs = [r for r in results if r["map"] == m and r["opponent"] == opp]
            report["matchups"][f"{m}:{args.focal}_vs_{opp}"] = summarize(gs, args.focal)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        games_path = args.output.with_name(args.output.stem + "_games.jsonl")
        with games_path.open("w") as f:
            for r in results:
                r = {k: v for k, v in r.items() if k != "fc_ms"}
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
