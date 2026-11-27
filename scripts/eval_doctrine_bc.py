#!/usr/bin/env python3
"""Evaluate per-doctrine BC clones on 48x24 against scripted doctrines and random.

For each clone (``--clone DOCTRINE=path/to/best.pt``, repeatable) plays:
  - ``teacher``: vs the scripted agent forced to the clone's own doctrine
  - ``others``:  vs the other five doctrines, split evenly
  - ``random``:  vs the random agent
Seats alternate. The clone samples from its policy (``policy.act``), the way
the v0.4.x BC checkpoints are meant to be used. Fresh 48x24 maps per seed
(seeds start at --seed, disjoint from the value-harvest seeds by default).

Reports W/L/D, average turns, and each side's buy mix so a clone's fleet can
be compared with its teacher's.

    PYTHONPATH=. python scripts/eval_doctrine_bc.py \\
        --clone RAIDER_BLITZ=runs/bc_doctrines_v0.6.0/RAIDER_BLITZ/best.pt \\
        --games 20 --workers 8 --output-dir runs/bc_doctrines_v0.6.0/eval
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from battleboats.agents.registry import make_agent  # noqa: E402
from battleboats.agents.scripted_agent import DOCTRINE_NAMES, engine_accepts  # noqa: E402
from battleboats.core.actions import BuildShipAction, CapturePortAction, EndTurnAction  # noqa: E402
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.ui.maps import resolve_map  # noqa: E402

MAX_ACTIONS_PER_TURN = 400
_POLICIES: Dict[str, Any] = {}


def _policy(path: str):
    import torch

    from battleboats.training.checkpoints import load_checkpoint

    if path not in _POLICIES:
        torch.set_num_threads(1)
        net, _ = load_checkpoint(Path(path), torch.device("cpu"))
        net.eval()
        net.greedy = False
        _POLICIES[path] = net
    return _POLICIES[path]


def _policy_action(net, engine, pid: int):
    import torch

    from battleboats.envs.action_masks import ActionMasks

    masks = ActionMasks(engine, pid)
    with torch.no_grad():
        a, v, t, _lp, _val = net.act(masks.tokens, masks.pad_mask, masks, skill_idx=0, globals=masks.globals)
    return masks.to_action(asset_idx=a, verb_idx=v, target_idx=t)


def play(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        os.nice(19)
    except OSError:
        pass
    try:
        import torch

        seed = int(job["seed"])
        torch.manual_seed(seed)
        clone_seat = int(job["clone_seat"])
        with tempfile.TemporaryDirectory() as td:
            rm = resolve_map(job["map"], seed=seed, map_dir=Path(td))
            max_turns = int(rm.recommended_max_turns)
            max_steps = 8000 * max_turns // 80
            engine = gameEngine(str(rm.path))
            engine.reset(seed=seed)
        engine._refresh_sightings()
        teacher_side = job["checkpoint"] == "scripted"
        net = None if teacher_side else _policy(job["checkpoint"])
        focal = make_agent("scripted_agent", doctrine=job["clone"]) if teacher_side else None
        opp_kind = job["opponent"]
        opp = make_agent("random") if opp_kind == "random" else make_agent("scripted_agent", doctrine=opp_kind)
        rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
        buys = {0: Counter(), 1: Counter()}
        captures = {0: 0, 1: 0}
        per_turn: Counter = Counter()
        forced_end = {0: 0, 1: 0}
        steps = 0
        t0 = time.time()
        while engine.winner is None and engine.turn < max_turns and steps < max_steps:
            pid = int(engine.current_player)
            if pid == clone_seat:
                if focal is not None:
                    action = focal.act(engine, pid, rngs[pid])
                    if not engine_accepts(engine, pid, action):
                        action = EndTurnAction()
                else:
                    action = _policy_action(net, engine, pid)
            else:
                action = opp.act(engine, pid, rngs[pid])
                if not engine_accepts(engine, pid, action):
                    action = EndTurnAction()
            per_turn[(pid, engine.turn)] += 1
            if per_turn[(pid, engine.turn)] > MAX_ACTIONS_PER_TURN and not isinstance(action, EndTurnAction):
                forced_end[pid] += 1
                action = EndTurnAction()
            if isinstance(action, BuildShipAction):
                buys[pid][action.ship_type.value] += 1
            if isinstance(action, CapturePortAction):
                captures[pid] += 1
            engine.step(action)
            steps += 1
        winner = engine.winner
        res = "D" if winner is None else ("W" if winner == clone_seat else "L")
        return {
            "status": "ok",
            **{k: job[k] for k in ("clone", "bucket", "opponent", "seed", "clone_seat")},
            "winner": winner,
            "result": res,
            "turns": int(engine.turn),
            "steps": steps,
            "clone_buys": dict(buys[clone_seat]),
            "opp_buys": dict(buys[1 - clone_seat]),
            "clone_captures": captures[clone_seat],
            "opp_captures": captures[1 - clone_seat],
            "forced_end": forced_end[clone_seat],
            "wall_sec": time.time() - t0,
        }
    except Exception as exc:
        return {"status": "crash", **job, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc(limit=8)}


def build_jobs(clones: Dict[str, str], games: int, seed: int, map_spec: str, buckets: List[str]) -> List[Dict[str, Any]]:
    jobs = []
    for doc, ckpt in clones.items():
        others = [d for d in DOCTRINE_NAMES if d != doc]
        for bucket in buckets:
            for i in range(games):
                if bucket == "teacher":
                    opp = doc
                elif bucket == "others":
                    opp = others[i % len(others)]
                else:
                    opp = "random"
                jobs.append(
                    {
                        "clone": doc,
                        "checkpoint": ckpt,
                        "bucket": bucket,
                        "opponent": opp,
                        "seed": seed + i,  # same seeds/maps across clones and buckets
                        "clone_seat": i % 2,
                        "map": map_spec,
                    }
                )
    return jobs


def _wld(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter(r["result"] for r in rs)
    n = len(rs)
    return {
        "n": n,
        "W": c["W"],
        "L": c["L"],
        "D": c["D"],
        "win_rate": (c["W"] / n) if n else None,
        "score": ((c["W"] + 0.5 * c["D"]) / n) if n else None,
        "avg_turns": (sum(r["turns"] for r in rs) / n) if n else None,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    out: Dict[str, Any] = {"n_ok": len(ok), "n_crash": len(results) - len(ok), "clones": {}}
    by = defaultdict(list)
    for r in ok:
        by[r["clone"]].append(r)
    for doc, rs in by.items():
        d: Dict[str, Any] = {}
        for bucket in ("teacher", "others", "random"):
            b = [r for r in rs if r["bucket"] == bucket]
            if not b:
                continue
            d[bucket] = _wld(b)
            cb: Counter = Counter()
            for r in b:
                cb.update(r["clone_buys"])
            d[bucket]["clone_buys"] = dict(cb.most_common())
            if bucket == "teacher":
                tb: Counter = Counter()
                for r in b:
                    tb.update(r["opp_buys"])
                d[bucket]["teacher_buys"] = dict(tb.most_common())
            if bucket == "others":
                per = defaultdict(list)
                for r in b:
                    per[r["opponent"]].append(r)
                d[bucket]["per_opponent"] = {k: _wld(v) for k, v in per.items()}
        d["forced_end"] = sum(r["forced_end"] for r in rs)
        out["clones"][doc] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--clone",
        action="append",
        required=True,
        help="DOCTRINE=checkpoint.pt (repeatable). DOCTRINE=scripted plays the teacher itself (baseline).",
    )
    ap.add_argument("--games", type=int, default=20, help="Games per bucket per clone.")
    ap.add_argument("--buckets", type=str, default="teacher,others,random")
    ap.add_argument("--map", type=str, default="48x24")
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    clones: Dict[str, str] = {}
    for spec in args.clone:
        doc, _, path = spec.partition("=")
        doc = doc.strip().upper()
        if doc not in DOCTRINE_NAMES:
            raise SystemExit(f"unknown doctrine {doc!r}")
        clones[doc] = "scripted" if path.strip() == "scripted" else str(Path(path).resolve())
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    jobs = build_jobs(clones, args.games, args.seed, args.map, buckets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games_path = args.output_dir / "games.jsonl"
    print(f"{len(jobs)} games, {args.workers} workers", flush=True)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(games_path, "w") as fh:
        futs = [ex.submit(play, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            fh.write(json.dumps({k: v for k, v in r.items() if k != "tb"}) + "\n")
            fh.flush()
            if r["status"] != "ok":
                print("CRASH", r.get("error"), r.get("tb"), flush=True)
            else:
                print(
                    f"{i}/{len(jobs)} {r['clone']} {r['bucket']} vs {r['opponent']} seat={r['clone_seat']} "
                    f"{r['result']} turns={r['turns']} {r['wall_sec']:.0f}s  elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    summary = summarize(results)
    summary["elapsed_sec"] = time.time() - t0
    summary["config"] = {k: str(v) for k, v in vars(args).items()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
