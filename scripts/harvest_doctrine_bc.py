#!/usr/bin/env python3
"""Per-doctrine behavior-cloning labels from the mixed-doctrine value harvest.

The 72k value harvest (``scripts/harvest_value.py``, 48x24, all 36 ordered
doctrine pairs, 2000 games each) stores every seat's fogged observation and the
full action sequence of each game. This script replays a subset of those games
on the Python engine and labels each acting seat's action in the pointer
policy's (asset, verb, target) space. Rows are written per doctrine, so each
personality gets its own BC set with games against all six doctrines from both
seats.

Observations are the stored ``build_observation(engine, acting_seat)`` tensors
(the same ``build_entity_tokens`` / ``build_globals`` the policy sees). Use
``--verify-tokens`` to rebuild them during replay and check they match.

    PYTHONPATH=. python scripts/harvest_doctrine_bc.py \\
        --source runs/value_harvest_48x24_fc_mixed_72k_v0.5.0 \\
        --games-per-pair 50 --workers 10 --output-dir runs/bc_doctrines_v0.6.0
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
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402

from battleboats.agents.scripted_agent import DOCTRINE_NAMES, canonical_build_action  # noqa: E402
from battleboats.core.actions import (  # noqa: E402
    BuildShipAction,
    CapturePortAction,
    MerchantLoadAction,
    MerchantUnloadAction,
)
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.envs.action_masks import ActionMasks  # noqa: E402
from battleboats.envs.observation import build_entity_tokens, build_globals  # noqa: E402
from battleboats.training.policy import VERB_NAMES  # noqa: E402
from battleboats.training.tactical_bc import GAMMA_DEFAULT, discounted_returns, write_game_npz  # noqa: E402

# Verbs whose (asset, verb, target) triple can collide with another legal action.
_ROUNDTRIP_TYPES = (BuildShipAction, CapturePortAction, MerchantLoadAction, MerchantUnloadAction)


def _decode(rec):
    from harvest_value import decode_action

    return decode_action(rec)


def label_game(job: Dict[str, Any]) -> Dict[str, Any]:
    """Replay one value-harvest shard and write one BC shard per seat."""
    try:
        os.nice(19)
    except OSError:
        pass
    src = Path(job["src"])
    out_root = Path(job["out_root"])
    verify = bool(job.get("verify_tokens"))
    gamma = float(job.get("gamma", GAMMA_DEFAULT))
    try:
        with np.load(src, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            map_text = str(z["map_json"])
            actions = json.loads(str(z["actions_json"]))
            tokens_data = z["tokens_data"]
            splits = z["row_splits"]
            globals_arr = z["globals"]
            seat_arr = z["seat"]
            acting_arr = z["acting_seat"]
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "map.json"
            mp.write_text(map_text)
            engine = gameEngine(str(mp))
            engine.reset(seed=int(meta["seed"]))
        engine._refresh_sightings()
        doctrines = {int(k): v for k, v in meta["doctrines"].items()}
        rows: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        errors: Counter = Counter()
        mismatched_tokens = 0
        for step, rec in enumerate(actions):
            pid = int(engine.current_player)
            r = 2 * step + pid
            if int(seat_arr[r]) != pid or int(acting_arr[r]) != pid:
                raise RuntimeError(f"row layout mismatch at step {step}")
            action = _decode(rec)
            labelled = canonical_build_action(engine, pid, action)
            masks = ActionMasks(engine, pid)
            try:
                a, v, t = masks.factor(labelled)
                ok = True
                if isinstance(labelled, _ROUNDTRIP_TYPES):
                    masks._ensure_built()
                    if masks._table.get((a, v, t)) != labelled:
                        errors["roundtrip_mismatch"] += 1
                        ok = False
            except Exception as exc:
                errors[f"unencodable:{type(exc).__name__}"] += 1
                ok = False
            tok = tokens_data[int(splits[r]) : int(splits[r + 1])]
            if verify:
                fresh = build_entity_tokens(engine, pid).astype(np.float16)
                g = build_globals(engine, pid)
                if fresh.shape != tok.shape or not np.array_equal(fresh, tok) or not np.allclose(g, globals_arr[r]):
                    mismatched_tokens += 1
            if ok:
                rows[pid].append(
                    {
                        "tokens": tok,
                        "globals": globals_arr[r],
                        "action": [int(a), int(v), int(t)],
                        "verb": VERB_NAMES[int(v)],
                    }
                )
            engine.step(action)
        winner = engine.winner
        if winner != meta.get("winner"):
            raise RuntimeError(f"replay winner {winner} != harvest winner {meta.get('winner')}")
        out = []
        for pid in (0, 1):
            z = 0.0 if winner is None else (1.0 if winner == pid else -1.0)
            rets = discounted_returns(len(rows[pid]), z, gamma)
            for row, ret in zip(rows[pid], rets):
                row["return"] = float(ret)
            doc = doctrines[pid]
            gid = int(meta["game_idx"]) * 2 + pid
            rel = Path(doc) / "games" / f"game_{gid:06d}.npz"
            m = {
                "game_idx": gid,
                "source_game_idx": int(meta["game_idx"]),
                "seat": pid,
                "doctrine": doc,
                "opponent": doctrines[1 - pid],
                "winner": winner,
                "result": "W" if z > 0 else ("L" if z < 0 else "D"),
                "turns": int(meta["turns"]),
                "n_labels": len(rows[pid]),
                "verb_counts": dict(Counter(r["verb"] for r in rows[pid])),
                "path": str(rel.relative_to(doc)),
            }
            write_game_npz(out_root / rel, rows[pid], m)
            out.append(m)
        return {
            "status": "ok",
            "source_game_idx": int(meta["game_idx"]),
            "seats": out,
            "errors": dict(errors),
            "token_mismatch": mismatched_tokens,
            "decisions": len(actions),
        }
    except Exception as exc:
        return {"status": "crash", "src": str(src), "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc(limit=6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--games-per-pair", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--verify-tokens", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Debug: label only this many games.")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    by_pair: Dict[tuple, List[int]] = defaultdict(list)
    with open(args.source / "progress.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status", "ok") != "ok" or not rec.get("doctrines"):
                continue
            d = rec["doctrines"]
            by_pair[(d["0"], d["1"])].append(int(rec["game_idx"]))
    rng = random.Random(args.seed)
    chosen: List[int] = []
    for a in DOCTRINE_NAMES:
        for b in DOCTRINE_NAMES:
            ids = sorted(by_pair[(a, b)])
            rng.shuffle(ids)
            chosen.extend(ids[: args.games_per_pair])
    chosen.sort()
    if args.limit:
        chosen = chosen[: args.limit]
    jobs = [
        {
            "src": str(args.source / "games" / f"game_{g:06d}.npz"),
            "out_root": str(out),
            "verify_tokens": args.verify_tokens,
        }
        for g in chosen
    ]
    print(f"labelling {len(jobs)} games with {args.workers} workers -> {out}", flush=True)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(label_game, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            results.append(res)
            if res["status"] != "ok":
                print("CRASH", res.get("src"), res.get("error"), flush=True)
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"{i}/{len(jobs)} games  {el:.0f}s  eta {el / i * (len(jobs) - i):.0f}s", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    errs: Counter = Counter()
    for r in ok:
        errs.update(r["errors"])
    summary: Dict[str, Any] = {
        "source": str(args.source),
        "games_per_pair": args.games_per_pair,
        "games_ok": len(ok),
        "games_crash": len(results) - len(ok),
        "decisions": sum(r["decisions"] for r in ok),
        "label_errors": dict(errs),
        "token_mismatch_rows": sum(r["token_mismatch"] for r in ok),
        "verify_tokens": args.verify_tokens,
        "elapsed_sec": time.time() - t0,
        "doctrines": {},
    }
    for doc in DOCTRINE_NAMES:
        seats = sorted((s for r in ok for s in r["seats"] if s["doctrine"] == doc), key=lambda s: s["game_idx"])
        ids = [s["game_idx"] for s in seats if s["n_labels"] > 0]
        srng = random.Random(args.seed + 1)
        srng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * args.val_frac))) if ids else 0
        split = {"seed": args.seed, "val_frac": args.val_frac, "train": sorted(ids[n_val:]), "val": sorted(ids[:n_val])}
        split["n_train"], split["n_val"] = len(split["train"]), len(split["val"])
        ddir = out / doc
        ddir.mkdir(parents=True, exist_ok=True)
        verbs: Counter = Counter()
        results_c: Counter = Counter()
        for s in seats:
            verbs.update(s["verb_counts"])
            results_c[s["result"]] += 1
        dman = {
            "doctrine": doc,
            "source": str(args.source),
            "n_game_seats": len(seats),
            "n_labels": sum(s["n_labels"] for s in seats),
            "verb_counts": dict(verbs),
            "teacher_results": dict(results_c),
            "games": seats,
        }
        (ddir / "manifest.json").write_text(json.dumps(dman))
        (ddir / "split.json").write_text(json.dumps(split, indent=2))
        summary["doctrines"][doc] = {k: v for k, v in dman.items() if k != "games"} | {
            "n_train_games": split["n_train"],
            "n_val_games": split["n_val"],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
