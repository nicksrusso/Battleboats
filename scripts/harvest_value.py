"""Harvest fogged observations, terminal outcomes, and replay traces for value-net training.

Plays games (default 48x24, 80% fog_commander self-play + 20% fog_commander vs
scripted with seats alternated). At every decision, records BOTH seats'
fog-of-war observations from ``battleboats.envs.observation`` (post-fix
coastline encoding), paired with each seat's final outcome (+1/-1/0),
discounted return, turn count, and enough metadata (map JSON, seed, full
action sequence) to replay the game deterministically.

Each game is written atomically to ``games/game_<idx>.npz`` using float16
entity tokens. Live per-game summaries append to ``progress.jsonl``, and
``manifest.json`` + ``split.json`` (90/10 by game) are written at completion.

Usage:
    PYTHONPATH=. python scripts/harvest_value.py \\
        --map 48x24 --self-play 4000 --vs-scripted 1000 --workers 10 \\
        --output-dir runs/value_harvest_48x24_v0.5.0
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from battleboats.agents.fog_commander import engine_accepts  # noqa: E402
from battleboats.agents.registry import AGENT_KINDS, make_agent  # noqa: E402
from battleboats.core.actions import (  # noqa: E402
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
from battleboats.core.gameEngine import gameEngine  # noqa: E402
from battleboats.core.shipyard.ship_type import ShipType  # noqa: E402
from battleboats.envs.observation import (  # noqa: E402
    GLOBALS_DIM,
    TOKEN_DIM,
    build_observation,
)
from battleboats.ui.maps import resolve_map  # noqa: E402

OBS_ENCODING_VERSION = "v0.5.0-fog-coastline-fix"
GAMMA_DEFAULT = 0.99
MAX_ACTIONS_PER_TURN = 400


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def encode_action(action: Action) -> list:
    if isinstance(action, MoveAction):
        return ["m", int(action.ship_id), int(action.destination[0]), int(action.destination[1])]
    if isinstance(action, AttackAction):
        return ["a", int(action.attacker_id), int(action.target_id)]
    if isinstance(action, BuildShipAction):
        return [
            "b",
            int(action.port[0]),
            int(action.port[1]),
            int(action.spawn_position[0]),
            int(action.spawn_position[1]),
            action.ship_type.value,
        ]
    if isinstance(action, BuildPortAction):
        return ["p", int(action.builder_ship_id), int(action.target[0]), int(action.target[1])]
    if isinstance(action, CapturePortAction):
        return ["c", int(action.landing_ship_id), int(action.target[0]), int(action.target[1])]
    if isinstance(action, MerchantLoadAction):
        return ["l", int(action.merchant_id), int(action.port[0]), int(action.port[1])]
    if isinstance(action, MerchantUnloadAction):
        return ["u", int(action.merchant_id), int(action.port[0]), int(action.port[1])]
    if isinstance(action, EndTurnAction):
        return ["e"]
    raise TypeError(f"unsupported action type: {type(action).__name__}")


def decode_action(rec: Sequence[Any]) -> Action:
    tag = rec[0]
    if tag == "m":
        return MoveAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "a":
        return AttackAction(int(rec[1]), int(rec[2]))
    if tag == "b":
        return BuildShipAction(
            (int(rec[1]), int(rec[2])), (int(rec[3]), int(rec[4])), ShipType(rec[5])
        )
    if tag == "p":
        return BuildPortAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "c":
        return CapturePortAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "l":
        return MerchantLoadAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "u":
        return MerchantUnloadAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "e":
        return EndTurnAction()
    raise ValueError(f"unknown encoded action tag: {tag!r}")


def replay_shard(shard_path: Path) -> Tuple[gameEngine, Dict[str, Any]]:
    """Deterministically replay a harvested .npz shard and return (engine, meta)."""
    with np.load(shard_path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        map_data = json.loads(str(z["map_json"]))
        encoded_actions = json.loads(str(z["actions_json"]))
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "map.json"
        mp.write_text(json.dumps(map_data))
        engine = gameEngine(str(mp))
        engine.reset(seed=int(meta["seed"]))
    engine._refresh_sightings()
    for rec in encoded_actions:
        engine.step(decode_action(rec))
    return engine, meta


def load_value_shard(shard_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(shard_path, allow_pickle=False) as z:
        arrays = {
            k: z[k]
            for k in z.files
            if k not in ("meta_json", "map_json", "actions_json")
        }
        meta = json.loads(str(z["meta_json"]))
    return arrays, meta


def _shard_path(out_dir: Path, game_idx: int) -> Path:
    return out_dir / "games" / f"game_{game_idx:06d}.npz"


def _outcome_for(winner: Optional[int], seat: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == seat else -1.0


def _play_and_save(job: Dict[str, Any]) -> Dict[str, Any]:
    game_idx = int(job["game_idx"])
    seed = int(job["seed"])
    map_spec = str(job["map_spec"])
    actors = {0: str(job["actors"][0]), 1: str(job["actors"][1])}
    gamma = float(job["gamma"])
    out_path = Path(job["out_path"])
    commit = str(job["commit"])

    try:
        with tempfile.TemporaryDirectory() as td:
            rm = resolve_map(map_spec, seed=seed, map_dir=Path(td))
            map_text = rm.path.read_text()
            max_turns = int(rm.recommended_max_turns)
            max_steps = int(job.get("max_steps") or (8000 * max_turns // 80))
            engine = gameEngine(str(rm.path))
            engine.reset(seed=seed)
        engine._refresh_sightings()

        rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
        agents = {pid: make_agent(actors[pid]) for pid in (0, 1)}

        tokens_list: List[np.ndarray] = []
        globals_list: List[np.ndarray] = []
        seat_list: List[int] = []
        acting_list: List[int] = []
        turn_list: List[int] = []
        step_list: List[int] = []
        encoded_actions: List[list] = []
        buys = {0: Counter(), 1: Counter()}
        per_turn = Counter()
        illegal = {0: 0, 1: 0}
        forced_end = {0: 0, 1: 0}

        steps = 0
        while engine.winner is None and engine.turn < max_turns and steps < max_steps:
            pid = engine.current_player
            legal = engine.enumerate_legal(pid)
            for s in (0, 1):
                obs = build_observation(engine, s)
                tokens_list.append(np.asarray(obs["entity_tokens"], dtype=np.float16))
                globals_list.append(np.asarray(obs["globals"], dtype=np.float32))
                seat_list.append(s)
                acting_list.append(pid)
                turn_list.append(int(engine.turn))
                step_list.append(steps)

            action = agents[pid].act(engine, pid, rngs[pid])
            per_turn[(pid, engine.turn)] += 1
            if not engine_accepts(engine, pid, action, legal):
                illegal[pid] += 1
                action = EndTurnAction()
            elif per_turn[(pid, engine.turn)] > MAX_ACTIONS_PER_TURN and not isinstance(
                action, EndTurnAction
            ):
                forced_end[pid] += 1
                action = EndTurnAction()
            if isinstance(action, BuildShipAction):
                buys[pid][action.ship_type.value] += 1
            encoded_actions.append(encode_action(action))
            engine.step(action)
            steps += 1

        winner = engine.winner
        final_turn = int(engine.turn)
        n_dec = steps
        n_rows = len(tokens_list)
        outcome_by_seat = {0: _outcome_for(winner, 0), 1: _outcome_for(winner, 1)}

        lengths = [int(t.shape[0]) for t in tokens_list]
        splits = np.zeros(n_rows + 1, dtype=np.int32)
        if n_rows:
            splits[1:] = np.cumsum(lengths, dtype=np.int32)
            tokens_data = np.concatenate(tokens_list, axis=0)
            globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
        else:
            tokens_data = np.zeros((0, TOKEN_DIM), dtype=np.float16)
            globals_arr = np.zeros((0, GLOBALS_DIM), dtype=np.float32)

        seat_arr = np.asarray(seat_list, dtype=np.int8)
        acting_arr = np.asarray(acting_list, dtype=np.int8)
        turn_arr = np.asarray(turn_list, dtype=np.int16)
        step_arr = np.asarray(step_list, dtype=np.int32)
        outcome_arr = np.asarray(
            [outcome_by_seat[s] for s in seat_list], dtype=np.float32
        )
        if n_dec > 0:
            expo = np.asarray([(n_dec - 1) - st for st in step_list], dtype=np.float32)
            returns_arr = outcome_arr * (np.float32(gamma) ** expo)
        else:
            returns_arr = np.zeros((0,), dtype=np.float32)

        doctrines = {
            pid: getattr(agents[pid], "doctrine_for", lambda _p: None)(pid)
            for pid in (0, 1)
        }
        reason = "home_capture" if winner is not None else (
            "turn_cap" if final_turn >= max_turns else "step_cap"
        )
        meta = {
            "game_idx": game_idx,
            "kind": job["kind"],
            "seed": seed,
            "map_spec": map_spec,
            "width": int(engine.map.width),
            "height": int(engine.map.height),
            "max_turns": max_turns,
            "max_steps": max_steps,
            "actors": actors,
            "doctrines": doctrines,
            "winner": winner,
            "reason": reason,
            "turns": final_turn,
            "num_decisions": n_dec,
            "num_rows": n_rows,
            "outcome_by_seat": outcome_by_seat,
            "gamma": gamma,
            "buys": {pid: dict(buys[pid]) for pid in (0, 1)},
            "illegal": illegal,
            "forced_end": forced_end,
            "obs_encoding": OBS_ENCODING_VERSION,
            "commit": commit,
            "path": str(out_path.relative_to(Path(job["out_root"]))),
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            tokens_data=tokens_data,
            row_splits=splits,
            globals=globals_arr,
            seat=seat_arr,
            acting_seat=acting_arr,
            turn=turn_arr,
            step_idx=step_arr,
            outcome=outcome_arr,
            returns=returns_arr,
            actions_json=np.asarray(json.dumps(encoded_actions, separators=(",", ":"))),
            map_json=np.asarray(map_text),
            meta_json=np.asarray(json.dumps(meta, separators=(",", ":"))),
        )
        tmp_path.replace(out_path)
        meta["status"] = "ok"
        meta["bytes"] = int(out_path.stat().st_size)
        return meta
    except Exception as exc:
        return {
            "game_idx": game_idx,
            "kind": job["kind"],
            "seed": seed,
            "actors": actors,
            "status": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def build_schedule(
    *,
    map_spec: str,
    n_self_play: int,
    n_vs_scripted: int,
    n_vs_random: int,
    seed: int,
    gamma: float,
    out_dir: Path,
    commit: str,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    idx = 0
    for i in range(n_self_play):
        jobs.append(
            {
                "game_idx": idx,
                "kind": "self_play",
                "seed": seed + i,
                "map_spec": map_spec,
                "actors": {0: "fog_commander", 1: "fog_commander"},
                "gamma": gamma,
                "out_root": str(out_dir),
                "out_path": str(_shard_path(out_dir, idx)),
                "commit": commit,
            }
        )
        idx += 1
    for i in range(n_vs_scripted):
        fc_seat = i % 2
        actors = {fc_seat: "fog_commander", 1 - fc_seat: "scripted"}
        jobs.append(
            {
                "game_idx": idx,
                "kind": "vs_scripted",
                "seed": seed + 100_000 + i,
                "map_spec": map_spec,
                "actors": actors,
                "gamma": gamma,
                "out_root": str(out_dir),
                "out_path": str(_shard_path(out_dir, idx)),
                "commit": commit,
            }
        )
        idx += 1
    for i in range(n_vs_random):
        fc_seat = i % 2
        actors = {fc_seat: "fog_commander", 1 - fc_seat: "random"}
        jobs.append(
            {
                "game_idx": idx,
                "kind": "vs_random",
                "seed": seed + 200_000 + i,
                "map_spec": map_spec,
                "actors": actors,
                "gamma": gamma,
                "out_root": str(out_dir),
                "out_path": str(_shard_path(out_dir, idx)),
                "commit": commit,
            }
        )
        idx += 1
    return jobs


def write_split(out_dir: Path, game_recs: Sequence[Dict[str, Any]], *, seed: int, val_frac: float = 0.1) -> Dict[str, Any]:
    gids = sorted(int(r["game_idx"]) for r in game_recs if r.get("status") == "ok")
    rng = random.Random(seed)
    shuffled = list(gids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac))) if len(shuffled) >= 2 else 0
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    split = {
        "seed": seed,
        "val_frac": val_frac,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_games": train_ids,
        "val_games": val_ids,
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))
    return split


def _summarize(results: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    crashes = [r for r in results if r.get("status") != "ok"]
    decisive = [r for r in ok if r.get("winner") is not None]
    by_kind: Dict[str, Dict[str, int]] = {}
    buys_by_agent: Dict[str, Counter] = {}
    seat_wins = Counter()
    for r in ok:
        k = r["kind"]
        d = by_kind.setdefault(k, {"games": 0, "decisive": 0, "seat0": 0, "seat1": 0, "fc_wins": 0})
        d["games"] += 1
        if r["winner"] is not None:
            d["decisive"] += 1
            d[f"seat{r['winner']}"] += 1
            seat_wins[r["winner"]] += 1
            if r["actors"][str(r["winner"])] == "fog_commander" if str(r["winner"]) in r["actors"] else r["actors"][r["winner"]] == "fog_commander":
                d["fc_wins"] += 1
        for pid in (0, 1):
            agent_name = r["actors"][str(pid)] if str(pid) in r["actors"] else r["actors"][pid]
            c = buys_by_agent.setdefault(agent_name, Counter())
            c.update(r["buys"][str(pid)] if str(pid) in r["buys"] else r["buys"][pid])
    return {
        "n_games": len(results),
        "n_ok": len(ok),
        "n_crash": len(crashes),
        "decisive": len(decisive),
        "decisive_pct": round(100.0 * len(decisive) / len(ok), 2) if ok else 0.0,
        "seat_wins": {str(s): seat_wins[s] for s in (0, 1)},
        "total_decisions": sum(int(r["num_decisions"]) for r in ok),
        "total_rows": sum(int(r["num_rows"]) for r in ok),
        "total_bytes": sum(int(r.get("bytes", 0)) for r in ok),
        "avg_turns": round(sum(r["turns"] for r in ok) / len(ok), 2) if ok else 0.0,
        "avg_turns_decisive": round(sum(r["turns"] for r in decisive) / len(decisive), 2) if decisive else 0.0,
        "illegal": sum(sum((r.get("illegal") or {}).values()) for r in ok),
        "forced_end": sum(sum((r.get("forced_end") or {}).values()) for r in ok),
        "by_kind": by_kind,
        "buys": {k: dict(sorted(v.items())) for k, v in buys_by_agent.items()},
        "elapsed_sec": round(elapsed, 2),
    }


def _load_existing_meta(shard_path: Path) -> Optional[Dict[str, Any]]:
    if not shard_path.is_file():
        return None
    try:
        _, meta = load_value_shard(shard_path)
        meta["status"] = "ok"
        meta["bytes"] = int(shard_path.stat().st_size)
        return meta
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="48x24", help="Map preset (default: 48x24)")
    ap.add_argument("--self-play", type=int, default=4000)
    ap.add_argument("--vs-scripted", type=int, default=1000)
    ap.add_argument("--vs-random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/value_harvest_48x24_v0.5.0"),
    )
    args = ap.parse_args(argv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_head()
    jobs = build_schedule(
        map_spec=args.map,
        n_self_play=args.self_play,
        n_vs_scripted=args.vs_scripted,
        n_vs_random=args.vs_random,
        seed=args.seed,
        gamma=args.gamma,
        out_dir=out_dir,
        commit=commit,
    )
    total = len(jobs)
    results_by_idx: Dict[int, Dict[str, Any]] = {}
    pending: List[Dict[str, Any]] = []
    if args.resume:
        for job in jobs:
            existing = _load_existing_meta(Path(job["out_path"]))
            if existing is not None:
                results_by_idx[int(job["game_idx"])] = existing
            else:
                pending.append(job)
    else:
        pending = jobs

    run_cfg = {
        "map": args.map,
        "self_play": args.self_play,
        "vs_scripted": args.vs_scripted,
        "vs_random": args.vs_random,
        "seed": args.seed,
        "gamma": args.gamma,
        "workers": args.workers,
        "obs_encoding": OBS_ENCODING_VERSION,
        "commit": commit,
        "total_games": total,
        "resumed_existing": len(results_by_idx),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    print(
        f"Starting value harvest: dir={out_dir} map={args.map} total={total} "
        f"(self_play={args.self_play}, vs_scripted={args.vs_scripted}, vs_random={args.vs_random}) "
        f"workers={args.workers} resumed={len(results_by_idx)} pending={len(pending)} "
        f"obs={OBS_ENCODING_VERSION}",
        flush=True,
    )

    progress_path = out_dir / "progress.jsonl"
    mode = "a" if args.resume and progress_path.exists() else "w"
    t0 = time.time()
    new_done = 0
    dec_count = sum(1 for r in results_by_idx.values() if r.get("status") == "ok" and r.get("winner") is not None)
    ok_count = sum(1 for r in results_by_idx.values() if r.get("status") == "ok")
    crash_count = sum(1 for r in results_by_idx.values() if r.get("status") != "ok")
    rows_count = sum(int(r.get("num_rows", 0)) for r in results_by_idx.values())
    bytes_count = sum(int(r.get("bytes", 0)) for r in results_by_idx.values())

    with progress_path.open(mode) as pf:
        if args.workers <= 1:
            it = (_play_and_save(job) for job in pending)
            for res in it:
                gid = int(res["game_idx"])
                results_by_idx[gid] = res
                new_done += 1
                if res.get("status") == "ok":
                    ok_count += 1
                    if res.get("winner") is not None:
                        dec_count += 1
                    rows_count += int(res["num_rows"])
                    bytes_count += int(res["bytes"])
                else:
                    crash_count += 1
                pf.write(json.dumps(res, separators=(",", ":")) + "\n")
                pf.flush()
                done = len(results_by_idx)
                if new_done % args.log_every == 0 or done == total:
                    el = max(1e-6, time.time() - t0)
                    rate = new_done / el
                    rem = (len(pending) - new_done) / rate if rate > 0 else 0.0
                    dpct = 100.0 * dec_count / ok_count if ok_count else 0.0
                    print(
                        f"[{done}/{total}] ok={ok_count} crash={crash_count} decisive={dec_count} ({dpct:.1f}%) "
                        f"rows={rows_count:,} disk={bytes_count/1e6:.1f}MB "
                        f"rate={rate:.2f} games/s elapsed={el/60:.1f}m eta={rem/60:.1f}m",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_play_and_save, job) for job in pending]
                for fut in as_completed(futs):
                    res = fut.result()
                    gid = int(res["game_idx"])
                    results_by_idx[gid] = res
                    new_done += 1
                    if res.get("status") == "ok":
                        ok_count += 1
                        if res.get("winner") is not None:
                            dec_count += 1
                        rows_count += int(res["num_rows"])
                        bytes_count += int(res["bytes"])
                    else:
                        crash_count += 1
                    pf.write(json.dumps(res, separators=(",", ":")) + "\n")
                    pf.flush()
                    done = len(results_by_idx)
                    if new_done % args.log_every == 0 or done == total:
                        el = max(1e-6, time.time() - t0)
                        rate = new_done / el
                        rem = (len(pending) - new_done) / rate if rate > 0 else 0.0
                        dpct = 100.0 * dec_count / ok_count if ok_count else 0.0
                        print(
                            f"[{done}/{total}] ok={ok_count} crash={crash_count} decisive={dec_count} ({dpct:.1f}%) "
                            f"rows={rows_count:,} disk={bytes_count/1e6:.1f}MB "
                            f"rate={rate:.2f} games/s elapsed={el/60:.1f}m eta={rem/60:.1f}m",
                            flush=True,
                        )

    ordered = [results_by_idx[i] for i in sorted(results_by_idx)]
    summary = _summarize(ordered, time.time() - t0)
    split = write_split(out_dir, ordered, seed=args.seed, val_frac=0.1)
    manifest = {**run_cfg, **summary, "split": {k: split[k] for k in ("n_train", "n_val", "seed", "val_frac")}, "games": ordered}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({**summary, "split": manifest["split"]}, indent=2), flush=True)
    return 0 if summary["n_crash"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
