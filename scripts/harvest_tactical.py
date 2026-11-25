#!/usr/bin/env python3
"""Harvest tactical-scripted decisions for behavior cloning.

Plays games on the v0.2.2 map pool (same episode caps as eval_tactical)
and writes packed per-game ``.npz`` shards under ``--output-dir``. Only
the tactical seat's actions are labels (both seats, when both are
tactical). Observations are the policy's fog tensors.

    PYTHONPATH=<repo> python scripts/harvest_tactical.py \
        --self-play 2000 --vs-policy 400 --vs-random 200 \
        --output-dir runs/bc_tactical/harvest --workers 12

vs-policy loads ``--checkpoint`` on CPU. Pre-v0.5.0 official
checkpoints and the v0.2.2 map pool were archived; pass those paths
explicitly (see ``battleboats.training.checkpoints.NO_CHECKPOINT_MSG``).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Harvest workers must not oversubscribe BLAS/torch against each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from battleboats.agents.fog_commander import FogCommander, canonical_build_action
from battleboats.agents.random_agent import random_action
from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import BuildShipAction, CapturePortAction
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.training.selfplay_purchases import hull_name
from battleboats.training.tactical_bc import (
    GAMMA_DEFAULT,
    VERB_GROUP,
    discounted_returns,
    encode_tactical_action,
    write_game_npz,
)

LABEL_KINDS = ("tactical", "fog_commander")  # scripted seats whose actions are labels


def _play_game(
    map_path: Path,
    *,
    max_turns: int,
    max_steps: int,
    seed: int,
    actors: Dict[int, str],
    policy=None,
    device=None,
    gamma: float = GAMMA_DEFAULT,
) -> Dict[str, Any]:
    env = BattleboatsAEC(str(map_path), max_turns=max_turns)
    if hasattr(env, "prime_legal_actions"):
        env.prime_legal_actions = False
    env.reset(seed=seed)
    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    fog_commander = FogCommander()

    def action_for(pid: int):
        kind = actors[pid]
        if kind == "tactical":
            return tactical_scripted_action(env.engine, pid, rngs[pid])
        if kind == "fog_commander":
            return fog_commander.act(env.engine, pid, rngs[pid])
        if kind == "random":
            return random_action(env.engine, pid, rngs[pid])
        import torch

        masks = ActionMasks(env.engine, pid, device=device)
        with torch.no_grad():
            a_idx, v_idx, t_idx, _logp, _value = policy.act(
                masks.tokens, masks.pad_mask, masks, skill_idx=0, globals=masks.globals
            )
        return masks.to_action(asset_idx=a_idx, verb_idx=v_idx, target_idx=t_idx)

    rows_by_pid: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    errors: List[Dict[str, Any]] = []
    verb_counts: Counter = Counter()
    buys: Dict[int, Counter] = {0: Counter(), 1: Counter()}
    captures = {0: 0, 1: 0}
    home_captures = {0: 0, 1: 0}
    steps = 0
    last_winner = None

    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        pid = env._player_id(agent)
        engine = env.engine
        # Match tactical / policy: sightings must be current before we
        # build the observation or choose an action.
        engine._refresh_sightings()
        action = action_for(pid)
        if actors[pid] in LABEL_KINDS:
            # The policy has one buy per (port, type); fog_commander picks its own
            # spawn tile, so its buys are labelled as the canonical legal buy.
            encoded = encode_tactical_action(engine, pid, canonical_build_action(engine, pid, action))
            if encoded.ok:
                rows_by_pid[pid].append(
                    {
                        "tokens": encoded.tokens,
                        "globals": encoded.globals,
                        "action": encoded.action_triple,
                        "verb": encoded.verb,
                        "turn": int(engine.turn),
                    }
                )
                verb_counts[encoded.verb] += 1
            else:
                errors.append(
                    {
                        "pid": pid,
                        "turn": int(engine.turn),
                        "action": type(action).__name__,
                        "error": encoded.error,
                    }
                )
        if isinstance(action, BuildShipAction):
            buys[pid][hull_name(action.ship_type)] += 1
        if isinstance(action, CapturePortAction):
            captures[pid] += 1
            if action.target == engine.players[1 - pid].home_port:
                home_captures[pid] += 1
        env.step(action)
        steps += 1
        if engine.winner is not None or engine.turn >= max_turns or steps >= max_steps:
            last_winner = engine.winner
            break

    winner = env.engine.winner if env.engine.winner is not None else last_winner
    turn = int(env.engine.turn)
    try:
        env.close()
    except Exception:
        pass

    labeled_rows: List[Dict[str, Any]] = []
    for pid, rows in rows_by_pid.items():
        if actors[pid] not in LABEL_KINDS or not rows:
            continue
        if winner is None:
            z = 0.0
        elif winner == pid:
            z = 1.0
        else:
            z = -1.0
        rets = discounted_returns(len(rows), z, gamma)
        for row, ret in zip(rows, rets):
            row["return"] = float(ret)
            labeled_rows.append(row)

    error_kinds = Counter(e["error"].split(":")[0] for e in errors)
    return {
        "rows": labeled_rows,
        "winner": winner,
        "reason": "home_capture" if winner is not None else "timeout",
        "turn": turn,
        "steps": steps,
        "actors": {0: actors[0], 1: actors[1]},
        "n_labels": len(labeled_rows),
        "n_errors": len(errors),
        "error_kinds": dict(error_kinds),
        "errors_sample": errors[:8],
        "verb_counts": dict(verb_counts),
        "buys": {0: dict(buys[0]), 1: dict(buys[1])},
        "captures": captures,
        "home_captures": home_captures,
    }


def _schedule_jobs(
    maps,
    cfg,
    *,
    n_self: int,
    n_vs_policy: int,
    n_vs_random: int,
    seed: int,
    labeler: str = "tactical",
) -> List[Dict[str, Any]]:
    # Re-importing is fine; the caller already has maps/cfg. Build one
    # combined schedule: self-play, then vs-policy, then vs-random, each
    # with its own seed offset so they don't share openings.
    from scripts.eval_tactical import _schedule as make_sched

    jobs: List[Dict[str, Any]] = []
    for kind, n, offset in (
        ("self_play", n_self, 0),
        ("vs_policy", n_vs_policy, 100_000),
        ("vs_random", n_vs_random, 200_000),
    ):
        if n <= 0:
            continue
        for raw in make_sched(maps, cfg, n, seed + offset):
            job = dict(raw)
            job["kind"] = kind
            if kind == "self_play":
                job["actors"] = {0: labeler, 1: labeler}
            elif kind == "vs_policy":
                seat = job["seat"]  # labelled seat (same interleaving as eval)
                job["actors"] = {seat: labeler, 1 - seat: "policy"}
            else:
                seat = job["seat"]
                job["actors"] = {seat: labeler, 1 - seat: "random"}
            jobs.append(job)
    return jobs


def _worker_init(ckpt: Optional[str], threads: int = 1):
    """Load the v0.3.0 policy once per process (vs-policy jobs only)."""
    global _WORKER_POLICY, _WORKER_DEVICE
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _WORKER_POLICY = None
    _WORKER_DEVICE = torch.device("cpu")
    if ckpt:
        from battleboats.training.checkpoints import load_checkpoint

        policy, _ = load_checkpoint(Path(ckpt), _WORKER_DEVICE)
        policy.eval()
        _WORKER_POLICY = policy


def _run_job(payload: Tuple[Dict[str, Any], str, float]) -> Dict[str, Any]:
    job, out_dir, gamma = payload
    out = Path(out_dir)
    gid = int(job["i_global"])
    dest = out / f"game_{gid:05d}.npz"
    try:
        policy = _WORKER_POLICY if "policy" in job["actors"].values() else None
        played = _play_game(
            job["path"],
            max_turns=job["max_turns"],
            max_steps=job["max_steps"],
            seed=job["seed"],
            actors=job["actors"],
            policy=policy,
            device=_WORKER_DEVICE if policy is not None else None,
            gamma=gamma,
        )
        meta = {
            "game_idx": gid,
            "kind": job["kind"],
            "size": job["size"],
            "seed": job["seed"],
            "map": str(job["path"]),
            "actors": played["actors"],
            "winner": played["winner"],
            "reason": played["reason"],
            "turn": played["turn"],
            "steps": played["steps"],
            "n_labels": played["n_labels"],
            "n_errors": played["n_errors"],
            "error_kinds": played["error_kinds"],
            "verb_counts": played["verb_counts"],
            "buys": played["buys"],
            "captures": played["captures"],
            "home_captures": played["home_captures"],
            "errors_sample": played["errors_sample"],
        }
        write_game_npz(dest, played["rows"], meta)
        meta["path"] = dest.name
        meta["status"] = "ok"
        return meta
    except Exception as exc:
        return {
            "game_idx": gid,
            "kind": job["kind"],
            "size": job.get("size"),
            "path": dest.name,
            "n_labels": 0,
            "n_errors": 0,
            "status": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-play", type=int, default=2000)
    parser.add_argument("--vs-policy", type=int, default=400)
    parser.add_argument("--vs-random", type=int, default=200)
    parser.add_argument("--maps-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/bc_tactical/harvest"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--labeler",
        choices=sorted(LABEL_KINDS),
        default="tactical",
        help="scripted agent whose decisions become labels (default: tactical)",
    )
    args = parser.parse_args()

    from battleboats.training.checkpoints import NO_CHECKPOINT_MSG, no_archived
    from battleboats.training.selfplay import load_selfplay_config
    from scripts.eval_tactical import _list_maps

    if args.maps_dir is None:
        raise SystemExit(no_archived("maps-dir"))
    if args.config is None:
        raise SystemExit(no_archived("config"))
    if args.vs_policy > 0 and args.checkpoint is None:
        raise SystemExit(NO_CHECKPOINT_MSG)

    cfg = load_selfplay_config(args.config)
    maps = _list_maps(args.maps_dir)
    jobs = _schedule_jobs(
        maps,
        cfg,
        n_self=args.self_play,
        n_vs_policy=args.vs_policy,
        n_vs_random=args.vs_random,
        seed=args.seed,
        labeler=args.labeler,
    )
    # Unique game_idx across kinds (self-play first).
    for i, job in enumerate(jobs):
        job["i_global"] = i

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        pending = []
        for job in jobs:
            dest = out / f"game_{job['i_global']:05d}.npz"
            if dest.is_file():
                continue
            pending.append(job)
        print(
            f"resume: {len(jobs) - len(pending)} done, {len(pending)} remaining",
            flush=True,
        )
        jobs_to_run = pending
    else:
        jobs_to_run = jobs

    need_policy = any("policy" in j["actors"].values() for j in jobs_to_run)
    ckpt = str(args.checkpoint) if need_policy else None
    t0 = time.time()
    results: List[Dict[str, Any]] = []

    # Already-finished shards (resume): read their meta from npz.
    if args.resume:
        from battleboats.training.tactical_bc import load_game_npz

        done = {j["i_global"] for j in jobs_to_run}
        for job in jobs:
            if job["i_global"] in done:
                continue
            dest = out / f"game_{job['i_global']:05d}.npz"
            try:
                _arr, meta = load_game_npz(dest)
                meta["path"] = dest.name
                meta.setdefault("status", "ok")
                results.append(meta)
            except Exception as exc:
                results.append(
                    {
                        "game_idx": job["i_global"],
                        "path": dest.name,
                        "n_labels": 0,
                        "n_errors": 0,
                        "status": "crash",
                        "error": f"resume_read:{exc}",
                    }
                )

    payloads = [(job, str(out), float(args.gamma)) for job in jobs_to_run]
    if not payloads:
        print("nothing to run", flush=True)
    elif args.workers <= 1:
        _worker_init(ckpt, threads=1)
        for i, payload in enumerate(payloads, 1):
            rec = _run_job(payload)
            results.append(rec)
            if i % 20 == 0 or i == len(payloads):
                print(
                    f"  [{i}/{len(payloads)}] kind={rec.get('kind')} "
                    f"labels={rec.get('n_labels')} errors={rec.get('n_errors')} "
                    f"status={rec.get('status')}",
                    flush=True,
                )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(ckpt, 1),
        ) as pool:
            for i, rec in enumerate(pool.imap_unordered(_run_job, payloads), 1):
                results.append(rec)
                if i % 20 == 0 or i == len(payloads):
                    print(
                        f"  [{i}/{len(payloads)}] kind={rec.get('kind')} "
                        f"labels={rec.get('n_labels')} errors={rec.get('n_errors')} "
                        f"status={rec.get('status')}",
                        flush=True,
                    )

    results.sort(key=lambda r: int(r.get("game_idx", 0)))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_crash = sum(1 for r in results if r.get("status") == "crash")
    n_labels = sum(int(r.get("n_labels", 0)) for r in results)
    n_errors = sum(int(r.get("n_errors", 0)) for r in results)
    error_kinds: Counter = Counter()
    verbs: Counter = Counter()
    groups: Counter = Counter()
    by_kind: Counter = Counter()
    for r in results:
        by_kind[r.get("kind") or "unknown"] += 1
        for k, v in (r.get("error_kinds") or {}).items():
            error_kinds[k] += int(v)
        for k, v in (r.get("verb_counts") or {}).items():
            verbs[k] += int(v)
            groups[VERB_GROUP.get(k, k)] += int(v)

    manifest = {
        "commit": "c1b9124d72d0c724bbf7e2eb607af2c5bebc690e",
        "maps_dir": str(args.maps_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.vs_policy else None,
        "seed": args.seed,
        "gamma": args.gamma,
        "n_self_play": args.self_play,
        "n_vs_policy": args.vs_policy,
        "n_vs_random": args.vs_random,
        "n_games": len(results),
        "n_ok": n_ok,
        "n_crash": n_crash,
        "n_labels": n_labels,
        "n_errors": n_errors,
        "error_rate": (
            (n_errors / (n_errors + n_labels)) if (n_errors + n_labels) else 0.0
        ),
        "error_kinds": dict(error_kinds),
        "verb_counts": dict(verbs),
        "verb_groups": dict(groups),
        "games_by_kind": dict(by_kind),
        "elapsed_sec": time.time() - t0,
        "games": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # Split is written by the trainer (or here for convenience).
    from battleboats.training.tactical_bc import write_split

    split = write_split(out, seed=args.seed, val_frac=0.1)
    summary = {k: v for k, v in manifest.items() if k != "games"}
    summary["split"] = {k: split[k] for k in ("n_train", "n_val", "seed", "val_frac")}
    print(json.dumps(summary, indent=2, default=str))
    print(
        f"wrote {out / 'manifest.json'} labels={n_labels} errors={n_errors}", flush=True
    )


if __name__ == "__main__":
    main()
