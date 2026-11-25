#!/usr/bin/env python3
"""Evaluate a tactical BC checkpoint vs tactical, a frozen policy, and itself.

Uses the same caps and 25-per-(seat, size) schedule as ``eval_tactical.py``.
The BC policy is the ``policy`` actor (greedy by default; pass --sample to
draw from the policy like PPO eval).

    PYTHONPATH=<repo> python scripts/eval_tactical_bc.py \
        --checkpoint path/to/bc_policy.pt \
        --v030-checkpoint path/to/opponent.pt \
        --maps-dir path/to/maps \
        --config path/to/selfplay_config.yaml \
        --output-dir runs/eval/bc_policy/

Pre-v0.5.0 official checkpoints and the v0.2.2 map pool were archived;
see ``battleboats.training.checkpoints.NO_CHECKPOINT_MSG``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import (AttackAction, BuildShipAction,
                                      CapturePortAction)
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.training.checkpoints import (NO_CHECKPOINT_MSG, no_archived,
                                              load_checkpoint)
from battleboats.training.selfplay import load_selfplay_config
from battleboats.training.selfplay_purchases import hull_name
from battleboats.training.manifest import append_eval
from scripts.eval_tactical import (_finalize_survivors, _list_maps,
                                   _new_tracker, _note_births, _note_deaths,
                                   _schedule, summarize)



def _play(
    map_path: Path,
    *,
    max_turns: int,
    max_steps: int,
    seed: int,
    actors: Dict[int, str],
    policies: Dict[str, Any],
    device,
    greedy: bool,
) -> Dict[str, Any]:
    env = BattleboatsAEC(str(map_path), max_turns=max_turns)
    if hasattr(env, "prime_legal_actions"):
        env.prime_legal_actions = False
    env.reset(seed=seed)
    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    track_pids = (0, 1)
    tracker = _new_tracker(track_pids)
    buys = {0: Counter(), 1: Counter()}
    steps = 0
    last_winner = None

    def action_for(pid: int):
        kind = actors[pid]
        if kind == "tactical":
            return tactical_scripted_action(env.engine, pid, rngs[pid])
        policy = policies[kind]
        was = getattr(policy, "greedy", False)
        policy.greedy = greedy
        masks = ActionMasks(env.engine, pid, device=device)
        with torch.no_grad():
            a_idx, v_idx, t_idx, _logp, _value = policy.act(
                masks.tokens, masks.pad_mask, masks, skill_idx=0, globals=masks.globals
            )
        policy.greedy = was
        return masks.to_action(asset_idx=a_idx, verb_idx=v_idx, target_idx=t_idx)

    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        pid = env._player_id(agent)
        engine = env.engine
        before = {p: set(engine.players[p].owned_ship_ids) for p in (0, 1)}
        action = action_for(pid)
        if isinstance(action, BuildShipAction):
            buys[pid][hull_name(action.ship_type)] += 1
        if isinstance(action, CapturePortAction):
            ship = engine.ships.get(action.landing_ship_id)
            if ship is not None and ship.type is ShipType.LANDING:
                tracker[pid]["stats"]["port_captures"] += 1
                if action.target == engine.players[1 - pid].home_port:
                    tracker[pid]["stats"]["home_captures"] += 1
        if isinstance(action, AttackAction):
            ship = engine.ships.get(action.attacker_id)
            if ship is not None and ship.type is ShipType.LANDING:
                tracker[pid]["stats"]["ship_attacks"] += 1
        env.step(action)
        steps += 1
        _note_births(engine, pid, tracker, before[pid])
        for victim in (0, 1):
            if victim == pid:
                continue
            _note_deaths(
                engine,
                victim,
                tracker,
                before[victim],
                isinstance(action, AttackAction),
            )
        if engine.winner is not None or engine.turn >= max_turns or steps >= max_steps:
            last_winner = engine.winner
            break

    _finalize_survivors(env.engine, tracker)
    winner = env.engine.winner if env.engine.winner is not None else last_winner
    turn = int(env.engine.turn)
    reason = "home_capture" if winner is not None else "timeout"
    try:
        env.close()
    except Exception:
        pass
    landing = {}
    for pid in track_pids:
        s = tracker[pid]["stats"]
        landing[pid] = {
            "bought": s["bought"],
            "killed_by_hostile": s["killed_by_hostile"],
            "survived": s["survived"],
            "port_captures": s["port_captures"],
            "home_captures": s["home_captures"],
            "ship_attacks": s["ship_attacks"],
            "lifespans_killed": s["lifespans_killed"],
            "lifespans_all": s["lifespans_all"],
        }
    return {
        "winner": winner,
        "reason": reason,
        "turn": turn,
        "steps": steps,
        "buys": {0: dict(buys[0]), 1: dict(buys[1])},
        "landing": landing,
        "actors": {0: actors[0], 1: actors[1]},
    }


def _run(jobs, actors_fn, policies, device, greedy, label) -> List[Dict[str, Any]]:
    games = []
    for j, job in enumerate(jobs):
        actors = actors_fn(job)
        rec = _play(
            job["path"],
            max_turns=job["max_turns"],
            max_steps=job["max_steps"],
            seed=job["seed"],
            actors=actors,
            policies=policies,
            device=device,
            greedy=greedy,
        )
        rec["size"] = job["size"]
        rec["seat"] = job["seat"]
        rec["actors"] = actors
        games.append(rec)
        print(
            f"  [{label}] {j + 1}/{len(jobs)} size={job['size']} "
            f"seat={job['seat']} winner={rec['winner']} turn={rec['turn']} "
            f"reason={rec['reason']}",
            flush=True,
        )
    return games


def _focalize(games: List[Dict[str, Any]], focal: str) -> List[Dict[str, Any]]:
    """Rewrite W/L so summarize() treats ``focal`` as the scripted/focal seat."""
    out = []
    for g in games:
        rec = dict(g)
        actors = {int(k): v for k, v in g["actors"].items()}
        focal_pid = next(p for p, k in actors.items() if k == focal)
        rec["scripted_pid"] = focal_pid
        # summarize uses winner vs implicit? It uses scripted_pid if present
        # for by_seat, and buys['focal'] from actors. Look at summarize...
        rec["_focal_pid"] = focal_pid
        out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v030-checkpoint", type=Path, default=None)
    parser.add_argument("--maps-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--self-play-games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--sample", action="store_true", help="Sample BC actions instead of greedy."
    )
    parser.add_argument("--skip-vs-tactical", action="store_true")
    parser.add_argument("--skip-vs-v030", action="store_true")
    parser.add_argument("--skip-self-play", action="store_true")
    parser.add_argument(
        "--vs-checkpoint",
        type=Path,
        default=None,
        help="Optional extra frozen policy to play (e.g. v0.4.0).",
    )
    parser.add_argument("--vs-label", type=str, default="frozen")
    args = parser.parse_args()
    if args.maps_dir is None:
        raise SystemExit(no_archived("maps-dir"))
    if args.config is None:
        raise SystemExit(no_archived("config"))
    if not args.skip_vs_v030 and args.v030_checkpoint is None:
        raise SystemExit(NO_CHECKPOINT_MSG)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    greedy = not args.sample

    cfg = load_selfplay_config(args.config)
    maps = _list_maps(args.maps_dir)
    bc, _ = load_checkpoint(args.checkpoint, device)
    bc.eval()
    policies = {"bc": bc}
    if not args.skip_vs_v030:
        v030, _ = load_checkpoint(args.v030_checkpoint, device)
        v030.eval()
        policies["v030"] = v030
    if args.vs_checkpoint is not None:
        extra, _ = load_checkpoint(args.vs_checkpoint, device)
        extra.eval()
        policies["frozen"] = extra

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "bc_checkpoint": str(args.checkpoint),
        "v030_checkpoint": str(args.v030_checkpoint),
        "vs_checkpoint": str(args.vs_checkpoint) if args.vs_checkpoint else None,
        "vs_label": args.vs_label,
        "maps_dir": str(args.maps_dir),
        "config": str(args.config),
        "seed": args.seed,
        "greedy": greedy,
    }
    games_out: Dict[str, Any] = {}

    def vs_tactical_actors(job):
        # job['seat'] is the *tactical* seat, BC is the other, matching
        # "25 per seat x size" from the BC perspective if we swap...
        # We want 25 BC-as-seat-0 and 25 BC-as-seat-1 per size.
        # eval_tactical's seat is the scripted/tactical seat. Use it as
        # the BC seat so the schedule matches "25 per (BC seat, size)".
        bc_seat = job["seat"]
        return {bc_seat: "bc", 1 - bc_seat: "tactical"}

    def vs_v030_actors(job):
        bc_seat = job["seat"]
        return {bc_seat: "bc", 1 - bc_seat: "v030"}

    def self_actors(job):
        return {0: "bc", 1: "bc"}

    if not args.skip_vs_tactical:
        jobs = _schedule(maps, cfg, args.games, args.seed)
        games = _run(jobs, vs_tactical_actors, policies, device, greedy, "vs tactical")
        games_out["vs_tactical"] = games
        payload["vs_tactical"] = summarize(games, focal="bc", label="BC vs tactical")
        # summarize looks at actors for focal buy mix — it keys on
        # actors[pid]==focal. Our actor name is 'bc'. Good if summarize
        # uses the focal string against actors values.
        print(
            "VS TACTICAL",
            json.dumps(payload["vs_tactical"], indent=2, default=str)[:2000],
            flush=True,
        )

    if not args.skip_vs_v030:
        jobs = _schedule(maps, cfg, args.games, args.seed)
        games = _run(jobs, vs_v030_actors, policies, device, greedy, "vs v0.3.0")
        games_out["vs_v0.3.0"] = games
        payload["vs_v0.3.0"] = summarize(games, focal="bc", label="BC vs v0.3.0")
        print(
            "VS V0.3.0",
            json.dumps(payload["vs_v0.3.0"], indent=2, default=str)[:2000],
            flush=True,
        )

    if args.vs_checkpoint is not None:

        def vs_frozen_actors(job):
            bc_seat = job["seat"]
            return {bc_seat: "bc", 1 - bc_seat: "frozen"}

        jobs = _schedule(maps, cfg, args.games, args.seed)
        label = args.vs_label
        games = _run(jobs, vs_frozen_actors, policies, device, greedy, f"vs {label}")
        key = f"vs_{label}"
        games_out[key] = games
        payload[key] = summarize(games, focal="bc", label=f"BC vs {label}")
        print(
            f"VS {label.upper()}",
            json.dumps(payload[key], indent=2, default=str)[:2000],
            flush=True,
        )

    if not args.skip_self_play:
        jobs = _schedule(maps, cfg, args.self_play_games, args.seed + 10_000)
        games = _run(jobs, self_actors, policies, device, greedy, "BC self-play")
        games_out["self_play"] = games
        payload["self_play"] = summarize(games, focal="bc", label="BC vs BC")
        print(
            "SELF-PLAY",
            json.dumps(payload["self_play"], indent=2, default=str)[:2000],
            flush=True,
        )

    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "games.json").write_text(
        json.dumps({**payload, "games": games_out}, indent=2, default=str)
    )
    append_eval(payload, checkpoint=args.checkpoint, name=out.name)
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
