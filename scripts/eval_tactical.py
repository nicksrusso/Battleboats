"""Smoke-test the tactical scripted agent against v0.3.0, itself, and random.

    poetry run python scripts/eval_tactical.py \\
        --checkpoint runs/checkpoints/v0.3.0_26348ee.pt \\
        --maps-dir runs/selfplay_mixed_v0.2.2/maps \\
        --config runs/selfplay_mixed_v0.2.2/config.yaml \\
        --output-dir runs/eval/scripted_v_v0.3.0/

100 games vs the checkpoint (50 per seat, 16x8 and 20x10 from the v0.2.2
pool, per-map max_turns / max_steps from that run's config), plus 20
scripted-vs-scripted and 20 scripted-vs-random sanity games. Reports
W/L/D with Wilson CIs, per-seat and per-size splits, turn stats, buy
mix, and landing-ship capture / survival stats for the scripted seat.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prefer the repo that contains this script (worktrees included) over an
# editable install that may still point at another checkout.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from battleboats.agents.random_agent import random_action
from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import AttackAction, BuildShipAction, CapturePortAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import episode_limits
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.selfplay import load_selfplay_config
from battleboats.training.selfplay_purchases import hull_name


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def _size_of(path: Path) -> str:
    data = json.loads(path.read_text())
    return f"{int(data['width'])}x{int(data['height'])}"


def _list_maps(maps_dir: Path) -> List[Path]:
    maps = sorted(p for p in maps_dir.glob("map_*.json"))
    if not maps:
        raise FileNotFoundError(f"no map_*.json in {maps_dir}")
    return maps


def _limits(path: Path, cfg) -> Tuple[int, int, str]:
    data = json.loads(path.read_text())
    turns, steps = episode_limits(
        data,
        max_turns=cfg.max_turns,
        max_steps=cfg.max_steps,
        ref_home_distance=cfg.ref_home_distance,
    )
    return turns, steps, f"{int(data['width'])}x{int(data['height'])}"


def _empty_landing_stats() -> Dict[str, Any]:
    return {
        "bought": 0,
        "killed_by_hostile": 0,
        "survived": 0,
        "port_captures": 0,
        "home_captures": 0,
        "ship_attacks": 0,
        "lifespans_killed": [],
        "lifespans_all": [],
    }


def _new_tracker(scripted_pids: Tuple[int, ...]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for pid in scripted_pids:
        out[pid] = {
            "ids": {},  # sid -> birth_turn
            "stats": _empty_landing_stats(),
        }
    return out


def _note_births(
    engine, pid: int, tracker: Dict[int, Dict[str, Any]], before: set
) -> None:
    if pid not in tracker:
        return
    after = set(engine.players[pid].owned_ship_ids)
    for sid in after - before:
        ship = engine.ships.get(sid)
        if ship is None or ship.type is not ShipType.LANDING:
            continue
        birth = ship.built_turn if ship.built_turn >= 0 else engine.turn
        tracker[pid]["ids"][sid] = birth
        tracker[pid]["stats"]["bought"] += 1


def _note_deaths(
    engine,
    victim_pid: int,
    tracker: Dict[int, Dict[str, Any]],
    before: set,
    killed_by_hostile: bool,
) -> None:
    if victim_pid not in tracker:
        return
    after = set(engine.players[victim_pid].owned_ship_ids)
    # Only known landing ships. Other hulls also leave `before - after`
    # when they die; they must not be counted as landings.
    lost = (before - after) & set(tracker[victim_pid]["ids"])
    for sid in lost:
        birth = tracker[victim_pid]["ids"].pop(sid)
        life = max(0, engine.turn - birth)
        tracker[victim_pid]["stats"]["lifespans_killed"].append(life)
        tracker[victim_pid]["stats"]["lifespans_all"].append(life)
        if killed_by_hostile:
            tracker[victim_pid]["stats"]["killed_by_hostile"] += 1


def _finalize_survivors(engine, tracker: Dict[int, Dict[str, Any]]) -> None:
    for pid, rec in tracker.items():
        for sid, birth in list(rec["ids"].items()):
            life = max(0, engine.turn - birth)
            rec["stats"]["lifespans_all"].append(life)
            rec["stats"]["survived"] += 1
        rec["ids"].clear()


def _buy_name(ship_type: ShipType) -> str:
    return hull_name(ship_type)


def play_game(
    map_path: Path,
    *,
    max_turns: int,
    max_steps: int,
    seed: int,
    actors: Dict[int, str],
    policy=None,
    device=None,
) -> Dict[str, Any]:
    """Play one game. ``actors[pid]`` is ``tactical``, ``random``, or ``policy``."""
    env = BattleboatsAEC(str(map_path), max_turns=max_turns)
    if hasattr(env, "prime_legal_actions"):
        env.prime_legal_actions = False
    env.reset(seed=seed)
    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    scripted_pids = tuple(pid for pid, kind in actors.items() if kind == "tactical")
    tracker = _new_tracker(scripted_pids)
    buys = {0: Counter(), 1: Counter()}
    # No landings exist at a standard opening.
    steps = 0
    last_winner = None

    def action_for(pid: int):
        kind = actors[pid]
        if kind == "tactical":
            return tactical_scripted_action(env.engine, pid, rngs[pid])
        if kind == "random":
            return random_action(env.engine, pid, rngs[pid])
        masks = ActionMasks(env.engine, pid, device=device)
        with torch.no_grad():
            a_idx, v_idx, t_idx, _logp, _value = policy.act(
                masks.tokens, masks.pad_mask, masks, skill_idx=0
            )
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
            buys[pid][_buy_name(action.ship_type)] += 1
        if pid in tracker:
            if isinstance(action, CapturePortAction):
                ship = engine.ships.get(action.landing_ship_id)
                if ship is not None and ship.type is ShipType.LANDING:
                    tracker[pid]["stats"]["port_captures"] += 1
                    opp_home = engine.players[1 - pid].home_port
                    if action.target == opp_home:
                        tracker[pid]["stats"]["home_captures"] += 1
            if isinstance(action, AttackAction):
                ship = engine.ships.get(action.attacker_id)
                if ship is not None and ship.type is ShipType.LANDING:
                    tracker[pid]["stats"]["ship_attacks"] += 1
        env.step(action)
        steps += 1
        # Births for the acting player (new buys).
        _note_births(engine, pid, tracker, before[pid])
        # Deaths: a missing landing after an enemy AttackAction is a hostile kill.
        for victim in (0, 1):
            if victim == pid:
                continue
            killed = isinstance(action, AttackAction)
            _note_deaths(engine, victim, tracker, before[victim], killed)
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
    for pid in scripted_pids:
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


def _mean(xs: List[float]) -> Optional[float]:
    return float(st.mean(xs)) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return float(st.median(xs)) if xs else None


def summarize(games: List[Dict[str, Any]], *, focal: str, label: str) -> Dict[str, Any]:
    """Summarize from the point of view of seats whose actor is ``focal``."""
    n = len(games)
    # Outcome is from the focal seat. If both seats are focal (scripted vs
    # scripted), seat 0 is the reporting seat so W/L stay well-defined.
    wins = losses = draws = 0
    win_turns = []
    loss_turns = []
    all_turns = []
    reasons = Counter()
    by_seat = {0: Counter(), 1: Counter()}
    by_size: Dict[str, Counter] = defaultdict(Counter)
    buys_focal: Counter = Counter()
    buys_other: Counter = Counter()
    landing_acc = _empty_landing_stats()

    for g in games:
        size = g["size"]
        all_turns.append(g["turn"])
        reasons[g["reason"]] += 1
        # Reporting seat: first seat whose actor is focal, else seat 0.
        report_pid = 0
        for pid in (0, 1):
            if g["actors"][pid] == focal:
                report_pid = pid
                break
        winner = g["winner"]
        if winner is None:
            result = "D"
            draws += 1
        elif winner == report_pid:
            result = "W"
            wins += 1
            win_turns.append(g["turn"])
        else:
            result = "L"
            losses += 1
            loss_turns.append(g["turn"])
        by_seat[report_pid][result] += 1
        # Also split "as seat k" only when that seat is the focal actor.
        by_size[size][result] += 1
        for pid in (0, 1):
            bucket = buys_focal if g["actors"][pid] == focal else buys_other
            for hull, c in g["buys"][pid].items():
                bucket[hull] += c
        for pid, rec in g["landing"].items():
            if g["actors"].get(pid) != focal:
                continue
            landing_acc["bought"] += rec["bought"]
            landing_acc["killed_by_hostile"] += rec["killed_by_hostile"]
            landing_acc["survived"] += rec["survived"]
            landing_acc["port_captures"] += rec["port_captures"]
            landing_acc["home_captures"] += rec["home_captures"]
            landing_acc["ship_attacks"] += rec["ship_attacks"]
            landing_acc["lifespans_killed"].extend(rec["lifespans_killed"])
            landing_acc["lifespans_all"].extend(rec["lifespans_all"])

    def wld(counter: Counter, total: int) -> Dict[str, Any]:
        w, l, d = counter["W"], counter["L"], counter["D"]
        return {
            "n": total,
            "wins": w,
            "losses": l,
            "draws": d,
            "win_rate": w / total if total else 0.0,
            "wilson95_win_rate": list(wilson(w, total)),
        }

    # Per-seat: vs a non-focal opponent, report the seat that played as
    # focal. For scripted vs scripted both seats are focal; seat 0 is the
    # headline W/L seat and seat 1 is shown from its own perspective.
    seat_stats = {}
    for seat in (0, 1):
        c = Counter()
        for g in games:
            report_pid = 0
            for pid in (0, 1):
                if g["actors"][pid] == focal:
                    report_pid = pid
                    break
            if report_pid != seat and not (
                g["actors"][0] == focal and g["actors"][1] == focal
            ):
                continue
            if g["actors"][0] == focal and g["actors"][1] == focal:
                # only attach to the actual report seat (0)
                if seat != 0:
                    # still record this seat's own W/L for s-vs-s
                    winner = g["winner"]
                    if winner is None:
                        c["D"] += 1
                    elif winner == seat:
                        c["W"] += 1
                    else:
                        c["L"] += 1
                    continue
            winner = g["winner"]
            if winner is None:
                c["D"] += 1
            elif winner == report_pid:
                c["W"] += 1
            else:
                c["L"] += 1
        seat_n = sum(c.values())
        seat_stats[f"seat_{seat}"] = wld(c, seat_n)

    size_stats = {}
    for size, c in sorted(by_size.items()):
        size_stats[size] = wld(c, sum(c.values()))

    bought = landing_acc["bought"]
    killed = landing_acc["killed_by_hostile"]
    return {
        "label": label,
        "n": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n if n else 0.0,
        "wilson95_win_rate": list(wilson(wins, n)),
        "score_draw_half": (wins + 0.5 * draws) / n if n else 0.0,
        "by_seat": seat_stats,
        "by_size": size_stats,
        "mean_turn": _mean(all_turns),
        "median_turn": _median(all_turns),
        "median_win_turn": _median(win_turns),
        "median_loss_turn": _median(loss_turns),
        "reasons": dict(reasons),
        "buy_mix_focal": dict(sorted(buys_focal.items())),
        "buy_mix_opponent": dict(sorted(buys_other.items())),
        "landing_ships": {
            "bought": bought,
            "killed_by_hostile": killed,
            "survived": landing_acc["survived"],
            "kill_rate": killed / bought if bought else 0.0,
            "survive_rate": landing_acc["survived"] / bought if bought else 0.0,
            "port_captures": landing_acc["port_captures"],
            "home_captures": landing_acc["home_captures"],
            "ship_attacks": landing_acc["ship_attacks"],
            "mean_lifespan_killed": _mean(landing_acc["lifespans_killed"]),
            "mean_lifespan_all": _mean(landing_acc["lifespans_all"]),
            "median_lifespan_all": _median(landing_acc["lifespans_all"]),
        },
    }


def _schedule(maps: List[Path], cfg, n_games: int, seed: int) -> List[Dict[str, Any]]:
    """Alternate seats, cycle maps, keep size balance as even as possible."""
    by_size: Dict[str, List[Path]] = defaultdict(list)
    meta = {}
    for path in maps:
        turns, steps, size = _limits(path, cfg)
        by_size[size].append(path)
        meta[path] = (turns, steps, size)
    sizes = sorted(by_size)
    jobs = []
    for i in range(n_games):
        size = sizes[i % len(sizes)]
        pool = by_size[size]
        path = pool[(i // len(sizes)) % len(pool)]
        turns, steps, size = meta[path]
        # Cycle size fastest and seat every len(sizes) games so seat and
        # board size are not confounded (25 games per (seat, size) at N=100).
        jobs.append(
            {
                "i": i,
                "path": path,
                "max_turns": turns,
                "max_steps": steps,
                "size": size,
                "seed": seed + i,
                "seat": (i // max(1, len(sizes))) % 2,
            }
        )
    return jobs


def run_vs_policy(
    jobs: List[Dict[str, Any]],
    policy,
    device,
    *,
    label: str,
) -> List[Dict[str, Any]]:
    games = []
    for j, job in enumerate(jobs):
        scripted_pid = job["seat"]
        actors = {scripted_pid: "tactical", 1 - scripted_pid: "policy"}
        rec = play_game(
            job["path"],
            max_turns=job["max_turns"],
            max_steps=job["max_steps"],
            seed=job["seed"],
            actors=actors,
            policy=policy,
            device=device,
        )
        rec["size"] = job["size"]
        rec["scripted_pid"] = scripted_pid
        games.append(rec)
        print(
            f"  [{label}] {j + 1}/{len(jobs)} size={job['size']} "
            f"scripted_seat={scripted_pid} winner={rec['winner']} "
            f"turn={rec['turn']} reason={rec['reason']}",
            flush=True,
        )
    return games


def run_pair(
    jobs: List[Dict[str, Any]],
    *,
    kind0: str,
    kind1: str,
    label: str,
) -> List[Dict[str, Any]]:
    games = []
    for j, job in enumerate(jobs):
        # Alternate seats with the same schedule used vs the checkpoint.
        if job["seat"] == 0:
            actors = {0: kind0, 1: kind1}
            scripted_pid = 0 if kind0 == "tactical" else 1
        else:
            actors = {0: kind1, 1: kind0}
            scripted_pid = 1 if kind0 == "tactical" else 0
        rec = play_game(
            job["path"],
            max_turns=job["max_turns"],
            max_steps=job["max_steps"],
            seed=job["seed"],
            actors=actors,
        )
        rec["size"] = job["size"]
        rec["scripted_pid"] = scripted_pid
        games.append(rec)
        print(
            f"  [{label}] {j + 1}/{len(jobs)} size={job['size']} "
            f"winner={rec['winner']} turn={rec['turn']} reason={rec['reason']}",
            flush=True,
        )
    return games


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/checkpoints/v0.3.0_26348ee.pt"),
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("runs/selfplay_mixed_v0.2.2/maps"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("runs/selfplay_mixed_v0.2.2/config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/eval/scripted_v_v0.3.0"),
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sanity-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    cfg = load_selfplay_config(args.config)
    maps = _list_maps(args.maps_dir)
    print(f"maps={len(maps)} checkpoint={args.checkpoint} device=cpu", flush=True)

    policy, _ = load_checkpoint(args.checkpoint, device)
    policy.eval()

    vs_jobs = _schedule(maps, cfg, args.games, args.seed)
    vs_games = run_vs_policy(vs_jobs, policy, device, label="vs v0.3.0")
    vs_summary = summarize(vs_games, focal="tactical", label="tactical vs v0.3.0")

    ss_jobs = _schedule(maps, cfg, args.sanity_games, args.seed + 10_000)
    ss_games = run_pair(
        ss_jobs, kind0="tactical", kind1="tactical", label="scripted vs scripted"
    )
    ss_summary = summarize(ss_games, focal="tactical", label="tactical vs tactical")

    sr_jobs = _schedule(maps, cfg, args.sanity_games, args.seed + 20_000)
    sr_games = run_pair(
        sr_jobs, kind0="tactical", kind1="random", label="scripted vs random"
    )
    sr_summary = summarize(sr_games, focal="tactical", label="tactical vs random")

    payload = {
        "checkpoint": str(args.checkpoint),
        "maps_dir": str(args.maps_dir),
        "config": str(args.config),
        "seed": args.seed,
        "design": {
            "buys": "per-game uniform build queue: sample types equally, buy the head when cash and a legal port allow it, otherwise save",
            "strength": "local power = sum strength * max(0, 1 - d/R) within R=max(6, attack_range); engage if ours > theirs * 1.2",
            "noise": "softmax over per-ship move scores with temperature 0.5",
            "landings": "target the nearest known capturable port (any enemy or neutral, by water-path distance); always flee combat ships; never AttackAction",
            "fog": "same sightings as policy observation (known_enemy_ships / known_enemy_ports); stale last-seen positions decay over 4 turns",
        },
        "vs_v0.3.0": vs_summary,
        "scripted_vs_scripted": ss_summary,
        "scripted_vs_random": sr_summary,
        "games": {
            "vs_v0.3.0": vs_games,
            "scripted_vs_scripted": ss_games,
            "scripted_vs_random": sr_games,
        },
    }
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    full_path = out_dir / "games.json"
    # Split: summary without per-game blobs, full file with everything.
    summary = {k: v for k, v in payload.items() if k != "games"}
    summary_path.write_text(json.dumps(summary, indent=2, default=_jsonable))
    full_path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    print(json.dumps(summary, indent=2, default=_jsonable))
    print(f"wrote {summary_path} and {full_path}", flush=True)


if __name__ == "__main__":
    main()
