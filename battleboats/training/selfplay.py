"""Full-game PPO against a short list of recent snapshots.

The learner plays the latest weights. The opponent is, on most games, a
uniform draw from the last ``snapshot_pool`` checkpoints, and on the rest a
uniform random agent so the value head still sees decisive games. There is
no rating, no prioritized matchmaking, and no exploiter. A later
quality-diversity archive can replace ``RecentSnapshots.sample`` without
changing the checkpoint format or the policy network.

Collection goes through ``vec_collect``: worker processes keep several games
open, batch the forward passes, and stop at exactly ``rollout_decisions``
learner decisions. Cut games are bootstrapped and continue next update.
Opponents are drawn per game from the ring at that game's start.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.envs.skill_maps import write_full_map
from battleboats.training.checkpoints import (
    frozen_copy,
    load_checkpoint,
    model_config_of,
    save_checkpoint,
)
from battleboats.training.curriculum import ArchSpec, _infer_token_dim, _new_policy
from battleboats.training.device import resolve_device
from battleboats.training.loop import PPOHyper, apply_ppo
from battleboats.training.policy import PolicyNetwork, set_dropout
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.selfplay_purchases import (
    aggregate_purchases,
    format_purchase_line,
    purchase_mix,
)
from battleboats.training.vec_collect import make_collector


@dataclass
class SelfPlayConfig:
    run_dir: Path
    device: str = "auto"
    seed: int = 0
    arch: ArchSpec = field(default_factory=ArchSpec)
    ppo: PPOHyper = field(default_factory=PPOHyper)
    init_checkpoint: Optional[Path] = None
    width: int = 8
    height: int = 8
    map_count: int = 4
    updates: int = 200
    # Exact learner decisions per update. Games longer than this are cut
    # and bootstrapped, then continue next update (see vec_collect).
    rollout_decisions: int = 2048
    max_turns: int = 80
    # 400 cut full games near turn 45. 3000 lets an 80-turn game finish
    # on the turn cap. max_steps caps engine steps per game.
    max_steps: int = 3000
    snapshot_every: int = 5
    snapshot_pool: int = 4
    random_opponent_frac: float = 0.15
    save_every: int = 10
    wandb: bool = False
    # Random maps from core/config/generate_map.py. When set, width/height/
    # map_count are ignored, each episode draws a map uniformly from the pool,
    # and max_turns/max_steps are scaled per map by home distance over
    # ref_home_distance (the 8x8 boards' 14).
    map_pool: List[PoolSize] = field(default_factory=list)
    land_fraction: float = 0.02
    ref_home_distance: int = 14
    # Collection: 0 workers plays in this process. Each worker keeps
    # games_per_worker games open and batches their forward passes.
    num_workers: int = 0
    games_per_worker: int = 4
    collect_threads: int = 1
    ppo_threads: int = 8
    # Overrides every dropout rate after loading (None keeps the checkpoint's).
    # 0 makes the PPO re-score match the eval-mode rollout.
    dropout: Optional[float] = 0.0

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.init_checkpoint is not None:
            self.init_checkpoint = Path(self.init_checkpoint)


def load_selfplay_config(path: Path) -> SelfPlayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    arch = ArchSpec(**raw.get("arch", {}))
    ppo = PPOHyper(**raw.get("ppo", {}))
    init = raw.get("init_checkpoint") or None
    return SelfPlayConfig(
        run_dir=raw.get("run_dir", "runs/selfplay"),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 0)),
        arch=arch,
        ppo=ppo,
        init_checkpoint=Path(init) if init else None,
        width=int(raw.get("width", 8)),
        height=int(raw.get("height", 8)),
        map_count=int(raw.get("map_count", 4)),
        updates=int(raw.get("updates", 200)),
        rollout_decisions=int(raw.get("rollout_decisions", 2048)),
        max_turns=int(raw.get("max_turns", 80)),
        max_steps=int(raw.get("max_steps", 3000)),
        snapshot_every=int(raw.get("snapshot_every", 5)),
        snapshot_pool=int(raw.get("snapshot_pool", 4)),
        random_opponent_frac=float(raw.get("random_opponent_frac", 0.15)),
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
        map_pool=[PoolSize(**entry) for entry in raw.get("map_pool") or []],
        land_fraction=float(raw.get("land_fraction", 0.02)),
        ref_home_distance=int(raw.get("ref_home_distance", 14)),
        num_workers=int(raw.get("num_workers", 0)),
        games_per_worker=int(raw.get("games_per_worker", 4)),
        collect_threads=int(raw.get("collect_threads", 1)),
        ppo_threads=int(raw.get("ppo_threads", 8)),
        dropout=None if raw.get("dropout", 0.0) is None else float(raw.get("dropout", 0.0)),
    )


class RecentSnapshots:
    """Ring of the last K frozen policies, stored as ordinary checkpoints.

    ``sample`` draws uniformly. Files older than the ring stay on disk so a
    future archive can index them; they are just not sampled.
    """

    def __init__(self, directory: Path, pool_size: int, device: torch.device) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pool_size = max(1, pool_size)
        self.device = device
        self.paths: List[Path] = []
        self._cache: Dict[str, PolicyNetwork] = {}

    def add(self, policy: PolicyNetwork, *, tag: str) -> Path:
        path = self.directory / f"{tag}.pt"
        save_checkpoint(policy, path, extra={"tag": tag})
        self.paths.append(path)
        self._cache[str(path)] = frozen_copy(policy)
        while len(self.paths) > self.pool_size:
            dropped = self.paths.pop(0)
            self._cache.pop(str(dropped), None)
        return path

    def load_from_disk(self) -> int:
        """Keep the newest K ``update_*.pt`` files already in the directory."""
        found = sorted(self.directory.glob("update_*.pt"))
        self.paths = found[-self.pool_size :]
        self._cache.clear()
        return len(self.paths)

    def sample(self, rng: random.Random) -> Optional[PolicyNetwork]:
        if not self.paths:
            return None
        path = rng.choice(self.paths)
        key = str(path)
        if key not in self._cache:
            net, _ = load_checkpoint(path, self.device)
            self._cache[key] = frozen_copy(net)
        return self._cache[key]

    def names(self) -> List[str]:
        return [p.name for p in self.paths]

    def ring(self) -> Dict[str, PolicyNetwork]:
        """Every pooled snapshot by tag (file stem), loading any not cached."""
        out: Dict[str, PolicyNetwork] = {}
        for path in self.paths:
            key = str(path)
            if key not in self._cache:
                net, _ = load_checkpoint(path, self.device)
                self._cache[key] = frozen_copy(net)
            out[path.stem] = self._cache[key]
        return out


def run_selfplay(cfg: SelfPlayConfig, *, resume: bool = True) -> Dict[str, Any]:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    state_path = cfg.run_dir / "state.json"
    state = {"update": 0, "episode": 0}
    if resume and state_path.exists():
        state.update(json.loads(state_path.read_text()))

    map_dir = cfg.run_dir / "maps"
    if cfg.map_pool:
        maps = build_map_pool(map_dir, cfg.map_pool, seed=cfg.seed, land_fraction=cfg.land_fraction)
    else:
        maps = [
            write_full_map(map_dir / f"map_{i}.json", cfg.width, cfg.height, seed=cfg.seed + i)
            for i in range(max(1, cfg.map_count))
        ]
    map_info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        if cfg.map_pool:
            turns, steps = episode_limits(
                data,
                max_turns=cfg.max_turns,
                max_steps=cfg.max_steps,
                ref_home_distance=cfg.ref_home_distance,
            )
        else:
            turns, steps = cfg.max_turns, cfg.max_steps
        map_info.append({"size": f"{data['width']}x{data['height']}", "max_turns": turns, "max_steps": steps})
    env = BattleboatsAEC(maps[0], max_turns=cfg.max_turns)
    env.reset(seed=cfg.seed)

    latest = cfg.run_dir / "latest.pt"
    if state["update"] > 0 and latest.exists():
        policy, _ = load_checkpoint(latest, device)
        print(f"[selfplay] resume from update {state['update']}")
    elif cfg.init_checkpoint is not None and Path(cfg.init_checkpoint).exists():
        policy, _ = load_checkpoint(cfg.init_checkpoint, device)
        print(f"[selfplay] warm start from {cfg.init_checkpoint}")
    else:
        if cfg.init_checkpoint is not None:
            print(f"[selfplay] {cfg.init_checkpoint} missing — random init")
        policy = _new_policy(cfg.arch, _infer_token_dim(env), device)
    if cfg.dropout is not None:
        set_dropout(policy, cfg.dropout)

    snapshots = RecentSnapshots(cfg.run_dir / "snapshots", cfg.snapshot_pool, device)
    if snapshots.load_from_disk() == 0:
        # Seed the ring with the weights we are about to move away from.
        snapshots.add(policy, tag=f"update_{int(state['update']):05d}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    use_wandb = cfg.wandb
    if use_wandb:
        import wandb

        wandb.init(project="battleboats-selfplay", config={"run_dir": str(cfg.run_dir)})

    history = cfg.run_dir / "history.jsonl"
    ep = int(state["episode"])
    print(
        f"[selfplay] device={device} updates={cfg.updates} pool={cfg.snapshot_pool} "
        f"random_frac={cfg.random_opponent_frac} "
        + (
            f"map_pool={len(maps)} sizes={sorted({m['size'] for m in map_info})}"
            if cfg.map_pool
            else f"map={cfg.width}x{cfg.height}"
        )
    )

    collector = make_collector(
        maps,
        map_info,
        num_workers=cfg.num_workers,
        games_per_worker=cfg.games_per_worker,
        collect_threads=cfg.collect_threads,
        model_config=model_config_of(policy),
        seed=cfg.seed,
        episode_offset=ep,
        random_opponent_frac=cfg.random_opponent_frac,
        gamma=cfg.ppo.gamma,
        lam=cfg.ppo.lam,
    )
    try:
        return _train_loop(cfg, policy, optimizer, snapshots, collector, state, state_path, latest, history, device, use_wandb)
    finally:
        collector.close()
        try:
            env.close()
        except Exception:
            pass


def _train_loop(cfg, policy, optimizer, snapshots, collector, state, state_path, latest, history, device, use_wandb):
    for update in range(int(state["update"]), cfg.updates):
        t0 = time.perf_counter()
        if cfg.num_workers <= 0:
            torch.set_num_threads(max(1, cfg.collect_threads))
        policy.eval()
        collected = collector.collect(policy, snapshots.ring(), cfg.rollout_decisions)
        collect_s = time.perf_counter() - t0
        buffer = RolloutBuffer()
        for segment in collected.segments:
            buffer.add(segment)
        infos = collected.finished
        torch.set_num_threads(max(1, cfg.ppo_threads))
        t1 = time.perf_counter()
        metrics = apply_ppo(policy, optimizer, buffer, cfg.ppo, device)
        ppo_s = time.perf_counter() - t1
        wins = sum(1 for info in infos if info.get("outcome", 0) > 0)
        draws = sum(1 for info in infos if info.get("outcome", 0) == 0)
        n = max(1, len(infos))
        turns = [int(info.get("turn", 0)) for info in infos]
        step_counts = [int(info.get("steps", 0)) for info in infos]
        by_size: Dict[str, Dict[str, int]] = {}
        for info in infos:
            b = by_size.setdefault(info.get("map_size", "?"), {"games": 0, "wins": 0, "draws": 0})
            b["games"] += 1
            b["wins"] += int(info.get("outcome", 0) > 0)
            b["draws"] += int(info.get("outcome", 0) == 0)
        row = {
            "update": update + 1,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "games": len(infos),
            "max_turn": max(turns) if turns else 0,
            "mean_steps": (sum(step_counts) / len(step_counts)) if step_counts else 0.0,
            "pool": snapshots.names(),
            "by_size": by_size,
            "decisions": collected.decisions,
            "env_steps": collected.env_steps,
            "segments": len(collected.segments),
            "collect_s": collect_s,
            "ppo_s": ppo_s,
            # Learner purchases made during this update, finished games or not.
            **purchase_mix(aggregate_purchases([{"purchases": collected.purchases}])),
            **{f"ppo_{k}": v for k, v in metrics.items()},
        }
        with history.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[selfplay {update + 1}/{cfg.updates}] "
            f"win={row['win_rate']:.2f} draw={row['draw_rate']:.2f} "
            f"games={row['games']} max_turn={row['max_turn']} "
            f"mean_steps={row['mean_steps']:.0f} pool={len(snapshots.paths)} "
            f"ent={metrics['entropy']:.3f} dec={collected.decisions} env_steps={collected.env_steps} "
            f"collect={collect_s:.1f}s ppo={ppo_s:.1f}s"
            + (
                " sizes=" + ",".join(f"{k}:{v['games']}" for k, v in sorted(by_size.items()))
                if cfg.map_pool
                else ""
            )
            + " "
            + format_purchase_line(row)
        )
        if use_wandb:
            import wandb

            wandb.log({k: v for k, v in row.items() if k not in ("pool", "by_size", "purchases")})
        save_checkpoint(policy, latest, extra={"update": update + 1})
        if cfg.save_every and (update + 1) % cfg.save_every == 0:
            save_checkpoint(policy, cfg.run_dir / f"update_{update + 1}.pt")
        if (update + 1) % max(1, cfg.snapshot_every) == 0:
            snapshots.add(policy, tag=f"update_{update + 1:05d}")
        state = {"update": update + 1, "episode": collector.next_episode()}
        state_path.write_text(json.dumps(state, indent=2))

    if use_wandb:
        import wandb

        wandb.finish()
    return {"ok": True, "checkpoint": str(latest), "updates": cfg.updates, "pool": snapshots.names()}
