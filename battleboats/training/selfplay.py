"""Full-game PPO against a short list of recent snapshots.

The learner plays the latest weights. The opponent is, on most games, a
uniform draw from the last ``snapshot_pool`` checkpoints, and on the rest a
uniform random agent so the value head still sees decisive games. There is
no rating, no prioritized matchmaking, and no exploiter. A later
quality-diversity archive can replace ``RecentSnapshots.sample`` without
changing the checkpoint format or the policy network.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.skill_maps import write_full_map
from battleboats.training.checkpoints import frozen_copy, load_checkpoint, save_checkpoint
from battleboats.training.curriculum import ArchSpec, _infer_token_dim, _new_policy
from battleboats.training.device import resolve_device
from battleboats.training.loop import PPOHyper, apply_ppo, collect_episodes
from battleboats.training.policy import PolicyNetwork


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
    rollout_decisions: int = 2048
    max_turns: int = 80
    max_steps: int = 400
    snapshot_every: int = 5
    snapshot_pool: int = 4
    random_opponent_frac: float = 0.15
    save_every: int = 10
    wandb: bool = False

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
        max_steps=int(raw.get("max_steps", 400)),
        snapshot_every=int(raw.get("snapshot_every", 5)),
        snapshot_pool=int(raw.get("snapshot_pool", 4)),
        random_opponent_frac=float(raw.get("random_opponent_frac", 0.15)),
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
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
    maps = [
        write_full_map(map_dir / f"map_{i}.json", cfg.width, cfg.height, seed=cfg.seed + i)
        for i in range(max(1, cfg.map_count))
    ]
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
        f"random_frac={cfg.random_opponent_frac} map={cfg.width}x{cfg.height}"
    )

    def reset(episode_index: int) -> int:
        env.map_json_path = maps[episode_index % len(maps)]
        env.reset(seed=cfg.seed + episode_index)
        return episode_index % 2

    def opponent_for(episode_index: int):
        if rng.random() < cfg.random_opponent_frac:
            return None, None
        opp = snapshots.sample(rng)
        if opp is None:
            return None, None
        return opp, None

    for update in range(int(state["update"]), cfg.updates):
        buffer, infos, ep = collect_episodes(
            policy,
            env,
            rng,
            target_decisions=cfg.rollout_decisions,
            hyper=cfg.ppo,
            max_steps=cfg.max_steps,
            reset=reset,
            opponent_for=opponent_for,
            use_env_reward=False,
            episode_start=ep,
        )
        metrics = apply_ppo(policy, optimizer, buffer, cfg.ppo, device)
        wins = sum(1 for info in infos if info.get("outcome", 0) > 0)
        draws = sum(1 for info in infos if info.get("outcome", 0) == 0)
        n = max(1, len(infos))
        row = {
            "update": update + 1,
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "games": len(infos),
            "pool": snapshots.names(),
            **{f"ppo_{k}": v for k, v in metrics.items()},
        }
        with history.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[selfplay {update + 1}/{cfg.updates}] "
            f"win={row['win_rate']:.2f} draw={row['draw_rate']:.2f} "
            f"games={row['games']} pool={len(snapshots.paths)} "
            f"ent={metrics['entropy']:.3f}"
        )
        if use_wandb:
            import wandb

            wandb.log({k: v for k, v in row.items() if k != "pool"})
        save_checkpoint(policy, latest, extra={"update": update + 1})
        if cfg.save_every and (update + 1) % cfg.save_every == 0:
            save_checkpoint(policy, cfg.run_dir / f"update_{update + 1}.pt")
        if (update + 1) % max(1, cfg.snapshot_every) == 0:
            snapshots.add(policy, tag=f"update_{update + 1:05d}")
        state = {"update": update + 1, "episode": ep}
        state_path.write_text(json.dumps(state, indent=2))

    if use_wandb:
        import wandb

        wandb.finish()
    try:
        env.close()
    except Exception:
        pass
    return {"ok": True, "checkpoint": str(latest), "updates": cfg.updates, "pool": snapshots.names()}
