"""Size curriculum over sandboxes, one shared policy.

Order is merchant, then builder (buy a ship), then landing, then one combat
ship type at a time. A size advances when the rolling success rate on that
size clears the threshold. A fraction of episodes are smaller sizes of the
same sandbox, and a fraction are fresh episodes of earlier sandboxes (rehearsal).
The one-hot skill input is ``env.skill_index``.

Checkpoints:
    ``<run_dir>/<skill>/size_<N>.pt`` when that size clears
    ``<run_dir>/<skill>/final.pt`` when the sandbox finishes
    ``<run_dir>/latest.pt`` every update
    ``<run_dir>/summary.json`` rates per sandbox and size
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

from battleboats.envs.sandboxes import horizon, make_sandbox
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.checkpoints import load_checkpoint, save_checkpoint
from battleboats.training.curriculum import ArchSpec
from battleboats.training.device import resolve_device
from battleboats.training.loop import PPOHyper, apply_ppo
from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.rollout import collect_trajectory, compute_gae
from battleboats.training.sandbox_skills import SKILL_NAMES, skill_index
from battleboats.training.wandb_log import init_wandb

Role = str  # focus | mix | rehearsal


def choose_episode(
    rng: random.Random,
    *,
    focus_skill: str,
    focus_size: int,
    smaller_sizes: Sequence[int],
    earlier: Sequence[Tuple[str, int]],
    rehearsal_frac: float,
    size_mix_frac: float,
) -> Tuple[str, int, Role]:
    """Pick the sandbox and board size for one episode."""
    if earlier and rng.random() < rehearsal_frac:
        skill, size = earlier[rng.randrange(len(earlier))]
        return skill, int(size), "rehearsal"
    if smaller_sizes and rng.random() < size_mix_frac:
        return focus_skill, int(smaller_sizes[rng.randrange(len(smaller_sizes))]), "mix"
    return focus_skill, int(focus_size), "focus"


def window_cleared(window: Sequence[float], need: int, threshold: float) -> bool:
    if len(window) < need:
        return False
    return (sum(window) / len(window)) >= threshold


def size_advances(updates: int, window: Sequence[float], need: int, threshold: float, min_updates: int) -> bool:
    """A size clears only after enough updates and a full successful window."""
    return updates >= min_updates and window_cleared(window, need, threshold)


def _succeeded(skill: str, env) -> bool:
    if skill == "merchant":
        return bool(env.hauled)
    if skill == "builder":
        return bool(env.built)
    if skill == "landing":
        return bool(env.captured)
    return bool(env.killed)


@dataclass
class SandboxRunConfig:
    run_dir: Path
    skills: List[str]
    sizes: List[int]
    device: str = "cpu"
    seed: int = 0
    success_threshold: float = 0.8
    window: int = 64
    min_updates: int = 3
    max_episodes_per_size: int = 400
    decisions_per_update: int = 512
    rehearsal_frac: float = 0.15
    size_mix_frac: float = 0.2
    eval_every: int = 5
    eval_episodes: int = 8
    allow_unmet: bool = False
    save_every: int = 10
    wandb: bool = False
    wandb_project: str = "battleboats-skill-pipeline"
    arch: ArchSpec = field(default_factory=ArchSpec)
    ppo: PPOHyper = field(default_factory=PPOHyper)
    init_checkpoint: Optional[Path] = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.sizes = [int(s) for s in self.sizes]
        if not self.skills:
            raise ValueError("skills is empty")
        if not self.sizes:
            raise ValueError("sizes is empty")
        for name in self.skills:
            skill_index(name)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_updates < 1:
            raise ValueError("min_updates must be >= 1")


def load_sandbox_config(path: Path) -> SandboxRunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    arch = ArchSpec(**raw.get("arch", {}))
    ppo = PPOHyper(**raw.get("ppo", {}))
    init = raw.get("init_checkpoint") or None
    return SandboxRunConfig(
        run_dir=raw.get("run_dir", "runs/sandboxes"),
        skills=list(raw.get("skills", SKILL_NAMES)),
        sizes=list(raw.get("sizes", [3, 4, 5, 6, 8, 10])),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 0)),
        success_threshold=float(raw.get("success_threshold", 0.8)),
        window=int(raw.get("window", 64)),
        min_updates=int(raw.get("min_updates", 3)),
        max_episodes_per_size=int(raw.get("max_episodes_per_size", 400)),
        decisions_per_update=int(raw.get("decisions_per_update", 512)),
        rehearsal_frac=float(raw.get("rehearsal_frac", 0.15)),
        size_mix_frac=float(raw.get("size_mix_frac", 0.2)),
        eval_every=int(raw.get("eval_every", 5)),
        eval_episodes=int(raw.get("eval_episodes", 8)),
        allow_unmet=bool(raw.get("allow_unmet", False)),
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
        wandb_project=str(raw.get("wandb_project", "battleboats-skill-pipeline")),
        arch=arch,
        ppo=ppo,
        init_checkpoint=Path(init) if init else None,
    )


def run_sandboxes(cfg: SandboxRunConfig, *, resume: bool = True) -> Dict:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    state_path = cfg.run_dir / "state.json"
    state = _load_state(state_path) if resume and state_path.exists() else _fresh_state()
    logger = init_wandb(
        cfg.wandb,
        project=cfg.wandb_project,
        config={"run_dir": str(cfg.run_dir), "skills": cfg.skills, "sizes": cfg.sizes},
    )
    policy = _load_or_new(cfg, state, device)
    envs: Dict[Tuple[str, int], object] = {}
    print(
        f"[sandboxes] device={device} skills={cfg.skills} sizes={cfg.sizes} "
        f"resume_skill={state['skill_index']} resume_size={state['size_index']}"
    )
    try:
        while state["skill_index"] < len(cfg.skills):
            skill = cfg.skills[state["skill_index"]]
            done, policy = _run_skill(cfg, state, skill, policy, envs, rng, device, logger)
            if not done:
                state["ok"] = False
                _finish(cfg, state, policy, logger)
                return {"ok": False, "reason": state.get("reason", ""), "checkpoint": str(cfg.run_dir / "latest.pt")}
            state["skill_index"] += 1
            state["size_index"] = 0
            state["focus_window"] = []
            state["episodes_this_size"] = 0
            _save_state(state_path, state)
    finally:
        for env in envs.values():
            try:
                env.close()
            except Exception:
                pass
    state["ok"] = True
    state["reason"] = ""
    _finish(cfg, state, policy, logger)
    print(f"[sandboxes] finished. latest={cfg.run_dir / 'latest.pt'}")
    return {"ok": True, "reason": "", "checkpoint": str(cfg.run_dir / "latest.pt"), "summary": state["summary"]}


def _run_skill(cfg, state, skill, policy, envs, rng, device, logger):
    """Train every size of one sandbox. Returns (cleared, policy)."""
    while state["size_index"] < len(cfg.sizes):
        size = cfg.sizes[state["size_index"]]
        window: Deque[float] = deque(state.get("focus_window", []), maxlen=cfg.window)
        episodes = int(state.get("episodes_this_size", 0))
        updates = int(state.get("updates_this_size", 0))
        cleared = False
        while not cleared:
            if episodes >= cfg.max_episodes_per_size and updates >= cfg.min_updates:
                break
            batch_eps, metrics = _one_update(
                cfg, state, skill, size, policy, envs, rng, device, window
            )
            episodes += batch_eps
            updates += 1
            state["episodes_this_size"] = episodes
            state["updates_this_size"] = updates
            state["focus_window"] = list(window)
            rate = (sum(window) / len(window)) if window else 0.0
            record = {
                "skill": skill,
                "size": size,
                "update": updates,
                "episodes": episodes,
                "window_rate": rate,
                "window_n": len(window),
                **metrics,
            }
            _append_history(cfg, record)
            prior = " ".join(f"{k}={v:.2f}" for k, v in metrics.items() if k.startswith("retain_"))
            print(
                f"[{skill} {size}x{size} u{updates}] window={rate:.2f} "
                f"({len(window)}/{cfg.window}, gate {cfg.success_threshold:.2f}) "
                f"ent={metrics.get('entropy', 0):.3f} {prior}"
            )
            logger.log(
                {
                    "window_rate": rate,
                    "size": size,
                    "skill_index": state["skill_index"],
                    **{k: v for k, v in metrics.items() if isinstance(v, float)},
                }
            )
            save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"skill": skill, "size": size})
            if cfg.save_every and updates % cfg.save_every == 0:
                save_checkpoint(policy, cfg.run_dir / skill / f"update_{updates}.pt")
            _save_state(cfg.run_dir / "state.json", state)
            if size_advances(updates, window, cfg.window, cfg.success_threshold, cfg.min_updates):
                cleared = True
        rate = (sum(window) / len(window)) if window else 0.0
        gate_met = cleared
        if not gate_met and not cfg.allow_unmet:
            state["reason"] = (
                f"{skill} {size}x{size} stopped at window {rate:.2f} "
                f"< {cfg.success_threshold:.2f} after {episodes} episodes"
            )
            print(f"[sandboxes] STOP {state['reason']}")
            _record_size(state, skill, size, rate, gate_met=False)
            _write_summary(cfg, state)
            return False, policy
        _record_size(state, skill, size, rate, gate_met=gate_met)
        ckpt = cfg.run_dir / skill / f"size_{size}.pt"
        save_checkpoint(policy, ckpt, extra={"skill": skill, "size": size, "gate_met": gate_met})
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"skill": skill, "size": size})
        _write_summary(cfg, state)
        if not gate_met:
            print(f"[sandboxes] {skill} {size}x{size} missed the gate; allow_unmet advances")
        state["size_index"] += 1
        state["focus_window"] = []
        state["episodes_this_size"] = 0
        state["updates_this_size"] = 0
        _save_state(cfg.run_dir / "state.json", state)
    final = cfg.run_dir / skill / "final.pt"
    save_checkpoint(policy, final, extra={"skill": skill, "size": cfg.sizes[-1]})
    state["summary"].setdefault(skill, {})["cleared"] = all(
        state["summary"].get(skill, {}).get("sizes", {}).get(str(s), {}).get("gate_met", False)
        for s in cfg.sizes
    )
    _write_summary(cfg, state)
    return True, policy


def _one_update(cfg, state, focus_skill, focus_size, policy, envs, rng, device, window):
    earlier = _earlier(cfg, state)
    smaller = cfg.sizes[: state["size_index"]]
    buffer = RolloutBuffer()
    policy.eval()
    episodes = 0
    decisions = 0
    rehearsal_hits: Dict[str, List[float]] = {}
    while decisions < cfg.decisions_per_update and episodes < 64:
        skill, size, role = choose_episode(
            rng,
            focus_skill=focus_skill,
            focus_size=focus_size,
            smaller_sizes=smaller,
            earlier=earlier,
            rehearsal_frac=cfg.rehearsal_frac,
            size_mix_frac=cfg.size_mix_frac,
        )
        env = _env(cfg, envs, skill, size)
        seed = cfg.seed + int(state.get("episode_counter", 0))
        state["episode_counter"] = int(state.get("episode_counter", 0)) + 1
        _turns, max_steps = horizon(skill, size)
        env.reset(seed=seed)
        traj, info = collect_trajectory(
            policy,
            None,
            env,
            0,
            rng,
            max_steps=max_steps,
            use_env_reward=True,
        )
        if traj:
            compute_gae(traj, gamma=cfg.ppo.gamma, lam=cfg.ppo.lam)
            buffer.add(traj)
            decisions += len(traj)
        ok = 1.0 if _succeeded(skill, env) else 0.0
        if role == "focus":
            window.append(ok)
        elif role == "rehearsal":
            rehearsal_hits.setdefault(skill, []).append(ok)
        episodes += 1
    optimizer = state["_optimizer"]
    metrics = apply_ppo(policy, optimizer, buffer, cfg.ppo, device)
    for skill, hits in rehearsal_hits.items():
        metrics[f"retain_{skill}"] = sum(hits) / len(hits)
    if cfg.eval_episodes and int(state.get("updates_this_size", 0)) % max(1, cfg.eval_every) == 0:
        for skill, size in earlier:
            metrics[f"eval_{skill}"] = _eval_rate(policy, _env(cfg, envs, skill, size), cfg, size, skill)
    return episodes, metrics


def _eval_rate(policy, env, cfg, size, skill) -> float:
    _turns, max_steps = horizon(skill, size)
    hits = 0
    rng = random.Random(10_000 + size)
    policy.eval()
    for i in range(cfg.eval_episodes):
        env.reset(seed=20_000 + i)
        collect_trajectory(policy, None, env, 0, rng, max_steps=max_steps, use_env_reward=True)
        hits += 1 if _succeeded(skill, env) else 0
    return hits / max(1, cfg.eval_episodes)


def _earlier(cfg, state) -> List[Tuple[str, int]]:
    """Earlier sandboxes rehearsed at the largest size they actually cleared."""
    out = []
    for name in cfg.skills[: state["skill_index"]]:
        sizes_info = state["summary"].get(name, {}).get("sizes", {})
        cleared = [int(s) for s, rec in sizes_info.items() if rec.get("gate_met")]
        size = max(cleared) if cleared else cfg.sizes[0]
        out.append((name, size))
    return out


def _env(cfg, envs, skill, size):
    key = (skill, int(size))
    if key not in envs:
        path = cfg.run_dir / "maps" / f"{skill}_{size}.json"
        write_two_home_map(path, int(size), int(size))
        turns, _steps = horizon(skill, int(size))
        env = make_sandbox(skill, str(path), width=int(size), height=int(size), max_turns=turns)
        env.skill_index = skill_index(skill)
        envs[key] = env
    return envs[key]


def _load_or_new(cfg, state, device):
    latest = cfg.run_dir / "latest.pt"
    if state.get("updates_seen") and latest.exists():
        policy, _ = load_checkpoint(latest, device)
        print(f"[sandboxes] resume from {latest}")
    elif cfg.init_checkpoint and Path(cfg.init_checkpoint).exists() and state["skill_index"] == 0 and state["size_index"] == 0:
        policy, _ = load_checkpoint(cfg.init_checkpoint, device)
        print(f"[sandboxes] warm start {cfg.init_checkpoint}")
    else:
        probe = cfg.run_dir / "maps" / "probe.json"
        write_two_home_map(probe, 3, 3)
        env = make_sandbox(cfg.skills[0], str(probe), width=3, height=3, max_turns=4)
        env.reset(seed=0)
        from battleboats.envs.action_masks import ActionMasks

        dim = int(ActionMasks(env.engine, 0).tokens.shape[-1])
        policy = PolicyNetwork(
            token_dim=dim,
            d_model=cfg.arch.d_model,
            nhead=cfg.arch.nhead,
            num_layers=cfg.arch.num_layers,
            dim_feedforward=cfg.arch.dim_feedforward,
            dropout=cfg.arch.dropout,
            skill_count=len(SKILL_NAMES),
        ).to(device)
        env.close()
    state["_optimizer"] = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    return policy


def _record_size(state, skill, size, rate, *, gate_met: bool) -> None:
    block = state["summary"].setdefault(skill, {"sizes": {}, "cleared": False})
    block["sizes"][str(size)] = {"success_rate": rate, "gate_met": gate_met}


def _write_summary(cfg, state) -> None:
    payload = {
        "ok": bool(state.get("ok", False)),
        "reason": state.get("reason", ""),
        "checkpoint": str(cfg.run_dir / "latest.pt"),
        "skills": state["summary"],
        "order": cfg.skills,
        "sizes": cfg.sizes,
    }
    (cfg.run_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def _append_history(cfg, record) -> None:
    path = cfg.run_dir / "history.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _finish(cfg, state, policy, logger) -> None:
    if policy is not None:
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"stage": "final"})
    _write_summary(cfg, state)
    logger.finish()


def _fresh_state() -> Dict:
    return {
        "skill_index": 0,
        "size_index": 0,
        "focus_window": [],
        "episodes_this_size": 0,
        "updates_this_size": 0,
        "episode_counter": 0,
        "summary": {},
        "ok": False,
        "reason": "",
        "updates_seen": False,
    }


def _load_state(path: Path) -> Dict:
    data = json.loads(path.read_text())
    base = _fresh_state()
    base.update(data)
    base["updates_seen"] = True
    return base


def _save_state(path: Path, state: Dict) -> None:
    skip = {"_optimizer"}
    path.write_text(json.dumps({k: v for k, v in state.items() if k not in skip}, indent=2))
