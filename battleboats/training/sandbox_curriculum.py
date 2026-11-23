"""Size curriculum over sandboxes, one shared policy, no skill input.

Every sandbox is in the batch together. An update gives each sandbox the
same decision budget, then a size advances only when every sandbox's own
gate currently clears (a fresh check). Merchant, builder, landing, landing_build,
scouting, and each ``build_<type>`` drill clear on a live success window:
on-size episodes keep appending after a pass, and a drop below the
threshold reopens the drill. Combat clears on a plateau of smoothed
reward and value loss, and a combat pass stays passed for that size.
A fraction of each sandbox's episodes are smaller boards it has already
cleared (rehearsal and size mix). Rewards are divided by that sandbox's
running return scale so a drill with a larger return cannot own the
gradient.

The network is the full-game pointer net (``skill_count=0``). Ship type
comes from the ordinary observation tokens, not a task id. ``latest.pt``
loads in self-play with no shape change.

Checkpoints:
    ``<run_dir>/<skill>/size_<N>.pt`` when that size clears
    ``<run_dir>/<skill>/final.pt`` when every size has been recorded
    ``<run_dir>/latest.pt`` every update
    ``<run_dir>/summary.json`` rates per sandbox and size
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from battleboats.training.hull_explore import (
    HullExplore,
    empty_explore_stats,
    format_explore,
    merge_explore_stats,
    set_policy_explore,
)
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


def choose_mixed_episode(
    rng: random.Random,
    *,
    skills: Sequence[str],
    focus_size: int,
    smaller_sizes: Sequence[int],
    rehearsal_frac: float,
    size_mix_frac: float,
    skill: Optional[str] = None,
) -> Tuple[str, int, Role]:
    """One episode of one sandbox.

    The caller passes ``skill`` in round-robin order so every sandbox gets
    the same number of draws. ``rehearsal_frac`` and ``size_mix_frac`` pull
    a smaller board that size has already cleared. With no smaller board the
    episode stays on ``focus_size``.
    """
    if skill is None:
        if not skills:
            raise ValueError("skills is empty")
        skill = skills[rng.randrange(len(skills))]
    if smaller_sizes and rng.random() < rehearsal_frac:
        return skill, int(smaller_sizes[rng.randrange(len(smaller_sizes))]), "rehearsal"
    if smaller_sizes and rng.random() < size_mix_frac:
        return skill, int(smaller_sizes[rng.randrange(len(smaller_sizes))]), "mix"
    return skill, int(focus_size), "focus"


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    total = 0.0
    for reward in reversed(rewards):
        total = float(reward) + gamma * total
    return total


def return_scale(mean: float, m2: float, count: int) -> float:
    """Divisor for one sandbox's rewards.

    Welford mean and second moment of that sandbox's discounted returns.
    The divisor is ``max(1, |mean|, std)``. Returns at or below 1 (the sparse
    sandbox rewards) are left alone, and a sandbox whose returns are larger
    is scaled down onto that range. Fewer than two samples stays at 1 so the
    first episode does not set the scale.
    """
    if count < 2:
        return 1.0
    variance = m2 / count
    std = math.sqrt(variance) if variance > 0.0 else 0.0
    return max(1.0, abs(mean), std)


def observe_return(stats: Dict[str, float], raw_return: float) -> float:
    """Fold ``raw_return`` into ``stats`` and return the scale to apply."""
    count = int(stats.get("count", 0)) + 1
    mean = float(stats.get("mean", 0.0))
    m2 = float(stats.get("m2", 0.0))
    delta = raw_return - mean
    mean += delta / count
    m2 += delta * (raw_return - mean)
    stats["count"] = count
    stats["mean"] = mean
    stats["m2"] = m2
    return return_scale(mean, m2, count)


def scale_trajectory(traj, stats: Dict[str, float], gamma: float) -> float:
    """Divide this episode's rewards by the sandbox's running return scale.

    The stats see the raw discounted return. Returns that raw return.
    """
    raw = discounted_return([tr.reward for tr in traj], gamma)
    scale = observe_return(stats, raw)
    if scale != 1.0:
        for tr in traj:
            tr.reward = tr.reward / scale
    return raw


def window_cleared(window: Sequence[float], need: int, threshold: float) -> bool:
    if len(window) < need:
        return False
    return (sum(window) / len(window)) >= threshold


def size_advances(updates: int, window: Sequence[float], need: int, threshold: float, min_updates: int) -> bool:
    """A size clears only after enough updates and a full successful window."""
    return updates >= min_updates and window_cleared(window, need, threshold)


def _is_combat(skill: str) -> bool:
    return skill.startswith("combat_")


def smooth_series(values: Sequence[float], *, method: str, span: int) -> List[float]:
    """Trailing mean or EMA, one output per input.

    ``span`` is the trailing window for ``mean`` and the EMA span
    (``alpha = 2 / (span + 1)``). The first outputs use a shorter window
    so the series stays aligned with ``values``.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    if method not in ("mean", "ema"):
        raise ValueError(f"unknown smooth method {method!r}")
    data = [float(v) for v in values]
    if method == "ema":
        alpha = 2.0 / (float(span) + 1.0)
        out: List[float] = []
        acc = 0.0
        for i, value in enumerate(data):
            acc = value if i == 0 else (1.0 - alpha) * acc + alpha * value
            out.append(acc)
        return out
    out = []
    acc = 0.0
    for i, value in enumerate(data):
        acc += value
        if i >= span:
            acc -= data[i - span]
        out.append(acc / float(min(i + 1, span)))
    return out


def full_smooth(values: Sequence[float], *, method: str, span: int) -> List[float]:
    """Smoothed points whose window is full. Empty until ``span`` samples exist."""
    if len(values) < span:
        return []
    return smooth_series(values, method=method, span=span)[span - 1 :]


def plateaued(history: Sequence[float], n: int, threshold: float, *, higher_is_better: bool) -> bool:
    """True when ``history`` has not improved by ``threshold`` over the last ``n`` points.

    Improvement is ``(newer - older) / max(|older|, 1e-6)`` when higher is
    better, and the opposite for a loss. A short history is not a plateau.
    Callers pass a moving average, not the raw per-update series.
    """
    if n < 1 or len(history) < n:
        return False
    older = float(history[-n])
    newer = float(history[-1])
    base = max(abs(older), 1e-6)
    gain = (newer - older) / base if higher_is_better else (older - newer) / base
    return gain < threshold


def relative_gain(history: Sequence[float], n: int, *, higher_is_better: bool) -> float:
    """Signed improvement over the last ``n`` points. 1.0 until that span exists."""
    if n < 1 or len(history) < n:
        return 1.0
    older = float(history[-n])
    newer = float(history[-1])
    base = max(abs(older), 1e-6)
    return (newer - older) / base if higher_is_better else (older - newer) / base


def drill_done(
    skill: str,
    *,
    updates: int,
    window: Sequence[float],
    episodes: int,
    reward_trace: Sequence[float],
    value_trace: Sequence[float],
    success_threshold: float,
    need: int,
    min_updates: int,
    plateau_updates: int,
    plateau_threshold: float,
    combat_min_episodes: int,
    plateau_smooth: str = "mean",
    plateau_smooth_span: int = 8,
    already_passed: bool = False,
) -> bool:
    """Goal drills use the success window. Combat drills use a smoothed plateau.

    Merchant, builder, landing, landing_build, scouting, and ``build_<type>`` need ``need``
    on-size episodes at ``success_threshold``. Scouting's success bit is
    detection. A build drill's bit is buying the cued hull, not the speed reward.

    Combat compares moving averages of the per-update reward and of value
    loss. Each average has to have improved by less than ``plateau_threshold``
    across the last ``plateau_updates`` smoothed points, after
    ``combat_min_episodes``. The raw per-update means are not compared: a
    swing of a few tenths would keep a 2% test from ever firing.
    ``already_passed`` keeps a combat drill passed for the rest of this
    size. Wins are not an input.
    """
    if _is_combat(skill):
        if already_passed:
            return True
        if episodes < combat_min_episodes:
            return False
        reward_s = full_smooth(reward_trace, method=plateau_smooth, span=plateau_smooth_span)
        value_s = full_smooth(value_trace, method=plateau_smooth, span=plateau_smooth_span)
        return plateaued(
            reward_s, plateau_updates, plateau_threshold, higher_is_better=True
        ) and plateaued(
            value_s, plateau_updates, plateau_threshold, higher_is_better=False
        )
    return size_advances(updates, window, need, success_threshold, min_updates)


def _succeeded(skill: str, env) -> bool:
    if skill == "merchant":
        return bool(env.hauled)
    if skill == "builder":
        return bool(env.built)
    if skill in ("landing", "landing_build"):
        return bool(env.captured)
    if skill == "scouting":
        return bool(env.found)
    if skill.startswith("build_"):
        return bool(getattr(env, "built_target", False))
    return False


@dataclass(frozen=True)
class DecaySpec:
    """Linear per-turn reward decay. ``horizon`` 0 uses the episode length."""

    horizon: int = 0
    floor: float = 0.0


def _check_decay(spec: DecaySpec, where: str) -> None:
    if spec.horizon < 0:
        raise ValueError(f"{where} horizon must be >= 0")
    if not 0.0 <= spec.floor < 1.0:
        raise ValueError(f"{where} floor must be in [0, 1)")


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
    plateau_threshold: float = 0.02
    plateau_updates: int = 10
    plateau_smooth: str = "mean"
    plateau_smooth_span: int = 8
    combat_min_episodes: int = 32
    max_updates_per_size: int = 150
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
    reward_decay: DecaySpec = field(default_factory=DecaySpec)
    reward_decay_drills: Dict[str, DecaySpec] = field(default_factory=dict)
    arch: ArchSpec = field(default_factory=ArchSpec)
    ppo: PPOHyper = field(default_factory=PPOHyper)
    init_checkpoint: Optional[Path] = None
    # landing_build drill: "all" hulls buyable, or "landing" only.
    landing_build_hulls: str = "all"
    # Epsilon-uniform hull on training build decisions (hull_explore.py).
    # Never applied to eval episodes. eps 0 = off.
    hull_explore: HullExplore = field(default_factory=HullExplore)

    def decay_for(self, skill: str) -> DecaySpec:
        return self.reward_decay_drills.get(skill, self.reward_decay)

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
        if self.plateau_smooth not in ("mean", "ema"):
            raise ValueError("plateau_smooth must be 'mean' or 'ema'")
        if self.plateau_smooth_span < 1:
            raise ValueError("plateau_smooth_span must be >= 1")
        if self.plateau_updates < 1:
            raise ValueError("plateau_updates must be >= 1")
        if self.max_updates_per_size < 1:
            raise ValueError("max_updates_per_size must be >= 1")
        if self.landing_build_hulls not in ("all", "landing"):
            raise ValueError("landing_build_hulls must be 'all' or 'landing'")
        _check_decay(self.reward_decay, "reward_decay")
        for name, spec in self.reward_decay_drills.items():
            skill_index(name)
            _check_decay(spec, f"reward_decay.drills.{name}")


def load_sandbox_config(path: Path) -> SandboxRunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    arch = ArchSpec(**raw.get("arch", {}))
    ppo = PPOHyper(**raw.get("ppo", {}))
    init = raw.get("init_checkpoint") or None
    decay_raw = raw.get("reward_decay") or {}
    drills_raw = decay_raw.get("drills") or {}
    global_decay = DecaySpec(
        horizon=int(decay_raw.get("horizon", 0)),
        floor=float(decay_raw.get("floor", 0.0)),
    )
    drill_decay = {
        str(name): DecaySpec(
            horizon=int((spec or {}).get("horizon", global_decay.horizon)),
            floor=float((spec or {}).get("floor", global_decay.floor)),
        )
        for name, spec in drills_raw.items()
    }
    return SandboxRunConfig(
        run_dir=raw.get("run_dir", "runs/sandboxes"),
        skills=list(raw.get("skills", SKILL_NAMES)),
        sizes=list(raw.get("sizes", [3, 4, 5, 6, 8, 10])),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 0)),
        success_threshold=float(raw.get("success_threshold", 0.8)),
        window=int(raw.get("window", 64)),
        min_updates=int(raw.get("min_updates", 3)),
        plateau_threshold=float(raw.get("plateau_threshold", 0.02)),
        plateau_updates=int(raw.get("plateau_updates", 10)),
        plateau_smooth=str(raw.get("plateau_smooth", "mean")),
        plateau_smooth_span=int(raw.get("plateau_smooth_span", 8)),
        combat_min_episodes=int(raw.get("combat_min_episodes", 32)),
        max_updates_per_size=int(raw.get("max_updates_per_size", 150)),
        max_episodes_per_size=int(raw.get("max_episodes_per_size", 400)),
        decisions_per_update=int(raw.get("decisions_per_update", 512)),
        rehearsal_frac=float(raw.get("rehearsal_frac", 0.15)),
        size_mix_frac=float(raw.get("size_mix_frac", 0.2)),
        eval_every=int(raw.get("eval_every", 5)),
        eval_episodes=int(raw.get("eval_episodes", 8)),
        allow_unmet=bool(raw.get("allow_unmet", False)),
        landing_build_hulls=str(raw.get("landing_build_hulls", "all")),
        hull_explore=HullExplore.from_dict(raw.get("hull_explore")),
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
        wandb_project=str(raw.get("wandb_project", "battleboats-skill-pipeline")),
        reward_decay=global_decay,
        reward_decay_drills=drill_decay,
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
    fresh = not (resume and state_path.exists())
    state = _fresh_state() if fresh else _load_state(state_path)
    if fresh:
        history = cfg.run_dir / "history.jsonl"
        if history.exists():
            history.unlink()
    _ensure_runtime_state(state, cfg)
    logger = init_wandb(
        cfg.wandb,
        project=cfg.wandb_project,
        config={
            "run_dir": str(cfg.run_dir),
            "skills": cfg.skills,
            "sizes": cfg.sizes,
            "skill_count": 0,
        },
    )
    policy = _load_or_new(cfg, state, device)
    if int(getattr(policy, "skill_count", 0)) != 0:
        raise RuntimeError(
            f"sandbox warm-up requires skill_count=0, loaded {policy.skill_count}"
        )
    envs: Dict[Tuple[str, int], object] = {}
    print(
        f"[sandboxes] device={device} skills={cfg.skills} sizes={cfg.sizes} "
        f"mixed size_index={state['size_index']} skill_count=0 "
        f"wandb={'on' if logger.enabled else 'off'}"
    )
    if cfg.hull_explore.enabled:
        print(f"[sandboxes] hull exploration ON {cfg.hull_explore.describe()} (training episodes only)")
    try:
        while state["size_index"] < len(cfg.sizes):
            done, policy = _run_size(cfg, state, policy, envs, rng, device, logger)
            if not done:
                state["ok"] = False
                _finish(cfg, state, policy, logger)
                return {"ok": False, "reason": state.get("reason", ""), "checkpoint": str(cfg.run_dir / "latest.pt")}
            state["size_index"] += 1
            _reset_size_progress(state, cfg)
            _save_state(state_path, state)
        for skill in cfg.skills:
            save_checkpoint(
                policy,
                cfg.run_dir / skill / "final.pt",
                extra={"skill": skill, "size": cfg.sizes[-1], "skill_count": 0},
            )
            sizes_info = state["summary"].get(skill, {}).get("sizes", {})
            state["summary"].setdefault(skill, {"sizes": sizes_info, "cleared": False})
            state["summary"][skill]["cleared"] = all(
                sizes_info.get(str(s), {}).get("gate_met", False) for s in cfg.sizes
            )
        _write_summary(cfg, state)
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


def _run_size(cfg, state, policy, envs, rng, device, logger):
    """Train every sandbox at the current board size. Returns (cleared, policy)."""
    size = cfg.sizes[state["size_index"]]
    state["size_capped"] = False
    state["cap_kind"] = ""
    state["capped_skills"] = []
    windows = {
        skill: deque(state["windows"].get(skill, []), maxlen=cfg.window) for skill in cfg.skills
    }
    reward_windows = {
        skill: deque(state["reward_windows"].get(skill, []), maxlen=cfg.window) for skill in cfg.skills
    }
    while not _size_cleared(cfg, state, windows):
        updates = int(state.get("updates_this_size", 0))
        if updates >= cfg.max_updates_per_size:
            state["size_capped"] = True
            state["cap_kind"] = "updates"
            logger.log({"size": size, "update": updates, "size_cap": 1.0})
            if cfg.allow_unmet:
                print(
                    f"[sandboxes] WARNING {size}x{size} hit max_updates_per_size="
                    f"{cfg.max_updates_per_size}; advancing"
                )
            break
        capped = _capped_unmet_skills(cfg, state, windows)
        if updates >= cfg.min_updates and capped:
            state["size_capped"] = True
            state["cap_kind"] = "episodes"
            state["capped_skills"] = capped
            logger.log({"size": size, "update": updates, "episode_cap": 1.0})
            if cfg.allow_unmet:
                print(
                    f"[sandboxes] WARNING {size}x{size} hit max_episodes_per_size="
                    f"{cfg.max_episodes_per_size}; advancing"
                )
            break
        metrics = _one_mixed_update(
            cfg, state, size, policy, envs, rng, device, windows, reward_windows
        )
        updates += 1
        state["updates_this_size"] = updates
        state["windows"] = {skill: list(window) for skill, window in windows.items()}
        state["reward_windows"] = {
            skill: list(window) for skill, window in reward_windows.items()
        }
        for skill in cfg.skills:
            # Raw per-update means. The combat gate smooths these before the
            # plateau test; it does not compare the unsmoothed points.
            state["reward_trace"].setdefault(skill, []).append(
                float(metrics.get(f"{skill}/mean_reward", 0.0))
            )
            state["value_trace"].setdefault(skill, []).append(
                float(metrics.get(f"{skill}/value_loss", 0.0))
            )
        parts = []
        log_row = {"size": size, "update": updates, "entropy": metrics.get("entropy", 0.0)}
        for skill in cfg.skills:
            rate = metrics.get(f"{skill}/success_rate", 0.0)
            reward = metrics.get(f"{skill}/mean_reward", 0.0)
            done = _drill_done(cfg, state, skill, windows[skill])
            metrics[f"{skill}/gate"] = 1.0 if done else 0.0
            parts.append(
                f"{skill} {'done' if done else 'open'} {rate:.2f}/{reward:.2f}"
            )
            log_row[f"{skill}/success_rate"] = rate
            log_row[f"{skill}/mean_reward"] = reward
            log_row[f"{skill}/window_reward"] = metrics.get(f"{skill}/window_reward", 0.0)
            log_row[f"{skill}/reward_smooth"] = metrics.get(f"{skill}/reward_smooth", reward)
            log_row[f"{skill}/plateau_gain"] = metrics.get(f"{skill}/plateau_gain", 1.0)
            log_row[f"{skill}/value_loss"] = metrics.get(f"{skill}/value_loss", 0.0)
            log_row[f"{skill}/value_smooth"] = metrics.get(f"{skill}/value_smooth", 0.0)
            log_row[f"{skill}/value_plateau_gain"] = metrics.get(f"{skill}/value_plateau_gain", 1.0)
            log_row[f"{skill}/gate"] = metrics[f"{skill}/gate"]
            log_row[f"{skill}/return_scale"] = metrics.get(f"{skill}/return_scale", 1.0)
        metrics.pop("_explore_line", None)
        record = {"size": size, "update": updates, **metrics}
        _append_history(cfg, record)
        print(
            f"[size {size}x{size} u{updates}] "
            + " | ".join(parts)
            + f" ent={metrics.get('entropy', 0):.3f}"
            + (f" {metrics['_explore_line']}" if metrics.get("_explore_line") else "")
        )
        logger.log(log_row, step=int(state.get("updates_total", 0)) + updates)
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"size": size, "skill_count": 0})
        if cfg.save_every and updates % cfg.save_every == 0:
            save_checkpoint(policy, cfg.run_dir / f"update_{updates}.pt", extra={"size": size})
        _save_state(cfg.run_dir / "state.json", state)
    unmet = []
    forced = bool(state.get("size_capped"))
    for skill in cfg.skills:
        window = windows[skill]
        rate = (sum(window) / len(window)) if window else 0.0
        gate_met = _drill_done(cfg, state, skill, window)
        if not gate_met:
            unmet.append((skill, rate, int(state["episodes"].get(skill, 0))))
        _record_size(state, skill, size, rate, gate_met=gate_met, capped=forced and not gate_met)
        save_checkpoint(
            policy,
            cfg.run_dir / skill / f"size_{size}.pt",
            extra={"skill": skill, "size": size, "gate_met": gate_met, "skill_count": 0},
        )
    save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"size": size, "skill_count": 0})
    _write_summary(cfg, state)
    if unmet and not cfg.allow_unmet:
        state["reason"] = format_stop_reason(
            cfg,
            size,
            unmet,
            cap_kind=str(state.get("cap_kind") or ""),
            capped_skills=list(state.get("capped_skills") or []),
        )
        print(f"[sandboxes] STOP {state['reason']}")
        return False, policy
    for skill, rate, _episodes in unmet:
        if not forced:
            print(f"[sandboxes] {skill} {size}x{size} missed the gate; allow_unmet advances")
    return True, policy


def _one_mixed_update(cfg, state, focus_size, policy, envs, rng, device, windows, reward_windows):
    """Equal decision budget per sandbox, rewards scaled per sandbox."""
    smaller = cfg.sizes[: state["size_index"]]
    budget = max(1, cfg.decisions_per_update // len(cfg.skills))
    buffer = RolloutBuffer()
    policy.eval()
    metrics: Dict[str, float] = {}
    explore_on = cfg.hull_explore.enabled
    # Anneal on a run-wide update counter (updates_this_size resets per size).
    explore_update = int(state.get("explore_updates", 0))
    state["explore_updates"] = explore_update + 1
    eps = cfg.hull_explore.eps_at(explore_update)
    explore_total = empty_explore_stats()
    for skill in cfg.skills:
        if explore_on:
            set_policy_explore(policy, eps, cfg.hull_explore.logprob)
            policy.hull_explore_stats = empty_explore_stats()
        decisions = 0
        episodes = 0
        raw_returns: List[float] = []
        focus_hits: List[float] = []
        stats = state["return_stats"].setdefault(skill, {"mean": 0.0, "m2": 0.0, "count": 0})
        window = windows[skill]
        reward_window = reward_windows[skill]
        squared = 0.0
        valued = 0
        while decisions < budget and episodes < 32:
            _name, size, role = choose_mixed_episode(
                rng,
                skills=cfg.skills,
                focus_size=focus_size,
                smaller_sizes=smaller,
                rehearsal_frac=cfg.rehearsal_frac,
                size_mix_frac=cfg.size_mix_frac,
                skill=skill,
            )
            env = _env(cfg, envs, skill, size)
            seed = cfg.seed + int(state.get("episode_counter", 0))
            state["episode_counter"] = int(state.get("episode_counter", 0)) + 1
            _turns, max_steps = horizon(skill, size)
            env.reset(seed=seed)
            traj, _info = collect_trajectory(
                policy,
                None,
                env,
                0,
                rng,
                max_steps=max_steps,
                use_env_reward=True,
            )
            raw = 0.0
            if traj:
                raw = scale_trajectory(traj, stats, cfg.ppo.gamma)
                compute_gae(traj, gamma=cfg.ppo.gamma, lam=cfg.ppo.lam)
                buffer.add(traj)
                decisions += len(traj)
            raw_returns.append(raw)
            if traj:
                for tr in traj:
                    squared += (tr.value - tr.ret) ** 2
                    valued += 1
            ok = 1.0 if _succeeded(skill, env) else 0.0
            state["episodes"][skill] = int(state["episodes"].get(skill, 0)) + 1
            # Goal-drill windows stay live after a pass so a collapse can
            # reopen the gate. Combat has no success window; its plateau
            # latch is separate (combat_passed).
            if role == "focus":
                reward_window.append(raw)
                if not _is_combat(skill):
                    window.append(ok)
                    focus_hits.append(ok)
            episodes += 1
            if episodes >= 1 and decisions >= budget:
                break
        rate = (sum(window) / len(window)) if window else 0.0
        window_reward = (sum(reward_window) / len(reward_window)) if reward_window else 0.0
        mean_reward = sum(raw_returns) / len(raw_returns) if raw_returns else 0.0
        value_loss = (squared / valued) if valued else 0.0
        reward_smooth, reward_gain = _smooth_gain(
            list(state["reward_trace"].get(skill, [])) + [mean_reward],
            cfg,
            higher_is_better=True,
        )
        value_smooth, value_gain = _smooth_gain(
            list(state["value_trace"].get(skill, [])) + [value_loss],
            cfg,
            higher_is_better=False,
        )
        metrics[f"{skill}/success_rate"] = rate
        metrics[f"{skill}/mean_reward"] = mean_reward
        metrics[f"{skill}/window_reward"] = window_reward
        metrics[f"{skill}/value_loss"] = value_loss
        metrics[f"{skill}/reward_smooth"] = reward_smooth
        metrics[f"{skill}/value_smooth"] = value_smooth
        metrics[f"{skill}/plateau_gain"] = reward_gain
        metrics[f"{skill}/value_plateau_gain"] = value_gain
        metrics[f"{skill}/return_scale"] = return_scale(
            float(stats.get("mean", 0.0)), float(stats.get("m2", 0.0)), int(stats.get("count", 0))
        )
        metrics[f"{skill}/episodes"] = float(state["episodes"].get(skill, 0))
        if focus_hits:
            metrics[f"{skill}/batch_success"] = sum(focus_hits) / len(focus_hits)
        if explore_on:
            st = policy.hull_explore_stats
            metrics[f"{skill}/hull_builds"] = float(st["build"])
            metrics[f"{skill}/hull_explored"] = float(st["explored"])
            metrics[f"{skill}/landing_explored"] = float(st["explored_hulls"].get("landing", 0))
            metrics[f"{skill}/landing_policy"] = float(st["policy_hulls"].get("landing", 0))
            merge_explore_stats(explore_total, st)
            policy.hull_explore_stats = None
            set_policy_explore(policy, 0.0, cfg.hull_explore.logprob)
    if cfg.eval_episodes and int(state.get("updates_this_size", 0)) % max(1, cfg.eval_every) == 0:
        for skill in cfg.skills:
            metrics[f"{skill}/eval_success"] = _eval_rate(
                policy, _env(cfg, envs, skill, focus_size), cfg, focus_size, skill
            )
    if explore_on:
        metrics["hull_explore/eps"] = eps
        metrics["hull_explore/builds"] = float(explore_total["build"])
        metrics["hull_explore/explored"] = float(explore_total["explored"])
        for src in ("explored_hulls", "policy_hulls"):
            for name, n in explore_total[src].items():
                metrics[f"hull_explore/{src}/{name}"] = float(n)
        metrics["_explore_line"] = format_explore(explore_total, eps)
    optimizer = state["_optimizer"]
    ppo_metrics = apply_ppo(policy, optimizer, buffer, cfg.ppo, device)
    metrics.update(ppo_metrics)
    return metrics


def _eval_rate(policy, env, cfg, size, skill) -> float:
    _turns, max_steps = horizon(skill, size)
    hits = 0
    rng = random.Random(10_000 + size)
    policy.eval()
    # Eval measures the policy itself: no hull exploration, not counted.
    saved = (getattr(policy, "hull_explore_eps", 0.0), getattr(policy, "hull_explore_stats", None))
    policy.hull_explore_eps, policy.hull_explore_stats = 0.0, None
    try:
        for i in range(cfg.eval_episodes):
            env.reset(seed=20_000 + i)
            collect_trajectory(policy, None, env, 0, rng, max_steps=max_steps, use_env_reward=True)
            hits += 1 if _succeeded(skill, env) else 0
    finally:
        policy.hull_explore_eps, policy.hull_explore_stats = saved
    return hits / max(1, cfg.eval_episodes)


def _env(cfg, envs, skill, size):
    key = (skill, int(size))
    if key not in envs:
        path = cfg.run_dir / "maps" / f"{skill}_{size}.json"
        write_two_home_map(path, int(size), int(size))
        turns, _steps = horizon(skill, int(size))
        decay = cfg.decay_for(skill)
        env = make_sandbox(
            skill,
            str(path),
            width=int(size),
            height=int(size),
            max_turns=turns,
            decay_horizon=decay.horizon,
            decay_floor=decay.floor,
            landing_build_hulls=cfg.landing_build_hulls,
        )
        # The pointer net has no skill embedding. Zero this so a task id is
        # not stored on the transition. Ship type still sits in the token.
        env.skill_index = 0
        envs[key] = env
    return envs[key]


def _load_or_new(cfg, state, device):
    latest = cfg.run_dir / "latest.pt"
    if state.get("updates_seen") and latest.exists():
        policy, _ = load_checkpoint(latest, device)
        print(f"[sandboxes] resume from {latest}")
    elif cfg.init_checkpoint and Path(cfg.init_checkpoint).exists() and state["size_index"] == 0 and int(state.get("updates_this_size", 0)) == 0:
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
            skill_count=0,
        ).to(device)
        env.close()
    state["_optimizer"] = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    return policy


def _record_size(state, skill, size, rate, *, gate_met: bool, capped: bool = False) -> None:
    block = state["summary"].setdefault(skill, {"sizes": {}, "cleared": False})
    block["sizes"][str(size)] = {"success_rate": rate, "gate_met": gate_met, "capped": capped}


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
        "size_index": 0,
        "updates_this_size": 0,
        "windows": {},
        "reward_windows": {},
        "reward_trace": {},
        "value_trace": {},
        "combat_passed": {},
        "episodes": {},
        "return_stats": {},
        "episode_counter": 0,
        "summary": {},
        "ok": False,
        "reason": "",
        "updates_seen": False,
    }


def _ensure_runtime_state(state: Dict, cfg) -> None:
    state.setdefault("size_index", 0)
    state.setdefault("updates_this_size", 0)
    state.setdefault("windows", {})
    state.setdefault("reward_windows", {})
    state.setdefault("reward_trace", {})
    state.setdefault("value_trace", {})
    state.setdefault("combat_passed", {})
    state.setdefault("episodes", {})
    state.setdefault("return_stats", {})
    state.setdefault("episode_counter", 0)
    state.setdefault("summary", {})
    for skill in cfg.skills:
        state["windows"].setdefault(skill, [])
        state["reward_windows"].setdefault(skill, [])
        state["reward_trace"].setdefault(skill, [])
        state["value_trace"].setdefault(skill, [])
        state["combat_passed"].setdefault(skill, False)
        state["episodes"].setdefault(skill, 0)
        state["return_stats"].setdefault(skill, {"mean": 0.0, "m2": 0.0, "count": 0})


def _reset_size_progress(state: Dict, cfg) -> None:
    state["updates_this_size"] = 0
    state["windows"] = {skill: [] for skill in cfg.skills}
    state["reward_windows"] = {skill: [] for skill in cfg.skills}
    state["reward_trace"] = {skill: [] for skill in cfg.skills}
    state["value_trace"] = {skill: [] for skill in cfg.skills}
    state["combat_passed"] = {skill: False for skill in cfg.skills}
    state["episodes"] = {skill: 0 for skill in cfg.skills}
    state["size_capped"] = False
    state["cap_kind"] = ""
    state["capped_skills"] = []


def _smooth_gain(series: Sequence[float], cfg, *, higher_is_better: bool) -> Tuple[float, float]:
    smoothed = full_smooth(series, method=cfg.plateau_smooth, span=cfg.plateau_smooth_span)
    latest = smoothed[-1] if smoothed else (float(series[-1]) if series else 0.0)
    gain = relative_gain(smoothed, cfg.plateau_updates, higher_is_better=higher_is_better)
    return latest, gain


def _drill_done(cfg, state, skill: str, window) -> bool:
    passed = bool(state.get("combat_passed", {}).get(skill, False))
    done = drill_done(
        skill,
        updates=int(state.get("updates_this_size", 0)),
        window=window,
        episodes=int(state["episodes"].get(skill, 0)),
        reward_trace=state["reward_trace"].get(skill, []),
        value_trace=state["value_trace"].get(skill, []),
        success_threshold=cfg.success_threshold,
        need=cfg.window,
        min_updates=cfg.min_updates,
        plateau_updates=cfg.plateau_updates,
        plateau_threshold=cfg.plateau_threshold,
        combat_min_episodes=cfg.combat_min_episodes,
        plateau_smooth=cfg.plateau_smooth,
        plateau_smooth_span=cfg.plateau_smooth_span,
        already_passed=passed,
    )
    if done and _is_combat(skill):
        state.setdefault("combat_passed", {})[skill] = True
    return done


def _size_cleared(cfg, state, windows) -> bool:
    return all(_drill_done(cfg, state, skill, windows[skill]) for skill in cfg.skills)


def _window_rate(window) -> float:
    return (sum(window) / len(window)) if window else 0.0


def _capped_unmet_skills(cfg, state, windows):
    """Unmet drills that have reached ``max_episodes_per_size``."""
    hit = []
    for skill in cfg.skills:
        window = windows[skill]
        if _drill_done(cfg, state, skill, window):
            continue
        episodes = int(state["episodes"].get(skill, 0))
        if episodes >= cfg.max_episodes_per_size:
            hit.append((skill, _window_rate(window), episodes))
    return hit


def format_stop_reason(cfg, size, unmet, cap_kind: str = "", capped_skills=None) -> str:
    """Name the skill that hit the cap, not just the first unmet drill.

    On an episode cap this is every unmet skill at ``max_episodes_per_size``.
    On an update cap it is every skill that still fails its gate.
    """
    if cap_kind == "episodes" and capped_skills:
        rows = list(capped_skills)
        cap_name, cap_val = "max_episodes_per_size", cfg.max_episodes_per_size
    else:
        rows = list(unmet)
        cap_name = "max_updates_per_size" if cap_kind == "updates" else (
            "max_episodes_per_size" if cap_kind == "episodes" else "cap"
        )
        cap_val = (
            cfg.max_episodes_per_size
            if cap_kind == "episodes"
            else cfg.max_updates_per_size
        )
    if not rows:
        return f"{size}x{size} stopped (hit {cap_name}={cap_val})"
    parts = []
    for i, (skill, rate, episodes) in enumerate(rows):
        if i == 0:
            parts.append(
                f"{skill} {size}x{size} stopped at window {rate:.2f} "
                f"< {cfg.success_threshold:.2f} after {episodes} episodes"
            )
        else:
            parts.append(f"{skill} window {rate:.2f} after {episodes} episodes")
    return "; ".join(parts) + f" (hit {cap_name}={cap_val})"


def _unclear_and_capped(cfg, state, windows) -> bool:
    """True when an unmet drill has hit the per-size episode cap."""
    return bool(_capped_unmet_skills(cfg, state, windows))


def _load_state(path: Path) -> Dict:
    data = json.loads(path.read_text())
    base = _fresh_state()
    base.update(data)
    base["updates_seen"] = True
    return base


def _save_state(path: Path, state: Dict) -> None:
    skip = {"_optimizer"}
    path.write_text(json.dumps({k: v for k, v in state.items() if k not in skip}, indent=2))
