"""One shared policy: interleaved skill sandboxes, then the full game.

The sequential curriculum trains a skill and then overwrites it with the next
one. This pipeline keeps every mechanic in the same update by pooling equal
amounts of data from each sandbox and normalizing advantages per task, so a
dense merchant return cannot drown a single capture. The full-game stage
keeps mixing those sandboxes in, and is scored against a random opponent and
the scripted baseline.

The network, observation, and action heads are the full-game ones. Sandbox
verb filters only change the mask.

Checkpoints:
    ``<run_dir>/latest.pt``  updated every update (resume from here)
    ``<run_dir>/final.pt``   written when the run ends; this is the policy to load
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from battleboats.agents.scripted import scripted_action
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.factory import make_env
from battleboats.envs.skill_maps import write_skill_map
from battleboats.training.checkpoints import load_checkpoint, save_checkpoint
from battleboats.training.curriculum import SKILLS, ArchSpec, PPOHyper, episode_success
from battleboats.training.device import resolve_device
from battleboats.training.loop import apply_ppo, collect_episodes
from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.rollout import Transition
from battleboats.training.wandb_log import init_wandb
from battleboats.training.manifest import ManifestSession

SANDBOX_SKILLS = tuple(s for s in SKILLS if s != "full")
FINAL_NAME = "final.pt"


@dataclass
class TaskSpec:
    skill: str
    width: int
    height: int
    max_turns: int = 30
    max_steps: int = 120
    randomize: bool = True
    approach_coef: float = 0.0
    success_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.skill not in SANDBOX_SKILLS:
            raise ValueError(f"unknown sandbox {self.skill!r}; expected one of {SANDBOX_SKILLS}")


@dataclass
class StageSpec:
    name: str
    kind: str
    max_updates: int
    tasks: List[TaskSpec]
    eval_every: int = 5
    eval_episodes: int = 8
    decisions_per_update: int = 1536
    width: int = 6
    height: int = 6
    max_turns: int = 40
    max_steps: int = 250
    map_count: int = 2
    skill_frac: float = 0.5
    scripted_frac: float = 0.0
    win_threshold: float = 0.5
    scripted_threshold: float = 0.0
    retain_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("interleave", "full_mix"):
            raise ValueError(f"unknown stage kind {self.kind!r}")
        if not self.tasks:
            raise ValueError(f"stage {self.name!r} has no tasks")
        if not 0.0 <= self.skill_frac <= 1.0:
            raise ValueError("skill_frac must be in [0, 1]")
        if not 0.0 <= self.scripted_frac <= 1.0:
            raise ValueError("scripted_frac must be in [0, 1]")


@dataclass
class PipelineConfig:
    run_dir: Path
    device: str = "cpu"
    seed: int = 0
    arch: ArchSpec = field(default_factory=ArchSpec)
    ppo: PPOHyper = field(default_factory=PPOHyper)
    stages: List[StageSpec] = field(default_factory=list)
    init_checkpoint: Optional[Path] = None
    allow_unmet: bool = False
    save_every: int = 10
    wandb: bool = False
    wandb_project: str = "battleboats-skill-pipeline"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.init_checkpoint is not None:
            self.init_checkpoint = Path(self.init_checkpoint)


def final_checkpoint_path(cfg: PipelineConfig) -> Path:
    return cfg.run_dir / FINAL_NAME


def load_pipeline_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    arch = ArchSpec(**raw.get("arch", {}))
    ppo = PPOHyper(**raw.get("ppo", {}))
    stages = []
    for spec in raw["stages"]:
        tasks = [TaskSpec(**t) for t in spec.pop("tasks")]
        stages.append(StageSpec(tasks=tasks, **spec))
    init = raw.get("init_checkpoint") or None
    return PipelineConfig(
        run_dir=raw.get("run_dir", "runs/skill_pipeline"),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 0)),
        arch=arch,
        ppo=ppo,
        stages=stages,
        init_checkpoint=Path(init) if init else None,
        allow_unmet=bool(raw.get("allow_unmet", False)),
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
        wandb_project=str(raw.get("wandb_project", "battleboats-skill-pipeline")),
    )


def normalize_advantages(transitions: List[Transition]) -> None:
    """Zero-mean, unit-variance advantages, in place. Too-small batches are left raw."""
    n = len(transitions)
    if n < 2:
        return
    mean = sum(t.advantage for t in transitions) / n
    var = sum((t.advantage - mean) ** 2 for t in transitions) / n
    std = var ** 0.5
    for t in transitions:
        t.advantage = (t.advantage - mean) / (std + 1e-8)


def run_pipeline(cfg: PipelineConfig, *, resume: bool = True) -> Dict[str, Any]:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with ManifestSession(
        cfg.run_dir,
        kind="skill_pipeline",
        config=cfg,
        init_checkpoint=cfg.init_checkpoint,
    ) as session:
        result = _run_pipeline(cfg, resume=resume)
        final = cfg.run_dir / "final.pt"
        latest = cfg.run_dir / "latest.pt"
        ckpt = final if final.is_file() else (latest if latest.is_file() else None)
        session.complete(
            final_checkpoint=ckpt,
            updates_completed=None,
        )
        return result


def _run_pipeline(cfg: PipelineConfig, *, resume: bool = True) -> Dict[str, Any]:
    """Train every stage. Returns ``{"ok", "reason", "checkpoint", "summary"}``."""
    import random

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    state_path = cfg.run_dir / "state.json"
    state = _load_state(state_path) if resume and state_path.exists() else _fresh_state()
    history_path = cfg.run_dir / "history.jsonl"
    logger = init_wandb(
        cfg.wandb,
        project=cfg.wandb_project,
        config={"run_dir": str(cfg.run_dir), "stages": [s.name for s in cfg.stages]},
    )
    print(
        f"[skill_pipeline] device={device} stages={len(cfg.stages)} "
        f"resume_at={state['stage_index']} allow_unmet={cfg.allow_unmet}"
    )

    policy: Optional[PolicyNetwork] = None
    while state["stage_index"] < len(cfg.stages):
        idx = state["stage_index"]
        stage = cfg.stages[idx]
        outcome = _run_stage(cfg, stage, idx, state, device, rng, policy, history_path, logger)
        policy = outcome["policy"]
        if not outcome["promoted"]:
            state["ok"] = False
            state["reason"] = outcome["reason"]
            _save_state(state_path, state)
            _finish(cfg, policy, state, logger)
            return _result(cfg, state, ok=False)
        state["stage_index"] = idx + 1
        state["updates_done"] = 0
        state["episode_counter"] = 0
        state["completed"].append(outcome["record"])
        state["ok"] = True
        state["reason"] = ""
        _save_state(state_path, state)

    _finish(cfg, policy, state, logger)
    print(f"[skill_pipeline] finished. final={final_checkpoint_path(cfg)}")
    return _result(cfg, state, ok=True)


def _run_stage(cfg, stage, idx, state, device, rng, policy, history_path, logger):
    envs, maps = _open_stage(cfg, stage)
    stage_dir = cfg.run_dir / "stages" / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    in_progress = stage_dir / "latest.pt"

    if state["updates_done"] > 0 and in_progress.exists():
        policy, _ = load_checkpoint(in_progress, device)
        print(f"[skill_pipeline] resume {stage.name} from update {state['updates_done']}")
    elif policy is None:
        ckpt = cfg.init_checkpoint if idx == 0 else None
        if ckpt is not None and Path(ckpt).exists():
            policy, _ = load_checkpoint(ckpt, device)
            print(f"[skill_pipeline] warm start {stage.name} from {ckpt}")
        else:
            if ckpt is not None:
                print(f"[skill_pipeline] {ckpt} missing — random init")
            policy = _new_policy(cfg.arch, next(iter(envs.values())), device)
    else:
        print(f"[skill_pipeline] handoff into {stage.name}")

    assert policy is not None
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    ep = int(state["episode_counter"])
    start_update = int(state["updates_done"])
    promoted = False
    gate_met = False
    last_eval: Dict[str, float] = {}

    for update in range(start_update, stage.max_updates):
        pool = RolloutBuffer()
        train_rates: Dict[str, float] = {}
        for name, decisions, kind in _budgets(stage):
            if decisions <= 0:
                continue
            env, env_maps, max_steps, skill_name, use_env_reward, opponent = _bucket(
                stage, envs, maps, name, kind
            )
            transitions, infos, ep = _collect(
                policy,
                env,
                rng,
                decisions=decisions,
                hyper=cfg.ppo,
                max_steps=max_steps,
                maps=env_maps,
                seed0=cfg.seed + ep,
                episode_start=ep,
                skill=skill_name,
                use_env_reward=use_env_reward,
                opponent=opponent,
            )
            normalize_advantages(transitions)
            pool.add(transitions)
            train_rates[name] = _rate(infos)
        metrics = apply_ppo(policy, optimizer, pool, cfg.ppo, device)
        do_eval = stage.eval_episodes > 0 and (
            (update + 1) % max(1, stage.eval_every) == 0 or update + 1 == stage.max_updates
        )
        if do_eval:
            last_eval = _evaluate(policy, stage, envs, maps, cfg, update)
        shown = last_eval if do_eval else train_rates
        state["updates_done"] = update + 1
        state["episode_counter"] = ep
        record = {
            "stage": stage.name,
            "update": update + 1,
            "train": train_rates,
            "eval": last_eval if do_eval else {},
            **{f"ppo_{k}": v for k, v in metrics.items()},
        }
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        flat = " ".join(f"{k}={v:.2f}" for k, v in shown.items())
        print(
            f"[{stage.name} {update + 1}/{stage.max_updates}] {flat} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f}"
        )
        logger.log(_flat_log(record), step=update + idx * 10_000)
        save_checkpoint(policy, in_progress, extra={"stage": stage.name, "update": update + 1})
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"stage": stage.name, "update": update + 1})
        if cfg.save_every and (update + 1) % cfg.save_every == 0:
            save_checkpoint(policy, stage_dir / f"update_{update + 1}.pt")
        _save_state(cfg.run_dir / "state.json", state)
        if do_eval and _gate_ok(stage, last_eval):
            promoted = True
            gate_met = True
            break

    _close_envs(envs)
    if not promoted:
        if cfg.allow_unmet:
            promoted = True
            gate_met = False
            print(f"[skill_pipeline] {stage.name} missed the gate; allow_unmet hands the checkpoint on")
        else:
            reason = f"{stage.name} finished {stage.max_updates} updates below its gate ({last_eval})"
            print(f"[skill_pipeline] STOP {reason}")
            return {"promoted": False, "reason": reason, "policy": policy, "record": {}}

    return {
        "promoted": True,
        "reason": "",
        "policy": policy,
        "record": {
            "stage": stage.name,
            "kind": stage.kind,
            "gate_met": gate_met,
            "metrics": last_eval,
            "checkpoint": str(cfg.run_dir / "latest.pt"),
        },
    }


def _budgets(stage: StageSpec) -> List[Tuple[str, int, str]]:
    """(bucket name, decision count, kind). kind is skill | full_random | full_scripted."""
    total = max(1, int(stage.decisions_per_update))
    if stage.kind == "interleave":
        return _split(total, [(t.skill, "skill") for t in stage.tasks])
    skill_total = int(round(total * stage.skill_frac))
    full_total = total - skill_total
    scripted = int(round(full_total * stage.scripted_frac))
    random_n = full_total - scripted
    out = _split(skill_total, [(t.skill, "skill") for t in stage.tasks]) if skill_total else []
    if random_n:
        out.append(("win_random", random_n, "full_random"))
    if scripted:
        out.append(("win_scripted", scripted, "full_scripted"))
    return out


def _split(total: int, named: List[Tuple[str, str]]) -> List[Tuple[str, int, str]]:
    if not named or total <= 0:
        return []
    base, rem = divmod(total, len(named))
    out = []
    for i, (name, kind) in enumerate(named):
        n = base + (1 if i < rem else 0)
        if n:
            out.append((name, n, kind))
    return out


def _bucket(stage, envs, maps, name, kind):
    if kind == "skill":
        task = next(t for t in stage.tasks if t.skill == name)
        return envs[name], maps[name], task.max_steps, task.skill, True, "none"
    return envs["full"], maps["full"], stage.max_steps, "full", False, (
        "scripted" if kind == "full_scripted" else "random"
    )


def _collect(
    policy,
    env,
    rng,
    *,
    decisions,
    hyper,
    max_steps,
    maps,
    seed0,
    episode_start,
    skill,
    use_env_reward,
    opponent,
):
    def reset(episode_index: int) -> int:
        env.map_json_path = maps[episode_index % len(maps)]
        env.reset(seed=seed0 + episode_index)
        if skill == "full":
            return episode_index % 2
        return 0

    def opponent_for(_episode_index: int):
        if opponent == "scripted":
            return None, scripted_action
        return None, None

    def annotate(info, live):
        info = dict(info)
        info["success"] = episode_success(skill, info, live)
        return info

    buffer, infos, ep = collect_episodes(
        policy,
        env,
        rng,
        target_decisions=decisions,
        hyper=hyper,
        max_steps=max_steps,
        reset=reset,
        opponent_for=opponent_for,
        use_env_reward=use_env_reward,
        episode_start=episode_start,
        annotate=annotate,
    )
    return list(buffer.transitions), infos, ep


def _evaluate(policy, stage: StageSpec, envs, maps, cfg: PipelineConfig, update: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    seed_base = 1_000_000 + update * 1_000
    for task in stage.tasks:
        metrics[task.skill] = _eval_one(
            policy,
            envs[task.skill],
            maps[task.skill],
            episodes=stage.eval_episodes,
            hyper=cfg.ppo,
            max_steps=task.max_steps,
            seed0=seed_base + _skill_salt(task.skill),
            skill=task.skill,
            opponent="none",
        )
    if stage.kind == "full_mix":
        metrics["win_random"] = _eval_one(
            policy,
            envs["full"],
            maps["full"],
            episodes=stage.eval_episodes,
            hyper=cfg.ppo,
            max_steps=stage.max_steps,
            seed0=seed_base + 800,
            skill="full",
            opponent="random",
        )
        metrics["win_scripted"] = _eval_one(
            policy,
            envs["full"],
            maps["full"],
            episodes=stage.eval_episodes,
            hyper=cfg.ppo,
            max_steps=stage.max_steps,
            seed0=seed_base + 900,
            skill="full",
            opponent="scripted",
        )
    return metrics


def _eval_one(policy, env, maps, *, episodes, hyper, max_steps, seed0, skill, opponent) -> float:
    import random as _random

    infos = []
    eval_rng = _random.Random(seed0)
    for i in range(episodes):

        def reset(episode_index: int, i=i) -> int:
            env.map_json_path = maps[i % len(maps)]
            env.reset(seed=seed0 + i)
            if skill == "full":
                return i % 2
            return 0

        def opponent_for(_episode_index: int, opponent=opponent):
            if opponent == "scripted":
                return None, scripted_action
            return None, None

        def annotate(info, live, skill=skill):
            info = dict(info)
            info["success"] = episode_success(skill, info, live)
            return info

        _, batch, _ = collect_episodes(
            policy,
            env,
            eval_rng,
            target_decisions=1,
            hyper=hyper,
            max_steps=max_steps,
            reset=reset,
            opponent_for=opponent_for,
            use_env_reward=skill != "full",
            episode_start=0,
            annotate=annotate,
        )
        if batch:
            infos.append(batch[0])
    return _rate(infos)


def _gate_ok(stage: StageSpec, metrics: Dict[str, float]) -> bool:
    if stage.kind == "interleave":
        return all(metrics.get(t.skill, 0.0) >= t.success_threshold for t in stage.tasks)
    ok = metrics.get("win_random", 0.0) >= stage.win_threshold
    if stage.scripted_threshold > 0:
        ok = ok and metrics.get("win_scripted", 0.0) >= stage.scripted_threshold
    if stage.retain_threshold > 0:
        ok = ok and all(metrics.get(t.skill, 0.0) >= stage.retain_threshold for t in stage.tasks)
    return ok


def _rate(infos: List[dict]) -> float:
    if not infos:
        return 0.0
    return sum(1.0 if info.get("success") else 0.0 for info in infos) / len(infos)


def _open_stage(cfg: PipelineConfig, stage: StageSpec):
    map_dir = cfg.run_dir / "maps"
    envs = {}
    maps: Dict[str, List[str]] = {}
    for task in stage.tasks:
        path = write_skill_map(map_dir / f"{stage.name}_{task.skill}.json", task.skill, task.width, task.height)
        maps[task.skill] = [path]
        env = make_env(
            task.skill,
            path,
            max_turns=task.max_turns,
            randomize=task.randomize,
            approach_coef=task.approach_coef,
        )
        env.reset(seed=cfg.seed)
        envs[task.skill] = env
    if stage.kind == "full_mix":
        full_maps = []
        for i in range(max(1, stage.map_count)):
            full_maps.append(
                write_skill_map(
                    map_dir / f"{stage.name}_full_{i}.json",
                    "full",
                    stage.width,
                    stage.height,
                    seed=cfg.seed + i,
                )
            )
        full = make_env("full", full_maps[0], max_turns=stage.max_turns)
        full.reset(seed=cfg.seed)
        envs["full"] = full
        maps["full"] = full_maps
    return envs, maps


def _close_envs(envs) -> None:
    for env in envs.values():
        try:
            env.close()
        except Exception:
            pass


def _new_policy(arch: ArchSpec, env, device: torch.device) -> PolicyNetwork:
    masks = ActionMasks(env.engine, 0, device="cpu")
    return PolicyNetwork(
        token_dim=int(masks.tokens.shape[-1]),
        d_model=arch.d_model,
        nhead=arch.nhead,
        num_layers=arch.num_layers,
        dim_feedforward=arch.dim_feedforward,
        dropout=arch.dropout,
    ).to(device)


def _skill_salt(skill: str) -> int:
    return 50 * (SANDBOX_SKILLS.index(skill) + 1)


def _flat_log(record: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for prefix in ("train", "eval"):
        for key, value in record.get(prefix, {}).items():
            flat[f"{prefix}/{key}"] = float(value)
    for key, value in record.items():
        if key.startswith("ppo_"):
            flat[key] = float(value)
    return flat


def _finish(cfg: PipelineConfig, policy: Optional[PolicyNetwork], state: Dict[str, Any], logger) -> None:
    if policy is not None:
        save_checkpoint(
            policy,
            final_checkpoint_path(cfg),
            extra={"stage": "final", "ok": bool(state.get("ok"))},
        )
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"stage": "final"})
    summary = {
        "ok": bool(state.get("ok")),
        "reason": state.get("reason", ""),
        "checkpoint": str(final_checkpoint_path(cfg)),
        "completed": state.get("completed", []),
    }
    (cfg.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.finish()


def _result(cfg: PipelineConfig, state: Dict[str, Any], *, ok: bool) -> Dict[str, Any]:
    return {
        "ok": ok,
        "reason": state.get("reason", ""),
        "checkpoint": str(final_checkpoint_path(cfg)),
        "summary": state.get("completed", []),
    }


def _fresh_state() -> Dict[str, Any]:
    return {
        "stage_index": 0,
        "updates_done": 0,
        "episode_counter": 0,
        "completed": [],
        "ok": False,
        "reason": "",
    }


def _load_state(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    base = _fresh_state()
    base.update(data)
    return base


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2))
