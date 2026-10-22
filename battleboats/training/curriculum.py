"""Config-driven skill curriculum.

Stages are (skill, map size) pairs. A stage promotes when its metric clears
the threshold, or — if ``allow_unmet`` is set — when it exhausts its update
budget. The promoted checkpoint is the next stage's warm start. The last
stage is the full game.

Reward shaping stays inside the minigame envs (merchant throughput, optional
approach bonus on landing / combat / builder). The full-game stage uses the
sparse terminal ±1 from ``collect_trajectory``.

No league, ratings, or population ids. The checkpoint on disk is a plain
policy file a later quality-diversity run can load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.factory import make_env
from battleboats.envs.skill_maps import write_skill_map
from battleboats.training.checkpoints import frozen_copy, load_checkpoint, save_checkpoint
from battleboats.training.device import resolve_device
from battleboats.training.loop import PPOHyper, apply_ppo, collect_episodes
from battleboats.training.policy import PolicyNetwork

SKILLS = ("merchant", "landing", "combat", "builder", "full")


@dataclass
class ArchSpec:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


@dataclass
class StageSpec:
    skill: str
    width: int
    height: int
    success_threshold: float
    max_updates: int
    metric: str = "success_rate"
    eval_every: int = 5
    eval_episodes: int = 16
    max_turns: int = 40
    max_steps: int = 200
    rollout_decisions: int = 1024
    randomize: bool = False
    approach_coef: float = 0.0
    train_opponent: str = "none"  # none | random | snapshot. Skills ignore this.
    map_count: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        if self.skill not in SKILLS:
            raise ValueError(f"unknown skill {self.skill!r}; expected one of {SKILLS}")
        if not self.name:
            self.name = f"{self.skill}_{self.width}x{self.height}"
        if self.metric not in ("success_rate", "win_rate", "mean_return"):
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.train_opponent not in ("none", "random", "snapshot"):
            raise ValueError(f"unknown train_opponent {self.train_opponent!r}")


@dataclass
class CurriculumConfig:
    run_dir: Path
    device: str = "auto"
    seed: int = 0
    arch: ArchSpec = field(default_factory=ArchSpec)
    ppo: PPOHyper = field(default_factory=PPOHyper)
    stages: List[StageSpec] = field(default_factory=list)
    init_checkpoint: Optional[Path] = None
    allow_unmet: bool = False
    opponent_refresh: int = 5
    save_every: int = 10
    wandb: bool = False

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.init_checkpoint is not None:
            self.init_checkpoint = Path(self.init_checkpoint)


def load_curriculum_config(path: Path) -> CurriculumConfig:
    raw = yaml.safe_load(Path(path).read_text())
    arch = ArchSpec(**raw.get("arch", {}))
    ppo_raw = dict(raw.get("ppo", {}))
    opponent_refresh = int(ppo_raw.pop("opponent_refresh", raw.get("opponent_refresh", 5)))
    ppo = PPOHyper(**ppo_raw)
    stages = [StageSpec(**s) for s in raw["stages"]]
    init = raw.get("init_checkpoint") or None
    return CurriculumConfig(
        run_dir=raw.get("run_dir", "runs/curriculum"),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 0)),
        arch=arch,
        ppo=ppo,
        stages=stages,
        init_checkpoint=Path(init) if init else None,
        allow_unmet=bool(raw.get("allow_unmet", False)),
        opponent_refresh=opponent_refresh,
        save_every=int(raw.get("save_every", 10)),
        wandb=bool(raw.get("wandb", False)),
    )


def episode_success(skill: str, info: Dict[str, Any], env) -> bool:
    """Did this episode demonstrate the skill (or a win, for the full game)?"""
    if skill == "merchant":
        return float(info.get("reward_sum", 0.0)) > 0.0
    if skill == "landing":
        return info.get("winner") == 0 or bool(getattr(env, "_captured", False))
    if skill == "combat":
        return bool(getattr(env, "killed", False))
    if skill == "builder":
        return bool(getattr(env, "built", False))
    if skill == "full":
        return float(info.get("outcome", 0.0)) > 0.0
    raise ValueError(skill)


def _infer_token_dim(env) -> int:
    masks = ActionMasks(env.engine, 0, device="cpu")
    return int(masks.tokens.shape[-1])


def _new_policy(arch: ArchSpec, token_dim: int, device: torch.device) -> PolicyNetwork:
    return PolicyNetwork(
        token_dim=token_dim,
        d_model=arch.d_model,
        nhead=arch.nhead,
        num_layers=arch.num_layers,
        dim_feedforward=arch.dim_feedforward,
        dropout=arch.dropout,
    ).to(device)


def _stage_maps(cfg: CurriculumConfig, stage: StageSpec) -> List[str]:
    map_dir = cfg.run_dir / "maps"
    paths = []
    count = stage.map_count if stage.skill == "full" else 1
    for i in range(max(1, count)):
        path = map_dir / f"{stage.name}_{i}.json"
        paths.append(write_skill_map(path, stage.skill, stage.width, stage.height, seed=cfg.seed + i))
    return paths


def _metric_value(skill: str, infos: List[dict], env) -> Dict[str, float]:
    if not infos:
        return {"success_rate": 0.0, "win_rate": 0.0, "mean_return": 0.0, "episodes": 0.0}
    successes = []
    for info in infos:
        if "success" in info:
            successes.append(1.0 if info["success"] else 0.0)
        else:
            successes.append(1.0 if episode_success(skill, info, env) else 0.0)
    if skill == "full":
        returns = [float(info.get("outcome", 0.0)) for info in infos]
    else:
        returns = [float(info.get("reward_sum", 0.0)) for info in infos]
    rate = sum(successes) / len(successes)
    return {
        "success_rate": rate,
        "win_rate": rate if skill == "full" else rate,
        "mean_return": sum(returns) / len(returns),
        "episodes": float(len(infos)),
    }


def run_curriculum(cfg: CurriculumConfig, *, resume: bool = True) -> Dict[str, Any]:
    """Train every stage. Returns ``{"ok": bool, ...}``. Does not raise on a missed gate."""
    import random

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    state_path = cfg.run_dir / "state.json"
    state = _load_state(state_path) if resume and state_path.exists() else _fresh_state()
    history_path = cfg.run_dir / "history.jsonl"

    use_wandb = cfg.wandb
    if use_wandb:
        import wandb

        wandb.init(project="battleboats-curriculum", config={"run_dir": str(cfg.run_dir)})

    print(
        f"[curriculum] device={device} stages={len(cfg.stages)} "
        f"resume_at={state['stage_index']} allow_unmet={cfg.allow_unmet}"
    )

    policy: Optional[PolicyNetwork] = None
    model_config: Optional[dict] = None
    while state["stage_index"] < len(cfg.stages):
        idx = state["stage_index"]
        stage = cfg.stages[idx]
        outcome = _run_stage(
            cfg, stage, idx, state, device, rng, policy, model_config, history_path, use_wandb
        )
        policy = outcome["policy"]
        model_config = outcome["model_config"]
        if not outcome["promoted"]:
            state["ok"] = False
            state["reason"] = outcome["reason"]
            _save_state(state_path, state)
            if use_wandb:
                import wandb

                wandb.finish()
            return {"ok": False, "reason": outcome["reason"], "state": state}
        state["stage_index"] = idx + 1
        state["updates_done"] = 0
        state["episode_counter"] = 0
        state["completed"].append(outcome["record"])
        state["ok"] = True
        state["reason"] = ""
        _save_state(state_path, state)

    if use_wandb:
        import wandb

        wandb.finish()
    latest = cfg.run_dir / "latest.pt"
    print(f"[curriculum] finished. latest={latest}")
    return {"ok": True, "reason": "", "state": state, "checkpoint": str(latest)}


def _run_stage(cfg, stage, idx, state, device, rng, policy, model_config, history_path, use_wandb):
    maps = _stage_maps(cfg, stage)
    env = make_env(
        stage.skill,
        maps[0],
        max_turns=stage.max_turns,
        randomize=stage.randomize,
        approach_coef=stage.approach_coef,
    )
    env.reset(seed=cfg.seed)
    stage_dir = cfg.run_dir / "stages" / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    in_progress = stage_dir / "latest.pt"

    if state["updates_done"] > 0 and in_progress.exists():
        policy, model_config = load_checkpoint(in_progress, device)
        print(f"[curriculum] resume {stage.name} from update {state['updates_done']}")
    elif policy is None:
        ckpt = cfg.init_checkpoint if idx == 0 else None
        if ckpt is not None and Path(ckpt).exists():
            policy, model_config = load_checkpoint(ckpt, device)
            print(f"[curriculum] warm start {stage.name} from {ckpt}")
        else:
            if ckpt is not None:
                print(f"[curriculum] {ckpt} missing — random init")
            policy = _new_policy(cfg.arch, _infer_token_dim(env), device)
            model_config = None
    else:
        print(f"[curriculum] handoff into {stage.name}")

    assert policy is not None
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    opponent = None
    if stage.skill == "full" and stage.train_opponent == "snapshot":
        opponent = frozen_copy(policy)

    use_env_reward = stage.skill != "full"
    ep = int(state["episode_counter"])
    start_update = int(state["updates_done"])
    best_metric = -1e9
    last_stats = {"success_rate": 0.0, "win_rate": 0.0, "mean_return": 0.0}
    promoted = False
    gate_met = False

    def reset(episode_index: int) -> int:
        env.map_json_path = maps[episode_index % len(maps)]
        env.reset(seed=cfg.seed + episode_index)
        if stage.skill == "full":
            return episode_index % 2
        return 0

    def opponent_for(_episode_index: int):
        if stage.skill != "full":
            return None, None
        if stage.train_opponent == "snapshot":
            return opponent, None
        return None, None  # random (or "none", same thing on the full game)

    def annotate(info, live_env):
        info = dict(info)
        info["success"] = episode_success(stage.skill, info, live_env)
        return info

    for update in range(start_update, stage.max_updates):
        buffer, infos, ep = collect_episodes(
            policy,
            env,
            rng,
            target_decisions=stage.rollout_decisions,
            hyper=cfg.ppo,
            max_steps=stage.max_steps,
            reset=reset,
            opponent_for=opponent_for,
            use_env_reward=use_env_reward,
            episode_start=ep,
            annotate=annotate,
        )
        metrics = apply_ppo(policy, optimizer, buffer, cfg.ppo, device)
        train_stats = _metric_value(stage.skill, infos, env)
        do_eval = stage.eval_episodes > 0 and (
            (update + 1) % max(1, stage.eval_every) == 0 or update + 1 == stage.max_updates
        )
        if do_eval:
            last_stats = _evaluate(policy, env, rng, stage, maps, cfg, ep + 100000)
            ep += stage.eval_episodes
        else:
            last_stats = train_stats
        value = float(last_stats[stage.metric])
        best_metric = max(best_metric, value)
        state["updates_done"] = update + 1
        state["episode_counter"] = ep
        record = {
            "stage": stage.name,
            "update": update + 1,
            "metric": stage.metric,
            "value": value,
            "train_success": train_stats["success_rate"],
            **{f"ppo_{k}": v for k, v in metrics.items()},
        }
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[{stage.name} {update + 1}/{stage.max_updates}] "
            f"{stage.metric}={value:.3f} (gate {stage.success_threshold:.3f}) "
            f"train_ok={train_stats['success_rate']:.2f} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f}"
        )
        if use_wandb:
            import wandb

            wandb.log(record)
        save_checkpoint(policy, in_progress, extra={"stage": stage.name, "update": update + 1})
        save_checkpoint(policy, cfg.run_dir / "latest.pt", extra={"stage": stage.name, "update": update + 1})
        if cfg.save_every and (update + 1) % cfg.save_every == 0:
            save_checkpoint(policy, stage_dir / f"update_{update + 1}.pt")
        if (
            stage.skill == "full"
            and stage.train_opponent == "snapshot"
            and (update + 1) % max(1, cfg.opponent_refresh) == 0
        ):
            opponent = frozen_copy(policy)
        _save_state(cfg.run_dir / "state.json", state)
        if do_eval and value >= stage.success_threshold:
            promoted = True
            gate_met = True
            break

    if not promoted:
        if cfg.allow_unmet:
            promoted = True
            gate_met = False
            print(f"[curriculum] {stage.name} missed the gate; allow_unmet hands the checkpoint on")
        else:
            reason = (
                f"{stage.name} finished {stage.max_updates} updates at "
                f"{stage.metric}={best_metric:.3f} < {stage.success_threshold:.3f}"
            )
            print(f"[curriculum] STOP {reason}")
            return {
                "promoted": False,
                "reason": reason,
                "policy": policy,
                "model_config": model_config,
                "record": {},
            }

    try:
        env.close()
    except Exception:
        pass
    return {
        "promoted": True,
        "reason": "",
        "policy": policy,
        "model_config": model_config,
        "record": {
            "stage": stage.name,
            "skill": stage.skill,
            "gate_met": gate_met,
            "metric": stage.metric,
            "value": best_metric,
            "checkpoint": str(cfg.run_dir / "latest.pt"),
        },
    }


def _evaluate(policy, env, rng, stage: StageSpec, maps, cfg: CurriculumConfig, seed0: int) -> Dict[str, float]:
    """Greedy-free eval. Full-game eval is always vs random, so the gate is not 'beat yourself'."""
    import random as _random

    policy.eval()
    infos = []
    eval_rng = _random.Random(seed0)

    def reset(episode_index: int) -> int:
        env.map_json_path = maps[episode_index % len(maps)]
        env.reset(seed=seed0 + episode_index)
        if stage.skill == "full":
            return episode_index % 2
        return 0

    def opponent_for(_episode_index: int):
        return None, None

    def annotate(info, live_env):
        info = dict(info)
        info["success"] = episode_success(stage.skill, info, live_env)
        return info

    # One episode per call: collect_trajectory finishes the episode, then the
    # buffer already holds its decisions and the loop stops.
    for i in range(stage.eval_episodes):
        _, batch, _ = collect_episodes(
            policy,
            env,
            eval_rng,
            target_decisions=1,
            hyper=cfg.ppo,
            max_steps=stage.max_steps,
            reset=lambda ep, i=i: reset(i),
            opponent_for=opponent_for,
            use_env_reward=stage.skill != "full",
            episode_start=0,
            annotate=annotate,
        )
        if batch:
            infos.append(batch[0])
    return _metric_value(stage.skill, infos, env)


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


def config_as_dict(cfg: CurriculumConfig) -> Dict[str, Any]:
    """For tests and logging. Paths become strings."""
    raw = asdict(cfg)
    raw["run_dir"] = str(cfg.run_dir)
    raw["init_checkpoint"] = str(cfg.init_checkpoint) if cfg.init_checkpoint else None
    return raw
