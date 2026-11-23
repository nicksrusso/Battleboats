"""Buy-diversity sandbox helpers: score, cheap-type masking, PPO fine-tune.

The v0.4.0 behavior clone is close to uniform over *currently affordable*
hulls, so cheap types (destroyer / submarine / merchant, cost 100) are
legal far more often and dominate the mix. This module fine-tunes a warm
start in a no-opponent sandbox whose only reward is a fleet-diversity
score, with intermittent masking of those cheap types so the policy
experiences saving for expensive hulls.

Diversity is computed over all eight ShipType values, including landing
and builder. The goal is an equal purchase *rate* across every hull the
engine can build; dropping builder (or landing) would let the policy
ignore a 250-cost type and still look diverse. Landing ships are never
masked. The reported metric is ``max(0, 1 - std/mean)`` of the 8-count
vector (zeros included, so a missing type is imbalance). Dividing by the
mean means a bigger but still even fleet is not penalized.

A terminal-only reward would vanish under PPO's per-decision discount
(gamma=0.99 over a 50-turn episode with hundreds of decisions). Training
therefore pays the potential-based delta of the (clamped) score on each
buy; those deltas sum to the final score. A small coverage term,
``coverage_coef * unique/8``, is added to the training potential only so
the first few distinct types are not a long zero-reward prefix (the
clamped CV score stays 0 until at least five distinct types appear). The
logged / evaluated diversity score does not include coverage.

Evaluate unmasked. No combat or win reward is used in this sandbox.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml

from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.training.checkpoints import (frozen_copy, load_checkpoint,
                                              save_checkpoint)
from battleboats.training.loop import PPOHyper, apply_ppo
from battleboats.training.policy import VERB_TO_IDX, PolicyNetwork, set_dropout
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.rollout import collect_trajectory, compute_gae
from battleboats.training.selfplay_purchases import (empty_purchase_counts,
                                                     hull_name)
from battleboats.training.manifest import ManifestSession

# All eight buildable hulls, ShipType enum order. Include landing and builder:
# the warm start under-buys both 250-cost types, and "equal rate" is over
# every type the shipyard can produce.
BUYDIV_HULLS: Tuple[str, ...] = tuple(hull_name(t) for t in ShipType)

# Cost-100 types. Affordable from opening cash and after almost every income
# tick; never includes landing (cost 250).
CHEAP_HULLS: Tuple[str, ...] = ("destroyer", "submarine", "merchant")
CHEAP_SHIP_TYPES = frozenset(t for t in ShipType if hull_name(t) in CHEAP_HULLS)
HULL_COST: Dict[str, int] = {hull_name(t): int(BASE_STATS[t].cost) for t in ShipType}

if set(BUYDIV_HULLS) != {hull_name(t) for t in ShipType}:
    raise RuntimeError("BUYDIV_HULLS does not match ShipType")
if any(name not in HULL_COST for name in CHEAP_HULLS):
    raise RuntimeError(f"cheap hulls {CHEAP_HULLS} are not in ShipType")
if "landing" in CHEAP_HULLS:
    raise RuntimeError("landing ships must never be masked")


def _counts_vector(counts: Dict[str, int]) -> List[float]:
    return [float(counts.get(name, 0)) for name in BUYDIV_HULLS]


def diversity_raw(counts: Dict[str, int]) -> float:
    """Unclamped ``1 - std/mean`` over the eight-type count vector.

    Empty fleets return 0.0 (no mean). Population std (divide by n, not n-1).
    """
    xs = _counts_vector(counts)
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    n = float(len(xs))
    mean = total / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    if mean <= 0.0:
        return 0.0
    return 1.0 - (std / mean)


def diversity_score(counts: Dict[str, int]) -> float:
    """Reported metric: ``max(0, 1 - std/mean)`` over all eight types."""
    return max(0.0, diversity_raw(counts))


def coverage_bonus(counts: Dict[str, int], coef: float) -> float:
    """``coef * unique/8``. Training potential only; not part of the metric."""
    if coef <= 0.0:
        return 0.0
    unique = sum(1 for x in _counts_vector(counts) if x > 0.0)
    return float(coef) * (unique / float(len(BUYDIV_HULLS)))


def diversity_potential(counts: Dict[str, int], coverage_coef: float = 0.0) -> float:
    """Training potential: clamped diversity plus optional coverage bonus."""
    return diversity_score(counts) + coverage_bonus(counts, coverage_coef)


def gold_spent_by_type(counts: Dict[str, int]) -> Dict[str, int]:
    return {name: int(counts.get(name, 0)) * HULL_COST[name] for name in BUYDIV_HULLS}


def gold_spent_share(counts: Dict[str, int]) -> Dict[str, float]:
    spent = gold_spent_by_type(counts)
    total = float(sum(spent.values()))
    if total <= 0.0:
        return {name: 0.0 for name in BUYDIV_HULLS}
    return {name: spent[name] / total for name in BUYDIV_HULLS}


def mask_prob(update: int, total_updates: int, p_start: float, p_end: float) -> float:
    """Linear anneal of the per-episode cheap-type mask probability.

    ``update`` is 0-based. ``p_end`` is reached at the last update. Values
    are clamped to [0, 1].
    """
    if total_updates <= 1:
        return float(p_end)
    t = min(1.0, max(0.0, float(update) / float(total_updates - 1)))
    p = float(p_start) + t * (float(p_end) - float(p_start))
    return min(1.0, max(0.0, p))


def sum_counts(items: Iterable[Dict[str, int]]) -> Dict[str, int]:
    out = empty_purchase_counts()
    for counts in items:
        for name in BUYDIV_HULLS:
            out[name] = int(out.get(name, 0)) + int(counts.get(name, 0))
    return out


def mean_diversity(items: Sequence[Dict[str, int]]) -> float:
    if not items:
        return 0.0
    return sum(diversity_score(c) for c in items) / float(len(items))


@dataclass
class BuyDivConfig:
    init_checkpoint: str = "runs/checkpoints/v0.4.0_8ad2530.pt"
    run_dir: str = "runs/sandbox_buydiv_v0.4.0"
    updates: int = 300
    decisions_per_update: int = 1024
    horizon_turns: int = 50
    sizes: List[List[int]] = field(default_factory=lambda: [[16, 8], [20, 10]])
    p_start: float = 0.7
    p_end: float = 0.0
    coverage_coef: float = 0.25
    anchor_coef: float = 0.0
    lr: float = 1.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    hull_ent_coef: float = 0.05
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    dropout: float = 0.0
    seed: int = 0
    eval_every: int = 25
    eval_episodes: int = 16
    ckpt_every: int = 10
    window: int = 25
    max_minutes: float = 40.0
    device: str = "cpu"

    def ppo(self) -> PPOHyper:
        return PPOHyper(
            lr=self.lr,
            gamma=self.gamma,
            lam=self.lam,
            clip_eps=self.clip_eps,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            epochs=self.epochs,
            minibatch_size=self.minibatch_size,
            max_grad_norm=self.max_grad_norm,
            target_kl=self.target_kl,
            hull_ent_coef=self.hull_ent_coef,
        )


def load_buydiv_config(path: Path) -> BuyDivConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in BuyDivConfig.__dataclass_fields__.values()}
    extra = set(raw) - known
    if extra:
        raise ValueError(f"unknown buydiv config keys: {sorted(extra)}")
    return BuyDivConfig(**raw)


def apply_nonbuy_anchor(
    policy: PolicyNetwork,
    frozen: PolicyNetwork,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    coef: float,
    device: torch.device,
    minibatch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> float:
    """MSE on (log pi - log pi_init) for non-build_ship decisions only.

    Off by default. Use only if a later eval shows move / capture collapse.
    Buy decisions are excluded so the diversity objective can move the hull
    head. One extra epoch after the PPO update; returns the mean loss or 0.
    """
    if coef <= 0.0 or len(buffer) == 0:
        return 0.0
    build_idx = VERB_TO_IDX["build_ship"]
    losses: List[float] = []
    policy.train()
    frozen.eval()
    batch_size = min(int(minibatch_size), len(buffer))
    for batch, masks in buffer.iter_minibatches(batch_size, device=device):
        nonbuy = batch["verb_idx"] != build_idx
        if not bool(nonbuy.any()):
            continue
        logp_new, _, _ = policy.evaluate_actions(
            batch["tokens"],
            batch["pad_mask"],
            batch["asset_idx"],
            batch["verb_idx"],
            batch["target_idx"],
            masks,
            skill_idx=batch["skill_idx"],
            globals=batch["globals"],
        )
        with torch.no_grad():
            logp_init, _, _ = frozen.evaluate_actions(
                batch["tokens"],
                batch["pad_mask"],
                batch["asset_idx"],
                batch["verb_idx"],
                batch["target_idx"],
                masks,
                skill_idx=batch["skill_idx"],
                globals=batch["globals"],
            )
        loss = float(coef) * (logp_new - logp_init)[nonbuy].pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def _parse_sizes(sizes: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    out = []
    for pair in sizes:
        if len(pair) != 2:
            raise ValueError(f"size must be [width, height], got {pair}")
        out.append((int(pair[0]), int(pair[1])))
    if not out:
        raise ValueError("at least one map size is required")
    return out


def _steps_cap(turns: int) -> int:
    # Generous per-turn decision budget (moves + buys + endturn).
    return int(turns) * 80


def _episode_record(env, info: Dict[str, Any], masked: bool) -> Dict[str, Any]:
    counts = dict(getattr(env, "purchase_counts", info.get("purchases") or {}))
    return {
        "masked": bool(masked),
        "purchases": counts,
        "diversity": diversity_score(counts),
        "diversity_raw": diversity_raw(counts),
        "potential": float(
            getattr(env, "training_potential", diversity_potential(counts))
        ),
        "reward_sum": float(info.get("reward_sum", 0.0)),
        "gold_share": gold_spent_share(counts),
        "gold_spent": gold_spent_by_type(counts),
        "n_decisions": int(info.get("n_decisions", 0)),
        "turn": int(info.get("turn", 0)),
        "width": int(getattr(env, "width", 0)),
        "height": int(getattr(env, "height", 0)),
        "learner_id": int(getattr(env, "learner_id", 0)),
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "episodes": 0,
            "buy_counts": empty_purchase_counts(),
            "buy_mean": {name: 0.0 for name in BUYDIV_HULLS},
            "diversity_mean": 0.0,
            "gold_share": {name: 0.0 for name in BUYDIV_HULLS},
            "gold_spent": {name: 0 for name in BUYDIV_HULLS},
            "reward_mean": 0.0,
            "decisions_mean": 0.0,
        }
    buys = sum_counts(r["purchases"] for r in records)
    n = float(len(records))
    return {
        "episodes": len(records),
        "buy_counts": buys,
        "buy_mean": {name: buys[name] / n for name in BUYDIV_HULLS},
        "diversity_mean": sum(float(r["diversity"]) for r in records) / n,
        "gold_share": gold_spent_share(buys),
        "gold_spent": gold_spent_by_type(buys),
        "reward_mean": sum(float(r["reward_sum"]) for r in records) / n,
        "decisions_mean": sum(float(r["n_decisions"]) for r in records) / n,
    }


def mean_ci(values: Sequence[float], z: float = 1.96) -> Dict[str, float]:
    """Wald 95% CI for a mean (normal; fine at n≈100)."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    mean = sum(xs) / n
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "lo": mean,
            "hi": mean,
            "min": mean,
            "max": mean,
        }
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    half = float(z) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "lo": mean - half,
        "hi": mean + half,
        "min": min(xs),
        "max": max(xs),
    }


def summarize_validation(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Unmasked sandbox report: counts, gold share, diversity CI, min/max types."""
    base = summarize_records(records)
    scores = [float(r["diversity"]) for r in records]
    ci = mean_ci(scores)
    buys = base["buy_counts"]
    totals = [int(buys[name]) for name in BUYDIV_HULLS]
    if totals:
        tmin = min(totals)
        tmax = max(totals)
        min_type = [name for name in BUYDIV_HULLS if buys[name] == tmin]
        max_type = [name for name in BUYDIV_HULLS if buys[name] == tmax]
    else:
        tmin = tmax = 0
        min_type = max_type = []
    mean_count = (sum(totals) / len(BUYDIV_HULLS)) if BUYDIV_HULLS else 0.0
    # Per-episode spread across the eight types (zeros included).
    ep_mins: List[int] = []
    ep_maxs: List[int] = []
    for rec in records:
        vec = [int(rec["purchases"].get(name, 0)) for name in BUYDIV_HULLS]
        ep_mins.append(min(vec) if vec else 0)
        ep_maxs.append(max(vec) if vec else 0)
    ratio = (float(tmax) / float(tmin)) if tmin > 0 else float("inf")
    # Roughly even: every type is within a factor of 2 of the mean count,
    # and no type is missing.
    even = (
        bool(totals)
        and tmin > 0
        and all(0.5 * mean_count <= c <= 2.0 * mean_count for c in totals)
    )
    base.update(
        {
            "diversity": ci,
            "type_counts_min": tmin,
            "type_counts_max": tmax,
            "type_counts_min_types": min_type,
            "type_counts_max_types": max_type,
            "type_counts_ratio_max_min": ratio if math.isfinite(ratio) else None,
            "type_counts_mean": mean_count,
            "episode_type_min": mean_ci(ep_mins) if ep_mins else mean_ci([]),
            "episode_type_max": mean_ci(ep_maxs) if ep_maxs else mean_ci([]),
            "roughly_even": even,
        }
    )
    return base


def _env_for(cfg: BuyDivConfig, width: int, height: int):
    from battleboats.envs.sandboxes import BuyDiversitySandbox

    # Placeholder map; reset() rewrites a two-home layout into the temp file.
    placeholder = Path(cfg.run_dir) / "maps" / f"placeholder_{width}x{height}.json"
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    if not placeholder.exists():
        placeholder.write_text(
            json.dumps(
                {
                    "width": width,
                    "height": height,
                    "seed": 0,
                    "land": [[0, 0], [width - 1, height - 1]],
                    "ports": {"0": [[0, 0]], "1": [[width - 1, height - 1]]},
                }
            )
        )
    return BuyDiversitySandbox(
        str(placeholder),
        width=width,
        height=height,
        max_turns=int(cfg.horizon_turns),
        coverage_coef=float(cfg.coverage_coef),
    )


def run_unmasked_eval(
    policy: PolicyNetwork,
    cfg: BuyDivConfig,
    envs: Dict[Tuple[int, int], Any],
    rng: random.Random,
    device: torch.device,
    n_episodes: int,
    seed0: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sizes = _parse_sizes(cfg.sizes)
    records: List[Dict[str, Any]] = []
    policy.eval()
    for i in range(int(n_episodes)):
        width, height = sizes[i % len(sizes)]
        env = envs[(width, height)]
        learner_id = i % 2
        env.reset(
            seed=int(seed0) + i,
            options={"mask_cheap": False, "learner_id": learner_id},
        )
        _traj, info = collect_trajectory(
            policy,
            None,
            env,
            learner_id,
            rng,
            max_steps=_steps_cap(cfg.horizon_turns),
            use_env_reward=True,
            opponent_action=None,
        )
        records.append(_episode_record(env, info, masked=False))
    return records, summarize_records(records)


def run_buydiv(cfg: BuyDivConfig) -> Dict[str, Any]:
    """Fine-tune ``cfg.init_checkpoint`` in the buy-diversity sandbox."""
    if cfg.device != "cpu":
        raise ValueError("buy-diversity fine-tune is CPU-only")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    with ManifestSession(
        run_dir,
        kind="buydiv",
        config=cfg,
        init_checkpoint=cfg.init_checkpoint,
    ) as session:
        summary = _run_buydiv(cfg)
        latest = run_dir / "latest.pt"
        session.complete(
            final_checkpoint=latest if latest.is_file() else None,
            updates_completed=summary.get("updates_run"),
        )
        return summary


def _run_buydiv(cfg: BuyDivConfig) -> Dict[str, Any]:
    """Fine-tune ``cfg.init_checkpoint`` in the buy-diversity sandbox."""
    if cfg.device != "cpu":
        raise ValueError("buy-diversity fine-tune is CPU-only")
    device = torch.device("cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))

    policy, _model_cfg = load_checkpoint(Path(cfg.init_checkpoint), device)
    set_dropout(policy, float(cfg.dropout))
    policy.to(device)
    frozen = frozen_copy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    hyper = cfg.ppo()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    sizes = _parse_sizes(cfg.sizes)
    envs = {(w, h): _env_for(cfg, w, h) for w, h in sizes}

    history_path = run_dir / "history.jsonl"
    history_fh = history_path.open("a")
    started = time.time()
    episode_counter = 0
    window_train: List[Dict[str, Any]] = []
    window_unmasked: List[Dict[str, Any]] = []
    window_ppo: List[Dict[str, float]] = []
    report_windows: List[Dict[str, Any]] = []
    stop_reason = "completed"

    def _flush_window(update_end: int) -> Dict[str, Any]:
        ppo_keys = (
            "entropy",
            "approx_kl",
            "value_loss",
            "clip_frac",
            "policy_loss",
            "hull_entropy",
        )
        ppo_mean = {}
        if window_ppo:
            for key in ppo_keys:
                ppo_mean[key] = sum(float(m.get(key, 0.0)) for m in window_ppo) / len(
                    window_ppo
                )
        rec = {
            "update_end": update_end,
            "updates": len(window_ppo),
            "train_all": summarize_records(window_train),
            "train_unmasked": summarize_records(window_unmasked),
            "ppo": ppo_mean,
        }
        report_windows.append(rec)
        (run_dir / "window_report.json").write_text(
            json.dumps(report_windows, indent=2)
        )
        window_train.clear()
        window_unmasked.clear()
        window_ppo.clear()
        return rec

    try:
        for update in range(int(cfg.updates)):
            elapsed_min = (time.time() - started) / 60.0
            if elapsed_min >= float(cfg.max_minutes):
                stop_reason = f"max_minutes={cfg.max_minutes}"
                break
            p = mask_prob(update, cfg.updates, cfg.p_start, cfg.p_end)
            buffer = RolloutBuffer()
            policy.eval()
            decisions = 0
            ep_this = 0
            batch_records: List[Dict[str, Any]] = []
            cap = max(8, int(cfg.decisions_per_update) * 4 + 8)
            while decisions < int(cfg.decisions_per_update) and ep_this < cap:
                width, height = sizes[episode_counter % len(sizes)]
                env = envs[(width, height)]
                masked = rng.random() < p
                # Alternate seats so the policy cannot overfit a left-home
                # (seat 0) or right-home (seat 1) absolute-position habit.
                learner_id = episode_counter % 2
                env.reset(
                    seed=int(cfg.seed) + episode_counter,
                    options={"mask_cheap": masked, "learner_id": learner_id},
                )
                traj, info = collect_trajectory(
                    policy,
                    None,
                    env,
                    learner_id,
                    rng,
                    max_steps=_steps_cap(cfg.horizon_turns),
                    use_env_reward=True,
                )
                if traj:
                    compute_gae(traj, gamma=hyper.gamma, lam=hyper.lam)
                    buffer.add(traj)
                    decisions += len(traj)
                rec = _episode_record(env, info, masked=masked)
                batch_records.append(rec)
                window_train.append(rec)
                if not masked:
                    window_unmasked.append(rec)
                episode_counter += 1
                ep_this += 1

            ppo_metrics = apply_ppo(policy, optimizer, buffer, hyper, device)
            anchor_loss = apply_nonbuy_anchor(
                policy,
                frozen,
                buffer,
                optimizer,
                coef=float(cfg.anchor_coef),
                device=device,
                minibatch_size=hyper.minibatch_size,
                max_grad_norm=hyper.max_grad_norm,
            )
            ppo_metrics = dict(ppo_metrics)
            ppo_metrics["anchor_loss"] = float(anchor_loss)
            window_ppo.append(ppo_metrics)

            row = {
                "update": update + 1,
                "mask_p": p,
                "episodes": ep_this,
                "decisions": decisions,
                "elapsed_min": (time.time() - started) / 60.0,
                "batch": summarize_records(batch_records),
                "batch_unmasked": summarize_records(
                    [r for r in batch_records if not r["masked"]]
                ),
                **{f"ppo_{k}": v for k, v in ppo_metrics.items()},
            }
            if (update + 1) % int(cfg.eval_every) == 0 or update == 0:
                _recs, ev = run_unmasked_eval(
                    policy,
                    cfg,
                    envs,
                    rng,
                    device,
                    n_episodes=int(cfg.eval_episodes),
                    seed0=1_000_000 + (update + 1) * 100,
                )
                row["eval_unmasked"] = ev
                print(
                    f"[buydiv] u{update + 1} eval diversity={ev['diversity_mean']:.3f} "
                    f"buys={ev['buy_counts']}",
                    flush=True,
                )

            history_fh.write(json.dumps(row) + "\n")
            history_fh.flush()
            print(
                f"[buydiv] u{update + 1}/{cfg.updates} p={p:.3f} "
                f"div_all={row['batch']['diversity_mean']:.3f} "
                f"div_unmasked={row['batch_unmasked']['diversity_mean']:.3f} "
                f"ent={ppo_metrics['entropy']:.3f} kl={ppo_metrics['approx_kl']:.4f} "
                f"vf={ppo_metrics['value_loss']:.4f} dec={decisions} "
                f"min={row['elapsed_min']:.1f}",
                flush=True,
            )

            save_checkpoint(policy, run_dir / "latest.pt", extra={"update": update + 1})
            if (update + 1) % int(cfg.ckpt_every) == 0:
                save_checkpoint(
                    policy,
                    run_dir / f"update_{update + 1}.pt",
                    extra={"update": update + 1},
                )
            if (update + 1) % int(cfg.window) == 0:
                win = _flush_window(update + 1)
                print(
                    f"[buydiv] window u{update + 1} unmasked "
                    f"div={win['train_unmasked']['diversity_mean']:.3f} "
                    f"buys={win['train_unmasked']['buy_counts']} "
                    f"ent={win['ppo'].get('entropy', 0.0):.3f}",
                    flush=True,
                )
        else:
            stop_reason = "completed"
    finally:
        if window_ppo:
            _flush_window(int(row["update"]) if "row" in locals() else 0)
        history_fh.close()
        save_checkpoint(
            policy, run_dir / "latest.pt", extra={"stop_reason": stop_reason}
        )
        for env in envs.values():
            env.close()

    summary = {
        "stop_reason": stop_reason,
        "updates_run": int(row["update"]) if "row" in locals() else 0,
        "elapsed_min": (time.time() - started) / 60.0,
        "windows": report_windows,
        "init_checkpoint": cfg.init_checkpoint,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
