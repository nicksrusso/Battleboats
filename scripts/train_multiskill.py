"""Multi-skill PPO: interleave landing + merchant episodes into one shared policy.

Goal: one 3-headed policy that retains landing AND learns merchant, without a solo
merchant run overwriting the landing behavior (catastrophic forgetting).

The two tasks are imbalanced two independent ways, both handled here or merchant
swamps landing:
  1. Reward scale — merchant is dense (~90 return) vs landing sparse (±1).
  2. Transition count — a skilled merchant episode emits far more decisions.

Fixes: (1) COUNT-BALANCE — collect ~equal decisions per task per update; (2)
PER-TASK ADVANTAGE NORMALIZATION — normalize each task's advantages to unit scale
before pooling, so both contribute comparably regardless of raw reward magnitude
(ppo_update's per-minibatch norm would otherwise let merchant's ~90× advantages
crush landing's). Value loss still uses raw returns, so the shared value head is
merchant-biased — acceptable since per-task adv-norm keeps the POLICY balanced;
revisit with per-task return normalization if landing degrades.

Warm-starts from a landing checkpoint (landing retained; merchant's load/unload
verbs were never trained, so the merchant starts ~random and learns on top).

Usage:
    poetry run python3 scripts/train_multiskill.py \
        --landing-checkpoint runs/landing_checkpoints_5x5_rand/landing_latest.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from battleboats.envs.landing_skill_aec import LandingSkillAEC
from battleboats.envs.merchant_skill_aec import MerchantSkillAEC
from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.rollout import collect_trajectory, compute_gae

LEARNER_PID = 0


def load_policy(path: Path, device):
    """Return (policy, model_config); config is reused verbatim when saving."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PolicyNetwork(**ckpt["model_config"])
    net.load_state_dict(ckpt["model_state"])
    return net.to(device), ckpt["model_config"]


def normalize_advantages(transitions) -> None:
    """Normalize a task's advantages to zero-mean/unit-std, in place."""
    n = len(transitions)
    if n < 2:
        return
    mean = sum(t.advantage for t in transitions) / n
    var = sum((t.advantage - mean) ** 2 for t in transitions) / n
    std = var ** 0.5
    for t in transitions:
        t.advantage = (t.advantage - mean) / (std + 1e-8)


def collect_task(policy, env, rng, *, target_decisions, gamma, lam, max_steps, ep_seed):
    """Collect >= target_decisions from one env; GAE per episode. Returns
    (transitions, next_ep_seed, stats)."""
    transitions = []
    caps, rets, lens = [], [], []
    while len(transitions) < target_decisions:
        env.reset(seed=ep_seed)
        ep_seed += 1
        tr, info = collect_trajectory(
            policy, None, env, LEARNER_PID, rng, max_steps=max_steps, use_env_reward=True
        )
        compute_gae(tr, gamma=gamma, lam=lam)
        transitions.extend(tr)
        caps.append(int(info["winner"] == LEARNER_PID))
        rets.append(info["reward_sum"])
        lens.append(info["steps"])
    stats = {
        "episodes": len(rets),
        "capture_rate": sum(caps) / len(caps),
        "mean_return": sum(rets) / len(rets),
        "mean_len": sum(lens) / len(lens),
    }
    return transitions, ep_seed, stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landing-checkpoint", type=Path,
                   default=Path("runs/landing_checkpoints_5x5_rand/landing_latest.pt"),
                   help="Warm-start checkpoint (model_config authoritative).")
    p.add_argument("--landing-map", type=str,
                   default="battleboats/core/config/maps/map_landing_5x5.json")
    p.add_argument("--merchant-map", type=str,
                   default="battleboats/core/config/maps/map_merchant_3x2.json")
    p.add_argument("--merchant-randomize", action="store_true",
                   help="Randomize the satellite port each episode (home fixed). "
                        "Use with a larger merchant map (e.g. map_merchant_5x5.json).")
    p.add_argument("--landing-max-turns", type=int, default=60)
    p.add_argument("--landing-max-steps", type=int, default=300)
    p.add_argument("--merchant-max-turns", type=int, default=30)
    p.add_argument("--merchant-max-steps", type=int, default=200)
    # PPO
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--rollout-decisions", type=int, default=1024,
                   help="Total decisions per update; split evenly across the two tasks.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.02,
                   help="Moderate: enough to explore merchant verbs, not wreck landing.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=Path, default=Path("runs/multiskill_checkpoints"))
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="battleboats-multiskill")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    policy, model_config = load_policy(args.landing_checkpoint, device)
    print(f"[train_multiskill] warm-start from {args.landing_checkpoint} (config={model_config})")

    landing_env = LandingSkillAEC(
        args.landing_map, max_turns=args.landing_max_turns, randomize_target=True
    )
    merchant_env = MerchantSkillAEC(
        args.merchant_map, max_turns=args.merchant_max_turns,
        randomize_ports=args.merchant_randomize,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = RolloutBuffer()
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    per_task = args.rollout_decisions // 2
    land_ep = merch_ep = 0
    for update in range(args.updates):
        buffer.clear()
        policy.eval()

        land_tr, land_ep, l_stats = collect_task(
            policy, landing_env, rng, target_decisions=per_task,
            gamma=args.gamma, lam=args.lam, max_steps=args.landing_max_steps, ep_seed=land_ep,
        )
        merch_tr, merch_ep, m_stats = collect_task(
            policy, merchant_env, rng, target_decisions=per_task,
            gamma=args.gamma, lam=args.lam, max_steps=args.merchant_max_steps, ep_seed=merch_ep,
        )
        # Balance the two tasks' gradient contributions before pooling.
        normalize_advantages(land_tr)
        normalize_advantages(merch_tr)
        buffer.add(land_tr)
        buffer.add(merch_tr)

        policy.train()
        metrics = ppo_update(
            policy, optimizer, buffer,
            clip_eps=args.clip_eps, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
            epochs=args.epochs, minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm, device=device, target_kl=args.target_kl,
        )

        print(
            f"[update {update + 1}/{args.updates}] "
            f"LAND cap={l_stats['capture_rate']:.2f} len={l_stats['mean_len']:.1f} | "
            f"MERCH ret={m_stats['mean_return']:.1f} eps={m_stats['episodes']} | "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
        )

        if args.wandb:
            import wandb

            wandb.log(
                {
                    "landing/capture_rate": l_stats["capture_rate"],
                    "landing/mean_len": l_stats["mean_len"],
                    "landing/mean_return": l_stats["mean_return"],
                    "merchant/mean_return": m_stats["mean_return"],
                    "merchant/episodes": m_stats["episodes"],
                    "merchant/mean_len": m_stats["mean_len"],
                    **{f"ppo/{k}": v for k, v in metrics.items()},
                },
                step=update,
            )

        if not args.no_save and (update + 1) % args.save_every == 0:
            _save(policy, model_config, args.save_dir / f"multiskill_update_{update + 1}.pt")

    if not args.no_save:
        _save(policy, model_config, args.save_dir / "multiskill_latest.pt")


def _save(policy: PolicyNetwork, model_config: dict, path: Path) -> None:
    torch.save({"model_state": policy.state_dict(), "model_config": model_config}, path)


if __name__ == "__main__":
    main()
