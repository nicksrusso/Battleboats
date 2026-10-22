"""Minimal PPO training loop for the LandingSkillAEC sandbox.

Reuses the full-game training modules unchanged — collect_trajectory (single
learner, opponent auto-EndTurn'd inside the env so agent_pid=0 is the only seat),
compute_gae, RolloutBuffer, ppo_update — and wraps them in a from-scratch loop on
the tiny 2x3 map. No BC warm-start: this is a fresh skill, so the policy starts
random.

Sparse reward is handled entirely by collect_trajectory: it reads engine.winner
and puts +1 on the last decision when the learner captures the enemy home, 0 on
truncation. (When shaping lands later, collect_trajectory will need to read the
env's per-step rewards instead — not required for sparse.)

Usage:
    python scripts/train_landing.py --updates 50 --rollout-decisions 1024
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.landing_skill_aec import LandingSkillAEC
from battleboats.training.device import resolve_device
from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.rollout import collect_trajectory, compute_gae

DEFAULT_MAP = "battleboats/core/config/maps/map_landing_2x3.json"
LEARNER_PID = 0


def infer_token_dim(env: LandingSkillAEC) -> int:
    """Read the per-token feature width from a live mask (map-size independent)."""
    return int(ActionMasks(env.engine, LEARNER_PID, device="cpu").tokens.shape[-1])


def collect_rollouts(
    policy: PolicyNetwork,
    env: LandingSkillAEC,
    buffer: RolloutBuffer,
    rng: random.Random,
    *,
    target_decisions: int,
    gamma: float,
    lam: float,
    max_steps: int,
    episode_counter: int,
) -> tuple[int, dict[str, float]]:
    """Play sandbox episodes until the buffer holds >= target_decisions.

    For each episode: env.reset(seed=episode_counter) so runs are reproducible,
    then collect_trajectory(policy, opponent_policy=None, env, agent_pid=0, rng,
    max_steps), run compute_gae on the returned trajectory (per-episode, BEFORE
    buffer.add — GAE must not cross episode boundaries), and add it. Track
    per-episode capture (info["winner"] == 0) and decision count so we can report
    a capture-rate and mean steps-to-capture for this rollout.

    Returns:
        (updated episode_counter, stats dict)
    """
    episodes = 0
    captures = 0
    total_len = 0  # steps summed over ALL episodes (captures + truncations)
    capture_steps = 0  # steps summed over CAPTURED episodes only

    while len(buffer) < target_decisions:
        env.reset(seed=episode_counter)
        episode_counter += 1  # global seed, threaded across updates
        traj, info = collect_trajectory(
            policy,
            opponent_policy=None,
            env=env,
            agent_pid=LEARNER_PID,
            rng=rng,
            max_steps=max_steps,
        )
        compute_gae(traj, gamma=gamma, lam=lam)
        buffer.add(traj)
        episodes += 1  # this rollout's count
        captured = info["winner"] == LEARNER_PID
        captures += int(captured)
        total_len += info["steps"]
        if captured:
            capture_steps += info["steps"]

    stats = {
        "rollout/episodes": float(episodes),
        "rollout/decisions": float(len(buffer)),
        "rollout/capture_rate": captures / episodes,
        # Overall episode length (all outcomes) — a rollout-cost signal, not an
        # efficiency one; truncations inflate it, explained by (1 - capture_rate).
        "rollout/mean_len": total_len / episodes,
        # Directness among successes only — the clean efficiency signal. NaN when
        # the rollout captured nothing (no successes to average).
        "rollout/mean_steps_to_capture": (
            capture_steps / captures if captures else float("nan")
        ),
    }
    return episode_counter, stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-path", type=str, default=DEFAULT_MAP)
    p.add_argument("--randomize-target", action="store_true",
                   help="Place the enemy home on a random tile each episode (learner "
                        "home fixed). Forces relational navigation over route memorization.")
    p.add_argument("--max-turns", type=int, default=50, help="Env truncation budget.")
    p.add_argument("--max-steps", type=int, default=200, help="Engine-step cap per episode.")
    # PPO outer loop
    p.add_argument("--updates", type=int, default=50)
    p.add_argument(
        "--rollout-decisions", type=int, default=1024, help="Collect episodes until the buffer holds this many decisions."
    )
    p.add_argument("--epochs", type=int, default=4, help="K epochs per update.")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    # Small net for a small task
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dim-feedforward", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    # Warm start
    p.add_argument("--load-checkpoint", type=Path, default=None,
                   help="Warm-start from a self-describing {model_state, model_config} "
                        "checkpoint (e.g. a prior stage's landing_latest.pt). Its "
                        "model_config is authoritative over the --d-model/etc. flags.")
    # Infra
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, cuda:N, or auto. ROCm reports the AMD GPU as cuda.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=Path, default=Path("runs/landing_checkpoints"))
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--no-save", action="store_true",
                   help="Skip all checkpoint writes (throwaway diagnostic run). Also "
                        "avoids clobbering an existing landing_latest.pt in --save-dir.")
    return p


def save_checkpoint(policy: PolicyNetwork, args: argparse.Namespace, path: Path) -> None:
    """Self-describing checkpoint (same shape as BC/PPO so load_policy can read it)."""
    torch.save(
        {
            "model_state": policy.state_dict(),
            "model_config": {
                "token_dim": policy.proj.in_features,
                "d_model": policy.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
        },
        path,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    args.device = str(resolve_device(args.device))
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    env = LandingSkillAEC(
        args.map_path, max_turns=args.max_turns, randomize_target=args.randomize_target
    )
    env.reset(seed=args.seed)
    token_dim = infer_token_dim(env)

    if args.load_checkpoint is not None:
        # Warm start: the checkpoint's model_config is authoritative (the state
        # dict only fits that architecture); token_dim is map-independent so a
        # 2x3-trained policy loads straight onto this board.
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        policy = PolicyNetwork(**ckpt["model_config"])
        policy.load_state_dict(ckpt["model_state"])
        policy = policy.to(device)
        print(f"[train_landing] warm-start from {args.load_checkpoint} "
              f"(config={ckpt['model_config']})")
    else:
        policy = PolicyNetwork(
            token_dim=token_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = RolloutBuffer()
    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- main loop
    # Sanity signals on this sandbox: capture_rate should climb from the ~0.63
    # random baseline toward ~1.0 and mean_len should fall. An entropy spike with
    # high approx_kl/clip_frac means too-large a step or noise advantages — but
    # the reward here is dense enough that shouldn't bite.
    episode_counter = 0
    for update in range(args.updates):
        buffer.clear()  # on-policy: drop last update's stale data
        policy.eval()  # rollout: sample actions, dropout off
        episode_counter, roll_stats = collect_rollouts(
            policy,
            env,
            buffer,
            rng,
            target_decisions=args.rollout_decisions,
            gamma=args.gamma,
            lam=args.lam,
            max_steps=args.max_steps,
            episode_counter=episode_counter,
        )

        policy.train()  # update: normal training forward, dropout on
        metrics = ppo_update(
            policy,
            optimizer,
            buffer,
            clip_eps=args.clip_eps,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            device=device,
            target_kl=args.target_kl,
        )

        # ------------------------------------------------------------- logging
        print(
            f"[update {update + 1}/{args.updates}] "
            f"cap={roll_stats['rollout/capture_rate']:.2f} "
            f"cap_len={roll_stats['rollout/mean_steps_to_capture']:.1f} "
            f"len={roll_stats['rollout/mean_len']:.1f} "
            f"eps={roll_stats['rollout/episodes']:.0f} "
            f"dec={roll_stats['rollout/decisions']:.0f} | "
            f"pi_loss={metrics['policy_loss']:.4f} "
            f"v_loss={metrics['value_loss']:.4f} "
            f"ent={metrics['entropy']:.3f} "
            f"kl={metrics['approx_kl']:.4f} "
            f"clip={metrics['clip_frac']:.2f}"
        )

        if not args.no_save and (update + 1) % args.save_every == 0:
            save_checkpoint(policy, args, args.save_dir / f"landing_update_{update + 1}.pt")

    if not args.no_save:
        save_checkpoint(policy, args, args.save_dir / "landing_latest.pt")


if __name__ == "__main__":
    main()
