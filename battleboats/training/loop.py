"""Shared PPO collect + update used by the curriculum and by self-play.

Opponent selection stays outside this module. Callers pass a policy, ``None``
(random), or a callable. Nothing here records ratings or a population id, so a
later quality-diversity archive can supply opponents without touching the
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.rollout import collect_trajectory, compute_gae


@dataclass
class PPOHyper:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None


def collect_episodes(
    policy: PolicyNetwork,
    env,
    rng,
    *,
    target_decisions: int,
    hyper: PPOHyper,
    max_steps: int,
    reset: Callable[[int], int],
    opponent_for: Callable[[int], Tuple[Optional[PolicyNetwork], Optional[Callable]]],
    use_env_reward: bool,
    episode_start: int = 0,
    annotate: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
) -> Tuple[RolloutBuffer, List[Dict[str, Any]], int]:
    """Fill a buffer. ``reset(ep)`` resets the env and returns the learner seat.

    ``opponent_for(ep)`` returns ``(opponent_policy, opponent_action)``. A
    policy of None with no callable is uniform random. Stops once the buffer
    holds ``target_decisions`` or a safety cap of episodes is hit.
    """
    buffer = RolloutBuffer()
    infos: List[Dict[str, Any]] = []
    ep = episode_start
    cap = max(8, target_decisions * 4 + 8)
    policy.eval()
    while len(buffer) < target_decisions and (ep - episode_start) < cap:
        agent_pid = reset(ep)
        opponent_policy, opponent_action = opponent_for(ep)
        traj, info = collect_trajectory(
            policy,
            opponent_policy,
            env,
            agent_pid,
            rng,
            max_steps=max_steps,
            use_env_reward=use_env_reward,
            opponent_action=opponent_action,
        )
        if traj:
            compute_gae(traj, gamma=hyper.gamma, lam=hyper.lam)
            buffer.add(traj)
        info = dict(info)
        info["agent_pid"] = agent_pid
        if annotate is not None:
            info = annotate(info, env)
        infos.append(info)
        ep += 1
    return buffer, infos, ep


def apply_ppo(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    hyper: PPOHyper,
    device: torch.device,
) -> Dict[str, float]:
    """One PPO update. Empty buffers return zeros and do not step."""
    if len(buffer) == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "epochs_run": 0.0,
        }
    policy.train()
    metrics = ppo_update(
        policy,
        optimizer,
        buffer,
        clip_eps=hyper.clip_eps,
        vf_coef=hyper.vf_coef,
        ent_coef=hyper.ent_coef,
        epochs=hyper.epochs,
        minibatch_size=min(hyper.minibatch_size, len(buffer)),
        max_grad_norm=hyper.max_grad_norm,
        device=device,
        target_kl=hyper.target_kl,
    )
    return metrics
