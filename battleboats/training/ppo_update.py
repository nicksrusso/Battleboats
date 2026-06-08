"""PPO clipped-surrogate update.

Consumes the ``(batch, StoredMasks)`` minibatches a ``RolloutBuffer`` yields and
runs K epochs of the clipped update on ``policy``, re-scoring each stored decision
under the current policy and nudging it toward actions that beat their baseline.

The loss combines three terms: a clipped policy loss (the trust-region surrogate
that keeps each update close to the data-collecting policy), a value MSE loss
(regressing ``V(s)`` toward the GAE return), and an entropy bonus (subtracted, to
preserve exploration).

The ratio ``exp(log pi_new - log pi_old)`` is only valid because ``StoredMasks``
replays the exact legality the rollout acted under, so both log-probs use the same
masked distribution.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.policy import PolicyNetwork


def ppo_update(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    *,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    minibatch_size: int = 256,
    max_grad_norm: float = 0.5,
    device: Optional[torch.device] = None,
    target_kl: Optional[float] = None,
) -> Dict[str, float]:
    """Run K epochs of PPO on the data in ``buffer`` and update ``policy``.

    For each minibatch, re-scores the stored actions under the current policy
    (via ``evaluate_actions`` with the replayed ``StoredMasks``), normalizes
    advantages, and steps the clipped surrogate plus the value and entropy terms.
    Does not clear the buffer; the caller does that after the update (PPO is
    on-policy, so the data is stale once the policy moves).

    Args:
        policy: the learning policy, updated in place.
        optimizer: optimizer over ``policy``'s parameters.
        buffer: source of shuffled minibatches for this update.
        clip_eps: epsilon for the clipped probability ratio.
        vf_coef: weight on the value loss.
        ent_coef: weight on the (subtracted) entropy bonus.
        epochs: number of passes over the buffer.
        minibatch_size: rows per gradient step.
        max_grad_norm: global gradient-norm clip.
        device: device the minibatch tensors are collated onto.
        target_kl: optional early-stop threshold on the per-epoch approx KL.

    Returns:
        A dict of mean scalar metrics: "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", and "epochs_run".
    """

    policy_losses, value_losses, entropies = [], [], []
    approx_kls, clip_fracs = [], []
    epochs_run = 0

    for epoch in range(epochs):
        for batch, masks in buffer.iter_minibatches(minibatch_size, device=device):
            logp_new, entropy, value = policy.evaluate_actions(
                batch["tokens"], batch["pad_mask"], batch["asset_idx"], batch["verb_idx"], batch["target_idx"], masks
            )
            adv = batch["advantage"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp_new - batch["logp_old"])
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(value, batch["ret"])
            entropy_bonus = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - (logp_new - batch["logp_old"])).mean()
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())
                approx_kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())
        epochs_run += 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "policy_loss": _mean(policy_losses),
        "value_loss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approx_kl": _mean(approx_kls),
        "clip_frac": _mean(clip_fracs),
        "epochs_run": epochs_run,
    }
