"""Sample one legal action from a policy against a live engine state."""

from __future__ import annotations

import torch

from battleboats.core.actions import Action, EndTurnAction
from battleboats.envs.action_masks import ActionMasks
from battleboats.training.policy import PolicyNetwork


@torch.no_grad()
def policy_action(policy: PolicyNetwork, engine, player_id: int) -> Action:
    """Greedy-free sample: ``policy.act`` under the engine's legality masks.

    The policy is not switched to eval here; callers that want dropout off
    (rollout, play UI) set ``policy.eval()`` themselves.
    """
    device = next(policy.parameters()).device
    masks = ActionMasks(engine, player_id, device=device)
    if masks.n == 0:
        return EndTurnAction()
    asset_idx, verb_idx, target_idx, _logp, _value = policy.act(
        masks.tokens, masks.pad_mask, masks
    )
    return masks.to_action(
        asset_idx=asset_idx, verb_idx=verb_idx, target_idx=target_idx
    )
