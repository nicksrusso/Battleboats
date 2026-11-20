"""Self-describing policy checkpoints.

Same on-disk shape the BC and PPO scripts already write, plus optional
bookkeeping keys (``update``, ``optimizer``) that older loaders ignore:

    {"model_state": state_dict, "model_config": {token_dim, d_model, ...}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from battleboats.training.policy import PolicyNetwork


def model_config_of(policy: PolicyNetwork) -> Dict[str, Any]:
    """Rebuild the ``model_config`` dict from a live module.

    ``PolicyNetwork`` stores ``d_model`` directly; the rest is read off the
    first encoder layer so a checkpoint round-trips without the original CLI
    flags.
    """
    layer = policy.encoder.layers[0]
    return {
        "token_dim": int(policy.proj.in_features),
        "d_model": int(policy.d_model),
        "nhead": int(layer.self_attn.num_heads),
        "num_layers": len(policy.encoder.layers),
        "dim_feedforward": int(layer.linear1.out_features),
        "dropout": float(layer.dropout.p),
    }


def save_checkpoint(
    policy: PolicyNetwork,
    path: Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``policy`` to ``path``. ``extra`` is merged in (not required to load)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": policy.state_dict(),
        "model_config": model_config_of(policy),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[PolicyNetwork, Dict[str, Any]]:
    """Load a self-describing checkpoint. ``model_config`` wins over any caller arch."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in ckpt or "model_config" not in ckpt:
        raise ValueError(
            f"{path} is not a battleboats policy checkpoint "
            "(expected keys model_state and model_config)"
        )
    policy = PolicyNetwork(**ckpt["model_config"])
    policy.load_state_dict(ckpt["model_state"])
    policy.to(device)
    return policy, dict(ckpt["model_config"])


def frozen_copy(policy: PolicyNetwork) -> PolicyNetwork:
    """Deep-copied eval snapshot. Later steps on ``policy`` do not touch it."""
    import copy

    opp = copy.deepcopy(policy)
    opp.eval()
    for param in opp.parameters():
        param.requires_grad_(False)
    return opp
