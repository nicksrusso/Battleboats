"""Self-describing policy checkpoints.

Same on-disk shape the BC and PPO scripts already write, plus optional
bookkeeping keys (``update``, ``optimizer``) that older loaders ignore:

    {"model_state": state_dict, "model_config": {token_dim, d_model, ...}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from battleboats.envs.observation import TOKEN_DIM
from battleboats.training.policy import PolicyNetwork


# Pre-v0.5.0 official weights and run directories, packed 2026-11-24.
ARCHIVED_RUNS_TAR = (
    "~/Desktop/repos/battleboats_archive/runs_pre_v0.5.0_20261124.tar.zst"
)
ARCHIVED_RUNS_SHA256 = (
    "c0f63624052c78da0b48d413b23f2b15a6240f554e762adf5e70ca631db69c03"
)


def no_archived(kind: str = "checkpoint") -> str:
    """Error when a caller needed a path that used to live under runs/."""
    return f"no {kind}; old ones archived in {ARCHIVED_RUNS_TAR}"


NO_CHECKPOINT_MSG = no_archived("checkpoint")


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
        "skill_count": int(getattr(policy, "skill_count", 0)),
        "globals_dim": int(getattr(policy, "globals_dim", policy.global_proj.in_features)),
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
    """Load a self-describing checkpoint. ``model_config`` wins over any caller arch.

    Official checkpoints whose ``token_dim`` is smaller than the current
    observation width are warm-started: the first layer (``proj``) is widened
    and the new input columns are zero-initialized, so the old features
    behave as they did until training moves those weights.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in ckpt or "model_config" not in ckpt:
        raise ValueError(
            f"{path} is not a battleboats policy checkpoint "
            "(expected keys model_state and model_config)"
        )
    cfg = dict(ckpt["model_config"])
    state = dict(ckpt["model_state"])
    old_dim = int(cfg["token_dim"])
    if old_dim < TOKEN_DIM:
        cfg["token_dim"] = TOKEN_DIM
        policy = PolicyNetwork(**cfg)
        state = _widen_input_proj(state, policy, old_dim)
        print(f"[checkpoint] widened proj {old_dim} -> {TOKEN_DIM} (zero-init new columns)")
    elif old_dim > TOKEN_DIM:
        raise ValueError(
            f"{path} has token_dim={old_dim}, newer than current TOKEN_DIM={TOKEN_DIM}"
        )
    else:
        policy = PolicyNetwork(**cfg)
    current_keys = set(policy.state_dict())
    missing = [k for k in current_keys if k not in state]
    unexpected = [k for k in state if k not in current_keys]
    # Older checkpoints have no global token projection. Keep the freshly
    # initialized global_proj and load the rest.
    only_global_proj = missing and all(k.startswith("global_proj.") for k in missing) and not unexpected
    policy.load_state_dict(state, strict=not only_global_proj)
    if only_global_proj:
        print(f"[checkpoint] initialized new global_proj ({policy.globals_dim} -> {policy.d_model})")
    policy.to(device)
    return policy, cfg


def _widen_input_proj(
    state: Dict[str, Any],
    policy: PolicyNetwork,
    old_dim: int,
) -> Dict[str, Any]:
    """Copy old ``proj`` weights into a wider first layer; new columns stay 0."""
    key = "proj.weight"
    if key not in state:
        raise ValueError("checkpoint model_state is missing proj.weight")
    old_w = state[key]
    new_w = policy.proj.weight.detach().clone()
    if old_w.shape[0] != new_w.shape[0]:
        raise ValueError(
            f"proj out features {tuple(old_w.shape)} cannot widen to {tuple(new_w.shape)}"
        )
    if old_w.shape[1] != old_dim:
        raise ValueError(
            f"proj.weight width {old_w.shape[1]} does not match token_dim={old_dim}"
        )
    if new_w.shape[1] != TOKEN_DIM:
        raise ValueError(f"new proj width {new_w.shape[1]} != TOKEN_DIM={TOKEN_DIM}")
    new_w.zero_()
    new_w[:, :old_dim] = old_w
    state[key] = new_w
    return state


def frozen_copy(policy: PolicyNetwork) -> PolicyNetwork:
    """Deep-copied eval snapshot. Later steps on ``policy`` do not touch it."""
    import copy

    opp = copy.deepcopy(policy)
    opp.eval()
    for param in opp.parameters():
        param.requires_grad_(False)
    return opp
