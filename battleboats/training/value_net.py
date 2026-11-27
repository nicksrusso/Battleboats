"""Value net for v0.5.0: entity tokens + globals -> V(s) in [-1, 1].

Reuses ``TransformerValueModel`` (token projection, padding-masked
TransformerEncoder, masked mean-pool, MLP + tanh head) and adds two things:

* a globals token (cash, counts, turn, board size), prepended exactly like
  ``PolicyNetwork.encode`` does, so attention can read board-wide state;
* an optional auxiliary head (tanh) for the discounted return
  ``outcome * gamma**(decisions_left)``, a cheap "how soon" signal.

The main head predicts the game outcome from the observing seat's point of
view (+1 win, -1 loss, 0 draw). It is the leaf evaluator for the planned
fog-determinized MCTS.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from battleboats.envs.observation import GLOBALS_DIM, TOKEN_DIM
from battleboats.training.transformer import TransformerValueModel


class ValueNet(TransformerValueModel):
    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        globals_dim: int = GLOBALS_DIM,
        aux_head: bool = True,
    ):
        super().__init__(
            token_dim=token_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.config: Dict[str, Any] = dict(
            token_dim=token_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            globals_dim=globals_dim,
            aux_head=aux_head,
        )
        self.globals_dim = int(globals_dim)
        self.global_proj = nn.Linear(self.globals_dim, d_model) if self.globals_dim > 0 else None
        self.aux_head = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)) if aux_head else None
        )

    def encode(self, tokens: torch.Tensor, pad_mask: torch.Tensor, globals: Optional[torch.Tensor] = None):
        """Returns the pooled context (B, d_model). pad_mask True = real token."""
        assert pad_mask.dtype == torch.bool
        B = tokens.shape[0]
        x = self.proj(tokens)
        mask = pad_mask
        if self.global_proj is not None:
            g = tokens.new_zeros(B, self.globals_dim) if globals is None else globals.to(tokens.dtype)
            x = torch.cat([self.global_proj(g).unsqueeze(1), x], dim=1)
            mask = torch.cat([pad_mask.new_ones((B, 1)), pad_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1)
        return (x * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

    def forward_all(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor, globals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ctx = self.encode(tokens, pad_mask, globals)
        v = torch.tanh(self.value_head(ctx)).squeeze(-1)
        aux = torch.tanh(self.aux_head(ctx)).squeeze(-1) if self.aux_head is not None else None
        return v, aux

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor, globals: Optional[torch.Tensor] = None):
        return self.forward_all(tokens, pad_mask, globals)[0]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_value_checkpoint(path, model: ValueNet, **extra) -> None:
    payload = {"model_state": model.state_dict(), "model_config": dict(model.config), "kind": "value_net"}
    payload.update(extra)
    torch.save(payload, str(path))


def load_value_checkpoint(path, map_location="cpu") -> Tuple[ValueNet, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    model = ValueNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


@torch.no_grad()
def value_of_observations(model: ValueNet, entity_tokens_list, globals_list) -> list:
    """Batched V(s) for ``build_observation`` outputs (MCTS leaves).

    Pads variable-length token sets to the batch max. Tensors are placed on
    the model's device. A one-observation call is just a batch of size 1;
    vanilla UCT evaluates one leaf per iteration, but the batched path is
    here so a later parallel/batch search can amortize the forward.
    """
    if len(entity_tokens_list) != len(globals_list):
        raise ValueError("entity_tokens_list and globals_list must have the same length")
    if not entity_tokens_list:
        return []
    device = next(model.parameters()).device
    tok_arrays = [
        torch.as_tensor(tok, dtype=torch.float32, device=device) for tok in entity_tokens_list
    ]
    for i, arr in enumerate(tok_arrays):
        if arr.ndim != 2:
            raise ValueError(f"entity tokens[{i}] must have shape (N, TOKEN_DIM), got {tuple(arr.shape)}")
    max_n = max(arr.shape[0] for arr in tok_arrays)
    if max_n == 0:
        max_n = 1
    token_dim = tok_arrays[0].shape[1]
    B = len(tok_arrays)
    tokens = torch.zeros(B, max_n, token_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
    for i, arr in enumerate(tok_arrays):
        n = arr.shape[0]
        if n == 0:
            continue
        tokens[i, :n] = arr
        mask[i, :n] = True
    globals_t = torch.stack(
        [torch.as_tensor(g, dtype=torch.float32, device=device) for g in globals_list],
        dim=0,
    )
    return model(tokens, mask, globals_t).detach().cpu().tolist()


@torch.no_grad()
def value_of_observation(model: ValueNet, entity_tokens, globals_vec) -> float:
    """Single-position V(s) for a ``build_observation`` output (MCTS leaf)."""
    return float(value_of_observations(model, [entity_tokens], [globals_vec])[0])
