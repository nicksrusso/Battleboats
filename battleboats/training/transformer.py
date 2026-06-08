"""Transformer encoder + value head — variable-length entity tokens -> scalar value.

The drop-in successor to `SimpleMLP` (see `model.py`) for the SAME supervised
task: regress `mcts_root_value` in [-1, +1]. The only thing that changes is the
INPUT representation:

    SimpleMLP:           phi          (F,)            one hand-aggregated vector
    TransformerEncoder:  entity tokens (N, TOKEN_DIM)  one token per ship/port/sighting

The encoder lets each entity attend to every other entity, then we pool the
contextualized tokens into one vector and feed a value head — mirroring
`policy_architecture.md` §2 (encoder) and §6 (value head). We are building ONLY
the encoder + value head here, NOT the action heads. Smallest-thing-first, same
as the MLP: prove an attention encoder beats hand-engineered phi (val MSE 0.033)
before touching PPO / pointer heads.

THE ONE NEW MECHANIC: variable cardinality.
Different boards have different entity counts (N), so a batch is a ragged stack.
The Dataset/collate must pad each batch to the max N and produce a boolean
`pad_mask` (True = real token, False = padding). Two places must honor it:
  1. self-attention — pass `src_key_padding_mask` so real tokens never attend to
     padding (otherwise garbage padding rows leak into every embedding).
  2. pooling — a plain mean over N would average in the padding; pool over real
     tokens only.
Getting the mask polarity wrong is the classic silent bug here — torch's
`src_key_padding_mask` wants True = "ignore this position", which is the INVERSE
of our pad_mask. Flip it explicitly and it's worth an assert.

The encoder is permutation-invariant by design: entities are a SET, so there are
NO positional encodings on the token axis (unlike a language transformer). Token
order carries no meaning.
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerValueModel(nn.Module):
    """entity tokens (B, N, TOKEN_DIM) + pad_mask (B, N) -> value (B,) in [-1, 1].

    Architecture (fill in __init__ and forward below):
        tokens -> Linear projection to d_model
               -> TransformerEncoder (self-attention x num_layers), padding-masked
               -> masked mean-pool over the N axis  -> (B, d_model)
               -> value head MLP -> tanh            -> (B,)

    tanh bounds the output to the label range [-1, 1], same as SimpleMLP. The
    mcts_root_value labels sit in ~[-0.99, 0.99], so tanh fits without the
    saturation trap (don't stack a ReLU before it — that was the SimpleMLP bug).
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.proj = nn.Linear(token_dim, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, TOKEN_DIM)   pad_mask: (B, N) bool, True = real token.

        Returns (B,), NOT (B, 1) — same shape contract as SimpleMLP, so MSELoss
        against the (B,) targets doesn't silently broadcast. squeeze(-1) at the end.
        """

        x = self.proj(tokens)

        key_padding_mask = ~pad_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        m = pad_mask.unsqueeze(-1)
        summed = (x * m).sum(dim=1)
        counts = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = summed / counts
        out = self.value_head(pooled)

        return torch.tanh(out).squeeze(-1)
