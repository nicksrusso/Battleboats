from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """phi -> scalar value in [-1, +1].

    Architecture:
        x  ->  (x - mean) / std        # frozen input normalization
            ->  Linear(F, hidden) -> ReLU
            ->  Linear(hidden, hidden) -> ReLU
            ->  Linear(hidden, 1) -> tanh   # bound output to label range

    `mean` and `std` are stored as non-learnable buffers so they:
      - move to the right device with model.to(device)
      - get saved/loaded as part of state_dict()
      - are excluded from optimizer.param_groups (they're frozen stats,
        not parameters)
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if mean is None:
            mean = torch.zeros(in_features)
        if std is None:
            std = torch.ones(in_features)

        if mean.shape[0] != in_features:
            raise ValueError("Mean has wrong shape")

        if std.shape[0] != in_features:
            raise ValueError("STD has the wrong shape")

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: shape (B, F) -> shape (B,) scalar value per batch element.

        Make sure the returned tensor is shape (B,), NOT (B, 1) — the
        MSELoss target from the Dataset is shape (B,), and shape mismatch
        gives confusing broadcasting bugs instead of a clean error.
        """

        x = (x - self.mean) / self.std
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.tanh(x)

        return x.squeeze(-1)


def compute_normalization_stats(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std over a stack of phi vectors.

    Caller responsibility: pass in ONLY the training split's phi tensor.
    Computing stats over train + val leaks val into the normalization,
    which contaminates every "held-out" number downstream.

    Returns (mean[F], std[F]). Clamps std to a small floor so constant
    features (std=0) don't blow up in the model's forward pass.
    """

    mean = phi.mean(dim=0)
    std = phi.std(dim=0)
    std = std.clamp(min=1e-6)
    return mean, std
