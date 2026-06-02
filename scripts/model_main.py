"""Instantiate-and-forward smoke test for TrivialMLP.

Separate from train_mlp.py because this only exercises the model itself,
no data, no loss, no optimizer. Use it while implementing TrivialMLP to
verify that:

  - The constructor doesn't crash (buffer registration, Linear layers).
  - forward() accepts shape (B, F) and returns shape (B,) — NOT (B, 1).
  - The output values are in [-1, +1] (tanh at the end).
  - param count is what you expect (sanity-check you didn't accidentally
    build a 10M-param network).

Usage:
    poetry run python scripts/model_main.py
"""

from __future__ import annotations

import torch

from battleboats.agents.heuristics import FEATURE_KEYS
from battleboats.training.model import SimpleMLP


def main() -> None:
    in_features = len(FEATURE_KEYS)

    # Dummy normalization stats (no-op): zero mean, unit std. With these,
    # the model's normalize step is the identity, so any output weirdness
    # is the layers, not the stats.
    mean = torch.zeros(in_features)
    std = torch.ones(in_features)

    model = SimpleMLP(in_features=in_features, hidden=64, mean=mean, std=std)
    model.eval()  # no dropout / batchnorm here, but a good habit

    print("Model:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print()

    # Build a random batch — values in roughly [-1, 1] so we exercise
    # both positive and negative inputs to the tanh.
    B = 4
    x = torch.randn(B, in_features)
    print(f"Input  shape: {tuple(x.shape)}  dtype={x.dtype}")

    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}  dtype={y.dtype}")
    print(f"Output values: {y.tolist()}")
    print()

    # Hard-asserts: if these fail, the model is broken in a way that
    # WILL cause silent training bugs. Better to find out here.
    assert y.shape == (B,), f"Expected shape ({B},), got {tuple(y.shape)} — check squeeze(-1) in forward()."
    assert torch.all(y >= -1) and torch.all(y <= 1), "Output is outside [-1, +1] — missing tanh at the end?"
    print("All shape and range assertions passed.")


if __name__ == "__main__":
    main()
