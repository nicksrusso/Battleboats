"""Torch device selection that works on CPU and on ROCm.

ROCm PyTorch exposes the AMD GPU through the CUDA API: ``torch.cuda.is_available()``
is true and tensors live on ``torch.device("cuda")``. There is no ``nvidia-smi``
and no separate ``torch.device("rocm")``. ``auto`` uses that GPU when the
installed wheel can see it, and otherwise CPU.
"""

from __future__ import annotations

import torch


def resolve_device(name: str | None = None) -> torch.device:
    """Return a ``torch.device`` for ``cpu``, ``cuda``, or ``auto``.

    ``auto`` (the default) picks ``cuda`` when ``torch.cuda.is_available()`` —
    which is how a ROCm build reports an RX 7700 XT / 7800 XT — and CPU
    otherwise. ``cuda`` raises if the wheel cannot see a device, so a typo
    does not silently train on CPU. ``cpu`` is always CPU.
    """
    choice = (name or "auto").strip().lower()
    if choice in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" or choice.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {choice!r} was requested but torch.cuda.is_available() is False. "
                "On the RX 7700 XT / 7800 XT, install the ROCm wheel and, if the GPU is "
                "still hidden, export HSA_OVERRIDE_GFX_VERSION=11.0.0 (gfx1101)."
            )
        return torch.device(choice)
    raise ValueError(f"unknown device {name!r}; expected cpu, cuda, cuda:N, or auto")
