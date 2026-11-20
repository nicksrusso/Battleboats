"""Device selection: CPU in this environment, ROCm-as-cuda on Nick's box."""

import pytest
import torch

from battleboats.training.device import resolve_device


def test_cpu_is_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_auto_matches_cuda_availability() -> None:
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_blank_is_auto() -> None:
    assert resolve_device(None).type == resolve_device("auto").type
    assert resolve_device("").type == resolve_device("auto").type


def test_cuda_requested_when_missing() -> None:
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
        return
    with pytest.raises(RuntimeError, match="HSA_OVERRIDE_GFX_VERSION"):
        resolve_device("cuda")


def test_unknown_device() -> None:
    with pytest.raises(ValueError):
        resolve_device("rocm")
