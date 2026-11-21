"""Position encoding is board-size-agnostic, and older checkpoints warm-start."""

from pathlib import Path

import numpy as np
import pytest
import torch

from battleboats.envs.factory import make_env
from battleboats.envs.observation import (
    BOARD_SIZE_OFFSET,
    POSITION_OFFSET,
    POSITION_SCALE,
    TOKEN_DIM,
    build_entity_tokens,
)
from battleboats.envs.skill_maps import write_merchant_map
from battleboats.training.checkpoints import load_checkpoint, save_checkpoint
from battleboats.training.policy import PolicyNetwork

REPO = Path(__file__).resolve().parents[1]
V001_CANDIDATES = (
    REPO / "runs" / "checkpoints" / "v0.0.1_8f0a942.pt",
    REPO.parent / "Battleboats" / "runs" / "checkpoints" / "v0.0.1_8f0a942.pt",
)


def _tokens(tmp_path: Path, width: int, height: int) -> np.ndarray:
    path = write_merchant_map(tmp_path / f"{width}x{height}.json", width, height)
    env = make_env("full", path, max_turns=8)
    env.reset(seed=0)
    tokens = build_entity_tokens(env.engine, 0)
    env.close()
    return tokens


def _at(tokens: np.ndarray, x: int, y: int) -> np.ndarray:
    pos = tokens[:, POSITION_OFFSET : POSITION_OFFSET + 2]
    want = np.array([x / POSITION_SCALE, y / POSITION_SCALE], dtype=np.float32)
    hits = np.where(np.isclose(pos, want).all(axis=1))[0]
    assert len(hits) >= 1, f"no token at {(x, y)}"
    return tokens[hits[0]]


def test_positions_use_constant_scale_across_board_sizes(tmp_path: Path) -> None:
    small = _tokens(tmp_path, 8, 8)
    wide = _tokens(tmp_path, 16, 8)
    tall = _tokens(tmp_path, 48, 24)
    assert small.shape[-1] == wide.shape[-1] == tall.shape[-1] == TOKEN_DIM
    assert TOKEN_DIM == 30
    assert POSITION_SCALE == 256

    # Same cell, same features on every board — not x/width, which is 1/8
    # on 8x8 and 1/16 on 16x8.
    for tokens in (small, wide, tall):
        origin = _at(tokens, 0, 0)
        np.testing.assert_allclose(origin[POSITION_OFFSET : POSITION_OFFSET + 2], 0.0)
    np.testing.assert_allclose(
        _at(small, 0, 0)[POSITION_OFFSET : POSITION_OFFSET + 2],
        _at(wide, 0, 0)[POSITION_OFFSET : POSITION_OFFSET + 2],
    )

    for tokens, width, height in ((small, 8, 8), (wide, 16, 8), (tall, 48, 24)):
        satellite = _at(tokens, width - 1, 0)
        np.testing.assert_allclose(
            satellite[POSITION_OFFSET : POSITION_OFFSET + 2],
            np.array([(width - 1) / POSITION_SCALE, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(tokens[:, BOARD_SIZE_OFFSET], width / POSITION_SCALE)
        np.testing.assert_allclose(tokens[:, BOARD_SIZE_OFFSET + 1], height / POSITION_SCALE)

    # The far satellite would have been x/width = 1 on every board before.
    far_small = _at(small, 7, 0)[POSITION_OFFSET : POSITION_OFFSET + 2]
    far_wide = _at(wide, 15, 0)[POSITION_OFFSET : POSITION_OFFSET + 2]
    assert not np.allclose(far_small, far_wide)
    assert not np.allclose(
        small[0, BOARD_SIZE_OFFSET : BOARD_SIZE_OFFSET + 2],
        tall[0, BOARD_SIZE_OFFSET : BOARD_SIZE_OFFSET + 2],
    )


def test_old_checkpoint_widens_first_layer(tmp_path: Path) -> None:
    old_dim = TOKEN_DIM - 2
    old = PolicyNetwork(token_dim=old_dim, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    path = tmp_path / "old.pt"
    save_checkpoint(old, path)

    loaded, cfg = load_checkpoint(path, torch.device("cpu"))
    assert cfg["token_dim"] == TOKEN_DIM
    assert loaded.proj.in_features == TOKEN_DIM
    assert loaded.proj.weight.shape == (32, TOKEN_DIM)
    torch.testing.assert_close(loaded.proj.weight[:, :old_dim], old.proj.weight)
    assert torch.count_nonzero(loaded.proj.weight[:, old_dim:]) == 0
    torch.testing.assert_close(loaded.proj.bias, old.proj.bias)

    tokens = torch.randn(2, 5, TOKEN_DIM)
    with torch.no_grad():
        widened = loaded.proj(tokens)
        baseline = old.proj(tokens[:, :, :old_dim])
    torch.testing.assert_close(widened, baseline)


def test_v001_checkpoint_widens_when_present() -> None:
    path = next((p for p in V001_CANDIDATES if p.is_file()), None)
    if path is None:
        pytest.skip("v0.0.1 checkpoint is not on this machine")
    policy, cfg = load_checkpoint(path, torch.device("cpu"))
    assert cfg["token_dim"] == TOKEN_DIM
    assert policy.proj.in_features == TOKEN_DIM
    assert torch.count_nonzero(policy.proj.weight[:, TOKEN_DIM - 2 :]) == 0
