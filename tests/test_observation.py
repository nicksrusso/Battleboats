"""Sinusoidal position encoding is board-size-agnostic, and older checkpoints warm-start."""

from pathlib import Path

import numpy as np
import pytest
import torch

from battleboats.envs.factory import make_env
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.observation import (
    BOARD_SIZE_OFFSET,
    CASH_SCALE,
    GLOBALS_DIM,
    HOME_REL_FREQS,
    HOME_REL_OFFSET,
    HOME_REL_WAVELENGTHS,
    MAX_TURNS_SCALE,
    POSITION_FREQS,
    POSITION_MAX_WAVELENGTH,
    POSITION_MIN_WAVELENGTH,
    POSITION_OFFSET,
    POSITION_SCALE,
    POSITION_WAVELENGTHS,
    TOKEN_DIM,
    TOKEN_HOME_REL_DIM,
    TOKEN_POSITION_DIM,
    _write_position,
    build_entity_refs,
    build_entity_tokens,
    build_globals,
)
from battleboats.envs.skill_maps import write_merchant_map
from battleboats.training.checkpoints import load_checkpoint, save_checkpoint
from battleboats.training.policy import NUM_VERBS, PolicyNetwork

REPO = Path(__file__).resolve().parents[1]
V001_CANDIDATES = (
    REPO / "runs" / "checkpoints" / "v0.0.1_8f0a942.pt",
    REPO.parent / "Battleboats" / "runs" / "checkpoints" / "v0.0.1_8f0a942.pt",
)


def _tokens(tmp_path: Path, width: int, height: int):
    path = write_merchant_map(tmp_path / f"{width}x{height}.json", width, height)
    env = make_env("full", path, max_turns=8)
    env.reset(seed=0)
    tokens = build_entity_tokens(env.engine, 0)
    refs = build_entity_refs(env.engine, 0)
    home = env.engine.players[0].home_port
    env.close()
    return tokens, refs, home


def _port_token(tokens: np.ndarray, refs, pos: tuple) -> np.ndarray:
    for i, (kind, ident) in enumerate(refs):
        if kind in ("friendly_port", "enemy_port", "coastline") and ident == pos:
            return tokens[i]
    raise AssertionError(f"no port/coast token at {pos} in {refs}")


def _expected(x: float, y: float, home=(0, 0)) -> np.ndarray:
    row = np.zeros(TOKEN_DIM, dtype=np.float32)
    _write_position(row, x, y, home)
    return row


def test_wavelengths_are_geometric_from_2_to_4096() -> None:
    assert POSITION_FREQS == 8
    assert HOME_REL_FREQS == 4
    assert POSITION_WAVELENGTHS[0] == pytest.approx(POSITION_MIN_WAVELENGTH)
    assert POSITION_WAVELENGTHS[-1] == pytest.approx(POSITION_MAX_WAVELENGTH)
    assert HOME_REL_WAVELENGTHS[0] == pytest.approx(POSITION_MIN_WAVELENGTH)
    assert HOME_REL_WAVELENGTHS[-1] == pytest.approx(POSITION_MAX_WAVELENGTH)
    assert POSITION_MIN_WAVELENGTH == 2.0
    assert POSITION_MAX_WAVELENGTH == 4096.0
    ratios = [
        POSITION_WAVELENGTHS[i + 1] / POSITION_WAVELENGTHS[i]
        for i in range(len(POSITION_WAVELENGTHS) - 1)
    ]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)


def test_fourier_positions_are_board_size_agnostic(tmp_path: Path) -> None:
    small, small_refs, small_home = _tokens(tmp_path, 8, 8)
    wide, wide_refs, wide_home = _tokens(tmp_path, 16, 8)
    tall, tall_refs, tall_home = _tokens(tmp_path, 48, 24)
    assert small.shape[-1] == wide.shape[-1] == tall.shape[-1] == TOKEN_DIM
    assert TOKEN_DIM == 76
    assert TOKEN_POSITION_DIM == 32
    assert TOKEN_HOME_REL_DIM == 16
    assert POSITION_SCALE == 256
    assert small_home == wide_home == tall_home == (0, 0)

    origin_s = _port_token(small, small_refs, (0, 0))
    origin_w = _port_token(wide, wide_refs, (0, 0))
    origin_t = _port_token(tall, tall_refs, (0, 0))
    expected_origin = _expected(0, 0, (0, 0))
    # The same cell has the same absolute Fourier features on every board.
    for row in (origin_s, origin_w, origin_t):
        np.testing.assert_allclose(
            row[POSITION_OFFSET : POSITION_OFFSET + TOKEN_POSITION_DIM],
            expected_origin[POSITION_OFFSET : POSITION_OFFSET + TOKEN_POSITION_DIM],
            atol=1e-6,
        )
        # At own home, home-relative coords are 0: sin=0, cos=1 on every freq.
        home_rel = row[HOME_REL_OFFSET : HOME_REL_OFFSET + TOKEN_HOME_REL_DIM]
        np.testing.assert_allclose(home_rel[0::2], 0.0, atol=1e-6)
        np.testing.assert_allclose(home_rel[1::2], 1.0, atol=1e-6)

    # Adjacent cells differ a lot on the shortest wavelength (2 cells),
    # not by 1/256 ~= 0.004 as the previous linear scale did.
    adj = _expected(1, 0, (0, 0))
    origin = expected_origin
    # x-axis, shortest freq: (sin, cos) at λ=2. x=0 -> (0, 1); x=1 -> (0, -1).
    np.testing.assert_allclose(origin[POSITION_OFFSET : POSITION_OFFSET + 2], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(adj[POSITION_OFFSET : POSITION_OFFSET + 2], [0.0, -1.0], atol=1e-6)
    # Home-relative offset of +1 cell is the same on every board size.
    one_east = adj[HOME_REL_OFFSET : HOME_REL_OFFSET + TOKEN_HOME_REL_DIM]
    for tokens, refs, width in ((small, small_refs, 8), (wide, wide_refs, 16), (tall, tall_refs, 48)):
        satellite = _port_token(tokens, refs, (width - 1, 0))
        # Board-size features stay width/256, height/256.
        height = {8: 8, 16: 8, 48: 24}[width]
        np.testing.assert_allclose(tokens[:, BOARD_SIZE_OFFSET], width / POSITION_SCALE)
        np.testing.assert_allclose(tokens[:, BOARD_SIZE_OFFSET + 1], height / POSITION_SCALE)
        # Far satellites are different distances from home, so home-rel differs.
        sat_rel = satellite[HOME_REL_OFFSET : HOME_REL_OFFSET + TOKEN_HOME_REL_DIM]
        if width == 8:
            far_rel_small = sat_rel
        else:
            assert not np.allclose(sat_rel, far_rel_small)

    # A +1 cell home-relative encoding does not depend on board size.
    row = np.zeros(TOKEN_DIM, dtype=np.float32)
    _write_position(row, 1, 0, (0, 0))
    np.testing.assert_allclose(
        row[HOME_REL_OFFSET : HOME_REL_OFFSET + TOKEN_HOME_REL_DIM], one_east
    )
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


def test_build_globals_include_cash_turn_and_board_size(tmp_path: Path) -> None:
    path = write_merchant_map(tmp_path / "g.json", 16, 8)
    env = make_env("full", path, max_turns=8)
    env.reset(seed=0)
    g = build_globals(env.engine, 0)
    assert g.shape == (GLOBALS_DIM,)
    assert GLOBALS_DIM == 8
    assert g.dtype == np.float32
    np.testing.assert_allclose(g[0], env.engine.players[0].cash / CASH_SCALE)
    np.testing.assert_allclose(g[5], env.engine.turn / MAX_TURNS_SCALE)
    np.testing.assert_allclose(g[6], 16 / POSITION_SCALE)
    np.testing.assert_allclose(g[7], 8 / POSITION_SCALE)
    # Wider board changes only the size slots, not cash or turn at reset.
    wide = write_merchant_map(tmp_path / "wide.json", 48, 24)
    other = make_env("full", wide, max_turns=8)
    other.reset(seed=0)
    gw = build_globals(other.engine, 0)
    np.testing.assert_allclose(gw[6], 48 / POSITION_SCALE)
    np.testing.assert_allclose(gw[7], 24 / POSITION_SCALE)
    assert not np.allclose(g[6:8], gw[6:8])
    env.close()
    other.close()


def test_policy_global_token_keeps_entity_indices(tmp_path: Path) -> None:
    path = write_merchant_map(tmp_path / "idx.json", 8, 8)
    env = make_env("full", path, max_turns=8)
    env.reset(seed=0)
    masks = ActionMasks(env.engine, 0)
    policy = PolicyNetwork(
        token_dim=TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0
    )
    policy.eval()
    assert masks.globals.shape == (1, GLOBALS_DIM)
    with torch.no_grad():
        emb, ctx = policy.encode(masks.tokens, masks.pad_mask, masks.globals)
    assert emb.shape == (1, masks.n, 32)
    assert emb.shape[1] == masks.tokens.shape[1]
    assert 0 <= int(masks.asset.float().argmax()) < emb.shape[1]
    # Pointer-legal assets stay in entity-index range after the global token is prepended.
    legal = torch.where(masks.asset[0])[0]
    assert legal.numel() > 0
    assert int(legal.max()) < emb.shape[1]
    env.close()


def test_verb_logits_change_with_cash() -> None:
    torch.manual_seed(0)
    policy = PolicyNetwork(
        token_dim=TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0
    )
    policy.eval()
    tokens = torch.randn(1, 4, TOKEN_DIM)
    pad = torch.ones(1, 4, dtype=torch.bool)
    low = torch.zeros(1, GLOBALS_DIM)
    high = torch.zeros(1, GLOBALS_DIM)
    low[0, 0] = 50.0 / CASH_SCALE
    high[0, 0] = 250.0 / CASH_SCALE
    verb_mask = torch.ones(1, NUM_VERBS, dtype=torch.bool)
    with torch.no_grad():
        emb_lo, ctx_lo = policy.encode(tokens, pad, low)
        emb_hi, ctx_hi = policy.encode(tokens, pad, high)
        assert not torch.allclose(ctx_lo, ctx_hi)
        logits_lo = policy.head_verb(torch.cat([ctx_lo, emb_lo[:, 0]], dim=-1), verb_mask)
        logits_hi = policy.head_verb(torch.cat([ctx_hi, emb_hi[:, 0]], dim=-1), verb_mask)
    assert not torch.allclose(logits_lo, logits_hi)


def test_old_checkpoint_without_global_proj_loads(tmp_path: Path) -> None:
    policy = PolicyNetwork(
        token_dim=TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0
    )
    path = tmp_path / "no_global.pt"
    save_checkpoint(policy, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    for key in list(state):
        if key.startswith("global_proj."):
            del state[key]
    payload["model_config"].pop("globals_dim", None)
    torch.save(payload, path)

    loaded, cfg = load_checkpoint(path, torch.device("cpu"))
    assert "global_proj.weight" in loaded.state_dict()
    assert loaded.global_proj.in_features == GLOBALS_DIM
    torch.testing.assert_close(loaded.proj.weight, policy.proj.weight)
