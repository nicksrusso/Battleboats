"""Tactical behavior-cloning: action encoding, packed harvest, dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.envs.observation import (
    GLOBALS_DIM,
    TOKEN_DIM,
    build_entity_tokens,
    build_globals,
)
from battleboats.envs.skill_maps import write_full_map, write_two_home_map
from battleboats.training.checkpoints import load_checkpoint, save_checkpoint
from battleboats.training.policy import PolicyNetwork
from battleboats.training.tactical_bc import (
    V0_3_0_ARCH,
    TacticalBCDataset,
    collate_tactical_bc,
    discounted_returns,
    encode_tactical_action,
    write_game_npz,
    write_split,
)


def test_discounted_returns_terminal():
    r = discounted_returns(3, 1.0, gamma=0.5)
    assert r.shape == (3,)
    assert abs(float(r[2]) - 1.0) < 1e-6
    assert abs(float(r[1]) - 0.5) < 1e-6
    assert abs(float(r[0]) - 0.25) < 1e-6
    r0 = discounted_returns(2, -1.0, gamma=1.0)
    assert float(r0[0]) == -1.0 and float(r0[1]) == -1.0
    assert discounted_returns(0, 1.0).shape == (0,)


def test_encode_endturn_and_build_is_legal(tmp_path: Path) -> None:
    engine = gameEngine(write_two_home_map(tmp_path / "m.json", 6, 4))
    engine.reset(seed=0)
    engine.players[0].cash = 250
    legal = engine.enumerate_legal(0)
    end = next(a for a in legal if isinstance(a, EndTurnAction))
    encoded = encode_tactical_action(engine, 0, end)
    assert encoded.ok, encoded.error
    assert encoded.verb == "endturn"
    assert encoded.action_triple[2] == -1
    assert encoded.tokens.shape[1] == TOKEN_DIM
    assert encoded.globals.shape == (GLOBALS_DIM,)

    # A legal buy must factor as build_ship on a friendly port token.
    from battleboats.core.actions import BuildShipAction

    buys = [a for a in legal if isinstance(a, BuildShipAction)]
    assert buys
    encoded = encode_tactical_action(engine, 0, buys[0])
    assert encoded.ok, encoded.error
    assert encoded.verb == "build_ship"


def test_tactical_selfplay_labels_are_legal_and_encodable(tmp_path: Path) -> None:
    """Every tactical action in a short game is legal under the policy mask."""
    engine = gameEngine(write_full_map(tmp_path / "f.json", 8, 6, seed=1))
    engine.reset(seed=1)
    rngs = {0: random.Random(1), 1: random.Random(2)}
    n = 0
    errors = []
    verbs = set()
    while engine.winner is None and engine.turn < 6 and n < 80:
        pid = engine.current_player
        engine._refresh_sightings()
        legal = engine.enumerate_legal(pid)
        action = tactical_scripted_action(engine, pid, rngs[pid])
        assert action in legal
        encoded = encode_tactical_action(engine, pid, action)
        if not encoded.ok:
            errors.append((pid, engine.turn, type(action).__name__, encoded.error))
        else:
            verbs.add(encoded.verb)
            # Observation must be the policy's own tensors.
            tokens = build_entity_tokens(engine, pid)
            globals_vec = build_globals(engine, pid)
            assert np.allclose(encoded.tokens, tokens)
            assert np.allclose(encoded.globals, globals_vec)
        engine.step(action)
        n += 1
    assert n > 0
    assert not errors, errors
    assert "endturn" in verbs or "build_ship" in verbs


def test_packed_dataset_roundtrip(tmp_path: Path) -> None:
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    # Two tiny games, 2 and 1 rows.
    rows0 = [
        {
            "tokens": np.ones((3, TOKEN_DIM), dtype=np.float32),
            "globals": np.arange(GLOBALS_DIM, dtype=np.float32),
            "action": [0, 7, -1],  # endturn
            "verb": "endturn",
            "return": 0.5,
        },
        {
            "tokens": np.full((5, TOKEN_DIM), 2.0, dtype=np.float32),
            "globals": np.zeros((GLOBALS_DIM,), dtype=np.float32),
            "action": [1, 0, 2],  # move
            "verb": "move",
            "return": 1.0,
        },
    ]
    rows1 = [
        {
            "tokens": np.full((2, TOKEN_DIM), 3.0, dtype=np.float32),
            "globals": np.ones((GLOBALS_DIM,), dtype=np.float32),
            "action": [0, 2, 3],
            "verb": "build_ship",
            "return": -1.0,
        }
    ]
    write_game_npz(
        harvest / "game_00000.npz",
        rows0,
        {"game_idx": 0, "n_labels": 2, "kind": "self_play"},
    )
    write_game_npz(
        harvest / "game_00001.npz",
        rows1,
        {"game_idx": 1, "n_labels": 1, "kind": "self_play"},
    )
    manifest = {
        "games": [
            {"game_idx": 0, "path": "game_00000.npz", "n_labels": 2},
            {"game_idx": 1, "path": "game_00001.npz", "n_labels": 1},
        ]
    }
    (harvest / "manifest.json").write_text(json.dumps(manifest))
    write_split(harvest, seed=0, val_frac=0.5)
    ds = TacticalBCDataset(harvest)
    assert len(ds) == 3
    tokens, globals_t, action, ret = ds[0]
    assert tokens.shape == (3, TOKEN_DIM)
    assert globals_t.shape == (GLOBALS_DIM,)
    assert action.tolist() == [0, 7, -1]
    assert abs(float(ret) - 0.5) < 1e-6
    batch = collate_tactical_bc([ds[0], ds[1], ds[2]])
    tokens_b, pad, glob, acts, rets = batch
    assert tokens_b.shape[0] == 3
    assert tokens_b.shape[2] == TOKEN_DIM
    assert pad[0].sum() == 3 and pad[1].sum() == 5 and pad[2].sum() == 2
    assert acts.shape == (3, 3)
    # Split should keep both games across train/val (val_frac 0.5, 2 games).
    split = json.loads((harvest / "split.json").read_text())
    assert split["n_train"] + split["n_val"] == 2


def test_bc_checkpoint_roundtrips_load_checkpoint(tmp_path: Path) -> None:
    """Official save/load format: PPO can warm-start, value head is present."""
    cfg = dict(V0_3_0_ARCH)
    net = PolicyNetwork(**cfg)
    path = tmp_path / "bc.pt"
    save_checkpoint(net, path, extra={"source": "tactical_bc"})
    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    assert "value_head.0.weight" in net.state_dict()
    assert "value_head.0.weight" in loaded.state_dict()
    assert loaded.skill_count == 0
    assert loaded.globals_dim == GLOBALS_DIM
    # A forward value pass works.
    tokens = torch.zeros(1, 4, TOKEN_DIM)
    pad = torch.ones(1, 4, dtype=torch.bool)
    globs = torch.zeros(1, GLOBALS_DIM)
    emb, ctx = loaded.encode(tokens, pad, globs)
    val = loaded.value(ctx)
    assert val.shape == (1,)
    v = val.detach()
    assert float(v.clamp(-1, 1)[0]) == float(v[0])
