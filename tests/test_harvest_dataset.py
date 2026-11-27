"""HarvestDataset still loads old phi shards and accepts token-only rows."""

from __future__ import annotations

import json
from pathlib import Path

from battleboats.training.dataset import FEATURE_KEYS, HarvestDataset


def test_feature_keys_are_the_frozen_historical_order() -> None:
    assert FEATURE_KEYS == (
        "material_diff",
        "home_pressure_diff",
        "combat_balance",
        "econ_value_self",
        "merchant_count_value_self",
        "has_landing_self",
        "landing_pressure_self",
    )


def _write_shard(directory: Path, rows: list[dict], winner=0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shard = directory / "game_0.jsonl"
    with shard.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps({"_type": "game_footer", "winner": winner}) + "\n")


def test_token_mode_allows_missing_phi(tmp_path: Path) -> None:
    harvest = tmp_path / "tok"
    tokens = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    _write_shard(
        harvest,
        [
            {"game_idx": 0, "step": 0, "perspective": 0, "tokens": tokens, "mcts_root_value": 0.25, "action": [0, 1, -1]},
            {"game_idx": 0, "step": 0, "perspective": 1, "tokens": tokens, "mcts_root_value": -0.25, "action": None},
        ],
    )
    ds = HarvestDataset(harvest, load_tokens=True, target_key="mcts_root_value")
    assert len(ds) == 2
    tok, target = ds[0]
    assert tuple(tok.shape) == (2, 3)
    assert float(target) == 0.25


def test_phi_mode_skips_rows_without_phi(tmp_path: Path) -> None:
    harvest = tmp_path / "noph"
    tokens = [[0.0, 1.0]]
    _write_shard(
        harvest,
        [{"game_idx": 0, "step": 0, "perspective": 0, "tokens": tokens, "mcts_root_value": 0.1}],
    )
    ds = HarvestDataset(harvest)
    assert len(ds) == 0


def test_phi_mode_loads_complete_historical_phi(tmp_path: Path) -> None:
    harvest = tmp_path / "phi"
    phi = {k: float(i) for i, k in enumerate(FEATURE_KEYS)}
    _write_shard(
        harvest,
        [
            {"game_idx": 0, "step": 0, "perspective": 0, "phi": phi},
            {"game_idx": 0, "step": 0, "perspective": 1, "phi": {k: -v for k, v in phi.items()}},
        ],
        winner=0,
    )
    ds = HarvestDataset(harvest)
    assert len(ds) == 2
    features, target = ds[0]
    assert tuple(features.shape) == (len(FEATURE_KEYS),)
    assert list(features.tolist()) == [float(i) for i in range(len(FEATURE_KEYS))]
    assert float(target) == 1.0  # perspective 0 won
