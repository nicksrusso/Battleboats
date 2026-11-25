"""End-to-end checks for scripts/harvest_value.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from battleboats.envs.observation import GLOBALS_DIM, TOKEN_DIM, build_observation
from scripts.harvest_value import (
    OBS_ENCODING_VERSION,
    load_value_shard,
    main as harvest_main,
    replay_shard,
)


def test_harvest_value_records_both_seats_outcomes_and_replay(tmp_path: Path) -> None:
    out_dir = tmp_path / "harvest"
    rc = harvest_main(
        [
            "--map",
            "16x8",
            "--self-play",
            "2",
            "--vs-scripted",
            "2",
            "--workers",
            "1",
            "--log-every",
            "2",
            "--seed",
            "7",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    split = json.loads((out_dir / "split.json").read_text())
    assert manifest["n_games"] == 4
    assert manifest["n_ok"] == 4
    assert manifest["n_crash"] == 0
    assert manifest["illegal"] == 0
    assert manifest["obs_encoding"] == OBS_ENCODING_VERSION
    assert split["n_train"] + split["n_val"] == 4
    assert manifest["by_kind"]["self_play"]["games"] == 2
    assert manifest["by_kind"]["vs_scripted"]["games"] == 2

    # Check mixed seats in vs_scripted
    vs_games = [g for g in manifest["games"] if g["kind"] == "vs_scripted"]
    assert vs_games[0]["actors"] == {"0": "fog_commander", "1": "scripted"}
    assert vs_games[1]["actors"] == {"0": "scripted", "1": "fog_commander"}

    # Check shard contents and replay roundtrip
    for rec in manifest["games"]:
        shard = out_dir / rec["path"]
        arrays, meta = load_value_shard(shard)
        n_dec = meta["num_decisions"]
        n_rows = meta["num_rows"]
        assert n_rows == 2 * n_dec
        assert arrays["tokens_data"].shape[1] == TOKEN_DIM
        assert arrays["tokens_data"].dtype == np.float16
        assert arrays["row_splits"].shape == (n_rows + 1,)
        assert int(arrays["row_splits"][-1]) == arrays["tokens_data"].shape[0]
        assert arrays["globals"].shape == (n_rows, GLOBALS_DIM)
        assert list(arrays["seat"][:4]) == [0, 1, 0, 1]

        winner = meta["winner"]
        for s in (0, 1):
            expected = 0.0 if winner is None else (1.0 if winner == s else -1.0)
            mask = arrays["seat"] == s
            assert np.all(arrays["outcome"][mask] == expected)
            if n_dec > 0:
                last_idx = np.where(mask)[0][-1]
                assert arrays["returns"][last_idx] == expected

        eng, _ = replay_shard(shard)
        assert eng.winner == meta["winner"]
        assert eng.turn == meta["turns"]
        # Final observation matches current observation.py (post-fix)
        for s in (0, 1):
            obs = build_observation(eng, s)
            assert obs["entity_tokens"].shape[1] == TOKEN_DIM
