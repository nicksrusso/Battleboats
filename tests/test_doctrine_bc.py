"""Per-doctrine BC labels, the memory-mapped BC dataset, and the clone eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from battleboats.training.tactical_bc import CompactBCDataset
from scripts.harvest_value import main as harvest_value_main


@pytest.fixture(scope="module")
def doctrine_bc(tmp_path_factory):
    import scripts.harvest_doctrine_bc as hdb

    root = tmp_path_factory.mktemp("doctrine_bc")
    src = root / "value_harvest"
    rc = harvest_value_main(
        ["--map", "16x8", "--self-play", "3", "--workers", "1", "--log-every", "3", "--seed", "3",
         "--output-dir", str(src), "--backend", "python"]
    )
    assert rc == 0
    out = root / "bc"
    argv = sys.argv
    sys.argv = ["harvest_doctrine_bc.py", "--source", str(src), "--output-dir", str(out),
                "--games-per-pair", "1", "--workers", "1", "--verify-tokens"]
    try:
        hdb.main()
    finally:
        sys.argv = argv
    return src, out


def test_labels_cover_every_decision_and_match_replay(doctrine_bc) -> None:
    _src, out = doctrine_bc
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games_ok"] == 3 and summary["games_crash"] == 0
    assert summary["token_mismatch_rows"] == 0
    dropped = sum(summary["label_errors"].values())
    labels = sum(d["n_labels"] for d in summary["doctrines"].values())
    assert labels + dropped == summary["decisions"]
    assert sum(d["n_game_seats"] for d in summary["doctrines"].values()) == 6  # two seats per game
    for doc, d in summary["doctrines"].items():
        if not d["n_game_seats"]:
            continue
        man = json.loads((out / doc / "manifest.json").read_text())
        split = json.loads((out / doc / "split.json").read_text())
        assert sorted(split["train"] + split["val"]) == sorted(g["game_idx"] for g in man["games"] if g["n_labels"])
        for g in man["games"]:
            assert g["doctrine"] == doc
            with np.load(out / doc / g["path"]) as z:
                assert z["actions"].shape == (g["n_labels"], 3)
                assert int(z["row_splits"][-1]) == z["tokens_data"].shape[0]


def _biggest_doctrine_dir(out: Path) -> Path:
    dirs = [d for d in out.iterdir() if (d / "manifest.json").is_file()]
    return max(dirs, key=lambda d: json.loads((d / "manifest.json").read_text())["n_labels"])


def test_compact_dataset_mmap_matches_in_ram(doctrine_bc, tmp_path) -> None:
    _src, out = doctrine_bc
    doc = _biggest_doctrine_dir(out)
    a = CompactBCDataset(doc, None, 40, seed=1)
    b = CompactBCDataset(doc, None, 40, seed=1, cache_dir=tmp_path / "c")
    c = CompactBCDataset(doc, None, 40, seed=1, cache_dir=tmp_path / "c")  # reopens the cache
    assert a.n_rows_full > 40
    assert len(a) == len(b) == len(c) > 0
    for i in range(len(a)):
        for x, y, z in zip(a[i], b[i], c[i]):
            assert torch.equal(x, y) and torch.equal(y, z)
    nat = CompactBCDataset(doc, None, 30, seed=2, mode="natural")
    assert len(nat) == 30
    full = CompactBCDataset(doc, None, 0, seed=0)
    assert len(full) == full.n_rows_full


def test_eval_clone_plays_a_game(tmp_path) -> None:
    from battleboats.training.checkpoints import save_checkpoint
    from battleboats.training.tactical_bc import V0_3_0_ARCH
    from scripts.eval_doctrine_bc import build_jobs, play, summarize
    from scripts.train_tactical_bc import new_policy

    net = new_policy(dict(V0_3_0_ARCH), torch.device("cpu"))
    ckpt = tmp_path / "clone.pt"
    save_checkpoint(net, ckpt, extra={"source": "test"})
    jobs = build_jobs({"RAIDER_BLITZ": str(ckpt)}, 1, 5, "16x8", ["teacher", "random"])
    base = build_jobs({"RAIDER_BLITZ": "scripted"}, 1, 5, "16x8", ["random"])
    res = [play(j) for j in jobs + base]
    assert all(r["status"] == "ok" for r in res), res
    s = summarize(res)
    assert s["n_ok"] == 3 and "teacher" in s["clones"]["RAIDER_BLITZ"]
