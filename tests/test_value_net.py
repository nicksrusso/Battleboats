"""Value-net pipeline: split integrity, cache/batch shapes, metrics, overfit, train+resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from battleboats.envs.observation import GLOBALS_DIM, TOKEN_DIM
from battleboats.training.value_data import (
    SPLITS,
    ValueCache,
    ValueDataset,
    build_value_cache,
    collate_value,
    hand_features,
    simple_features,
    split_for_seed,
    value_metrics,
)
from battleboats.training.value_eval import Baselines, evaluate_split, gate
from battleboats.training.value_net import ValueNet, count_params, load_value_checkpoint, save_value_checkpoint


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    from scripts.harvest_value import main as harvest_main

    root = tmp_path_factory.mktemp("valuenet")
    hdir = root / "harvest"
    rc = harvest_main(["--map", "16x8", "--self-play", "4", "--vs-scripted", "2", "--workers", "1",
                       "--log-every", "10", "--seed", "3", "--output-dir", str(hdir)])
    assert rc == 0
    cdir = root / "cache"
    # Wide val/test fractions so 6 games populate several splits.
    meta = build_value_cache(hdir, cdir, k_pool=12, seed=0, val_frac=0.3, test_frac=0.3, workers=1)
    return hdir, cdir, meta


def test_split_is_deterministic_and_roughly_90_5_5():
    seeds = range(20000)
    a = [split_for_seed(s) for s in seeds]
    assert a == [split_for_seed(s) for s in seeds]
    frac = {s: a.count(s) / len(a) for s in SPLITS}
    assert abs(frac["train"] - 0.90) < 0.01 and abs(frac["val"] - 0.05) < 0.01 and abs(frac["test"] - 0.05) < 0.01


def test_cache_no_game_in_two_splits(tiny_cache):
    _, cdir, meta = tiny_cache
    cache = ValueCache(cdir)
    assert meta["games"] == 6 and len(cache.games) == 6
    seen = {}
    for s in SPLITS:
        rows = cache.split_rows(s, include_draws=True)
        games = set(cache.game[rows].tolist())
        seeds = {cache.games[g]["seed"] for g in games}
        for sd in seeds:
            assert sd not in seen, f"seed {sd} in {seen[sd]} and {s}"
            seen[sd] = s
        for g in games:
            assert cache.games[g]["split"] == s == split_for_seed(cache.games[g]["seed"], 0.3, 0.3)
    assert len(seen) == 6
    # Every row belongs to exactly one split.
    total = sum(len(cache.split_rows(s, include_draws=True)) for s in SPLITS)
    assert total == len(cache)


def test_cache_shapes_and_labels_match_shards(tiny_cache):
    hdir, cdir, _ = tiny_cache
    from scripts.harvest_value import load_value_shard

    cache = ValueCache(cdir)
    assert cache.tokens.shape == (int(cache.row_splits[-1]), TOKEN_DIM)
    assert cache.globals.shape == (len(cache), GLOBALS_DIM)
    per_game = np.bincount(cache.game, minlength=len(cache.games))
    assert (per_game <= 12).all() and (per_game > 0).all()
    assert set(np.unique(cache.outcome)).issubset({-1.0, 0.0, 1.0})
    assert ((cache.turn_frac >= 0) & (cache.turn_frac <= 1)).all()
    # A cached row equals some row of its shard (same tokens, outcome, seat).
    r = 0
    g = cache.games[int(cache.game[r])]
    arrays, meta = load_value_shard(hdir / g["path"])
    rs = arrays["row_splits"]
    tok = cache.row_tokens(r)
    hits = [i for i in range(len(rs) - 1) if rs[i + 1] - rs[i] == len(tok)
            and np.array_equal(arrays["tokens_data"][rs[i]:rs[i + 1]].astype(np.float32), tok)
            and arrays["seat"][i] == cache.seat[r]]
    assert hits and arrays["outcome"][hits[0]] == cache.outcome[r] == meta["outcome_by_seat"][str(int(cache.seat[r]))]


def test_dataset_sampling_and_collate(tiny_cache):
    _, cdir, _ = tiny_cache
    cache = ValueCache(cdir)
    ds = ValueDataset(cache, "train", include_draws=True, k_per_game=3, seed=1)
    if len(ds) == 0:
        pytest.skip("hash put no game in train")
    counts = np.bincount(cache.game[ds.rows])
    assert counts.max() <= 3
    r0 = ds.rows.copy()
    ds.set_epoch(1)
    assert len(ds.rows) == len(r0)
    b = collate_value([ds[i] for i in range(len(ds))])
    B, N, D = b["tokens"].shape
    assert D == TOKEN_DIM and b["pad_mask"].shape == (B, N) and b["globals"].shape == (B, GLOBALS_DIM)
    assert b["pad_mask"].sum(1).min() > 0
    m = ValueNet(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    v, aux = m.forward_all(b["tokens"], b["pad_mask"], b["globals"])
    assert v.shape == (B,) and aux.shape == (B,)
    assert (v.abs() <= 1).all()


def test_padding_does_not_change_value():
    torch.manual_seed(0)
    m = ValueNet(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0).eval()
    t = torch.randn(1, 5, TOKEN_DIM)
    g = torch.randn(1, GLOBALS_DIM)
    v1 = m(t, torch.ones(1, 5, dtype=torch.bool), g)
    tp = torch.cat([t, torch.randn(1, 3, TOKEN_DIM) * 50], dim=1)
    mp = torch.tensor([[True] * 5 + [False] * 3])
    v2 = m(tp, mp, g)
    assert torch.allclose(v1, v2, atol=1e-5)


def test_tiny_overfit():
    torch.manual_seed(0)
    B, N = 32, 7
    t = torch.randn(B, N, TOKEN_DIM)
    g = torch.randn(B, GLOBALS_DIM)
    mask = torch.ones(B, N, dtype=torch.bool)
    y = torch.where(torch.rand(B) > 0.5, 1.0, -1.0)
    m = ValueNet(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = torch.mean((m(t, mask, g) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert value_metrics(m(t, mask, g).detach().numpy(), y.numpy())["sign_acc"] == 1.0


def test_metrics_and_gate():
    y = np.array([1, -1, 1, -1, 0], dtype=float)
    m = value_metrics(np.zeros(5), y)
    assert m["sign_acc"] == 0.0 or m["sign_acc"] <= 0.5
    assert abs(m["brier"] - 0.25) < 1e-9 and abs(m["logloss"] - np.log(2)) < 1e-6 and m["n_decisive"] == 4
    good = value_metrics(np.array([0.8, -0.8, 0.6, -0.6, 0.0]), y)
    assert good["sign_acc"] == 1.0 and good["brier"] < 0.1
    g = gate(good, {"coin_flip": m}, {})
    assert g["pass"]
    assert not gate(m, {"coin_flip": good}, {})["pass"]


def test_hand_features_fixed_width(tiny_cache):
    _, cdir, _ = tiny_cache
    cache = ValueCache(cdir)
    f = hand_features(cache.row_tokens(0), cache.globals[0], int(cache.seat[0]))
    s = simple_features(cache.row_tokens(0), cache.globals[0], int(cache.seat[0]))
    assert f.ndim == 1 and np.isfinite(f).all() and s.shape == (7,)


def test_checkpoint_roundtrip(tmp_path):
    m = ValueNet(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0).eval()
    save_value_checkpoint(tmp_path / "v.pt", m, epoch=3)
    m2, payload = load_value_checkpoint(tmp_path / "v.pt")
    t = torch.randn(2, 4, TOKEN_DIM)
    mask = torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(m(t, mask), m2(t, mask)) and payload["epoch"] == 3
    assert count_params(m) == count_params(m2) > 0


def test_train_script_runs_and_resumes(tiny_cache, tmp_path):
    from scripts.train_value import main as train_main

    hdir, cdir, _ = tiny_cache
    cfg = Path(__file__).resolve().parents[1] / "configs" / "value_net_v0.5.0.yaml"
    run = tmp_path / "run"
    common = ["--set", f"data.harvest_dir={hdir}", f"data.cache_dir={cdir}", "data.include_draws=true",
              "data.k_per_game=4", "model.d_model=16", "model.nhead=2", "model.num_layers=1", "model.dim_feedforward=32",
              "train.batch_size=8", "train.torch_threads=1", "train.num_workers=0", "train.warmup_steps=2", "train.log_every=1",
              "eval.baseline_fit_rows=500"]
    assert train_main(["--config", str(cfg), "--run-dir", str(run)] + common + ["train.epochs=1"]) == 0
    assert (run / "last.pt").exists() and (run / "metrics.csv").exists() and (run / "summary.json").exists()
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 2
    # Resume to 2 epochs: appends exactly one epoch row.
    assert train_main(["--resume", str(run), "--set", "train.epochs=2", "--skip-test"]) == 0
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 3



def test_resolve_device_cpu_and_fallback():
    from battleboats.training.value_eval import resolve_device

    assert resolve_device("cpu").type == "cpu"
    # "auto" may be cuda when a GPU is present; still a valid torch device.
    d = resolve_device("auto")
    assert d.type in ("cpu", "cuda")
    if not __import__("torch").cuda.is_available():
        assert resolve_device("cuda").type == "cpu"


def test_large_config_param_count():
    import yaml
    from battleboats.training.value_net import ValueNet, count_params

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "value_net_v0.5.0_large.yaml").read_text())
    m = ValueNet(**cfg["model"])
    n = count_params(m)
    assert n == 574210
    assert cfg["data"]["include_draws"] is False and cfg["data"]["k_pool"] == 64
    assert cfg["train"]["device"] == "cuda" and cfg["train"]["patience"] >= 1


def test_train_early_stop_and_device_cpu(tiny_cache, tmp_path):
    from scripts.train_value import main as train_main

    hdir, cdir, _ = tiny_cache
    cfg = Path(__file__).resolve().parents[1] / "configs" / "value_net_v0.5.0.yaml"
    run = tmp_path / "run_es"
    rc = train_main(
        ["--config", str(cfg), "--run-dir", str(run), "--set",
         f"data.harvest_dir={hdir}", f"data.cache_dir={cdir}", "data.include_draws=true",
         "data.k_per_game=4", "model.d_model=16", "model.nhead=2", "model.num_layers=1",
         "model.dim_feedforward=32", "train.batch_size=8", "train.torch_threads=1",
         "train.num_workers=0", "train.warmup_steps=1", "train.log_every=1",
         "train.epochs=4", "train.patience=1", "train.min_delta=1.0",
         "train.device=cpu", "eval.baseline_fit_rows=500"]
    )
    assert rc == 0
    # min_delta=1.0 means epoch 1 cannot beat epoch 0 by that much, so we stop after stale=1.
    rows = [ln for ln in (run / "metrics.csv").read_text().strip().splitlines() if ln]
    assert 2 <= len(rows) <= 3  # header + 1 or 2 epoch rows
    assert (run / "summary.json").exists()
