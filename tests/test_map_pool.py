"""Random map pools from generate_map.py and self-play over them."""

import json
from pathlib import Path

from battleboats.envs.map_pool import (
    PoolSize,
    build_map_pool,
    check_map,
    episode_limits,
    home_distance,
    render_ascii,
    water_path_length,
)
from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.selfplay import SelfPlayConfig, load_selfplay_config, run_selfplay

SIZES = [PoolSize(16, 8, 3, ports=2), PoolSize(32, 16, 2, ports=3), PoolSize(48, 24, 2, ports=4)]


def test_pool_is_seeded_non_square_and_checked(tmp_path: Path) -> None:
    a = build_map_pool(tmp_path / "a", SIZES, seed=7)
    b = build_map_pool(tmp_path / "b", SIZES, seed=7)
    assert len(a) == 7
    assert len(list((tmp_path / "a").glob("*.json"))) == 7
    for pa, pb in zip(a, b):
        da, db = json.loads(Path(pa).read_text()), json.loads(Path(pb).read_text())
        assert da == db
        assert check_map(da) == (True, "ok")
        assert water_path_length(da) is not None
        assert home_distance(da) >= da["width"] // 2
        rows = render_ascii(da).splitlines()
        assert len(rows) == da["height"] and all(len(r) == da["width"] for r in rows)
    sizes = sorted({(json.loads(Path(p).read_text())["width"], json.loads(Path(p).read_text())["height"]) for p in a})
    assert sizes == [(16, 8), (32, 16), (48, 24)]


def test_walled_off_home_is_rejected() -> None:
    data = {
        "width": 8,
        "height": 4,
        "land": [[0, 0], [7, 3]] + [[3, y] for y in range(4)],
        "ports": {"0": [[0, 0]], "1": [[7, 3]]},
    }
    assert check_map(data) == (False, "no water path between homes")


def test_limits_scale_with_home_distance() -> None:
    near = {"ports": {"0": [[0, 0]], "1": [[7, 7]]}}
    far = {"ports": {"0": [[0, 0]], "1": [[27, 0]]}}
    assert episode_limits(near, max_turns=80, max_steps=3000, ref_home_distance=14) == (80, 3000)
    assert episode_limits(far, max_turns=80, max_steps=3000, ref_home_distance=14) == (160, 6000)


def test_bigmaps_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_selfplay_config(root / "configs" / "selfplay_bigmaps.yaml")
    assert [(s.width, s.height) for s in cfg.map_pool] == [(16, 8), (32, 16), (48, 24)]
    assert cfg.run_dir == Path("runs/selfplay_bigmaps")


def test_one_update_on_a_pool(tmp_path: Path) -> None:
    cfg = SelfPlayConfig(
        run_dir=tmp_path / "run",
        device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=8),
        map_pool=[PoolSize(8, 4, 2, ports=1), PoolSize(12, 6, 2, ports=2)],
        updates=1,
        rollout_decisions=8,
        max_turns=4,
        max_steps=24,
        ref_home_distance=14,
        snapshot_every=1,
    )
    result = run_selfplay(cfg)
    assert result["ok"]
    row = json.loads((tmp_path / "run" / "history.jsonl").read_text().splitlines()[-1])
    assert row["update"] == 1
    assert sum(v["games"] for v in row["by_size"].values()) == row["games"]
    assert set(row["by_size"]) <= {"8x4", "12x6"}
