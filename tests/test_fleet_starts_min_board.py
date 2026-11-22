"""Fleet starts are skipped on boards below fleet_starts.min_width x min_height."""

import json
import random
from pathlib import Path

import pytest

from battleboats.envs import observation as obs
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.fleet_starts import FleetStarts, apply_fleet_start
from battleboats.training.policy import PolicyNetwork
from battleboats.training.vec_collect import GameCollector

ROOT = Path(__file__).resolve().parents[1]
ALWAYS = dict(prob=1.0, budget=1200, landing_prob=1.0, radius=6, landing_forward_prob=0.5)


def _pool(tmp_path, sizes):
    maps = build_map_pool(tmp_path / "maps", list(sizes), seed=3, land_fraction=0.02)
    info = []
    for m in maps:
        data = json.loads(Path(m).read_text())
        t, s = episode_limits(data, max_turns=6, max_steps=80, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    return maps, info


def _engine(path, seed=0):
    env = BattleboatsAEC(path, max_turns=50)
    env.reset(seed=seed)
    return env.engine


def test_defaults_and_allows():
    fs = FleetStarts()
    assert (fs.min_width, fs.min_height) == (16, 8)
    assert not fs.allows(12, 6) and not fs.allows(16, 6) and not fs.allows(12, 8)
    assert fs.allows(16, 8) and fs.allows(32, 16) and fs.allows(48, 24)
    raw = FleetStarts.from_dict({"prob": 0.5, "min_width": 10, "min_height": 5})
    assert (raw.min_width, raw.min_height) == (10, 5) and raw.allows(12, 6)
    assert FleetStarts.from_dict({"prob": 0.5}).min_width == 16
    with pytest.raises(ValueError):
        FleetStarts.from_dict({"min_width": 0})


def test_small_board_gets_normal_start(tmp_path):
    maps, _ = _pool(tmp_path, [PoolSize(12, 6, 3, ports=2)])
    for i in range(30):
        eng = _engine(maps[i % len(maps)], seed=i)
        before = [len(p.owned_ship_ids) for p in eng.players]
        assert apply_fleet_start(eng, FleetStarts(**ALWAYS), random.Random(i)) is None
        assert eng.fleet_start_forward is False
        assert [len(p.owned_ship_ids) for p in eng.players] == before
    # Lowering the threshold turns them back on for 12x6.
    eng = _engine(maps[0])
    assert apply_fleet_start(eng, FleetStarts(**ALWAYS, min_width=12, min_height=6), random.Random(0)) is not None


@pytest.mark.parametrize("size", [PoolSize(16, 8, 1, ports=2), PoolSize(32, 16, 1, ports=3), PoolSize(48, 24, 1, ports=4)])
def test_big_boards_unchanged(tmp_path, size):
    # Same rng -> same fleet with or without the (default) threshold.
    maps, _ = _pool(tmp_path, [size])
    for i in range(10):
        a = _engine(maps[0], seed=i)
        b = _engine(maps[0], seed=i)
        fa = apply_fleet_start(a, FleetStarts(**ALWAYS), random.Random(i))
        fb = apply_fleet_start(b, FleetStarts(**ALWAYS, min_width=1, min_height=1), random.Random(i))
        assert fa == fb and fa is not None
        assert a.fleet_start_forward == b.fleet_start_forward
        pos = lambda e: sorted((e.ships[s].type.name, e.ships[s].position) for p in e.players for s in p.owned_ship_ids)
        assert pos(a) == pos(b)


def test_collector_mixed_pool_skips_only_small(tmp_path):
    maps, info = _pool(tmp_path, [PoolSize(12, 6, 2, ports=2), PoolSize(16, 8, 2, ports=2)])
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    col = GameCollector(maps, info, model_config=model_config_of(net), seed=0, games=1, fleet_starts=FleetStarts(**ALWAYS))
    col._make_slots()
    slot = col.slots[0]
    seen = {"12x6": 0, "16x8": 0}
    for _ in range(40):
        col._start(slot)
        size = slot.size
        seen[size] += 1
        assert slot.fleet_start == (size == "16x8")
        if size == "12x6":
            assert not slot.fleet_forward
    assert seen["12x6"] > 0 and seen["16x8"] > 0


def test_config_and_log(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, load_selfplay_config, run_selfplay

    cfg = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets_landing.yaml")
    assert (cfg.fleet_starts.min_width, cfg.fleet_starts.min_height) == (16, 8)
    run = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=32),
        map_pool=[PoolSize(12, 6, 2, ports=2)], updates=2, rollout_decisions=150,
        max_turns=6, max_steps=120, num_workers=0, games_per_worker=3, ppo_threads=2,
        fleet_starts=FleetStarts(**ALWAYS),
    )
    assert run_selfplay(run)["ok"]
    out = capsys.readouterr().out
    assert "min_board=16x8" in out and "fleet starts OFF on ['12x6']" in out
    for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines():
        assert set(json.loads(line)["by_start"]) <= {"normal"}
