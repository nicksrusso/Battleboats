"""map_gen.min_port_enemy_home_water: every port's spawn tile (home and
satellites) must be at least N water moves from a tile next to the enemy home,
so a landing ship bought on turn 0 cannot rush the enemy home on turn 0/1."""

import json
from pathlib import Path

import pytest
import yaml

from battleboats.core.config.generate_map import generate_map_json, port_enemy_home_water
from battleboats.envs.map_pool import (
    PoolSize,
    build_map_pool,
    check_map,
    earliest_capture_turn,
    spawn_enemy_home_water,
    water_path_length,
)

ROOT = Path(__file__).resolve().parents[1]


def _yaml(tmp_path, w, h, ports=2, extra=None):
    m = {"sizeX": w, "sizeY": h, "landFraction": 0.02, "numPorts": ports, **(extra or {})}
    p = tmp_path / f"cfg_{w}x{h}.yaml"
    p.write_text(yaml.safe_dump({"Map": m}))
    return p


def test_port_enemy_home_water_counts_satellite_spawns():
    # 10x3 open water. Homes far apart; player 1's satellite sits next to home 0's
    # water. Spawn (3,1) of satellite (4,1) is 1 move from (2,1), next to home 0 (1,1).
    ports = {"0": [[1, 1]], "1": [[9, 1], [4, 1]]}
    land = [[1, 1], [9, 1], [4, 1]]
    assert port_enemy_home_water(10, 3, land, ports) == 1
    # Without the satellite the nearest spawn is (8,1) -> (2,1): 6 moves.
    assert port_enemy_home_water(10, 3, [[1, 1], [9, 1]], {"0": [[1, 1]], "1": [[9, 1]]}) == 6


def test_port_enemy_home_water_routes_around_land():
    # A wall at x=2 (rows 0-1) forces a detour through row 2.
    land = [[0, 0], [4, 0], [2, 0], [2, 1]]
    ports = {"0": [[0, 0]], "1": [[4, 0]]}
    # Spawn (1,0) -> (1,1)(1,2)(2,2)(3,2)(3,1)(3,0) next to home 1 = 6 moves (2 without the wall).
    assert port_enemy_home_water(5, 3, land, ports) == 6
    assert port_enemy_home_water(5, 3, [[0, 0], [4, 0]], ports) == 2


@pytest.mark.parametrize("moves,turn", [(0, 0), (3, 0), (4, 1), (6, 1), (7, 2), (9, 2), (10, 3)])
def test_earliest_capture_turn(moves, turn):
    assert earliest_capture_turn(moves) == turn


def test_check_map_rejects_close_spawn():
    # Symmetric satellites at x=5/6 put spawns 2 moves from the enemy home's water.
    data = {"width": 12, "height": 3, "land": [[1, 1], [6, 1], [10, 1], [5, 1]],
            "ports": {"0": [[1, 1], [6, 1]], "1": [[10, 1], [5, 1]]}}
    assert check_map(data) == (True, "ok")
    assert spawn_enemy_home_water(data) == 2
    ok, why = check_map(data, min_port_enemy_home_water=7)
    assert not ok and "water moves from the enemy home" in why


def test_zero_keeps_the_old_draw(tmp_path):
    cfg = _yaml(tmp_path, 12, 6)
    for seed in range(20):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        generate_map_json(str(cfg), str(a), seed=seed)
        generate_map_json(str(cfg), str(b), seed=seed, min_port_enemy_home_water=0)
        assert a.read_text() == b.read_text()


def test_yaml_key_enforces_the_minimum(tmp_path):
    cfg = _yaml(tmp_path, 12, 6, extra={"minPortEnemyHomeWater": 7})
    for seed in range(30):
        out = tmp_path / f"m{seed}.json"
        try:
            generate_map_json(str(cfg), str(out), seed=seed)
        except ValueError:
            continue
        d = spawn_enemy_home_water(json.loads(out.read_text()))
        assert d is None or d >= 7


def test_12x6_pool_has_no_turn0_or_turn1_capture(tmp_path):
    paths = build_map_pool(tmp_path, [PoolSize(12, 6, 30, ports=2)], seed=0, min_port_enemy_home_water=7)
    assert len(paths) == 30
    for p in paths:
        data = json.loads(Path(p).read_text())
        d = spawn_enemy_home_water(data)
        assert d is not None and d >= 7
        assert earliest_capture_turn(d) >= 2
        assert water_path_length(data) >= 7
        assert check_map(data, 7)[0]


@pytest.mark.parametrize("w,h,ports", [(8, 8, 2), (10, 10, 2), (16, 8, 2), (32, 16, 3), (48, 24, 4)])
def test_other_pool_sizes_meet_the_minimum(tmp_path, w, h, ports):
    paths = build_map_pool(tmp_path, [PoolSize(w, h, 8, ports=ports)], seed=0, min_port_enemy_home_water=7)
    assert len(paths) == 8
    assert min(spawn_enemy_home_water(json.loads(Path(p).read_text())) for p in paths) >= 7


def test_infeasible_small_board_raises_instead_of_looping(tmp_path):
    cfg = _yaml(tmp_path, 4, 4, extra={"minPortEnemyHomeWater": 7})
    with pytest.raises(ValueError, match="no port placement"):
        generate_map_json(str(cfg), str(tmp_path / "x.json"), seed=0)
    with pytest.raises(RuntimeError, match="min_port_enemy_home_water=7"):
        build_map_pool(tmp_path / "pool", [PoolSize(4, 4, 1, ports=2)], seed=0, min_port_enemy_home_water=7)


def test_selfplay_config_reads_map_gen(tmp_path):
    from battleboats.training.selfplay import load_selfplay_config

    raw = yaml.safe_load((ROOT / "configs" / "selfplay_bigmaps.yaml").read_text())
    assert load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps.yaml").min_port_enemy_home_water == 7
    raw.pop("map_gen", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_selfplay_config(p).min_port_enemy_home_water == 0
    raw["map_gen"] = {"min_port_enemy_home_water": 9}
    p.write_text(yaml.safe_dump(raw))
    assert load_selfplay_config(p).min_port_enemy_home_water == 9


def test_selfplay_pool_uses_and_logs_the_minimum(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, run_selfplay

    run = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=32),
        map_pool=[PoolSize(12, 6, 4, ports=2)], updates=1, rollout_decisions=64,
        max_turns=4, max_steps=60, num_workers=0, games_per_worker=2, ppo_threads=1,
        min_port_enemy_home_water=7,
    )
    assert run_selfplay(run)["ok"]
    out = capsys.readouterr().out
    assert "map_gen min_port_enemy_home_water=7" in out and "earliest_capture_turn>=2" in out
    for p in (tmp_path / "run" / "maps").glob("map_*.json"):
        assert spawn_enemy_home_water(json.loads(p.read_text())) >= 7
