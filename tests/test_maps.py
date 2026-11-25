"""Map generation, pool checks, spawn-distance rules, and config wiring."""

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
    episode_limits,
    home_distance,
    render_ascii,
    spawn_enemy_home_water,
    water_path_length,
)
from battleboats.training.selfplay import load_selfplay_config

ROOT = Path(__file__).resolve().parents[1]
SIZES = [PoolSize(16, 8, 3, ports=2), PoolSize(32, 16, 2, ports=3), PoolSize(48, 24, 2, ports=4)]


def _yaml(tmp_path, w, h, ports=2, extra=None):
    m = {"sizeX": w, "sizeY": h, "landFraction": 0.02, "numPorts": ports, **(extra or {})}
    p = tmp_path / f"cfg_{w}x{h}.yaml"
    p.write_text(yaml.safe_dump({"Map": m}))
    return p


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
    sizes = sorted(
        {
            (json.loads(Path(p).read_text())["width"], json.loads(Path(p).read_text())["height"])
            for p in a
        }
    )
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
    cfg = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps.yaml")
    assert [(s.width, s.height) for s in cfg.map_pool] == [(16, 8), (32, 16), (48, 24)]
    assert cfg.run_dir == Path("runs/selfplay_bigmaps")


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


# A bought ship cannot act on its build turn, so every rush lands a turn later than
# under the old buy-then-move rule (old: 0,0,1,1,2,2,3).
@pytest.mark.parametrize("moves,turn", [(0, 1), (3, 1), (4, 2), (6, 2), (7, 3), (9, 3), (10, 4)])
def test_earliest_capture_turn(moves, turn):
    assert earliest_capture_turn(moves) == turn


def test_check_map_rejects_close_spawn():
    # Symmetric satellites at x=5/6 put spawns 2 moves from the enemy home's water.
    data = {
        "width": 12,
        "height": 3,
        "land": [[1, 1], [6, 1], [10, 1], [5, 1]],
        "ports": {"0": [[1, 1], [6, 1]], "1": [[10, 1], [5, 1]]},
    }
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
        assert earliest_capture_turn(d) >= 3
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
    raw = yaml.safe_load((ROOT / "configs" / "selfplay_bigmaps.yaml").read_text())
    assert load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps.yaml").min_port_enemy_home_water == 7
    raw.pop("map_gen", None)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_selfplay_config(p).min_port_enemy_home_water == 0
    raw["map_gen"] = {"min_port_enemy_home_water": 9}
    p.write_text(yaml.safe_dump(raw))
    assert load_selfplay_config(p).min_port_enemy_home_water == 9
