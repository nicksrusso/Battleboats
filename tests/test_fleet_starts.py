"""Fleet-start placement: config gates, and spawned ships can act on turn 0."""

import json
import random

import pytest

from battleboats.core.actions import MoveAction
from battleboats.core.gameEngine import gameEngine
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool
from battleboats.training.fleet_starts import FleetStarts, apply_fleet_start

ALWAYS = dict(prob=1.0, budget=1200, landing_prob=1.0, radius=6, landing_forward_prob=0.5)


def _write_engine(tmp_path, w, h, ports, land=None, name="m.json"):
    land = land if land is not None else [p for ps in ports.values() for p in ps]
    path = tmp_path / name
    path.write_text(json.dumps({"width": w, "height": h, "seed": 0, "land": land, "ports": ports}))
    e = gameEngine(str(path))
    e.reset(seed=0)
    return e


def _engine_from_path(path, seed=0):
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


def test_fleet_start_ships_act_on_turn_0(tmp_path):
    e = _write_engine(tmp_path, 16, 8, {"0": [[1, 3], [5, 6]], "1": [[14, 4], [10, 1]]})
    fleet = apply_fleet_start(
        e, FleetStarts(prob=1.0, budget=1200, landing_prob=1.0, radius=6), random.Random(0)
    )
    assert fleet is not None
    mine = [s for s in e.ships.values() if s.owner == 0]
    assert mine and not any(s.bought_this_turn for s in mine)
    assert any(isinstance(a, MoveAction) and e.ships[a.ship_id].owner == 0 for a in e.enumerate_legal(0))


def test_small_board_gets_normal_start(tmp_path):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(12, 6, 3, ports=2)], seed=3, land_fraction=0.02)
    for i in range(12):
        eng = _engine_from_path(maps[i % len(maps)], seed=i)
        before = [len(p.owned_ship_ids) for p in eng.players]
        assert apply_fleet_start(eng, FleetStarts(**ALWAYS), random.Random(i)) is None
        assert eng.fleet_start_forward is False
        assert [len(p.owned_ship_ids) for p in eng.players] == before
    eng = _engine_from_path(maps[0])
    assert apply_fleet_start(eng, FleetStarts(**ALWAYS, min_width=12, min_height=6), random.Random(0)) is not None


def test_big_board_unchanged_with_or_without_threshold(tmp_path):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(16, 8, 1, ports=2)], seed=3, land_fraction=0.02)
    for i in range(5):
        a = _engine_from_path(maps[0], seed=i)
        b = _engine_from_path(maps[0], seed=i)
        fa = apply_fleet_start(a, FleetStarts(**ALWAYS), random.Random(i))
        fb = apply_fleet_start(b, FleetStarts(**ALWAYS, min_width=1, min_height=1), random.Random(i))
        assert fa == fb and fa is not None
        assert a.fleet_start_forward == b.fleet_start_forward

        def positions(eng):
            return sorted(
                (eng.ships[s].type.name, eng.ships[s].position)
                for p in eng.players
                for s in p.owned_ship_ids
            )

        assert positions(a) == positions(b)
