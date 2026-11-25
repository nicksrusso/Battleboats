"""Parity between the Rust core and the Python reference.

Skipped entirely when the ``battleboats_core`` extension is not built, so a
pure-Python install still passes the suite. The Rust backend is not registered
in ``tests/golden/backend.py`` yet: step, legal actions, combat, and fog are
still Python-only. What is ported (canonical JSON / SHA-256, compact actions,
map load, coastline, and a fresh reset) is checked here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pytest

from battleboats.envs.observation import _coastline_scan, _coastline_tiles
from tests.golden.backend import (
    PythonEngineBackend,
    canonical_json_bytes,
    encode_action,
)
from tests.golden.conftest import write_map_json

bb = pytest.importorskip("battleboats_core")

FIXTURE_PATH = Path(__file__).resolve().parent / "golden" / "fixtures" / "replays.json.gz"
BACKEND = PythonEngineBackend()


def _canon(state: dict) -> str:
    return canonical_json_bytes(state).decode("utf-8")


def _assert_same(got: str, expected: str, label: str) -> None:
    if got == expected:
        return
    limit = min(len(got), len(expected))
    index = next((i for i in range(limit) if got[i] != expected[i]), limit)
    start = max(0, index - 60)
    raise AssertionError(
        f"{label} diverged at character {index}\n"
        f"  python: {expected[start:index + 80]!r}\n"
        f"  rust:   {got[start:index + 80]!r}"
    )


@lru_cache(maxsize=1)
def _games() -> tuple:
    with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(payload["games"])


def test_rule_tables_match_python() -> None:
    assert bb.schema_version() == 1
    assert bb.starting_cash() == BACKEND.starting_cash()
    assert bb.port_production() == BACKEND.port_production()
    assert bb.cash_per_material() == BACKEND.cash_per_material()
    assert bb.merchant_capacity() == BACKEND.merchant_capacity()
    assert tuple(bb.ship_types()) == BACKEND.ship_types()
    types = BACKEND.ship_types()
    for ship_type in types:
        assert bb.base_stats(ship_type) == BACKEND.base_stats(ship_type)
    for attacker in types:
        for defender in types:
            assert bb.attack_modifier(attacker, defender) == BACKEND.attack_modifier(
                attacker, defender
            )
    keys = list(types) + ["Port"]
    for spotter in keys:
        for target in keys:
            assert bb.spot_distance(spotter, target) == BACKEND.spot_distance(spotter, target)


def test_compact_actions_match_python() -> None:
    actions = [
        BACKEND.move_action(3, (1, 4)),
        BACKEND.attack_action(1, 9),
        BACKEND.build_ship_action((0, 0), (1, 0), "Destroyer"),
        BACKEND.build_ship_action((2, 2), (2, 3), "Merchant"),
        BACKEND.build_ship_action((4, 1), (5, 1), "Landing"),
        BACKEND.build_ship_action((1, 1), (1, 2), "Builder"),
        BACKEND.build_port_action(4, (2, 2)),
        BACKEND.capture_port_action(5, (8, 1)),
        BACKEND.merchant_load_action(6, (3, 3)),
        BACKEND.merchant_unload_action(6, (0, 0)),
        BACKEND.end_turn_action(),
    ]
    for action in actions:
        record = encode_action(action)
        assert json.loads(bb.roundtrip_action(record)) == record
    with pytest.raises(ValueError, match="Unknown encoded action tag"):
        bb.roundtrip_action(["nope"])


def test_canonical_float_spelling_matches_cpython() -> None:
    samples = [
        0.0,
        1.0,
        2.0,
        4.0,
        0.1,
        0.0001,
        1e-5,
        1.5e-5,
        1e-6,
        1e-7,
        10.0,
        1e15,
        1e16,
        1.5e16,
        2.25**4,
    ]
    for value in samples:
        assert bb.format_canonical_float(value) == json.dumps(value)


def test_scenario_reset_matches_python(tmp_path: Path) -> None:
    path = write_map_json(
        tmp_path / "scenario.json",
        12,
        6,
        [[0, 0], [2, 0]],
        [[11, 5]],
        extra_land=[[4, 2]],
    )
    scenario = {
        "seed": 7,
        "map_path": str(path),
        "player_0": {
            "cash": 40,
            "ships": [
                {"type": "DESTROYER", "position": [1, 1]},
                {"type": "MERCHANT", "position": [3, 2]},
            ],
        },
        "player_1": {
            "cash": 80,
            "ships": [{"type": "SUBMARINE", "position": [8, 3]}],
        },
    }
    python = BACKEND.create(path, kill_curve_k=4.0)
    python.reset_from_scenario(scenario)
    rust = bb.Game(str(path), kill_curve_k=4.0)
    rust.reset_from_scenario(scenario)
    expected = _canon(python.canonical_state())
    _assert_same(rust.canonical_json(), expected, "scenario reset")
    assert rust.state_hash() == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert '"kill_curve_k":4.0' in expected
    assert " " not in rust.canonical_json()

    shuffled = json.loads(expected)
    shuffled["ships"] = list(reversed(shuffled["ships"]))
    shuffled["terrain_land"] = list(reversed(shuffled["terrain_land"]))
    shuffled["players"] = list(reversed(shuffled["players"]))
    messy = json.dumps(shuffled)
    _assert_same(bb.canonical_json_from_json(messy), expected, "shuffled canonical state")


def _assert_map_queries(python_engine, rust_game) -> None:
    game_map = python_engine.map
    width, height = game_map.width, game_map.height
    assert rust_game.width == width
    assert rust_game.height == height
    land = [
        [x, y]
        for x in range(width)
        for y in range(height)
        if int(game_map.terrain[x, y]) == 1
    ]
    assert rust_game.terrain_land() == land
    assert rust_game.home_ports() == [
        [int(player.home_port[0]), int(player.home_port[1])] for player in python_engine.players
    ]
    for x in range(width):
        for y in range(height):
            terrain = int(game_map.terrain[x, y])
            owner = int(game_map.port_owner[x, y])
            assert rust_game.is_water(x, y) == (terrain == 0)
            assert rust_game.is_land(x, y) == (terrain == 1)
            assert rust_game.is_port(x, y) == (owner != -1)
            assert rust_game.port_owner(x, y) == owner
    assert rust_game.in_bounds(-1, 0) is False
    with pytest.raises(IndexError):
        rust_game.is_water(width, 0)
    for x, y, radius in (
        (0, 0, 1),
        (width // 2, height // 2, 2),
        (width - 1, height - 1, 3),
    ):
        expect = [[int(a), int(b)] for a, b in game_map.tiles_within((x, y), radius)]
        assert rust_game.tiles_within(x, y, radius) == expect
        assert rust_game.manhattan(0, 0, x, y) == game_map.manhattan((0, 0), (x, y))
    assert rust_game.coastline_scan() == [
        [int(a), int(b)] for a, b in _coastline_scan(game_map)
    ]
    for player_id in (None, 0, 1):
        expect = [[int(a), int(b)] for a, b in _coastline_tiles(python_engine, player_id)]
        assert rust_game.coastline(player_id) == expect


def test_replay_reset_and_every_step_hash(tmp_path: Path) -> None:
    """Rust reset matches Python, and every recorded step's canonical JSON rehashes."""
    seen_maps: set[str] = set()
    games = _games()
    assert len(games) >= 50
    steps = 0
    for game in games:
        game_id = game["id"]
        map_path = tmp_path / f"{game_id}.json"
        map_path.write_text(json.dumps(game["map"]))
        python = BACKEND.create(map_path, kill_curve_k=float(game["kill_curve_k"]))
        python.reset(seed=int(game["seed"]))
        rust = bb.Game(str(map_path), kill_curve_k=float(game["kill_curve_k"]))
        rust.reset(seed=int(game["seed"]))

        initial = _canon(python.canonical_state())
        _assert_same(rust.canonical_json(), initial, f"{game_id} initial")
        assert rust.state_hash() == game["initial_hash"]
        assert hashlib.sha256(initial.encode("utf-8")).hexdigest() == game["initial_hash"]
        if "seed" in game["map"]:
            assert rust.map_seed == int(game["map"]["seed"])

        signature = json.dumps(game["map"], sort_keys=True)
        if signature not in seen_maps:
            seen_maps.add(signature)
            _assert_map_queries(python.raw, rust)

        for step_idx, record in enumerate(game["actions"]):
            python.step(BACKEND.decode_action(record))
            raw = _canon(python.canonical_state())
            rust_hash = bb.state_hash_from_json(raw)
            expect = game["step_hashes"][step_idx]
            if rust_hash != expect:
                got = bb.canonical_json_from_json(raw)
                _assert_same(got, raw, f"{game_id} step {step_idx} action {record}")
                raise AssertionError(f"{game_id} step {step_idx} hash {rust_hash} != {expect}")
            steps += 1
        assert python.state_hash() == game["final_state_hash"]
    assert steps == sum(len(game["actions"]) for game in games)
    assert steps > 1000
