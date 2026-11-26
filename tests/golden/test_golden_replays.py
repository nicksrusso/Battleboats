"""Golden seeded full-game replay parity tests.

Replays recorded action sequences from ``tests/golden/fixtures/replays.json.gz``
through every registered ``EngineBackend`` and verifies per-step canonical
state hashes, fog-of-war sightings at observation checkpoints, and final
game outcomes.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.golden.backend import EngineBackend

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "replays.json.gz"


@lru_cache(maxsize=1)
def _load_fixture_payload() -> Dict[str, Any]:
    with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def _fixture_ids() -> List[str]:
    return [g["id"] for g in _load_fixture_payload()["games"]]


def _games_by_id() -> Dict[str, Dict[str, Any]]:
    return {g["id"]: g for g in _load_fixture_payload()["games"]}


def test_golden_replay_fixture_coverage() -> None:
    payload = _load_fixture_payload()
    games = payload["games"]
    assert len(games) >= 50
    presets = {g["map_preset"] for g in games}
    assert {"16x8", "32x16", "48x24"} <= presets
    matchups = " ".join(g["matchup"] for g in games)
    for required in (
        "RAIDER_BLITZ",
        "ECONOMIC_BOOM",
        "WOLFPACK_STEALTH",
        "CAPITAL_ARMADA",
        "ISLAND_HOPPER",
        "BALANCED_ADAPTIVE",
        "random",
    ):
        assert required in matchups
    winners = {g["winner"] for g in games}
    assert {0, 1} <= winners


@pytest.mark.parametrize("game_id", _fixture_ids())
def test_golden_replay_exact_parity(
    backend: EngineBackend, tmp_path: Path, game_id: str
) -> None:
    game = _games_by_id()[game_id]
    map_path = tmp_path / f"{game_id}.json"
    map_path.write_text(json.dumps(game["map"]))

    inst = backend.create(map_path, kill_curve_k=float(game["kill_curve_k"]))
    inst.reset(seed=int(game["seed"]))
    # Rust uses its own seeded RNG. Recorded rolls keep the state hash identical
    # without a NumPy PCG64 port. Python ignores the hook and rolls for real.
    if hasattr(inst, "set_combat_outcomes"):
        inst.set_combat_outcomes(game["combat_outcomes"])
    assert inst.state_hash() == game["initial_hash"], f"{game_id}: initial hash mismatch"

    obs_by_step = {cp["step_idx"]: cp["seats"] for cp in game["obs_checkpoints"]}
    step_hashes = game["step_hashes"]

    for step_idx, rec in enumerate(game["actions"]):
        act = backend.decode_action(rec)
        inst.step(act)
        actual_hash = inst.state_hash()
        expected_hash = step_hashes[step_idx]
        assert actual_hash == expected_hash, (
            f"{game_id} ({game['matchup']}) diverged at step {step_idx} action={rec}"
        )

        if step_idx in obs_by_step:
            for seat_expected in obs_by_step[step_idx]:
                pid = int(seat_expected["player_id"])
                assert inst.visible_enemy_ships(pid) == seat_expected["visible_enemy_ships"]
                assert inst.known_enemy_ships(pid) == seat_expected["known_enemy_ships"]
                assert inst.visible_enemy_ports(pid) == seat_expected["visible_enemy_ports"]
                assert inst.known_enemy_ports(pid) == seat_expected["known_enemy_ports"]

    final_state = inst.canonical_state()
    assert final_state["turn"] == game["final_turn"]
    assert final_state["winner"] == game["winner"]
    assert inst.state_hash() == game["final_state_hash"]
