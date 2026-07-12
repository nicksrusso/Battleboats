"""Smoke tests for LandingSkillAEC (self-contained tiny map fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from battleboats.core.actions import CapturePortAction, EndTurnAction, MoveAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.landing_skill_aec import LandingSkillAEC


def _write_tiny_map(path: Path) -> None:
    """2x3 map: P0 home (0,0), P1 home (0,2); a water lane down column 0.

    Layout (x right, y down):
        (0,0) land home0 | (1,0) water
        (0,1) water      | (1,1) water
        (0,2) land home1 | (1,2) water
    """
    data = {
        "width": 2,
        "height": 3,
        "seed": 0,
        "land": [[0, 0], [0, 2]],
        "ports": {
            "0": [[0, 0]],
            "1": [[0, 2]],
        },
    }
    path.write_text(json.dumps(data))


@pytest.fixture
def map_path(tmp_path: Path) -> str:
    p = tmp_path / "landing_tiny.json"
    _write_tiny_map(p)
    return str(p)


def test_reset_spawns_landing_ship(map_path: str) -> None:
    env = LandingSkillAEC(map_path)
    env.reset(seed=0)

    assert env.agents == ["player_0"]
    assert env.agent_selection == "player_0"
    landings = [
        s
        for s in env.engine.ships.values()
        if s.type is ShipType.LANDING and s.owner == 0
    ]
    assert len(landings) == 1
    legal = env.infos["player_0"]["legal_actions"]
    assert legal


def test_capture_enemy_home_rewards_and_terminates(map_path: str) -> None:
    # Spawn adjacent to enemy home (0,2) so one capture wins.
    env = LandingSkillAEC(
        map_path,
        landing_position=(1, 2),  # water adjacent to enemy home (0,2)
        capture_reward=1.0,
        max_turns=20,
    )
    env.reset(seed=0)
    sid = next(iter(env.engine.players[0].owned_ship_ids))

    env.step(CapturePortAction(landing_ship_id=sid, target=(0, 2)))

    assert env.terminations["player_0"] is True
    assert env.rewards["player_0"] == 1.0
    assert env._cumulative_rewards["player_0"] == 1.0
    assert env.engine.winner == 0


def test_move_then_capture(map_path: str) -> None:
    env = LandingSkillAEC(map_path, landing_position=(1, 1), max_turns=20)
    env.reset(seed=0)
    sid = next(iter(env.engine.players[0].owned_ship_ids))

    # Move down the lane to sit adjacent to enemy home (0,2), then capture.
    env.step(MoveAction(sid, (1, 2)))
    assert env.terminations["player_0"] is False
    env.step(CapturePortAction(landing_ship_id=sid, target=(0, 2)))
    assert env.terminations["player_0"] is True
    assert env._cumulative_rewards["player_0"] == 1.0


def test_randomize_target_varies_but_keeps_spawn_stable(map_path: str) -> None:
    env = LandingSkillAEC(map_path, randomize_target=True, max_turns=20)
    targets, spawns, homes = set(), set(), set()
    for s in range(12):
        env.reset(seed=s)
        homes.add(env.engine.players[0].home_port)
        targets.add(env.engine.players[1].home_port)
        ship = next(iter(env.engine.players[0].owned_ship_ids))
        spawns.add(env.engine.ships[ship].position)
        # target always revealed to the learner
        assert env.engine.players[1].home_port in env.engine.players[0].port_sightings
    assert len(targets) > 1  # target actually moves
    assert len(homes) == 1 and len(spawns) == 1  # learner home + spawn fixed
    env.close()


def test_randomize_target_reproducible(map_path: str) -> None:
    env = LandingSkillAEC(map_path, randomize_target=True, max_turns=20)
    env.reset(seed=7)
    first = env.engine.players[1].home_port
    env.reset(seed=7)
    assert env.engine.players[1].home_port == first
    env.close()


def test_auto_opponent_endturn_returns_to_learner(map_path: str) -> None:
    env = LandingSkillAEC(map_path, landing_position=(1, 1), max_turns=10)
    env.reset(seed=0)
    env.step(EndTurnAction())
    assert env.engine.current_player == 0
    assert env.agent_selection == "player_0"
    assert env.terminations["player_0"] is False
