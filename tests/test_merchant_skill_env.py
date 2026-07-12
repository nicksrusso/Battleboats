"""Smoke tests for MerchantSkillAEC (self-contained tiny map fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from battleboats.core.actions import EndTurnAction, MerchantLoadAction, MerchantUnloadAction, MoveAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.merchant_skill_aec import MerchantSkillAEC


def _write_tiny_map(path: Path) -> None:
    """3x2 map: P0 home (0,0), P0 resource port (2,0), P1 home (2,1); rest water.

    Layout (x right, y down):
        (0,0) land home0 | (1,0) water | (2,0) land port0
        (0,1) water      | (1,1) water | (2,1) land home1
    """
    data = {
        "width": 3,
        "height": 2,
        "seed": 0,
        "land": [[0, 0], [2, 0], [2, 1]],
        "ports": {
            "0": [[0, 0], [2, 0]],
            "1": [[2, 1]],
        },
    }
    path.write_text(json.dumps(data))


@pytest.fixture
def map_path(tmp_path: Path) -> str:
    p = tmp_path / "merchant_tiny.json"
    _write_tiny_map(p)
    return str(p)


def test_reset_spawns_merchant_and_stockpile(map_path: str) -> None:
    env = MerchantSkillAEC(map_path, stockpile=50, starting_cargo=0)
    env.reset(seed=0)

    assert env.agents == ["player_0"]
    assert env.agent_selection == "player_0"
    merchants = [
        s
        for s in env.engine.ships.values()
        if s.type is ShipType.MERCHANT and s.owner == 0
    ]
    assert len(merchants) == 1
    assert env.engine.ports[(2, 0)].stockpile == 50
    legal = env.infos["player_0"]["legal_actions"]
    assert legal  # at least endturn / moves


def test_deliver_with_starting_cargo_rewards_and_terminates(map_path: str) -> None:
    # Spawn merchant already next to home with cargo — one unload wins.
    env = MerchantSkillAEC(
        map_path,
        stockpile=0,
        starting_cargo=10,
        merchant_position=(1, 0),  # water adjacent to home (0,0)
        deliver_reward=1.0,
        terminate_on_delivery=True,
        max_turns=20,
    )
    env.reset(seed=0)
    mid = next(iter(env.engine.players[0].owned_ship_ids))
    ship = env.engine.ships[mid]
    assert ship.cargo == 10
    assert ship.position == (1, 0)

    cash_before = env.engine.players[0].cash
    env.step(MerchantUnloadAction(merchant_id=mid, port=(0, 0)))

    assert env.terminations["player_0"] is True
    assert env.rewards["player_0"] == 1.0
    assert env._cumulative_rewards["player_0"] == 1.0
    assert env.engine.players[0].cash > cash_before
    assert env.engine.ships[mid].cargo == 0


def test_load_then_move_then_unload(map_path: str) -> None:
    env = MerchantSkillAEC(
        map_path,
        stockpile=25,
        starting_cargo=0,
        merchant_position=(1, 0),  # between home (0,0) and resource (2,0)
        terminate_on_delivery=True,
        max_turns=30,
    )
    env.reset(seed=0)
    mid = next(iter(env.engine.players[0].owned_ship_ids))

    # Move adjacent to resource port (2,0) → water (1,0) is already between;
    # from (1,0) resource is adjacent via (2,0) manhattan 1? (1,0)-(2,0)=1 yes.
    env.step(MerchantLoadAction(merchant_id=mid, port=(2, 0)))
    assert env.engine.ships[mid].cargo == 25
    assert env.terminations["player_0"] is False

    # Unload only at home — move next to home if needed. (1,0) already adj to (0,0).
    env.step(MerchantUnloadAction(merchant_id=mid, port=(0, 0)))
    assert env.terminations["player_0"] is True
    assert env._cumulative_rewards["player_0"] == 1.0


def test_throughput_accumulates_without_terminating(map_path: str) -> None:
    # Default (throughput) mode: each home unload pays materials * (deliver/CAP);
    # a full 100-load = 1.0, and the episode does NOT end on a delivery.
    env = MerchantSkillAEC(
        map_path,
        stockpile=1000,
        starting_cargo=10,
        merchant_position=(1, 0),  # adjacent to both home (0,0) and satellite (2,0)
        deliver_reward=1.0,
        max_turns=30,
    )
    env.reset(seed=0)
    mid = next(iter(env.engine.players[0].owned_ship_ids))

    # Deliver the 10 starting cargo: reward 10/100, no termination.
    env.step(MerchantUnloadAction(merchant_id=mid, port=(0, 0)))
    assert env.terminations["player_0"] is False
    assert env.rewards["player_0"] == pytest.approx(10 / 100)

    # Reload a full 100 from the satellite, deliver again: reward 100/100 = 1.0.
    env.step(MerchantLoadAction(merchant_id=mid, port=(2, 0)))
    assert env.engine.ships[mid].cargo == 100
    env.step(MerchantUnloadAction(merchant_id=mid, port=(0, 0)))
    assert env.terminations["player_0"] is False
    assert env.rewards["player_0"] == pytest.approx(1.0)

    # Throughput accrues across deliveries.
    assert env._cumulative_rewards["player_0"] == pytest.approx(10 / 100 + 1.0)


def test_randomize_ports_moves_satellite_keeps_home(map_path: str) -> None:
    env = MerchantSkillAEC(map_path, randomize_ports=True, max_turns=20)
    homes, satellites, spawns = set(), set(), set()
    for s in range(12):
        env.reset(seed=s)
        p0 = env.engine.players[0]
        home = p0.home_port
        sats = [p for p in p0.owned_port_positions if p != home]
        homes.add(home)
        satellites.update(sats)
        ship = next(iter(p0.owned_ship_ids))
        spawns.add(env.engine.ships[ship].position)
        # satellite is a friendly port with stockpile (loadable target)
        assert len(sats) == 1
        assert env.engine.ports[sats[0]].stockpile == env.stockpile
    assert len(homes) == 1  # home fixed
    assert len(spawns) == 1  # spawn fixed (near home)
    assert len(satellites) > 1  # satellite actually moves
    env.close()


def test_auto_opponent_endturn_returns_to_learner(map_path: str) -> None:
    env = MerchantSkillAEC(map_path, starting_cargo=10, merchant_position=(1, 0), max_turns=10)
    env.reset(seed=0)
    # End learner turn → env should auto-end p1 and come back to p0
    env.step(EndTurnAction())
    assert env.engine.current_player == 0
    assert env.agent_selection == "player_0"
    assert env.terminations["player_0"] is False
