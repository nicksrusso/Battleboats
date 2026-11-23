"""Tactical scripted agent: legal actions, builder target, fight-or-run, landing flee."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from battleboats.agents.tactical import (TacticalConfig, choose_build_action,
                                         choose_builder_target,
                                         inverse_cost_type_weights,
                                         local_combat_power, should_engage,
                                         tactical_scripted_action)
from battleboats.core.actions import (AttackAction, BuildPortAction,
                                      BuildShipAction, CapturePortAction,
                                      EndTurnAction, MerchantLoadAction,
                                      MerchantUnloadAction, MoveAction)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.skill_maps import (write_full_map, write_merchant_map,
                                         write_two_home_map)

# Near-greedy so tests of flee / attack / builder pathing are deterministic.
GREEDY = TacticalConfig(radius=6, margin=1.2, temperature=0.0)


def _act(engine, pid, rng, cfg=GREEDY):
    return tactical_scripted_action(engine, pid, rng, cfg)


def test_tactical_actions_are_legal(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "f.json", 6, 6, seed=1))
    engine.reset(seed=1)
    rng = random.Random(0)
    taken = 0
    while engine.winner is None and engine.turn < 8 and taken < 80:
        pid = engine.current_player
        legal = engine.enumerate_legal(pid)
        action = _act(engine, pid, rng)
        assert action in legal
        engine.step(action)
        taken += 1
    assert taken > 0


def test_builder_targets_nearest_empty_land_to_home(tmp_path: Path) -> None:
    """Home (0,0); empty land at (0,2) dist 2 and (3,0) dist 3. Build the closer one."""
    path = tmp_path / "b.json"
    path.write_text(
        '{"width": 6, "height": 4, "seed": 0,'
        ' "land": [[0, 0], [5, 3], [0, 2], [3, 0]],'
        ' "ports": {"0": [[0, 0]], "1": [[5, 3]]}}'
    )
    engine = gameEngine(str(path))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    engine.players[1].cash = 0
    builder = engine._spawn_ship(0, ShipType.BUILDER, (2, 2))
    target = choose_builder_target(engine, 0, builder.position)
    assert target == (0, 2)

    rng = random.Random(1)
    built = False
    for _ in range(16):
        assert engine.current_player == 0
        legal = engine.enumerate_legal(0)
        action = _act(engine, 0, rng)
        assert action in legal
        if isinstance(action, BuildPortAction):
            assert action.target == (0, 2)
            engine.step(action)
            built = True
            break
        if isinstance(action, EndTurnAction):
            engine.step(action)
            if engine.current_player == 1:
                engine.step(EndTurnAction())
            continue
        engine.step(action)
    assert built
    assert (0, 2) in engine.players[0].owned_port_positions
    assert (3, 0) not in engine.players[0].owned_port_positions


def test_flees_when_outmatched(tmp_path: Path) -> None:
    engine = gameEngine(write_two_home_map(tmp_path / "c.json", 5, 3))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    hunter = engine._spawn_ship(0, ShipType.DESTROYER, (2, 1))
    enemy = engine._spawn_ship(1, ShipType.BATTLESHIP, (3, 1))
    ours, theirs = local_combat_power(engine, 0, hunter.position, 6)
    assert theirs > ours
    assert not should_engage(ours, theirs, 1.2)

    rng = random.Random(0)
    action = _act(engine, 0, rng)
    legal = engine.enumerate_legal(0)
    assert action in legal
    assert not isinstance(action, AttackAction)
    assert isinstance(action, MoveAction)
    before = engine.map.manhattan(hunter.position, enemy.position)
    after = engine.map.manhattan(action.destination, enemy.position)
    assert after > before


def test_attacks_when_stronger(tmp_path: Path) -> None:
    engine = gameEngine(write_two_home_map(tmp_path / "c.json", 5, 3))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    engine._spawn_ship(0, ShipType.BATTLESHIP, (2, 1))
    engine._spawn_ship(1, ShipType.MERCHANT, (3, 1))
    action = _act(engine, 0, random.Random(0))
    assert isinstance(action, AttackAction)
    assert action in engine.enumerate_legal(0)


def test_landing_adjacent_to_enemy_moves_away_rather_than_attacking(
    tmp_path: Path,
) -> None:
    """A landing next to a hostile combat ship must flee, never attack or stay."""
    engine = gameEngine(write_two_home_map(tmp_path / "l.json", 5, 3))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    landing = engine._spawn_ship(0, ShipType.LANDING, (2, 1))
    enemy = engine._spawn_ship(1, ShipType.DESTROYER, (3, 1))
    rng = random.Random(0)
    action = _act(engine, 0, rng)
    legal = engine.enumerate_legal(0)
    assert action in legal
    assert not isinstance(action, AttackAction)
    assert isinstance(action, MoveAction)
    assert action.ship_id == landing.id
    before = engine.map.manhattan(landing.position, enemy.position)
    after = engine.map.manhattan(action.destination, enemy.position)
    assert after > before
    # And it must not spend the turn capturing either while threatened,
    # even if a capture is somehow legal.
    assert not isinstance(action, CapturePortAction)

    # Same decision with noise still never attacks and never closes.
    for seed in range(12):
        noisy = TacticalConfig(radius=6, margin=1.2, temperature=0.5)
        action = _act(engine, 0, random.Random(seed), noisy)
        assert action in legal
        assert not isinstance(action, AttackAction)
        if isinstance(action, MoveAction):
            assert engine.map.manhattan(action.destination, enemy.position) >= before


def test_landing_never_issues_attack_action(tmp_path: Path) -> None:
    engine = gameEngine(write_two_home_map(tmp_path / "l.json", 5, 3))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    engine._spawn_ship(0, ShipType.LANDING, (1, 1))
    engine._spawn_ship(1, ShipType.MERCHANT, (2, 1))
    # Landings have attack_range 0, so the engine should not even list an
    # attack; the policy must also never invent one.
    legal = engine.enumerate_legal(0)
    assert not any(isinstance(a, AttackAction) for a in legal)
    action = _act(engine, 0, random.Random(0))
    assert action in legal
    assert not isinstance(action, AttackAction)


def test_buy_weights_are_inverse_cost() -> None:
    types = tuple(ShipType)
    weights = inverse_cost_type_weights(types)
    # Equal expected gold: weight * cost is constant.
    spends = [weights[t] * BASE_STATS[t].cost for t in types]
    assert max(spends) - min(spends) < 1e-12
    # Cheap hulls must be more probable than expensive ones.
    assert (
        weights[ShipType.DESTROYER]
        > weights[ShipType.CRUISER]
        > weights[ShipType.CARRIER]
    )
    assert (
        weights[ShipType.MERCHANT]
        == weights[ShipType.DESTROYER]
        == weights[ShipType.SUBMARINE]
    )


def test_choose_build_samples_inverse_cost() -> None:
    # One legal buy per type, all affordable. Empirical spend should be even.
    options = [
        BuildShipAction(port=(0, 0), spawn_position=(1, 0), ship_type=t)
        for t in ShipType
    ]
    rng = random.Random(0)
    counts = Counter(choose_build_action(options, rng).ship_type for _ in range(8000))
    spends = {t: counts[t] * BASE_STATS[t].cost for t in ShipType}
    mean = sum(spends.values()) / len(spends)
    # Within 20% of the mean is plenty for 8000 inverse-cost draws.
    for t, spend in spends.items():
        assert abs(spend - mean) / mean < 0.20, (t, spend, mean, dict(counts))


def test_merchant_loads_at_satellite_and_unloads_at_home(tmp_path: Path) -> None:
    path = write_merchant_map(tmp_path / "m.json", 6, 5)
    engine = gameEngine(path)
    engine.reset(seed=0)
    engine.players[0].cash = 0
    me = engine.players[0]
    satellite = next(p for p in me.owned_port_positions if p != me.home_port)
    engine.ports[satellite].stockpile = 100
    # Spawn adjacent to the satellite so the first action can be a load.
    spawn = next(
        p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        for p in [(satellite[0] + dx, satellite[1] + dy)]
        if engine.map.in_bounds(p) and engine.map.is_water(p)
    )
    ship = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    rng = random.Random(0)
    loaded = False
    unloaded = False
    for _ in range(40):
        if engine.current_player != 0:
            engine.step(EndTurnAction())
            continue
        action = _act(engine, 0, rng)
        assert action in engine.enumerate_legal(0)
        if isinstance(action, MerchantLoadAction):
            assert action.port == satellite
            loaded = True
        if isinstance(action, MerchantUnloadAction):
            assert action.port == me.home_port
            unloaded = True
        engine.step(action)
        if unloaded:
            break
    assert loaded and unloaded
    assert ship.cargo == 0
    assert engine.players[0].cash > 0
