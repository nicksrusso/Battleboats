"""Tactical scripted agent: legal actions, builder target, fight-or-run, landing flee."""

from __future__ import annotations

import random
from collections import Counter, deque
from pathlib import Path

from battleboats.agents.tactical import (BUILDABLE_TYPES, TacticalConfig,
                                         build_queue_for, choose_build_action,
                                         choose_builder_target,
                                         local_combat_power,
                                         sample_build_queue, should_engage,
                                         tactical_scripted_action)
from battleboats.core.actions import (AttackAction, BuildPortAction,
                                      BuildShipAction, CapturePortAction,
                                      EndTurnAction, MerchantLoadAction,
                                      MerchantUnloadAction, MoveAction)
from battleboats.core.gameEngine import gameEngine
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


def test_build_queue_samples_are_uniform() -> None:
    types = tuple(BUILDABLE_TYPES)
    rng = random.Random(0)
    counts = Counter(sample_build_queue(rng, 8000, types))
    assert set(counts) == set(types)
    expected = 8000 / len(types)
    # Equal probability, not equal spend. 20% is plenty for 8000 draws.
    for t, n in counts.items():
        assert abs(n - expected) / expected < 0.20, (t, n, expected, dict(counts))
    # Cheap and expensive hulls must be about equally frequent.
    assert abs(counts[ShipType.DESTROYER] - counts[ShipType.CARRIER]) / expected < 0.20


def test_buy_follows_queue_order(tmp_path: Path) -> None:
    # Two ports a side so later buys still have a spawn tile after the
    # first ships occupy the water next to home (bought ships can't move).
    engine = gameEngine(write_full_map(tmp_path / "q.json", 6, 6, seed=0))
    engine.reset(seed=0)
    engine.players[0].cash = 1000
    engine.players[1].cash = 0
    planned = [ShipType.LANDING, ShipType.DESTROYER, ShipType.MERCHANT]
    queue = build_queue_for(engine, 0)
    queue.clear()
    queue.extend(planned)
    rng = random.Random(0)
    bought = []
    for expected in planned:
        action = _act(engine, 0, rng)
        assert isinstance(action, BuildShipAction)
        assert action.ship_type is expected
        assert action in engine.enumerate_legal(0)
        engine.step(action)
        bought.append(action.ship_type)
    assert bought == planned


def test_saves_when_short_of_queue_head(tmp_path: Path) -> None:
    engine = gameEngine(write_two_home_map(tmp_path / "q.json", 5, 3))
    engine.reset(seed=0)
    # Carrier costs 250; 200 can still buy a destroyer/merchant/sub, but must not.
    engine.players[0].cash = 200
    engine.players[1].cash = 0
    queue = build_queue_for(engine, 0)
    queue.clear()
    queue.extend([ShipType.CARRIER, ShipType.DESTROYER])
    legal = engine.enumerate_legal(0)
    assert any(
        isinstance(a, BuildShipAction) and a.ship_type is ShipType.DESTROYER
        for a in legal
    )
    assert not any(
        isinstance(a, BuildShipAction) and a.ship_type is ShipType.CARRIER
        for a in legal
    )
    action = _act(engine, 0, random.Random(0))
    assert action in legal
    assert not isinstance(action, BuildShipAction)
    assert isinstance(action, EndTurnAction)
    assert engine.players[0].cash == 200
    assert list(queue)[:2] == [ShipType.CARRIER, ShipType.DESTROYER]


def test_skips_unbuildable_queue_head() -> None:
    # Carrier is not available on this "map"; the destroyer behind it is.
    options = [
        BuildShipAction(port=(0, 0), spawn_position=(1, 0), ship_type=t)
        for t in (ShipType.DESTROYER, ShipType.MERCHANT)
    ]
    queue = deque([ShipType.CARRIER, ShipType.DESTROYER, ShipType.MERCHANT])
    action = choose_build_action(
        engine=None,
        player_id=0,
        options=options,
        rng=random.Random(0),
        queue=queue,
        types=(ShipType.DESTROYER, ShipType.MERCHANT),
    )
    assert isinstance(action, BuildShipAction)
    assert action.ship_type is ShipType.DESTROYER
    assert list(queue) == [ShipType.MERCHANT]


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


def test_decision_invariant_to_hidden_enemy(tmp_path: Path) -> None:
    """Perturbing an unseen enemy ship must not change the chosen action."""
    # Homes at (0,0) and (19,9), Manhattan 28, outside port detection (16).
    engine = gameEngine(write_two_home_map(tmp_path / "fog.json", 20, 10))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    engine.players[1].cash = 0
    engine._spawn_ship(0, ShipType.DESTROYER, (1, 0))
    hidden = engine._spawn_ship(1, ShipType.BATTLESHIP, (19, 8))
    engine._refresh_sightings()
    assert engine.known_enemy_ships(0) == []
    assert engine.known_enemy_ports(0) == []

    visible = engine.clone()
    perturbed = engine.clone()
    ship = perturbed.ships[hidden.id]
    new_pos = (18, 9)
    perturbed.map.relocate_ship(ship.id, ship.position, new_pos)
    ship.position = new_pos
    perturbed._spawn_ship(1, ShipType.CARRIER, (18, 8))

    a1 = _act(visible, 0, random.Random(0))
    a2 = _act(perturbed, 0, random.Random(0))
    assert a1 in visible.enumerate_legal(0)
    assert a2 == a1
    # And the perturbation stayed hidden.
    assert hidden.id not in {s.ship_id for s in visible.known_enemy_ships(0)}
    assert all(
        s.position not in ((18, 9), (18, 8)) for s in perturbed.known_enemy_ships(0)
    )


def test_landing_explores_far_side_not_unsighted_home(tmp_path: Path) -> None:
    """With no port sightings, a landing explores opposite our home, not the hidden enemy home."""
    # P0 home (0, 4), P1 home (19, 0): far-side explore target is (19, 5), not (19, 0).
    # Homes are 23 apart, so neither port is in the other's detection range.
    path = tmp_path / "explore.json"
    path.write_text(
        '{"width": 20, "height": 10, "seed": 0,'
        ' "land": [[0, 4], [19, 0]],'
        ' "ports": {"0": [[0, 4]], "1": [[19, 0]]}}'
    )
    engine = gameEngine(str(path))
    engine.reset(seed=0)
    engine.players[0].cash = 0
    engine.players[1].cash = 0
    landing = engine._spawn_ship(0, ShipType.LANDING, (1, 4))
    engine._refresh_sightings()
    assert engine.known_enemy_ports(0) == []
    action = _act(engine, 0, random.Random(0))
    assert isinstance(action, MoveAction)
    assert action.ship_id == landing.id
    explore = (19, 5)
    hidden_home = (19, 0)
    before_explore = engine.map.manhattan(landing.position, explore)
    after_explore = engine.map.manhattan(action.destination, explore)
    before_home = engine.map.manhattan(landing.position, hidden_home)
    after_home = engine.map.manhattan(action.destination, hidden_home)
    assert after_explore < before_explore
    # Must not be uniquely chasing the hidden home (a diagonal step toward
    # (19, 5) also shrinks y-distance to (19, 0) by 0, or x by 1 equally).
    # A greedy step toward (19, 0) would decrease y; toward (19, 5) decreases
    # x and keeps y or moves y toward 5. From (1, 4) that's +x, not -y.
    assert action.destination[1] >= landing.position[1]
    assert after_home <= before_home or after_explore < before_explore
