"""Scripted baseline: legal actions, randomized buy queue, determinism."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from battleboats.agents.scripted import (
    BUYABLE_TYPES,
    build_queue_for,
    sample_build_cycle,
    scripted_action,
)
from battleboats.core.actions import (
    BuildShipAction,
    EndTurnAction,
)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.skill_maps import write_full_map, write_two_home_map


def test_buyable_types_cover_the_fleet() -> None:
    assert set(BUYABLE_TYPES) == set(ShipType)
    assert len(BUYABLE_TYPES) == len(set(BUYABLE_TYPES))


def test_sample_build_cycle_is_a_shuffle_of_every_type() -> None:
    rng = random.Random(0)
    first = sample_build_cycle(rng)
    second = sample_build_cycle(rng)
    assert sorted(first, key=lambda t: t.name) == sorted(BUYABLE_TYPES, key=lambda t: t.name)
    assert sorted(second, key=lambda t: t.name) == sorted(BUYABLE_TYPES, key=lambda t: t.name)
    # Same seed always yields the same first cycle; a second draw differs.
    assert first == sample_build_cycle(random.Random(0))
    assert first != second
    counts = Counter()
    for seed in range(200):
        counts.update(sample_build_cycle(random.Random(seed)))
    expected = 200
    for t, n in counts.items():
        assert n == expected, (t, n, dict(counts))


def test_scripted_buy_follows_queue_order(tmp_path: Path) -> None:
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
        action = scripted_action(engine, 0, rng)
        assert isinstance(action, BuildShipAction)
        assert action.ship_type is expected
        assert action in engine.enumerate_legal(0)
        engine.step(action)
        bought.append(action.ship_type)
    assert bought == planned


def test_scripted_saves_when_short_of_queue_head(tmp_path: Path) -> None:
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
    action = scripted_action(engine, 0, random.Random(0))
    assert action in legal
    assert not isinstance(action, BuildShipAction)
    assert isinstance(action, EndTurnAction)
    assert engine.players[0].cash == 200
    assert list(queue)[:2] == [ShipType.CARRIER, ShipType.DESTROYER]


def test_scripted_is_deterministic_per_seed(tmp_path: Path) -> None:
    path = write_full_map(tmp_path / "d.json", 6, 6, seed=1)

    def play(seed: int):
        engine = gameEngine(path)
        engine.reset(seed=seed)
        rng = random.Random(seed)
        actions = []
        taken = 0
        while engine.winner is None and engine.turn < 10 and taken < 80:
            pid = engine.current_player
            action = scripted_action(engine, pid, rng)
            actions.append((pid, type(action).__name__, getattr(action, "ship_type", None)))
            engine.step(action)
            taken += 1
        return actions, engine.winner, engine.turn

    a, winner_a, turn_a = play(7)
    b, winner_b, turn_b = play(7)
    c, _, _ = play(8)
    assert a == b
    assert (winner_a, turn_a) == (winner_b, turn_b)
    assert a != c


def test_scripted_queue_resets_on_engine_reset(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "r.json", 6, 6, seed=0))
    engine.reset(seed=0)
    queue = build_queue_for(engine, 0)
    queue.extend([ShipType.CARRIER])
    engine.reset(seed=1)
    fresh = build_queue_for(engine, 0)
    assert list(fresh) == []
    assert fresh is not queue


def test_scripted_buys_varied_types_and_refills(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "v.json", 8, 8, seed=0))
    engine.reset(seed=0)
    engine.players[1].cash = 0
    rngs = {0: random.Random(0), 1: random.Random(1)}
    bought = []
    # Enough cash to walk through more than one full cycle even after
    # the first bought ships block a spawn tile until we end the turn.
    for _ in range(200):
        if engine.winner is not None:
            break
        pid = engine.current_player
        if pid == 0:
            engine.players[0].cash = 1000
        action = scripted_action(engine, pid, rngs[pid])
        if pid == 0 and isinstance(action, BuildShipAction):
            bought.append(action.ship_type)
        engine.step(action)
    assert len(bought) >= len(BUYABLE_TYPES) * 2
    counts = Counter(bought)
    assert set(counts) == set(BUYABLE_TYPES)
    # Shuffled cycles keep rates nearly equal; leftover partial cycle is at most 1.
    assert max(counts.values()) - min(counts.values()) <= 1
