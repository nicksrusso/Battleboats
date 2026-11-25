"""Fog Commander agent: fog invariance, determinism, legality, doctrine sampling."""

from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from battleboats.agents.fog_commander import (
    ALL_TYPES,
    DOCTRINE_NAMES,
    FogCommander,
    FogCommanderConfig,
    canonical_build_action,
    engine_accepts,
    fog_commander_action,
    fog_commander_agent,
    init_memory,
)
from battleboats.agents.random_agent import random_action
from battleboats.agents.registry import make_agent
from battleboats.agents.scripted import scripted_action
from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import BuildShipAction, EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_data import PORT, spot_distance
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.ui.maps import resolve_map

OPPONENTS = {
    "scripted": scripted_action,
    "random": random_action,
    "tactical": tactical_scripted_action,
}


def _engine(tmp_path: Path, preset: str, seed: int) -> gameEngine:
    rm = resolve_map(preset, seed=seed, map_dir=tmp_path)
    e = gameEngine(str(rm.path))
    e.reset(seed=seed)
    return e


def _sees(engine: gameEngine, pid: int, pos, target_type) -> bool:
    for sid in engine.players[pid].owned_ship_ids:
        s = engine.ships[sid]
        if engine.map.manhattan(s.position, pos) <= spot_distance(s.type, target_type):
            return True
    for ppos in engine.players[pid].owned_port_positions:
        if engine.map.manhattan(ppos, pos) <= spot_distance(PORT, target_type):
            return True
    return False


def _hidden_water(engine: gameEngine, pid: int, ship_type: ShipType):
    m = engine.map
    out = []
    for x in range(m.width):
        for y in range(m.height):
            p = (x, y)
            if m.is_water(p) and not m.is_occupied(p) and not _sees(engine, pid, p, ship_type):
                out.append(p)
    return out


def _perturb_hidden(engine: gameEngine, pid: int, rng: random.Random) -> gameEngine:
    """Clone ``engine`` and change only state ``pid`` cannot see."""
    e2 = engine.clone()
    opp = 1 - pid
    hidden = sorted(
        sid for sid in e2.players[opp].owned_ship_ids if not e2._can_observer_see(pid, e2.ships[sid])
    )
    for sid in hidden:
        ship = e2.ships[sid]
        spots = _hidden_water(e2, pid, ship.type)
        if spots:
            dst = rng.choice(spots)
            e2.map.relocate_ship(sid, ship.position, dst)
            ship.position = dst
        ship.cargo = rng.randrange(0, 101) if ship.type == ShipType.MERCHANT else ship.cargo
        ship.has_attacked = not ship.has_attacked
        ship.tiles_moved_this_turn = rng.randrange(0, ship.stats.speed + 1)
    if len(hidden) >= 2:  # remove one hidden ship without touching pid's (stale) records
        victim = e2.ships.pop(hidden[0])
        e2.players[opp].owned_ship_ids.discard(victim.id)
        e2.map.remove_ship(victim.position)
    spots = _hidden_water(e2, pid, ShipType.SUBMARINE)
    if spots:
        e2._spawn_ship(opp, ShipType.SUBMARINE, rng.choice(spots))
    e2.players[opp].cash += 777
    for pos, port in e2.ports.items():
        if port.owner == opp:
            port.stockpile += 50
    e2.rng = np.random.default_rng(rng.randrange(10**9))
    # The engine's canonical spawn tile points at the opponent home; move it.
    e2.players[opp].home_port = (e2.map.width - 1 - e2.players[opp].home_port[0], e2.players[opp].home_port[1])
    e2._refresh_sightings()
    return e2


def _seat_view_equal(a: gameEngine, b: gameEngine, pid: int) -> bool:
    pa, pb = a.players[pid], b.players[pid]
    return pa.sightings == pb.sightings and pa.port_sightings == pb.port_sightings


def _memory_signature(mem):
    return (
        mem.doctrine.name,
        mem.planned_buy,
        dict(mem.built_counts),
        mem.ports_built,
        dict(mem.builder_assignments),
        dict(mem.pruned),
        frozenset(mem.checked_coast),
        mem.steps_in_turn,
        {k: frozenset(v) for k, v in mem.visited.items()},
        frozenset(mem.sidestepped),
    )


@pytest.mark.parametrize("preset,opponent,seed", [
    ("32x16", "self", 11),
    ("32x16", "scripted", 12),
    ("16x8", "self", 13),
    ("48x24", "random", 14),
])
def test_fog_invariance_hidden_state_never_changes_the_action(tmp_path, preset, opponent, seed):
    engine = _engine(tmp_path, preset, seed)
    fc = FogCommander()
    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    prng = random.Random(seed)
    checks = 0
    steps = 0
    while engine.winner is None and engine.turn < 45 and steps < 2500:
        pid = engine.current_player
        is_fc = opponent == "self" or pid == 0
        if is_fc:
            e1 = engine.clone()
            e2 = _perturb_hidden(engine, pid, prng)
            assert _seat_view_equal(e1, e2, pid), "perturbation leaked into the seat's view"
            a1_agent, a2_agent = FogCommander(), FogCommander()
            a1_agent.memory = copy.deepcopy(fc.memory)
            a2_agent.memory = copy.deepcopy(fc.memory)
            r1, r2 = random.Random(), random.Random()
            r1.setstate(rngs[pid].getstate())
            r2.setstate(rngs[pid].getstate())
            a1 = a1_agent.act(e1, pid, r1)
            a2 = a2_agent.act(e2, pid, r2)
            assert a1 == a2, f"step {steps} turn {engine.turn}: {a1} != {a2}"
            assert r1.getstate() == r2.getstate()
            if pid in a1_agent.memory:
                assert _memory_signature(a1_agent.memory[pid]) == _memory_signature(a2_agent.memory[pid])
            checks += 1
        fn = fc.act if is_fc else OPPONENTS[opponent]
        engine.step(fn(engine, pid, rngs[pid]))
        steps += 1
    assert checks >= 20


def test_spawn_tile_ignores_true_enemy_home(tmp_path):
    engine = _engine(tmp_path, "32x16", 5)
    fc = FogCommander()
    rng = random.Random(1)
    for _ in range(4000):
        if engine.winner is not None:
            break
        pid = engine.current_player
        mem = copy.deepcopy(fc.memory)
        state = rng.getstate()
        a = fc.act(engine, pid, rng)
        if isinstance(a, BuildShipAction):
            moved = engine.clone()
            opp = 1 - pid
            moved.players[opp].home_port = (0, 0) if moved.players[opp].home_port != (0, 0) else (1, 1)
            other = FogCommander()
            other.memory = mem
            r2 = random.Random()
            r2.setstate(state)
            assert other.act(moved, pid, r2) == a
            assert engine_accepts(engine, pid, a)
            return
        engine.step(a)
    pytest.fail("fog_commander never bought a ship")


def test_unsighted_ports_are_land_not_water(tmp_path):
    engine = _engine(tmp_path, "32x16", 21)
    for pid in (0, 1):
        mem = init_memory(engine, pid, random.Random(0), FogCommanderConfig())
        hidden_enemy_ports = [
            pos for pos, port in engine.ports.items()
            if port.owner != pid and pos not in engine.players[pid].port_sightings
        ]
        for pos in engine.ports:
            assert pos in mem.land and pos not in mem.water
        for pos in hidden_enemy_ports:
            assert pos in mem.coast  # a search candidate, never navigable water


def _play(tmp_path, preset, seed, kinds, max_turns=60, functional=False):
    engine = _engine(tmp_path, preset, seed)
    rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
    agents = {pid: make_agent(kinds[pid]) for pid in (0, 1)}
    log = []
    bad = []
    per_turn = Counter()
    while engine.winner is None and engine.turn < max_turns and len(log) < 20000:
        pid = engine.current_player
        legal = engine.enumerate_legal(pid)
        if functional and kinds[pid] == "fog_commander":
            a = fog_commander_action(engine, pid, rngs[pid])
        else:
            a = agents[pid].act(engine, pid, rngs[pid])
        if kinds[pid] == "fog_commander":
            per_turn[(pid, engine.turn)] += 1
            if not engine_accepts(engine, pid, a, legal):
                bad.append((engine.turn, a))
            if not isinstance(a, BuildShipAction):
                assert a in legal
        log.append(a)
        engine.step(a)
    return engine, log, bad, per_turn, agents


@pytest.mark.parametrize("kinds", [("fog_commander", "fog_commander"), ("fog_commander", "scripted"),
                                   ("random", "fog_commander")])
def test_deterministic_per_seed(tmp_path, kinds):
    _, log1, _, _, _ = _play(tmp_path / "a", "32x16", 3, kinds, max_turns=40)
    _, log2, _, _, _ = _play(tmp_path / "b", "32x16", 3, kinds, max_turns=40)
    assert log1 == log2
    _, log3, _, _, _ = _play(tmp_path / "c", "32x16", 4, kinds, max_turns=40)
    assert log3 != log1


def test_functional_wrapper_matches_class_and_resets(tmp_path):
    kinds = ("fog_commander", "fog_commander")
    _, log1, _, _, _ = _play(tmp_path / "a", "16x8", 8, kinds, max_turns=30)
    engine, log2, _, _, _ = _play(tmp_path / "b", "16x8", 8, kinds, max_turns=30, functional=True)
    assert log1 == log2
    agent = fog_commander_agent(engine)
    assert agent is not None and set(agent.memory) == {0, 1}
    engine.reset(seed=8)  # new players -> stale memory must not be reused
    assert fog_commander_agent(engine) is None
    assert fog_commander_agent(engine.clone()) is None


GAMES = [
    ("16x8", s, k) for s, k in zip(range(100, 112), [("fog_commander", "fog_commander"), ("fog_commander", "scripted"),
                                                     ("scripted", "fog_commander"), ("fog_commander", "random"),
                                                     ("random", "fog_commander"), ("tactical", "fog_commander")] * 2)
] + [
    ("open-8x8", s, ("fog_commander", "fog_commander")) for s in range(200, 206)
] + [
    ("32x16", s, k) for s, k in zip(range(300, 306), [("fog_commander", "fog_commander"), ("scripted", "fog_commander"),
                                                     ("fog_commander", "random")] * 2)
] + [
    ("48x24", 400, ("fog_commander", "fog_commander")),
    ("24x12", 401, ("fog_commander", "scripted")),
]


@pytest.mark.parametrize("preset,seed,kinds", GAMES)
def test_no_illegal_actions(tmp_path, preset, seed, kinds):
    engine, log, bad, per_turn, _ = _play(tmp_path, preset, seed, kinds, max_turns=60)
    assert bad == []
    assert max(per_turn.values()) <= FogCommanderConfig().max_steps_per_turn + 1
    assert log, "no actions taken"


def test_doctrine_sampling_covers_all_and_is_seeded(tmp_path):
    engine = _engine(tmp_path, "16x8", 1)
    names = Counter()
    for seed in range(300):
        m1 = init_memory(engine, 0, random.Random(seed), FogCommanderConfig())
        m2 = init_memory(engine, 0, random.Random(seed), FogCommanderConfig())
        assert m1.doctrine == m2.doctrine and m1.jitter == m2.jitter
        assert set(m1.jitter) == set(ALL_TYPES)
        assert all(0.85 <= v <= 1.15 for v in m1.jitter.values())
        names[m1.doctrine.name] += 1
    assert set(names) == set(DOCTRINE_NAMES)
    assert min(names.values()) >= 25  # ~50 each when uniform
    forced = init_memory(engine, 0, random.Random(0), FogCommanderConfig(doctrine="CAPITAL_ARMADA"))
    assert forced.doctrine.name == "CAPITAL_ARMADA"


def test_seats_sample_doctrines_independently(tmp_path):
    pairs = Counter()
    for seed in range(40):
        engine = _engine(tmp_path, "16x8", 2)
        fc = FogCommander()
        rngs = {0: random.Random(seed * 2 + 1), 1: random.Random(seed * 2 + 2)}
        while len(fc.memory) < 2 and engine.winner is None:
            pid = engine.current_player
            engine.step(fc.act(engine, pid, rngs[pid]))
        pairs[(fc.doctrine_for(0), fc.doctrine_for(1))] += 1
    assert len({a for a, _ in pairs}) >= 4 and len({b for _, b in pairs}) >= 4
    assert sum(v for (a, b), v in pairs.items() if a != b) >= 20


def test_buys_span_many_types(tmp_path):
    buys = Counter()
    for seed in range(6):
        _, log, _, _, _ = _play(tmp_path / str(seed), "32x16", 500 + seed, ("fog_commander", "fog_commander"),
                                max_turns=60)
        buys.update(a.ship_type for a in log if isinstance(a, BuildShipAction))
    assert len(buys) >= 6, buys


def test_canonical_build_action_maps_to_legal(tmp_path):
    engine = _engine(tmp_path, "16x8", 9)
    legal = engine.enumerate_legal(0)
    b = next(a for a in legal if isinstance(a, BuildShipAction))
    px, py = b.port
    alt = [(px + dx, py + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    alt = [p for p in alt if engine.map.in_bounds(p) and engine.map.is_water(p) and not engine.map.is_occupied(p)
           and p != b.spawn_position]
    if not alt:
        pytest.skip("port has a single spawn tile on this map")
    other = BuildShipAction(b.port, alt[0], b.ship_type)
    assert other not in legal
    assert engine_accepts(engine, 0, other, legal)
    assert canonical_build_action(engine, 0, other) == b
    assert canonical_build_action(engine, 0, EndTurnAction()) == EndTurnAction()


def test_harvest_labels_fog_commander_decisions(tmp_path):
    from scripts.harvest_tactical import _play_game

    rm = resolve_map("16x8", seed=5, map_dir=tmp_path)
    r = _play_game(rm.path, max_turns=30, max_steps=3000, seed=5,
                   actors={0: "fog_commander", 1: "fog_commander"})
    assert r["n_labels"] > 0
    assert r["n_errors"] == 0, r["error_kinds"]
    assert r["verb_counts"].get("build_ship", 0) > 0
