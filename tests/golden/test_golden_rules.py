"""Language-agnostic, behavior-level golden parity tests for the engine API.

Every test interacts only through ``EngineBackend`` / ``EngineInstance`` and
canonical JSON records so both Python and future Rust/PyO3 backends can run the
identical suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from battleboats.agents.random_agent import random_action
from battleboats.envs.battleboats_aec import BattleboatsAEC
from tests.golden.backend import EngineBackend, EngineInstance
from tests.golden.conftest import write_map_json


def _ship_by_owner(state: Dict[str, Any], owner: int) -> List[Dict[str, Any]]:
    return [s for s in state["ships"] if s["owner"] == owner]


def _port_at(state: Dict[str, Any], pos: Tuple[int, int]) -> Dict[str, Any]:
    target = [pos[0], pos[1]]
    for p in state["ports"]:
        if p["position"] == target:
            return p
    raise KeyError(f"No port at {pos}")


def _end_round(engine: EngineInstance, backend: EngineBackend) -> None:
    engine.step(backend.end_turn_action())
    engine.step(backend.end_turn_action())


# ------------------------------------------------------------------ rule tables
def test_rule_constants_and_ship_stats(backend: EngineBackend) -> None:
    assert backend.starting_cash() == 250
    assert backend.port_production() == 25
    assert backend.cash_per_material() == 2
    assert backend.merchant_capacity() == 100

    expected_types = {
        "Carrier": (6, 250, 100.0, 12),
        "Battleship": (3, 200, 125.0, 4),
        "Cruiser": (5, 150, 75.0, 3),
        "Destroyer": (5, 100, 50.0, 3),
        "Submarine": (3, 125, 50.0, 3),
        "Merchant": (3, 100, 0.0, 0),
        "Landing": (3, 250, 0.0, 0),
        "Builder": (3, 250, 0.0, 0),
    }
    assert set(backend.ship_types()) == set(expected_types)
    for name, (speed, cost, strength, arange) in expected_types.items():
        stats = backend.base_stats(name)
        assert stats == {
            "speed": speed,
            "cost": cost,
            "strength": strength,
            "attack_range": arange,
        }


def test_attack_modifiers_and_spot_matrix_asymmetry(backend: EngineBackend) -> None:
    assert backend.attack_modifier("Destroyer", "Submarine") == 2.0
    assert backend.attack_modifier("Submarine", "Destroyer") == 0.5
    assert backend.attack_modifier("Submarine", "Battleship") == 4.0
    assert backend.attack_modifier("Carrier", "Battleship") == 1.5
    assert backend.attack_modifier("Cruiser", "Cruiser") == 1.0

    # Asymmetric spotting: Destroyer spots Sub at 3; Sub spots Destroyer at 2.
    assert backend.spot_distance("Destroyer", "Submarine") == 3.0
    assert backend.spot_distance("Submarine", "Destroyer") == 2.0
    # Sub spots Battleship at 4; Battleship spots Sub at 1.
    assert backend.spot_distance("Submarine", "Battleship") == 4.0
    assert backend.spot_distance("Battleship", "Submarine") == 1.0
    # Port spotting
    assert backend.spot_distance("Port", "Destroyer") == 12.0
    assert backend.spot_distance("Port", "Submarine") == 1.0
    assert backend.spot_distance("Port", "Port") == 16.0


# ------------------------------------------------------------------ combat
def test_combat_defenseless_always_dies_and_non_combat_cannot_attack(
    backend: EngineBackend, tmp_path: Path
) -> None:
    mp = write_map_json(tmp_path / "m.json", 12, 6, [[0, 0]], [[11, 5]])
    e = backend.create(mp)
    for non_combat in ("Merchant", "Landing", "Builder"):
        e.reset_from_scenario(
            {
                "seed": 7,
                "map_path": str(mp),
                "player_0": {
                    "cash": 0,
                    "ships": [
                        {"type": "BATTLESHIP", "position": [4, 2]},
                        {"type": non_combat.upper(), "position": [4, 3]},
                    ],
                },
                "player_1": {
                    "cash": 0,
                    "ships": [{"type": non_combat.upper(), "position": [5, 2]}],
                },
            }
        )
        s0 = e.canonical_state()
        bship = next(s for s in s0["ships"] if s["type"] == "Battleship")
        nc0 = next(s for s in s0["ships"] if s["owner"] == 0 and s["type"] == non_combat)
        nc1 = next(s for s in s0["ships"] if s["owner"] == 1)

        # Non-combat ship cannot attack
        e.step(backend.attack_action(nc0["id"], nc1["id"]))
        assert len(e.canonical_state()["ships"]) == 3

        # Strength-0 defender always dies
        e.step(backend.attack_action(bship["id"], nc1["id"]))
        s1 = e.canonical_state()
        assert [s["id"] for s in s1["ships"]] == [bship["id"], nc0["id"]]
        bship_after = next(s for s in s1["ships"] if s["id"] == bship["id"])
        assert bship_after["has_attacked"] is True


def test_combat_once_per_turn_range_friendly_fire_and_wrong_owner(
    backend: EngineBackend, tmp_path: Path
) -> None:
    mp = write_map_json(tmp_path / "m.json", 20, 8, [[0, 0]], [[19, 7]])
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 1,
            "map_path": str(mp),
            "player_0": {
                "cash": 0,
                "ships": [
                    {"type": "BATTLESHIP", "position": [5, 3]},
                    {"type": "BUILDER", "position": [6, 3]},
                ],
            },
            "player_1": {
                "cash": 0,
                "ships": [
                    {"type": "BUILDER", "position": [5, 4]},   # dist 1 (in range)
                    {"type": "BUILDER", "position": [5, 5]},   # dist 2 (in range)
                    {"type": "BUILDER", "position": [10, 3]},  # dist 5 (out of range 4)
                ],
            },
        }
    )
    st = e.canonical_state()
    p0_bship = next(s["id"] for s in st["ships"] if s["owner"] == 0 and s["type"] == "Battleship")
    p0_friend = next(s["id"] for s in st["ships"] if s["owner"] == 0 and s["type"] == "Builder")
    p1_ships = sorted(
        (s for s in st["ships"] if s["owner"] == 1), key=lambda s: s["position"][0] + s["position"][1]
    )
    t_near, t_second, t_far = p1_ships[0]["id"], p1_ships[1]["id"], p1_ships[2]["id"]

    # Friendly fire -> no-op
    e.step(backend.attack_action(p0_bship, p0_friend))
    assert len(e.canonical_state()["ships"]) == 5

    # Out of range -> no-op
    e.step(backend.attack_action(p0_bship, t_far))
    assert len(e.canonical_state()["ships"]) == 5

    # Wrong owner attacking -> no-op
    e.step(backend.attack_action(t_near, p0_bship))
    assert len(e.canonical_state()["ships"]) == 5

    # Legal attack kills t_near
    e.step(backend.attack_action(p0_bship, t_near))
    alive_ids = {s["id"] for s in e.canonical_state()["ships"]}
    assert t_near not in alive_ids
    assert t_second in alive_ids

    # Second attack same turn -> no-op
    e.step(backend.attack_action(p0_bship, t_second))
    assert t_second in {s["id"] for s in e.canonical_state()["ships"]}

    # Opponent ending turn does NOT reset P0's has_attacked; P0's own end-turn DOES.
    e.step(backend.end_turn_action())  # P0 -> P1 (resets P0 flags at P0's end_turn)
    b_after_p0 = next(s for s in e.canonical_state()["ships"] if s["id"] == p0_bship)
    assert b_after_p0["has_attacked"] is False
    e.step(backend.end_turn_action())  # P1 -> P0
    e.step(backend.attack_action(p0_bship, t_second))
    assert t_second not in {s["id"] for s in e.canonical_state()["ships"]}


def test_only_attack_what_you_spot_yourself_rule(backend: EngineBackend, tmp_path: Path) -> None:
    """Carrier has attack_range=12, but spot_distance(Carrier, Destroyer)=3.

    Even when a friendly port (spot_distance(Port, Destroyer)=12) spots the
    enemy Destroyer so it is in visible_enemy_ships(0), the Carrier MUST NOT
    be able to attack until the Carrier moves within its own spotting distance.
    """
    mp = write_map_json(tmp_path / "m.json", 24, 10, [[1, 4]], [[22, 4]])
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {
                "cash": 0,
                "ships": [{"type": "CARRIER", "position": [2, 4]}],
            },
            "player_1": {
                "cash": 0,
                "ships": [{"type": "DESTROYER", "position": [7, 4]}],
            },
        }
    )
    # Distance from P0 home port (1, 4) to enemy destroyer (7, 4) is 6 <= 12 -> spotted by port!
    # Distance from Carrier (2, 4) to enemy destroyer (7, 4) is 5 <= 12 (attack_range),
    # but 5 > spot_distance(Carrier, Destroyer) == 3.
    vis = e.visible_enemy_ships(0)
    assert len(vis) == 1
    st = e.canonical_state()
    carrier_id = next(s["id"] for s in st["ships"] if s["owner"] == 0)
    enemy_id = next(s["id"] for s in st["ships"] if s["owner"] == 1)

    legal = e.enumerate_legal(0)
    assert backend.attack_action(carrier_id, enemy_id) not in legal
    h_before = e.state_hash()
    e.step(backend.attack_action(carrier_id, enemy_id))
    assert e.state_hash() == h_before  # no-op

    # Move Carrier 2 tiles east to (4, 4): now Manhattan distance is 3 == spot_distance(Carrier, Destroyer).
    e.step(backend.move_action(carrier_id, (3, 4)))
    e.step(backend.move_action(carrier_id, (4, 4)))
    legal_after = e.enumerate_legal(0)
    assert backend.attack_action(carrier_id, enemy_id) in legal_after


@pytest.mark.parametrize(
    "att_type,def_type,k,expected_p,tol",
    [
        ("SUBMARINE", "SUBMARINE", 2.0, 0.50, 0.08),
        ("DESTROYER", "SUBMARINE", 2.0, 0.80, 0.08),
        ("SUBMARINE", "DESTROYER", 2.0, 0.20, 0.08),
        ("CRUISER", "DESTROYER", 1.0, 2.25 / 3.25, 0.08),
        ("CRUISER", "DESTROYER", 4.0, (2.25**4) / (1.0 + 2.25**4), 0.05),
    ],
)
def test_combat_pk_formula_and_kill_curve_k(
    backend: EngineBackend,
    tmp_path: Path,
    att_type: str,
    def_type: str,
    k: float,
    expected_p: float,
    tol: float,
) -> None:
    """Verify P(kill) = x^k / (1 + x^k) across type modifiers and k values using public step()."""
    mp = write_map_json(tmp_path / "pk.json", 24, 10, [[0, 0]], [[23, 9]])
    e = backend.create(mp, kill_curve_k=k)
    n_pairs = 10
    p0_ships = [{"type": att_type, "position": [2 + 2 * i, 4]} for i in range(n_pairs)]
    p1_ships = [{"type": def_type, "position": [2 + 2 * i, 5]} for i in range(n_pairs)]
    kills = 0
    trials = 0
    for seed in range(30):
        e.reset_from_scenario(
            {
                "seed": seed,
                "map_path": str(mp),
                "player_0": {"cash": 0, "ships": p0_ships},
                "player_1": {"cash": 0, "ships": p1_ships},
            }
        )
        st = e.canonical_state()
        attackers = sorted((s for s in st["ships"] if s["owner"] == 0), key=lambda s: s["position"][0])
        defenders = sorted((s for s in st["ships"] if s["owner"] == 1), key=lambda s: s["position"][0])
        for a, d in zip(attackers, defenders):
            e.step(backend.attack_action(a["id"], d["id"]))
        remaining_p1 = sum(1 for s in e.canonical_state()["ships"] if s["owner"] == 1)
        kills += n_pairs - remaining_p1
        trials += n_pairs
    rate = kills / trials
    assert abs(rate - expected_p) < tol, f"expected ~{expected_p:.3f}, got {rate:.3f}"


# ------------------------------------------------------------------ movement
def test_movement_rules_and_speed_budget(backend: EngineBackend, tmp_path: Path) -> None:
    mp = write_map_json(
        tmp_path / "move.json",
        10,
        6,
        [[0, 0]],
        [[9, 5]],
        extra_land=[[3, 2]],
    )
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {
                "cash": 0,
                "ships": [
                    {"type": "SUBMARINE", "position": [2, 2]},  # speed 3
                    {"type": "BUILDER", "position": [2, 3]},    # blocks (2, 3)
                ],
            },
            "player_1": {"cash": 0, "ships": []},
        }
    )
    st = e.canonical_state()
    sub = next(s for s in st["ships"] if s["type"] == "Submarine")
    sid = sub["id"]

    # Diagonal, 2-tile jump, land tile (3,2), occupied tile (2,3), out-of-bounds -> all no-ops
    for bad in [(3, 3), (4, 2), (3, 2), (2, 3), (-1, 2)]:
        h = e.state_hash()
        e.step(backend.move_action(sid, bad))
        assert e.state_hash() == h

    # Move 3 cardinal tiles (speed 3): (2,1) -> (3,1) -> (4,1)
    for dst in [(2, 1), (3, 1), (4, 1)]:
        e.step(backend.move_action(sid, dst))
    sub_after = next(s for s in e.canonical_state()["ships"] if s["id"] == sid)
    assert sub_after["position"] == [4, 1]
    assert sub_after["tiles_moved_this_turn"] == 3

    # 4th move in same turn -> no-op
    h = e.state_hash()
    e.step(backend.move_action(sid, (5, 1)))
    assert e.state_hash() == h

    # End P0 turn -> P0's tiles_moved_this_turn resets; P1 ends turn -> P0 moves again
    _end_round(e, backend)
    sub_r2 = next(s for s in e.canonical_state()["ships"] if s["id"] == sid)
    assert sub_r2["tiles_moved_this_turn"] == 0
    e.step(backend.move_action(sid, (5, 1)))
    assert next(s for s in e.canonical_state()["ships"] if s["id"] == sid)["position"] == [5, 1]


def test_bought_ship_waits_one_turn_but_scenario_ship_acts_immediately(
    backend: EngineBackend, tmp_path: Path
) -> None:
    mp = write_map_json(tmp_path / "fresh.json", 10, 4, [[1, 1]], [[8, 2]])
    e = backend.create(mp)
    e.reset(seed=0)
    # P0 starts with 250 cash -> buy a Destroyer (cost 100)
    legal_buys = [
        a for a in e.enumerate_legal(0)
        if backend.encode_action(a)[0] == "b" and backend.encode_action(a)[5] == "Destroyer"
    ]
    assert len(legal_buys) == 1
    e.step(legal_buys[0])
    st = e.canonical_state()
    assert st["players"][0]["cash"] == 150
    dd = st["ships"][0]
    assert dd["bought_this_turn"] is True
    assert dd["built_turn"] == 0
    # Bought ship has 0 legal actions this turn and direct move is no-op
    assert not any(
        backend.encode_action(a)[0] in ("m", "a", "p", "c", "l", "u")
        for a in e.enumerate_legal(0)
    )
    h = e.state_hash()
    e.step(backend.move_action(dd["id"], (dd["position"][0], dd["position"][1] + 1)))
    assert e.state_hash() == h

    _end_round(e, backend)
    dd_r2 = e.canonical_state()["ships"][0]
    assert dd_r2["bought_this_turn"] is False
    assert any(backend.encode_action(a)[0] == "m" for a in e.enumerate_legal(0))


# ------------------------------------------------------------------ visibility / fog
def test_fog_visibility_stale_witness_kill_and_port_sightings(
    backend: EngineBackend, tmp_path: Path
) -> None:
    # Wide board so ports at (0,0) and (39,9) don't interfere around x=18..25
    mp = write_map_json(tmp_path / "fog.json", 40, 10, [[0, 0]], [[39, 9]])
    e = backend.create(mp)

    # 1. Fresh reset: no sightings
    e.reset(seed=0)
    assert e.visible_enemy_ships(0) == []
    assert e.known_enemy_ships(0) == []
    assert e.visible_enemy_ports(0) == []
    assert e.known_enemy_ports(0) == []

    # 2. Asymmetric detection: Sub (0) at (18, 5) vs Battleship (1) at (21, 5) -> dist 3
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {"cash": 0, "ships": [{"type": "SUBMARINE", "position": [18, 5]}]},
            "player_1": {"cash": 0, "ships": [{"type": "BATTLESHIP", "position": [21, 5]}]},
        }
    )
    assert len(e.visible_enemy_ships(0)) == 1
    assert e.visible_enemy_ships(1) == []
    assert e.known_enemy_ships(1) == []

    # 3. Sighting goes stale when scout moves away; turn_seen freezes across rounds
    st = e.canonical_state()
    sub_id = next(s["id"] for s in st["ships"] if s["owner"] == 0)
    bship_id = next(s["id"] for s in st["ships"] if s["owner"] == 1)
    e.step(backend.move_action(sub_id, (17, 5)))  # dist 4 <= 4 -> still fresh
    assert e.known_enemy_ships(0)[0]["fresh"] is True
    e.step(backend.move_action(sub_id, (16, 5)))  # dist 5 > 4 -> stale!
    stale = e.known_enemy_ships(0)[0]
    assert stale["fresh"] is False
    assert stale["position"] == [21, 5]
    assert stale["turn_seen"] == 0
    assert e.visible_enemy_ships(0) == []

    for _ in range(3):
        _end_round(e, backend)
    persisted = e.known_enemy_ships(0)[0]
    assert persisted["fresh"] is False
    assert persisted["turn_seen"] == 0
    assert persisted["ship_id"] == bship_id


def test_witnessed_vs_unwitnessed_kill_and_port_capture_sightings(
    backend: EngineBackend, tmp_path: Path
) -> None:
    mp = write_map_json(tmp_path / "fog2.json", 30, 8, [[0, 0]], [[29, 7], [20, 4]])
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {
                "cash": 0,
                "ships": [
                    {"type": "SUBMARINE", "position": [10, 4]},  # spot_dist(Sub, Builder)=2
                    {"type": "BATTLESHIP", "position": [22, 4]},
                    {"type": "LANDING", "position": [20, 5]},
                ],
            },
            "player_1": {
                "cash": 0,
                "ships": [
                    {"type": "BUILDER", "position": [12, 4]},     # witnessed by P0 Sub at dist 2
                    {"type": "BUILDER", "position": [22, 5]},     # witnessed by P0 Bship at dist 1
                    {"type": "DESTROYER", "position": [19, 4]},   # witnesses capture of (20,4)
                ],
            },
        }
    )
    st = e.canonical_state()
    p0_sub = next(s["id"] for s in st["ships"] if s["owner"] == 0 and s["type"] == "Submarine")
    p0_bship = next(s["id"] for s in st["ships"] if s["owner"] == 0 and s["type"] == "Battleship")
    p0_land = next(s["id"] for s in st["ships"] if s["owner"] == 0 and s["type"] == "Landing")
    p1_b1 = next(s["id"] for s in st["ships"] if s["owner"] == 1 and s["position"] == [12, 4])
    p1_b2 = next(s["id"] for s in st["ships"] if s["owner"] == 1 and s["position"] == [22, 5])

    # Witnessed port sighting of (20, 4)
    p0_ports = {tuple(p["position"]): p for p in e.visible_enemy_ports(0)}
    assert (20, 4) in p0_ports and p0_ports[(20, 4)]["is_home"] is False

    # Move P0 sub away so p1_b1 becomes stale for P0
    e.step(backend.move_action(p0_sub, (9, 4)))
    assert not next(s for s in e.known_enemy_ships(0) if s["ship_id"] == p1_b1)["fresh"]

    # Witnessed kill of p1_b2 clears P0's sighting of p1_b2
    e.step(backend.attack_action(p0_bship, p1_b2))
    assert all(s["ship_id"] != p1_b2 for s in e.known_enemy_ships(0))

    # P0 captures (20, 4): removed from P0 port_sightings, and P1 witness sees owner flipped to 0
    e.step(backend.capture_port_action(p0_land, (20, 4)))
    assert all(p["position"] != [20, 4] for p in e.known_enemy_ports(0))
    p1_seen = next(p for p in e.visible_enemy_ports(1) if p["position"] == [20, 4])
    assert p1_seen["last_known_owner"] == 0


# ------------------------------------------------------------------ legal actions & canonical spawn
def test_legal_actions_canonical_spawn_and_noncanonical_step(
    backend: EngineBackend, tmp_path: Path
) -> None:
    # P0 home at (2, 2) has 4 water neighbors; P1 home at (9, 2).
    # Closest to (9, 2) is (3, 2).
    mp = write_map_json(tmp_path / "spawn.json", 12, 5, [[2, 2]], [[9, 2]])
    e = backend.create(mp)
    e.reset(seed=0)

    legal = e.enumerate_legal(0)
    assert e.enumerate_legal(1) == []  # not current player
    assert backend.end_turn_action() in legal

    buys = [a for a in legal if backend.encode_action(a)[0] == "b"]
    # Starting cash 250 -> all 8 ship types affordable
    assert len(buys) == 8
    assert all(backend.encode_action(a)[3:5] == [3, 2] for a in buys)

    # Non-canonical valid adjacent water spawn (1, 2) is ALSO accepted by step()
    e.step(backend.build_ship_action((2, 2), (1, 2), "Destroyer"))
    st = e.canonical_state()
    assert len(st["ships"]) == 1
    assert st["ships"][0]["position"] == [1, 2]
    assert st["players"][0]["cash"] == 150


# ------------------------------------------------------------------ economy, ports, builders, merchants, win
def test_economy_merchant_builder_and_capture_win(
    backend: EngineBackend, tmp_path: Path
) -> None:
    # P0 home (1, 1), P0 satellite (3, 1), empty land (5, 1); P1 home (9, 1)
    mp = write_map_json(
        tmp_path / "econ.json",
        12,
        4,
        [[1, 1], [3, 1]],
        [[9, 1]],
        extra_land=[[5, 1]],
    )
    e = backend.create(mp)
    e.reset(seed=0)
    assert e.canonical_state()["players"][0]["cash"] == 250

    # Complete round 0 -> on P0's turn start (after P1 ends turn), P0 gets +50 cash and sat (3,1) gets +25 stockpile.
    # Also on P1's turn start (after P0 ends turn), P1 gets +50 cash.
    e.step(backend.end_turn_action())
    st_p1 = e.canonical_state()
    assert st_p1["current_player"] == 1
    assert st_p1["turn"] == 0
    assert st_p1["players"][1]["cash"] == 300

    e.step(backend.end_turn_action())
    st_p0 = e.canonical_state()
    assert st_p0["current_player"] == 0
    assert st_p0["turn"] == 1
    assert st_p0["players"][0]["cash"] == 300
    assert _port_at(st_p0, (3, 1))["stockpile"] == 25

    # Scenario with merchant, builder, landing to verify load/unload, build port, capture home win
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {
                "cash": 100,
                "ships": [
                    {"type": "MERCHANT", "position": [3, 2]},
                    {"type": "BUILDER", "position": [5, 2]},
                    {"type": "LANDING", "position": [8, 1]},
                ],
            },
            "player_1": {"cash": 100, "ships": []},
        }
    )
    # Advance 2 rounds so (3,1) accumulates 50 stockpile
    _end_round(e, backend)
    _end_round(e, backend)
    st = e.canonical_state()
    merch = next(s for s in st["ships"] if s["type"] == "Merchant")
    builder = next(s for s in st["ships"] if s["type"] == "Builder")
    landing = next(s for s in st["ships"] if s["type"] == "Landing")
    assert _port_at(st, (3, 1))["stockpile"] == 50

    # Merchant loads 50 at satellite (3, 1)
    e.step(backend.merchant_load_action(merch["id"], (3, 1)))
    st = e.canonical_state()
    assert _port_at(st, (3, 1))["stockpile"] == 0
    assert next(s for s in st["ships"] if s["id"] == merch["id"])["cargo"] == 50
    # Unload at satellite (non-home) is illegal & no-op
    assert backend.merchant_unload_action(merch["id"], (3, 1)) not in e.enumerate_legal(0)
    e.step(backend.merchant_unload_action(merch["id"], (3, 1)))
    assert next(s for s in e.canonical_state()["ships"] if s["id"] == merch["id"])["cargo"] == 50

    # Move merchant to (1, 2) adjacent to home (1, 1) and unload -> +100 cash
    e.step(backend.move_action(merch["id"], (2, 2)))
    e.step(backend.move_action(merch["id"], (1, 2)))
    cash_before = e.canonical_state()["players"][0]["cash"]
    e.step(backend.merchant_unload_action(merch["id"], (1, 1)))
    st = e.canonical_state()
    assert st["players"][0]["cash"] == cash_before + 50 * 2
    assert next(s for s in st["ships"] if s["id"] == merch["id"])["cargo"] == 0

    # Builder builds port at (5, 1) -> builder consumed, port created & locked_this_turn
    e.step(backend.build_port_action(builder["id"], (5, 1)))
    st = e.canonical_state()
    assert all(s["id"] != builder["id"] for s in st["ships"])
    new_port = _port_at(st, (5, 1))
    assert new_port["owner"] == 0 and new_port["is_home"] is False and new_port["locked_this_turn"] is True
    assert not any(
        backend.encode_action(a)[0] == "b" and backend.encode_action(a)[1:3] == [5, 1]
        for a in e.enumerate_legal(0)
    )

    # Landing captures enemy home port (9, 1) -> terminal win for P0; subsequent steps no-op
    e.step(backend.capture_port_action(landing["id"], (9, 1)))
    assert e.is_terminal() is True
    assert e.canonical_state()["winner"] == 0
    assert e.enumerate_legal(0) == []
    h_term = e.state_hash()
    e.step(backend.end_turn_action())
    assert e.state_hash() == h_term


# ------------------------------------------------------------------ turn limits & draws
def test_turn_counter_and_aec_turn_limit_draw(backend: EngineBackend, tmp_path: Path) -> None:
    mp = write_map_json(tmp_path / "draw.json", 8, 4, [[0, 0]], [[7, 3]])
    e = backend.create(mp)
    e.reset(seed=0)
    for _ in range(10):
        _end_round(e, backend)
    assert e.canonical_state()["turn"] == 10
    assert e.is_terminal() is False
    assert e.canonical_state()["winner"] is None

    # AEC wrapper enforces max_turns truncation with 0.0 / 0.0 draw rewards
    if backend.name == "python":
        env = BattleboatsAEC(str(mp), max_turns=4)
        env.reset(seed=0)
        terminal_records = {}
        for agent in env.agent_iter():
            _, reward, term, trunc, _ = env.last()
            if term or trunc:
                terminal_records[agent] = (term, trunc, reward)
                env.step(None)
                continue
            env.step(env.infos[agent]["legal_actions"][-1])  # EndTurnAction
        assert env.engine.turn == 4
        assert env.engine.winner is None
        assert terminal_records == {
            "player_0": (False, True, 0.0),
            "player_1": (False, True, 0.0),
        }
        _ = random_action


# ------------------------------------------------------------------ clone & determinism
def test_clone_independence_and_seed_determinism(backend: EngineBackend, tmp_path: Path) -> None:
    mp = write_map_json(tmp_path / "clone.json", 16, 8, [[0, 0]], [[15, 7]])
    e1 = backend.create(mp)
    e2 = backend.create(mp)
    e1.reset(seed=123)
    e2.reset(seed=123)
    assert e1.state_hash() == e2.state_hash()

    # Identical action sequence -> identical state hashes
    for eng in (e1, e2):
        legal = eng.enumerate_legal(0)
        buy = next(a for a in legal if backend.encode_action(a)[0] == "b" and backend.encode_action(a)[5] == "Destroyer")
        eng.step(buy)
        eng.step(backend.end_turn_action())
        legal1 = eng.enumerate_legal(1)
        buy1 = next(a for a in legal1 if backend.encode_action(a)[0] == "b" and backend.encode_action(a)[5] == "Submarine")
        eng.step(buy1)
        eng.step(backend.end_turn_action())
    assert e1.state_hash() == e2.state_hash()

    # Clone produces identical hash, and mutating either leaves the other untouched
    orig_hash = e1.state_hash()
    cloned = e1.clone()
    assert cloned.state_hash() == orig_hash

    move_a = next(a for a in cloned.enumerate_legal(0) if backend.encode_action(a)[0] == "m")
    cloned.step(move_a)
    assert cloned.state_hash() != orig_hash
    assert e1.state_hash() == orig_hash

    e1.step(backend.end_turn_action())
    assert e1.canonical_state()["current_player"] == 1
    assert cloned.canonical_state()["current_player"] == 0


def test_merchant_capacity_cap_and_partial_load(backend: EngineBackend, tmp_path: Path) -> None:
    mp = write_map_json(tmp_path / "mcap.json", 10, 4, [[1, 1], [3, 1]], [[8, 2]])
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {"cash": 0, "ships": [{"type": "MERCHANT", "position": [3, 2]}]},
            "player_1": {"cash": 0, "ships": []},
        }
    )
    # 6 rounds -> satellite (3,1) accumulates 6 * 25 = 150 stockpile
    for _ in range(6):
        _end_round(e, backend)
    st = e.canonical_state()
    mid = st["ships"][0]["id"]
    assert _port_at(st, (3, 1))["stockpile"] == 150

    e.step(backend.merchant_load_action(mid, (3, 1)))
    st = e.canonical_state()
    assert st["ships"][0]["cargo"] == 100
    assert _port_at(st, (3, 1))["stockpile"] == 50

    # At capacity -> load is no longer legal and step is a no-op
    assert backend.merchant_load_action(mid, (3, 1)) not in e.enumerate_legal(0)
    h = e.state_hash()
    e.step(backend.merchant_load_action(mid, (3, 1)))
    assert e.state_hash() == h

    # Deliver 100 -> return -> load remaining 50
    e.step(backend.move_action(mid, (2, 2)))
    e.step(backend.move_action(mid, (1, 2)))
    e.step(backend.merchant_unload_action(mid, (1, 1)))
    _end_round(e, backend)
    e.step(backend.move_action(mid, (2, 2)))
    e.step(backend.move_action(mid, (3, 2)))
    e.step(backend.merchant_load_action(mid, (3, 1)))
    st = e.canonical_state()
    assert st["ships"][0]["cargo"] == 75
    assert _port_at(st, (3, 1))["stockpile"] == 0


def test_illegal_port_build_capture_and_enumerate_legal_soundness(
    backend: EngineBackend, tmp_path: Path
) -> None:
    mp = write_map_json(
        tmp_path / "legal_sound.json",
        10,
        5,
        [[1, 1]],
        [[8, 3], [5, 1]],
        extra_land=[[3, 1]],
    )
    e = backend.create(mp)
    e.reset_from_scenario(
        {
            "seed": 0,
            "map_path": str(mp),
            "player_0": {
                "cash": 150,  # can afford Merchant/Destroyer/Sub/Cruiser, NOT Battleship/Carrier/Landing/Builder
                "ships": [
                    {"type": "BUILDER", "position": [3, 2]},
                    {"type": "LANDING", "position": [5, 2]},
                ],
            },
            "player_1": {
                "cash": 0,
                "ships": [{"type": "BUILDER", "position": [5, 3]}],
            },
        }
    )
    st = e.canonical_state()
    builder_id = next(s["id"] for s in st["ships"] if s["type"] == "Builder" and s["owner"] == 0)
    landing_id = next(s["id"] for s in st["ships"] if s["type"] == "Landing")

    # Cannot build port on water (4,2), existing port (5,1), or out-of-range land (8,3)
    for bad_pos in [(4, 2), (5, 1), (8, 3)]:
        h = e.state_hash()
        e.step(backend.build_port_action(builder_id, bad_pos))
        assert e.state_hash() == h

    # Cannot capture own port (1,1) or out-of-range enemy port (8,3) or empty land (3,1)
    for bad_pos in [(1, 1), (8, 3), (3, 1)]:
        h = e.state_hash()
        e.step(backend.capture_port_action(landing_id, bad_pos))
        assert e.state_hash() == h

    # Cannot buy unaffordable ship (Carrier costs 250 > 150)
    h = e.state_hash()
    e.step(backend.build_ship_action((1, 1), (2, 1), "Carrier"))
    assert e.state_hash() == h

    # Every enumerated legal action mutates state when executed on a fresh clone
    legal = e.enumerate_legal(0)
    assert len(legal) > 0
    for act in legal:
        c = e.clone()
        h0 = c.state_hash()
        c.step(act)
        assert c.state_hash() != h0

    # Canonical state structural invariants
    assert st["schema_version"] == 1
    assert [s["id"] for s in st["ships"]] == sorted(s["id"] for s in st["ships"])
    land_set = {tuple(p) for p in st["terrain_land"]}
    for port in st["ports"]:
        assert tuple(port["position"]) in land_set
    ship_positions = [tuple(s["position"]) for s in st["ships"]]
    assert len(ship_positions) == len(set(ship_positions))
    assert all(pos not in land_set for pos in ship_positions)
