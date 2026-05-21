"""Smoke tests for port income tick and merchant load/unload."""
from pathlib import Path
from typing import List, Tuple

import pytest

from battleboats.core.actions import (
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
)
from battleboats.core.gameEngine import (
    CASH_PER_MATERIAL,
    MERCHANT_CAPACITY,
    PORT_PRODUCTION,
    STARTING_CASH,
    gameEngine,
)
from battleboats.core.shipyard.ship_type import ShipType


MAP_PATH = str(
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def _find_water_line(engine, length: int) -> List[Tuple[int, int]]:
    m = engine.map
    for y in range(m.height):
        for x in range(m.width - length + 1):
            line = [(x + i, y) for i in range(length)]
            if all(m.is_water(p) and not m.is_occupied(p) for p in line):
                return line
    return []


def _adjacent_water(engine, pos):
    m = engine.map
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        w = (pos[0] + dx, pos[1] + dy)
        if m.in_bounds(w) and m.is_water(w) and not m.is_occupied(w):
            return w
    return None


def _first_non_home_port(engine, player_id):
    home = engine.players[player_id].home_port
    for p in engine.players[player_id].owned_port_positions:
        if p != home:
            return p
    return None


@pytest.fixture
def engine():
    e = gameEngine(MAP_PATH)
    e.reset(seed=0)
    return e


# ---------------- income tick ----------------

def test_home_port_income_converts_to_cash_directly(engine):
    """At end of P0's turn, home port produces PORT_PRODUCTION * CASH_PER_MATERIAL cash."""
    # Start with only home port for clarity: zero out non-home ports.
    home = engine.players[0].home_port
    engine.players[0].owned_port_positions = {home}
    starting_cash = engine.players[0].cash
    engine.step(EndTurnAction())
    expected = starting_cash + PORT_PRODUCTION * CASH_PER_MATERIAL
    assert engine.players[0].cash == expected
    # Home port stockpile remains 0 (direct-to-cash).
    assert engine.ports[home].stockpile == 0


def test_non_home_port_accumulates_materials(engine):
    non_home = _first_non_home_port(engine, 0)
    assert non_home is not None
    before = engine.ports[non_home].stockpile
    engine.step(EndTurnAction())
    assert engine.ports[non_home].stockpile == before + PORT_PRODUCTION


def test_income_only_for_ending_player(engine):
    """End-of-turn tick applies to the ending player, not their opponent."""
    p1_non_home = _first_non_home_port(engine, 1)
    assert p1_non_home is not None
    before = engine.ports[p1_non_home].stockpile
    engine.step(EndTurnAction())  # P0 ends → P1's materials untouched
    assert engine.ports[p1_non_home].stockpile == before


def test_income_accumulates_across_multiple_turns(engine):
    non_home = _first_non_home_port(engine, 0)
    before = engine.ports[non_home].stockpile
    # Three full rounds for P0
    for _ in range(3):
        engine.step(EndTurnAction())  # P0 → P1
        engine.step(EndTurnAction())  # P1 → P0
    assert engine.ports[non_home].stockpile == before + 3 * PORT_PRODUCTION


# ---------------- merchant load ----------------

def test_merchant_loads_from_owned_non_home_port(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 60
    spawn = _adjacent_water(engine, non_home)
    assert spawn is not None
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    engine.step(MerchantLoadAction(merchant.id, non_home))
    assert merchant.cargo == 60
    assert engine.ports[non_home].stockpile == 0


def test_merchant_load_respects_capacity(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = MERCHANT_CAPACITY + 50
    spawn = _adjacent_water(engine, non_home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    engine.step(MerchantLoadAction(merchant.id, non_home))
    assert merchant.cargo == MERCHANT_CAPACITY
    assert engine.ports[non_home].stockpile == 50


def test_merchant_load_from_empty_port_noop(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 0
    spawn = _adjacent_water(engine, non_home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    engine.step(MerchantLoadAction(merchant.id, non_home))
    assert merchant.cargo == 0


def test_merchant_load_from_enemy_port_noop(engine):
    enemy_port = engine.players[1].home_port
    spawn = _adjacent_water(engine, enemy_port)
    assert spawn is not None
    # Seed enemy stockpile (test_economy creates fresh state)
    engine.ports[enemy_port].stockpile = 50
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    engine.step(MerchantLoadAction(merchant.id, enemy_port))
    assert merchant.cargo == 0
    assert engine.ports[enemy_port].stockpile == 50


# ---------------- merchant unload ----------------

def test_merchant_unloads_at_home_converts_to_cash(engine):
    home = engine.players[0].home_port
    spawn = _adjacent_water(engine, home)
    assert spawn is not None
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    merchant.cargo = 80
    starting_cash = engine.players[0].cash
    engine.step(MerchantUnloadAction(merchant.id, home))
    assert merchant.cargo == 0
    assert engine.players[0].cash == starting_cash + 80 * CASH_PER_MATERIAL
    # Home port stockpile stays 0.
    assert engine.ports[home].stockpile == 0


def test_merchant_unloads_at_non_home_deposits_materials(engine):
    """Ferry behavior: drop cargo at another owned port for later pickup."""
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 10
    spawn = _adjacent_water(engine, non_home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    merchant.cargo = 40
    starting_cash = engine.players[0].cash
    engine.step(MerchantUnloadAction(merchant.id, non_home))
    assert merchant.cargo == 0
    assert engine.ports[non_home].stockpile == 50
    assert engine.players[0].cash == starting_cash  # no conversion away from home


def test_merchant_unload_with_zero_cargo_noop(engine):
    home = engine.players[0].home_port
    spawn = _adjacent_water(engine, home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    starting_cash = engine.players[0].cash
    engine.step(MerchantUnloadAction(merchant.id, home))
    assert engine.players[0].cash == starting_cash


def test_merchant_unload_at_enemy_port_noop(engine):
    enemy_port = engine.players[1].home_port
    spawn = _adjacent_water(engine, enemy_port)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    merchant.cargo = 50
    engine.step(MerchantUnloadAction(merchant.id, enemy_port))
    assert merchant.cargo == 50  # unchanged
    assert engine.players[0].cash == STARTING_CASH


# ---------------- non-merchant ----------------

def test_non_merchant_cannot_load_or_unload(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 50
    spawn = _adjacent_water(engine, non_home)
    destroyer = engine._spawn_ship(0, ShipType.DESTROYER, spawn)
    engine.step(MerchantLoadAction(destroyer.id, non_home))
    assert destroyer.cargo == 0
    assert engine.ports[non_home].stockpile == 50
    destroyer.cargo = 30  # simulate (shouldn't happen normally)
    engine.step(MerchantUnloadAction(destroyer.id, non_home))
    assert destroyer.cargo == 30  # untouched


# ---------------- persistence + enumerate ----------------

def test_cargo_persists_across_turns(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 40
    spawn = _adjacent_water(engine, non_home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    engine.step(MerchantLoadAction(merchant.id, non_home))
    assert merchant.cargo == 40
    engine.step(EndTurnAction())  # P0 → P1
    engine.step(EndTurnAction())  # P1 → P0
    assert merchant.cargo == 40  # cargo not cleared by reset_turn_flags


def test_enumerate_legal_emits_load_and_unload(engine):
    non_home = _first_non_home_port(engine, 0)
    engine.ports[non_home].stockpile = 50
    spawn = _adjacent_water(engine, non_home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    legal = engine.enumerate_legal(0)
    assert any(
        isinstance(a, MerchantLoadAction) and a.merchant_id == merchant.id and a.port == non_home
        for a in legal
    )
    # No unload yet (cargo == 0)
    assert not any(isinstance(a, MerchantUnloadAction) for a in legal)
    # After loading, unload becomes legal
    engine.step(MerchantLoadAction(merchant.id, non_home))
    legal = engine.enumerate_legal(0)
    assert any(
        isinstance(a, MerchantUnloadAction) and a.merchant_id == merchant.id and a.port == non_home
        for a in legal
    )


def test_enumerate_legal_skips_load_at_home_port(engine):
    """Home port has no stockpile (direct-to-cash); never offer a load there."""
    home = engine.players[0].home_port
    spawn = _adjacent_water(engine, home)
    merchant = engine._spawn_ship(0, ShipType.MERCHANT, spawn)
    legal = engine.enumerate_legal(0)
    assert not any(
        isinstance(a, MerchantLoadAction) and a.port == home for a in legal
    )
