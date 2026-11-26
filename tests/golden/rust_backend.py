"""Rust `EngineBackend` for the golden suite.

Present only when the ``battleboats_core`` extension is built. Combat replays
inject ``combat_outcomes``; the engine RNG is not NumPy's PCG64.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import battleboats_core

from battleboats.core.shipyard.ship_type import ShipType
from tests.golden.backend import Pos


def _as_list(action: Any) -> List[Any]:
    if isinstance(action, list):
        return action
    return list(action)


@dataclass
class RustEngineInstance:
    raw: battleboats_core.Game

    def reset(self, seed: Optional[int] = None) -> None:
        self.raw.reset(None if seed is None else int(seed))

    def reset_from_scenario(self, scenario: Dict[str, Any]) -> None:
        self.raw.reset_from_scenario(scenario)

    def step(self, action: Any) -> None:
        self.raw.step(_as_list(action))

    def enumerate_legal(self, player_id: int) -> List[List[Any]]:
        return [list(action) for action in self.raw.enumerate_legal(int(player_id))]

    def is_terminal(self) -> bool:
        return bool(self.raw.is_terminal())

    def clone(self) -> "RustEngineInstance":
        return RustEngineInstance(self.raw.clone())

    def canonical_state(self) -> Dict[str, Any]:
        return json.loads(self.raw.canonical_json())

    def state_hash(self) -> str:
        return str(self.raw.state_hash())

    def set_combat_outcomes(self, outcomes: Sequence[Sequence[Any]]) -> None:
        self.raw.set_combat_outcomes([[int(step), bool(killed)] for step, killed in outcomes])

    def visible_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        return list(self.raw.visible_enemy_ships(int(player_id)))

    def known_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        return list(self.raw.known_enemy_ships(int(player_id)))

    def visible_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        return list(self.raw.visible_enemy_ports(int(player_id)))

    def known_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        return list(self.raw.known_enemy_ports(int(player_id)))


@dataclass(frozen=True)
class RustEngineBackend:
    name: str = "rust"

    def create(self, map_path: Union[str, Path], kill_curve_k: float = 2.0) -> RustEngineInstance:
        return RustEngineInstance(battleboats_core.Game(str(map_path), kill_curve_k=float(kill_curve_k)))

    def move_action(self, ship_id: int, destination: Pos) -> List[Any]:
        return ["m", int(ship_id), int(destination[0]), int(destination[1])]

    def attack_action(self, attacker_id: int, target_id: int) -> List[Any]:
        return ["a", int(attacker_id), int(target_id)]

    def build_ship_action(self, port: Pos, spawn_position: Pos, ship_type: str) -> List[Any]:
        return [
            "b",
            int(port[0]),
            int(port[1]),
            int(spawn_position[0]),
            int(spawn_position[1]),
            ShipType(ship_type).value,
        ]

    def build_port_action(self, builder_ship_id: int, target: Pos) -> List[Any]:
        return ["p", int(builder_ship_id), int(target[0]), int(target[1])]

    def capture_port_action(self, landing_ship_id: int, target: Pos) -> List[Any]:
        return ["c", int(landing_ship_id), int(target[0]), int(target[1])]

    def merchant_load_action(self, merchant_id: int, port: Pos) -> List[Any]:
        return ["l", int(merchant_id), int(port[0]), int(port[1])]

    def merchant_unload_action(self, merchant_id: int, port: Pos) -> List[Any]:
        return ["u", int(merchant_id), int(port[0]), int(port[1])]

    def end_turn_action(self) -> List[Any]:
        return ["e"]

    def encode_action(self, action: Any) -> List[Any]:
        return _as_list(action)

    def decode_action(self, rec: Sequence[Any]) -> List[Any]:
        return _as_list(rec)

    def starting_cash(self) -> int:
        return int(battleboats_core.starting_cash())

    def port_production(self) -> int:
        return int(battleboats_core.port_production())

    def cash_per_material(self) -> int:
        return int(battleboats_core.cash_per_material())

    def merchant_capacity(self) -> int:
        return int(battleboats_core.merchant_capacity())

    def ship_types(self) -> tuple:
        return tuple(battleboats_core.ship_types())

    def base_stats(self, ship_type: str) -> Dict[str, Any]:
        stats = battleboats_core.base_stats(ship_type)
        return {
            "speed": int(stats["speed"]),
            "cost": int(stats["cost"]),
            "strength": float(stats["strength"]),
            "attack_range": int(stats["attack_range"]),
        }

    def attack_modifier(self, attacker_type: str, defender_type: str) -> float:
        return float(battleboats_core.attack_modifier(attacker_type, defender_type))

    def spot_distance(self, spotter_key: str, target_key: str) -> float:
        return float(battleboats_core.spot_distance(spotter_key, target_key))
