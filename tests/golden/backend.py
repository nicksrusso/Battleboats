"""Backend-agnostic engine contract for golden parity tests.

Every engine implementation (currently ``"python"``, later ``"rust"`` via PyO3)
implements ``EngineBackend`` and ``EngineInstance``. Golden tests interact ONLY
through this surface and plain JSON-compatible dicts/records, never touching
Python-private helpers like ``_spawn_ship`` or ``_refresh_sightings``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from battleboats.core.actions import (
    Action,
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import (
    CASH_PER_MATERIAL,
    MERCHANT_CAPACITY,
    PORT_PRODUCTION,
    STARTING_CASH,
    gameEngine,
)
from battleboats.core.map.Map import LAND
from battleboats.core.shipyard.ship_data import (
    BASE_STATS,
    PORT,
    attack_modifier,
    spot_distance,
)
from battleboats.core.shipyard.ship_type import ShipType

Pos = Tuple[int, int]
SCHEMA_VERSION = 1


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON encoding shared across backends."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_canonical_state(state: Dict[str, Any]) -> str:
    """SHA-256 hex digest of a canonical_state() dictionary."""
    return hashlib.sha256(canonical_json_bytes(state)).hexdigest()


def encode_action(action: Action) -> List[Any]:
    """Encode an Action into the compact list representation used by golden replays."""
    if isinstance(action, MoveAction):
        return ["m", int(action.ship_id), int(action.destination[0]), int(action.destination[1])]
    if isinstance(action, AttackAction):
        return ["a", int(action.attacker_id), int(action.target_id)]
    if isinstance(action, BuildShipAction):
        return [
            "b",
            int(action.port[0]),
            int(action.port[1]),
            int(action.spawn_position[0]),
            int(action.spawn_position[1]),
            action.ship_type.value,
        ]
    if isinstance(action, BuildPortAction):
        return ["p", int(action.builder_ship_id), int(action.target[0]), int(action.target[1])]
    if isinstance(action, CapturePortAction):
        return ["c", int(action.landing_ship_id), int(action.target[0]), int(action.target[1])]
    if isinstance(action, MerchantLoadAction):
        return ["l", int(action.merchant_id), int(action.port[0]), int(action.port[1])]
    if isinstance(action, MerchantUnloadAction):
        return ["u", int(action.merchant_id), int(action.port[0]), int(action.port[1])]
    if isinstance(action, EndTurnAction):
        return ["e"]
    raise TypeError(f"Unsupported action type: {type(action).__name__}")


def decode_action(rec: Sequence[Any]) -> Action:
    """Decode a compact encoded action record back into a Python Action."""
    tag = rec[0]
    if tag == "m":
        return MoveAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "a":
        return AttackAction(int(rec[1]), int(rec[2]))
    if tag == "b":
        return BuildShipAction((int(rec[1]), int(rec[2])), (int(rec[3]), int(rec[4])), ShipType(rec[5]))
    if tag == "p":
        return BuildPortAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "c":
        return CapturePortAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "l":
        return MerchantLoadAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "u":
        return MerchantUnloadAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "e":
        return EndTurnAction()
    raise ValueError(f"Unknown encoded action tag: {tag!r}")


class EngineInstance(Protocol):
    """Language-agnostic contract for one live game engine instance."""

    def reset(self, seed: Optional[int] = None) -> None:
        ...

    def reset_from_scenario(self, scenario: Dict[str, Any]) -> None:
        ...

    def step(self, action: Any) -> None:
        ...

    def enumerate_legal(self, player_id: int) -> List[Any]:
        ...

    def is_terminal(self) -> bool:
        ...

    def clone(self) -> "EngineInstance":
        ...

    def canonical_state(self) -> Dict[str, Any]:
        ...

    def state_hash(self) -> str:
        ...

    def visible_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        ...

    def known_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        ...

    def visible_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        ...

    def known_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        ...


class EngineBackend(Protocol):
    """Factory + static rule-table surface for an engine backend."""

    name: str

    def create(self, map_path: Union[str, Path], kill_curve_k: float = 2.0) -> EngineInstance:
        ...

    def move_action(self, ship_id: int, destination: Pos) -> Any:
        ...

    def attack_action(self, attacker_id: int, target_id: int) -> Any:
        ...

    def build_ship_action(self, port: Pos, spawn_position: Pos, ship_type: str) -> Any:
        ...

    def build_port_action(self, builder_ship_id: int, target: Pos) -> Any:
        ...

    def capture_port_action(self, landing_ship_id: int, target: Pos) -> Any:
        ...

    def merchant_load_action(self, merchant_id: int, port: Pos) -> Any:
        ...

    def merchant_unload_action(self, merchant_id: int, port: Pos) -> Any:
        ...

    def end_turn_action(self) -> Any:
        ...

    def encode_action(self, action: Any) -> List[Any]:
        ...

    def decode_action(self, rec: Sequence[Any]) -> Any:
        ...

    def starting_cash(self) -> int:
        ...

    def port_production(self) -> int:
        ...

    def cash_per_material(self) -> int:
        ...

    def merchant_capacity(self) -> int:
        ...

    def ship_types(self) -> Tuple[str, ...]:
        ...

    def base_stats(self, ship_type: str) -> Dict[str, Any]:
        ...

    def attack_modifier(self, attacker_type: str, defender_type: str) -> float:
        ...

    def spot_distance(self, spotter_key: str, target_key: str) -> float:
        ...


class PythonEngineInstance:
    """Reference Python ``gameEngine`` wrapped behind the language-agnostic contract."""

    def __init__(self, engine: gameEngine) -> None:
        self.raw = engine

    def reset(self, seed: Optional[int] = None) -> None:
        self.raw.reset(seed=seed)

    def reset_from_scenario(self, scenario: Dict[str, Any]) -> None:
        self.raw.reset_from_scenario(scenario)

    def step(self, action: Action) -> None:
        self.raw.step(action)

    def enumerate_legal(self, player_id: int) -> List[Action]:
        return self.raw.enumerate_legal(player_id)

    def is_terminal(self) -> bool:
        return self.raw.is_terminal()

    def clone(self) -> "PythonEngineInstance":
        return PythonEngineInstance(self.raw.clone())

    def canonical_state(self) -> Dict[str, Any]:
        e = self.raw
        land_coords = [
            [int(x), int(y)] for x in range(e.map.width) for y in range(e.map.height) if e.map.terrain[x, y] == LAND
        ]
        players_out = []
        for p in e.players:
            ship_sightings = [
                {
                    "ship_id": int(s.ship_id),
                    "type": s.type.value,
                    "position": [int(s.position[0]), int(s.position[1])],
                    "turn_seen": int(s.turn_seen),
                    "fresh": bool(s.fresh),
                }
                for _, s in sorted(p.sightings.items(), key=lambda kv: kv[0])
            ]
            port_sightings = [
                {
                    "position": [int(s.position[0]), int(s.position[1])],
                    "last_known_owner": int(s.last_known_owner),
                    "is_home": bool(s.is_home),
                    "turn_seen": int(s.turn_seen),
                    "fresh": bool(s.fresh),
                }
                for _, s in sorted(p.port_sightings.items(), key=lambda kv: kv[0])
            ]
            players_out.append(
                {
                    "id": int(p.id),
                    "home_port": [int(p.home_port[0]), int(p.home_port[1])],
                    "cash": int(p.cash),
                    "ship_sightings": ship_sightings,
                    "port_sightings": port_sightings,
                }
            )
        ports_out = [
            {
                "position": [int(port.position[0]), int(port.position[1])],
                "owner": int(port.owner),
                "is_home": bool(port.is_home),
                "stockpile": int(port.stockpile),
                "locked_this_turn": bool(port.locked_this_turn),
            }
            for _, port in sorted(e.ports.items(), key=lambda kv: kv[0])
        ]
        ships_out = [
            {
                "id": int(s.id),
                "owner": int(s.owner),
                "type": s.type.value,
                "position": [int(s.position[0]), int(s.position[1])],
                "tiles_moved_this_turn": int(s.tiles_moved_this_turn),
                "has_attacked": bool(s.has_attacked),
                "cargo": int(s.cargo),
                "bought_this_turn": bool(s.bought_this_turn),
                "built_turn": int(s.built_turn),
            }
            for _, s in sorted(e.ships.items(), key=lambda kv: kv[0])
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "width": int(e.map.width),
            "height": int(e.map.height),
            "turn": int(e.turn),
            "current_player": int(e.current_player),
            "winner": None if e.winner is None else int(e.winner),
            "kill_curve_k": float(e.kill_curve_k),
            "next_ship_id": int(e._next_ship_id),
            "terrain_land": land_coords,
            "players": players_out,
            "ports": ports_out,
            "ships": ships_out,
        }

    def state_hash(self) -> str:
        return hash_canonical_state(self.canonical_state())

    def visible_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        ships = sorted(self.raw.visible_enemy_ships(player_id), key=lambda s: s.id)
        return [
            {
                "id": int(s.id),
                "type": s.type.value,
                "position": [int(s.position[0]), int(s.position[1])],
            }
            for s in ships
        ]

    def known_enemy_ships(self, player_id: int) -> List[Dict[str, Any]]:
        sightings = sorted(self.raw.known_enemy_ships(player_id), key=lambda s: s.ship_id)
        return [
            {
                "ship_id": int(s.ship_id),
                "type": s.type.value,
                "position": [int(s.position[0]), int(s.position[1])],
                "turn_seen": int(s.turn_seen),
                "fresh": bool(s.fresh),
            }
            for s in sightings
        ]

    def visible_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        ports = sorted(self.raw.visible_enemy_ports(player_id), key=lambda p: p.position)
        return [
            {
                "position": [int(p.position[0]), int(p.position[1])],
                "last_known_owner": int(p.last_known_owner),
                "is_home": bool(p.is_home),
                "turn_seen": int(p.turn_seen),
                "fresh": bool(p.fresh),
            }
            for p in ports
        ]

    def known_enemy_ports(self, player_id: int) -> List[Dict[str, Any]]:
        ports = sorted(self.raw.known_enemy_ports(player_id), key=lambda p: p.position)
        return [
            {
                "position": [int(p.position[0]), int(p.position[1])],
                "last_known_owner": int(p.last_known_owner),
                "is_home": bool(p.is_home),
                "turn_seen": int(p.turn_seen),
                "fresh": bool(p.fresh),
            }
            for p in ports
        ]


@dataclass(frozen=True)
class PythonEngineBackend:
    """Reference Python engine backend."""

    name: str = "python"

    def create(self, map_path: Union[str, Path], kill_curve_k: float = 2.0) -> PythonEngineInstance:
        return PythonEngineInstance(gameEngine(str(map_path), kill_curve_k=kill_curve_k))

    def move_action(self, ship_id: int, destination: Pos) -> MoveAction:
        return MoveAction(int(ship_id), (int(destination[0]), int(destination[1])))

    def attack_action(self, attacker_id: int, target_id: int) -> AttackAction:
        return AttackAction(int(attacker_id), int(target_id))

    def build_ship_action(self, port: Pos, spawn_position: Pos, ship_type: str) -> BuildShipAction:
        return BuildShipAction(
            (int(port[0]), int(port[1])),
            (int(spawn_position[0]), int(spawn_position[1])),
            ShipType(ship_type),
        )

    def build_port_action(self, builder_ship_id: int, target: Pos) -> BuildPortAction:
        return BuildPortAction(int(builder_ship_id), (int(target[0]), int(target[1])))

    def capture_port_action(self, landing_ship_id: int, target: Pos) -> CapturePortAction:
        return CapturePortAction(int(landing_ship_id), (int(target[0]), int(target[1])))

    def merchant_load_action(self, merchant_id: int, port: Pos) -> MerchantLoadAction:
        return MerchantLoadAction(int(merchant_id), (int(port[0]), int(port[1])))

    def merchant_unload_action(self, merchant_id: int, port: Pos) -> MerchantUnloadAction:
        return MerchantUnloadAction(int(merchant_id), (int(port[0]), int(port[1])))

    def end_turn_action(self) -> EndTurnAction:
        return EndTurnAction()

    def encode_action(self, action: Action) -> List[Any]:
        return encode_action(action)

    def decode_action(self, rec: Sequence[Any]) -> Action:
        return decode_action(rec)

    def starting_cash(self) -> int:
        return int(STARTING_CASH)

    def port_production(self) -> int:
        return int(PORT_PRODUCTION)

    def cash_per_material(self) -> int:
        return int(CASH_PER_MATERIAL)

    def merchant_capacity(self) -> int:
        return int(MERCHANT_CAPACITY)

    def ship_types(self) -> Tuple[str, ...]:
        return tuple(t.value for t in ShipType)

    def base_stats(self, ship_type: str) -> Dict[str, Any]:
        s = BASE_STATS[ShipType(ship_type)]
        return {
            "speed": int(s.speed),
            "cost": int(s.cost),
            "strength": float(s.strength),
            "attack_range": int(s.attack_range),
        }

    def attack_modifier(self, attacker_type: str, defender_type: str) -> float:
        return float(attack_modifier(ShipType(attacker_type), ShipType(defender_type)))

    def spot_distance(self, spotter_key: str, target_key: str) -> float:
        sk = PORT if spotter_key == PORT else ShipType(spotter_key)
        tk = PORT if target_key == PORT else ShipType(target_key)
        return float(spot_distance(sk, tk))


def available_backends() -> List[EngineBackend]:
    """Return all registered engine backends available in this environment."""
    backends: List[EngineBackend] = [PythonEngineBackend()]
    try:
        import battleboats_core  # noqa: F401
    except ImportError:
        return backends
    from tests.golden.rust_backend import RustEngineBackend

    backends.append(RustEngineBackend())
    return backends
