"""Hydration-free Rust engine facade for god-mode MCTS.

Wraps ``battleboats_core.Game`` with the clone/step/enumerate_legal surface
``godmode_mcts`` already uses on the Python engine. Leaf observations come
from the Rust ``Game.observation(seat)`` path that harvest used to train the
value net — there is no ``canonical_json`` view and no heuristic hydration.
"""

from __future__ import annotations

from typing import Any, List, Optional

from battleboats.core.actions import (
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import TOKEN_DIM


def encode_action(action: Any) -> list:
    """Compact list encoding used by the Rust engine and golden replays."""
    if isinstance(action, (list, tuple)):
        return list(action)
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
            action.ship_type.value if isinstance(action.ship_type, ShipType) else action.ship_type,
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


def decode_action(rec) -> Any:
    """Decode a compact action list into a Python Action object."""
    tag = rec[0]
    if tag == "m":
        return MoveAction(int(rec[1]), (int(rec[2]), int(rec[3])))
    if tag == "a":
        return AttackAction(int(rec[1]), int(rec[2]))
    if tag == "b":
        typ = rec[5]
        if not isinstance(typ, ShipType):
            typ = ShipType(typ)
        return BuildShipAction((int(rec[1]), int(rec[2])), (int(rec[3]), int(rec[4])), typ)
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


def uses_compact_actions(engine: Any) -> bool:
    """True when ``engine.step`` wants compact list actions, not Action objects."""
    if isinstance(engine, RustMctsEngine):
        return True
    typ = type(engine).__name__
    module = type(engine).__module__
    return typ == "Game" and "battleboats" in module


def wrap_engine(engine: Any) -> Any:
    """Wrap a raw ``battleboats_core.Game``; leave other engines unchanged."""
    if isinstance(engine, RustMctsEngine):
        return engine
    typ = type(engine).__name__
    module = type(engine).__module__
    if typ == "Game" and "battleboats" in module:
        return RustMctsEngine(engine)
    return engine


def current_player_of(engine: Any) -> int:
    value = engine.current_player
    return int(value() if callable(value) else value)


def winner_of(engine: Any) -> Optional[int]:
    value = engine.winner
    if callable(value):
        value = value()
    return None if value is None else int(value)


def turn_of(engine: Any) -> int:
    value = engine.turn
    return int(value() if callable(value) else value)


def observation_of(engine: Any, player_id: int):
    """Fogged observation of ``player_id`` as ``(entity_tokens, globals)``.

    Prefers Rust ``Game.observation``, which is parity-tested against
    ``build_observation`` and is the tensor harvest used for training.
    """
    import numpy as np
    from battleboats.envs.observation import build_observation

    obs_fn = getattr(engine, "observation", None)
    if callable(obs_fn):
        flat, globs, n = obs_fn(int(player_id))
        tokens = np.asarray(flat, dtype=np.float32)
        if n:
            tokens = tokens.reshape(int(n), TOKEN_DIM)
        else:
            tokens = np.zeros((0, TOKEN_DIM), dtype=np.float32)
        return tokens, np.asarray(globs, dtype=np.float32)
    obs = build_observation(engine, int(player_id))
    return obs["entity_tokens"], obs["globals"]


class RustMctsEngine:
    """``gameEngine``-shaped clone/step facade over ``battleboats_core.Game``.

    Deliberately does not expose a hydrated Python view of ships/ports/players
    from ``canonical_json``. Call ``list_ships`` / ``list_ports`` / ``list_players``
    or ``observation(pid)`` instead.
    """

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    @classmethod
    def create(cls, map_path: str, seed: int, kill_curve_k: float = 2.0) -> "RustMctsEngine":
        import battleboats_core

        raw = battleboats_core.Game(str(map_path), kill_curve_k=float(kill_curve_k))
        raw.reset(int(seed))
        return cls(raw)

    def clone(self) -> "RustMctsEngine":
        return RustMctsEngine(self.raw.clone())

    def step(self, action: Any = None, **kwargs: Any) -> None:
        if action is None:
            action = kwargs.get("action")
        self.raw.step(encode_action(action))

    def enumerate_legal(self, player_id: Optional[int] = None, **kwargs: Any) -> List[list]:
        if player_id is None:
            player_id = kwargs.get("player_id", self.current_player)
        return [list(action) for action in self.raw.enumerate_legal(int(player_id))]

    def is_terminal(self) -> bool:
        return bool(self.raw.is_terminal())

    def reset(self, seed: Optional[int] = None) -> None:
        self.raw.reset(None if seed is None else int(seed))

    def observation(self, player_id: int):
        """Same ``(flat, globals, n)`` tuple as ``battleboats_core.Game.observation``."""
        return self.raw.observation(int(player_id))

    def list_ships(self) -> List[dict]:
        return [dict(row) for row in self.raw.ships()]

    def list_ports(self) -> List[dict]:
        return [dict(row) for row in self.raw.ports()]

    def list_players(self) -> List[dict]:
        return [dict(row) for row in self.raw.players()]

    def ships(self) -> List[dict]:
        return self.list_ships()

    def ports(self) -> List[dict]:
        return self.list_ports()

    def players(self) -> List[dict]:
        return self.list_players()

    @property
    def current_player(self) -> int:
        return int(self.raw.current_player())

    @property
    def winner(self) -> Optional[int]:
        value = self.raw.winner()
        return None if value is None else int(value)

    @property
    def turn(self) -> int:
        return int(self.raw.turn())

    @property
    def width(self) -> int:
        return int(self.raw.width)

    @property
    def height(self) -> int:
        return int(self.raw.height)
