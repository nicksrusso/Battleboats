"""Tactical leaf rollout policy: landings go home, combat hunts landings."""

from __future__ import annotations

from types import SimpleNamespace

from battleboats.agents.rust_mcts_engine import encode_action
from battleboats.agents.tactical_rollout import apply_tactical_rollout, tactical_action
from battleboats.core.shipyard.ship_type import ShipType


class _Toy:
    def __init__(self, ships, ports, legal, current=0, terminal=False, winner=None):
        self._ships = ships
        self._ports = ports
        self._legal = [list(a) for a in legal]
        self.current_player = current
        self.winner = winner
        self._terminal = terminal
        self.steps: list[list] = []

    def list_ships(self):
        return list(self._ships)

    def list_ports(self):
        return list(self._ports)

    def enumerate_legal(self, player_id):
        del player_id
        return [list(a) for a in self._legal]

    def is_terminal(self) -> bool:
        return self._terminal

    def step(self, action):
        rec = list(action) if isinstance(action, (list, tuple)) else encode_action(action)
        self.steps.append(rec)
        # After an end-turn, there is nothing useful left to do in these toys.
        if rec == ["e"]:
            self._legal = [["e"]]


def test_landing_captures_enemy_home_when_legal() -> None:
    engine = _Toy(
        ships=[
            {"id": 1, "type": ShipType.LANDING, "owner": 0, "position": (2, 1), "has_attacked": False},
        ],
        ports=[
            {"position": (0, 0), "owner": 0, "is_home": True},
            {"position": (3, 1), "owner": 1, "is_home": True},
        ],
        legal=[["c", 1, 3, 1], ["m", 1, 2, 2], ["e"]],
    )
    assert tactical_action(engine, 0) == ["c", 1, 3, 1]


def test_landing_moves_toward_nearest_enemy_home() -> None:
    engine = _Toy(
        ships=[
            {"id": 1, "type": ShipType.LANDING, "owner": 0, "position": (0, 0), "has_attacked": False},
        ],
        ports=[
            {"position": (0, 5), "owner": 0, "is_home": True},
            {"position": (5, 0), "owner": 1, "is_home": True},
        ],
        legal=[["m", 1, 1, 0], ["m", 1, 0, 1], ["e"]],
    )
    # (1,0) is closer to the enemy home at (5,0) than (0,1).
    assert tactical_action(engine, 0) == ["m", 1, 1, 0]


def test_combat_attacks_threatening_landing_in_range() -> None:
    engine = _Toy(
        ships=[
            {"id": 2, "type": ShipType.DESTROYER, "owner": 0, "position": (2, 2), "has_attacked": False},
            {"id": 9, "type": ShipType.LANDING, "owner": 1, "position": (3, 2), "has_attacked": False},
        ],
        ports=[
            {"position": (0, 0), "owner": 0, "is_home": True},
            {"position": (8, 8), "owner": 1, "is_home": True},
        ],
        legal=[["a", 2, 9], ["m", 2, 3, 2], ["e"]],
    )
    assert tactical_action(engine, 0) == ["a", 2, 9]


def test_combat_chases_landing_when_out_of_range() -> None:
    engine = _Toy(
        ships=[
            {"id": 2, "type": ShipType.DESTROYER, "owner": 0, "position": (0, 0), "has_attacked": False},
            {"id": 9, "type": ShipType.LANDING, "owner": 1, "position": (5, 0), "has_attacked": False},
        ],
        ports=[
            {"position": (0, 0), "owner": 0, "is_home": True},
            {"position": (9, 9), "owner": 1, "is_home": True},
        ],
        legal=[["m", 2, 1, 0], ["m", 2, 0, 1], ["e"]],
    )
    assert tactical_action(engine, 0) == ["m", 2, 1, 0]


def test_other_units_end_turn() -> None:
    engine = _Toy(
        ships=[
            {"id": 3, "type": ShipType.MERCHANT, "owner": 0, "position": (1, 1), "has_attacked": False},
            {"id": 4, "type": ShipType.BUILDER, "owner": 0, "position": (2, 1), "has_attacked": False},
        ],
        ports=[{"position": (0, 0), "owner": 0, "is_home": True}],
        legal=[["m", 3, 2, 1], ["p", 4, 2, 2], ["e"]],
    )
    assert tactical_action(engine, 0) == ["e"]


def test_rollout_stops_at_terminal_and_counts_steps() -> None:
    engine = _Toy(
        ships=[{"id": 3, "type": ShipType.MERCHANT, "owner": 0, "position": (1, 1), "has_attacked": False}],
        ports=[{"position": (0, 0), "owner": 0, "is_home": True}],
        legal=[["e"]],
        terminal=False,
    )

    class _DieAfter(_Toy):
        def step(self, action):
            super().step(action)
            self._terminal = True
            self.winner = 0

    dying = _DieAfter(
        ships=engine._ships,
        ports=engine._ports,
        legal=engine._legal,
    )
    taken = apply_tactical_rollout(dying, 20)
    assert taken == 1
    assert dying.is_terminal()
    assert dying.steps == [["e"]]
