"""Per-event shaped rewards for full-game self-play.

Off by default (``enabled: false``), so terminal +1/-1/0 stays the only
reward unless a config turns this on. Each reward is measured from the engine
state just before and just after the acting player's own step, so it only
pays for what the engine actually did:

- ``build``: a ship was bought (cash went down by the hull cost and a new ship
  appeared). Reward ``build * cost / max_hull_cost``.
- ``load``: cargo actually taken on from a port stockpile. Reward
  ``load * units / MERCHANT_CAPACITY`` (a full hold pays ``load``). Stockpile
  only grows from non-home port production, so total load reward is bounded
  by what the player's ports produce.
- ``unload``: cargo actually delivered at the home port (the engine only
  allows home unloads). Reward ``unload * units / MERCHANT_CAPACITY``. A
  unit can be loaded once and delivered once, so load/unload loops earn
  nothing extra.
- ``port_build``: a builder turned a coastline tile into a new port.
- ``port_capture``: a landing ship took an enemy port. A capture that ends
  the game (enemy home port) gets no shaping; the terminal reward covers it.

Exploit guards: each port position pays ``port_build``/``port_capture`` at
most once per player per game (so a port traded back and forth pays once),
and each component has a per-game cap (``caps``), with a total per-game cap
on top. Only the acting player's step is scored, and only learner decisions
get the reward, on the transition of the action that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from battleboats.core.actions import (
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    MerchantLoadAction,
    MerchantUnloadAction,
)
from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.shipyard.ship_data import BASE_STATS

COMPONENTS = ("build", "load", "unload", "port_build", "port_capture")
MAX_HULL_COST = float(max(s.cost for s in BASE_STATS.values()))


def empty_totals() -> Dict[str, float]:
    return {k: 0.0 for k in COMPONENTS}


@dataclass
class RewardShaping:
    enabled: bool = False
    build: float = 0.0
    load: float = 0.0
    unload: float = 0.0
    port_build: float = 0.0
    port_capture: float = 0.0
    # Per-game cap on each component's summed reward (None = no cap).
    caps: Dict[str, Optional[float]] = field(default_factory=dict)
    # Cap on the summed shaping of one player in one game (None = no cap).
    total_cap: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RewardShaping":
        raw = dict(raw or {})
        weights = dict(raw.pop("weights", {}) or {})
        caps = dict(raw.pop("caps", {}) or {})
        enabled = bool(raw.pop("enabled", False))
        total_cap = raw.pop("total_cap", None)
        unknown = set(raw) | (set(weights) - set(COMPONENTS)) | (set(caps) - set(COMPONENTS))
        if unknown:
            raise ValueError(f"reward_shaping: unknown keys {sorted(unknown)}")
        return cls(
            enabled=enabled,
            caps={k: (None if v is None else float(v)) for k, v in caps.items()},
            total_cap=None if total_cap is None else float(total_cap),
            **{k: float(v) for k, v in weights.items()},
        )

    def weight(self, name: str) -> float:
        return float(getattr(self, name))


class ShapingTracker:
    """Per-game, per-player state: paid totals and ports already rewarded."""

    def __init__(self, cfg: RewardShaping) -> None:
        self.cfg = cfg
        self.totals: Dict[str, float] = empty_totals()
        self.ports_paid: Set[Tuple[str, Tuple[int, int]]] = set()

    def snapshot(self, engine, pid: int, action) -> Optional[Dict[str, Any]]:
        """State needed to score ``action`` (taken by ``pid``), read before the step."""
        if not self.cfg.enabled:
            return None
        player = engine.players[pid]
        if isinstance(action, BuildShipAction):
            return {"cash": player.cash, "ships": len(player.owned_ship_ids)}
        if isinstance(action, (MerchantLoadAction, MerchantUnloadAction)):
            ship = engine.ships.get(action.merchant_id)
            return {"cargo": None if ship is None else int(ship.cargo)}
        if isinstance(action, (BuildPortAction, CapturePortAction)):
            return {"owned": action.target in player.owned_port_positions}
        return None

    def score(self, engine, pid: int, action, pre: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Rewards earned by ``pid``'s ``action``, after caps. Updates totals."""
        if pre is None:
            return {}
        player = engine.players[pid]
        raw: Dict[str, float] = {}
        if isinstance(action, BuildShipAction):
            cost = BASE_STATS[action.ship_type].cost
            if pre["cash"] - player.cash == cost and len(player.owned_ship_ids) == pre["ships"] + 1:
                raw["build"] = self.cfg.build * cost / MAX_HULL_COST
        elif isinstance(action, MerchantLoadAction):
            ship = engine.ships.get(action.merchant_id)
            if ship is not None and pre["cargo"] is not None and ship.cargo > pre["cargo"]:
                raw["load"] = self.cfg.load * (ship.cargo - pre["cargo"]) / MERCHANT_CAPACITY
        elif isinstance(action, MerchantUnloadAction):
            ship = engine.ships.get(action.merchant_id)
            if ship is not None and pre["cargo"] and ship.cargo < pre["cargo"]:
                raw["unload"] = self.cfg.unload * (pre["cargo"] - ship.cargo) / MERCHANT_CAPACITY
        elif isinstance(action, (BuildPortAction, CapturePortAction)):
            name = "port_build" if isinstance(action, BuildPortAction) else "port_capture"
            key = (name, action.target)
            gained = not pre["owned"] and action.target in player.owned_port_positions
            if gained and engine.winner is None and key not in self.ports_paid:
                self.ports_paid.add(key)
                raw[name] = self.cfg.weight(name)
        out: Dict[str, float] = {}
        for name, value in raw.items():
            value = self._capped(name, value)
            if value != 0.0:
                self.totals[name] += value
                out[name] = value
        return out

    def _capped(self, name: str, value: float) -> float:
        cap = self.cfg.caps.get(name)
        if cap is not None:
            value = min(value, max(0.0, cap - self.totals[name]))
        if self.cfg.total_cap is not None:
            value = min(value, max(0.0, self.cfg.total_cap - sum(self.totals.values())))
        return value
