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
- ``kill`` (zero-sum): sinking an enemy ship pays ``+kill * cost / 250`` on
  the attack that sank it; losing one of your own ships costs
  ``-kill * cost / 250``. Ships only sink from attacks (a builder spent on a
  port is not a loss). A loss happens on the opponent's turn, so it is found
  by diffing the player's fleet at its next decision (``settle_losses``) and
  charged to the player's previous transition, the one whose outcome included
  the opponent's reply. A loss before the player's first decision has no
  transition to land on and is dropped.
- ``landing_progress`` (potential-based): with D the water (BFS) distance
  from the nearest of the player's landing ships to the enemy home port
  (1 = adjacent, can capture), and ``norm`` the water distance from the
  player's own home to the enemy home, the potential is
  ``phi = -landing_progress * min(D, norm) / norm``. No landing ship (none
  built yet, or all sunk) counts as D = norm, the distance from its own home,
  so phi lies in [-coef, 0]. Each learner transition gets ``phi' - phi``
  between its decision and the player's next one (settled like kill losses).
  Moving in and back out pays zero net, so oscillating can't be farmed. A sunk
  landing ship drops phi back to -coef, which takes back the progress it had
  earned, and building a new one at home is worth ~0, so there is no free
  refund. The full trip pays about ``coef`` once. The per-game cap clips the
  running net sum to [-cap, +cap]; it is not part of ``total_cap``.

Exploit guards: each port position pays ``port_build``/``port_capture`` at
most once per player per game (so a port traded back and forth pays once),
and each component has a per-game cap (``caps``). ``total_cap`` caps the
summed positive shaping of the non-kill components. Kill is capped on its
own, symmetrically: gains at ``+caps.kill`` and losses at ``-caps.kill``, so
both players' kill rewards stay mirror images. Only the acting player's step is scored, and only learner decisions
get the reward, on the transition of the action that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from battleboats.core.actions import (
    BuildPortAction,
    BuildShipAction,
    AttackAction,
    CapturePortAction,
    MerchantLoadAction,
    MerchantUnloadAction,
)
from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType

COMPONENTS = ("build", "load", "unload", "port_build", "port_capture", "kill", "landing_progress")
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
    kill: float = 0.0
    landing_progress: float = 0.0
    # Per-game cap on each component's summed reward (None = no cap).
    caps: Dict[str, Optional[float]] = field(default_factory=dict)
    # Cap on one player's summed positive non-kill shaping in one game
    # (None = no cap). Kill has its own symmetric cap in ``caps``.
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
        self.kill_gain = 0.0
        self.kill_loss = 0.0
        self.fleet: Optional[Dict[int, Any]] = None
        self.phi: Optional[float] = None
        self.phi0: Optional[float] = None
        self.progress_paid = 0.0
        self._dist: Optional[Dict[Tuple[int, int], int]] = None
        self._norm: Optional[int] = None

    # ----------------------------------------------------- landing distance
    def _distances(self, engine, pid: int) -> Tuple[Dict[Tuple[int, int], int], int]:
        """Water BFS distance to the enemy home (1 = adjacent), and the
        distance from ``pid``'s own home. Terrain is fixed in a game, so this
        is computed once per tracker (one tracker per game and player)."""
        if self._dist is None:
            m = engine.map
            target = engine.players[1 - pid].home_port
            dist: Dict[Tuple[int, int], int] = {}
            frontier = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (target[0] + dx, target[1] + dy)
                if m.in_bounds(nb) and m.is_water(nb):
                    dist[nb] = 1
                    frontier.append(nb)
            while frontier:
                nxt = []
                for pos in frontier:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = (pos[0] + dx, pos[1] + dy)
                        if nb not in dist and m.in_bounds(nb) and m.is_water(nb):
                            dist[nb] = dist[pos] + 1
                            nxt.append(nb)
                frontier = nxt
            home = engine.players[pid].home_port
            near = [
                dist[(home[0] + dx, home[1] + dy)]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (home[0] + dx, home[1] + dy) in dist
            ]
            self._dist = dist
            self._norm = max(1, min(near) if near else m.manhattan(home, target))
        return self._dist, self._norm

    def landing_distance(self, engine, pid: int) -> Tuple[Optional[int], int, int]:
        """(min water distance of ``pid``'s landing ships to the enemy home
        or None if it has none, norm, number of landing ships)."""
        dist, norm = self._distances(engine, pid)
        best, count = None, 0
        for sid in engine.players[pid].owned_ship_ids:
            ship = engine.ships[sid]
            if ship.type != ShipType.LANDING:
                continue
            count += 1
            d = dist.get(ship.position, norm)
            best = d if best is None else min(best, d)
        return best, norm, count

    def potential(self, engine, pid: int) -> float:
        if not (self.cfg.enabled and self.cfg.landing_progress):
            return 0.0
        d, norm, _ = self.landing_distance(engine, pid)
        d = norm if d is None else min(d, norm)
        return -self.cfg.landing_progress * d / norm

    def start(self, engine, pid: int) -> None:
        """Baseline fleet and potential with no reward (game start, or a
        first decision with no earlier transition to credit)."""
        self.mark_fleet(engine, pid)
        self.phi = self.potential(engine, pid)
        self.phi0 = self.phi

    def settle(self, engine, pid: int) -> Dict[str, float]:
        """Rewards that accrued since ``pid``'s previous decision (ships lost
        on the opponent's turn, landing progress). Credit them to that
        decision's transition."""
        out: Dict[str, float] = {}
        loss = self.settle_losses(engine, pid)
        if loss:
            out["kill"] = loss
        if self.cfg.enabled and self.cfg.landing_progress:
            new = self.potential(engine, pid)
            if self.phi is not None:
                delta = new - self.phi
                cap = self.cfg.caps.get("landing_progress")
                paid = self.progress_paid + delta
                if cap is not None:
                    paid = min(cap, max(-cap, paid))
                value = paid - self.progress_paid
                self.progress_paid = paid
                if value:
                    self.totals["landing_progress"] += value
                    out["landing_progress"] = value
            self.phi = new
        return out

    @property
    def scores_kills(self) -> bool:
        return self.cfg.enabled and self.cfg.kill != 0.0

    def mark_fleet(self, engine, pid: int) -> None:
        """Remember ``pid``'s fleet (call after its own step and at game start)."""
        if self.scores_kills:
            self.fleet = {sid: engine.ships[sid].type for sid in engine.players[pid].owned_ship_ids}

    def settle_losses(self, engine, pid: int) -> float:
        """Negative kill reward for ships ``pid`` lost since ``mark_fleet``.
        Re-marks the fleet. Returns 0 when nothing was lost."""
        if not self.scores_kills:
            return 0.0
        before = self.fleet
        self.mark_fleet(engine, pid)
        if before is None:
            return 0.0
        lost = [t for sid, t in before.items() if sid not in self.fleet]
        if not lost:
            return 0.0
        penalty = sum(self.cfg.kill * BASE_STATS[t].cost / MAX_HULL_COST for t in lost)
        cap = self.cfg.caps.get("kill")
        if cap is not None:
            penalty = min(penalty, max(0.0, cap - self.kill_loss))
        self.kill_loss += penalty
        self.totals["kill"] -= penalty
        return -penalty

    def snapshot(self, engine, pid: int, action) -> Optional[Dict[str, Any]]:
        """State needed to score ``action`` (taken by ``pid``), read before the step."""
        if not self.cfg.enabled:
            return None
        player = engine.players[pid]
        if isinstance(action, AttackAction):
            target = engine.ships.get(action.target_id)
            hit = target is not None and target.owner != pid and self.cfg.kill != 0.0
            return {"type": target.type if hit else None}
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
        if isinstance(action, AttackAction):
            if pre["type"] is not None and action.target_id not in engine.ships:
                raw["kill"] = self.cfg.kill * BASE_STATS[pre["type"]].cost / MAX_HULL_COST
        elif isinstance(action, BuildShipAction):
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
        if name == "kill":
            if cap is not None:
                value = min(value, max(0.0, cap - self.kill_gain))
            self.kill_gain += value
            return value
        if cap is not None:
            value = min(value, max(0.0, cap - self.totals[name]))
        if self.cfg.total_cap is not None:
            positive = sum(v for k, v in self.totals.items() if k not in ("kill", "landing_progress"))
            value = min(value, max(0.0, self.cfg.total_cap - positive))
        return value
