"""Random, fair fleet starts for full-game self-play.

With probability ``prob`` a self-play game starts with both sides already
holding the same random fleet. The composition is drawn once per game and
given to both players, so the total cost is identical. With probability
``landing_prob`` the fleet includes one landing ship; the rest of the budget
is filled with random combat hulls (carrier, battleship, cruiser, destroyer,
submarine) until less than the cheapest hull is left. Leftover budget is the
same for both sides and is not paid out.

Placement mirrors the two homes: each ship gets one sea distance d (BFS over
water from the tiles next to the home port, 1..``radius``), and each side puts
that ship on a random free water tile at distance d from its own home. If a
side has no free tile at d, the nearest free distance is used for that side;
if either side has no free tile at all, that hull is skipped for both. Cash, ports and the turn order are left as in a normal start.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType

COMBAT_HULLS = (
    ShipType.CARRIER,
    ShipType.BATTLESHIP,
    ShipType.CRUISER,
    ShipType.DESTROYER,
    ShipType.SUBMARINE,
)


@dataclass
class FleetStarts:
    prob: float = 0.0
    budget: int = 1200
    landing_prob: float = 0.5
    radius: int = 6

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "FleetStarts":
        raw = dict(raw or {})
        out = cls(
            prob=float(raw.pop("prob", 0.0)),
            budget=int(raw.pop("budget", 1200)),
            landing_prob=float(raw.pop("landing_prob", 0.5)),
            radius=int(raw.pop("radius", 6)),
        )
        if raw:
            raise ValueError(f"fleet_starts: unknown keys {sorted(raw)}")
        if not 0.0 <= out.prob <= 1.0:
            raise ValueError("fleet_starts.prob must be in [0, 1]")
        return out


def draw_fleet(rng: random.Random, budget: int, landing_prob: float) -> List[ShipType]:
    fleet: List[ShipType] = []
    left = int(budget)
    landing_cost = BASE_STATS[ShipType.LANDING].cost
    if left >= landing_cost and rng.random() < landing_prob:
        fleet.append(ShipType.LANDING)
        left -= landing_cost
    cheapest = min(BASE_STATS[t].cost for t in COMBAT_HULLS)
    while left >= cheapest:
        options = [t for t in COMBAT_HULLS if BASE_STATS[t].cost <= left]
        t = rng.choice(options)
        fleet.append(t)
        left -= BASE_STATS[t].cost
    return fleet


def fleet_cost(fleet: List[ShipType]) -> int:
    return sum(BASE_STATS[t].cost for t in fleet)


def sea_distances(engine, home: Tuple[int, int], radius: int) -> Dict[int, List[Tuple[int, int]]]:
    """Water tiles by BFS distance (over water) from the home port, 1..radius."""
    m = engine.map
    out: Dict[int, List[Tuple[int, int]]] = {}
    seen = {home}
    queue = deque()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (home[0] + dx, home[1] + dy)
        if m.in_bounds(nb) and m.is_water(nb) and nb not in seen:
            seen.add(nb)
            queue.append((nb, 1))
    while queue:
        pos, d = queue.popleft()
        out.setdefault(d, []).append(pos)
        if d >= radius:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (pos[0] + dx, pos[1] + dy)
            if nb not in seen and m.in_bounds(nb) and m.is_water(nb):
                seen.add(nb)
                queue.append((nb, d + 1))
    return out


def place_fleets(engine, fleet: List[ShipType], rng: random.Random, radius: int) -> List[List[int]]:
    """Spawn ``fleet`` for both players near their homes. Returns ship ids per player."""
    rings = [sea_distances(engine, engine.players[p].home_port, radius) for p in (0, 1)]
    common = sorted(set(rings[0]) & set(rings[1])) or sorted(set(rings[0]) | set(rings[1]))
    ids: List[List[int]] = [[], []]
    for ship_type in fleet:
        d = rng.choice(common) if common else 1
        spot0 = _free_tile(engine, rings[0], d, rng)
        spot1 = _free_tile(engine, rings[1], d, rng, avoid=spot0)
        if spot0 is None or spot1 is None:
            continue  # skip this hull for both sides so costs stay equal
        ids[0].append(engine._spawn_ship(0, ship_type, spot0).id)
        ids[1].append(engine._spawn_ship(1, ship_type, spot1).id)
    engine._refresh_sightings()
    return ids


def _free_tile(engine, ring: Dict[int, List[Tuple[int, int]]], d: int, rng: random.Random, avoid=None):
    for dist in sorted(ring, key=lambda x: (abs(x - d), x)):
        free = [t for t in ring[dist] if not engine.map.is_occupied(t) and t != avoid]
        if free:
            return rng.choice(free)
    return None


def apply_fleet_start(engine, cfg: FleetStarts, rng: random.Random) -> Optional[List[ShipType]]:
    """With probability ``cfg.prob`` give both sides the same random fleet.
    Returns the fleet, or None for a normal start."""
    if cfg.prob <= 0.0 or rng.random() >= cfg.prob:
        return None
    fleet = draw_fleet(rng, cfg.budget, cfg.landing_prob)
    place_fleets(engine, fleet, rng, cfg.radius)
    return fleet
