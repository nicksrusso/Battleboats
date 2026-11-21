"""Per-hull purchase counts for full-game self-play.

Cheap hulls are those whose yard cost is 100 (merchant, destroyer, submarine).
Everything else is expensive. The counters are how we tell whether the
learner only buys cheap ships; a later change may randomly mask cheaper
hulls on some turns if that shows up.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType

# Cost cutoff matches the typed-build split: 100 is cheap, 150+ is expensive.
CHEAP_COST = 100


def hull_name(ship_type: ShipType) -> str:
    return ship_type.name.lower()


def is_cheap_hull(ship_type: ShipType) -> bool:
    return int(BASE_STATS[ship_type].cost) <= CHEAP_COST


def empty_purchase_counts() -> Dict[str, int]:
    return {hull_name(ship): 0 for ship in ShipType}


def record_purchase(counts: Dict[str, int], ship_type: ShipType) -> None:
    name = hull_name(ship_type)
    counts[name] = int(counts.get(name, 0)) + 1


def aggregate_purchases(infos: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    totals = empty_purchase_counts()
    for info in infos:
        for name, n in (info.get("purchases") or {}).items():
            totals[name] = int(totals.get(name, 0)) + int(n)
    return totals


def purchase_mix(counts: Mapping[str, int]) -> Dict[str, Any]:
    by_hull = empty_purchase_counts()
    for name, n in counts.items():
        if name in by_hull:
            by_hull[name] = int(n)
    cheap = 0
    expensive = 0
    for ship in ShipType:
        n = by_hull[hull_name(ship)]
        if is_cheap_hull(ship):
            cheap += n
        else:
            expensive += n
    return {
        "purchases": by_hull,
        "purchases_cheap": cheap,
        "purchases_expensive": expensive,
    }


def format_purchase_line(mix: Mapping[str, Any]) -> str:
    by_hull = mix.get("purchases") or {}
    parts = [f"{name}:{count}" for name, count in by_hull.items() if int(count)]
    bought = ",".join(parts) if parts else "none"
    return (
        f"buys={bought} cheap={int(mix.get('purchases_cheap') or 0)} "
        f"expensive={int(mix.get('purchases_expensive') or 0)}"
    )
