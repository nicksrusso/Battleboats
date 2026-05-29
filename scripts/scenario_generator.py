"""Scenario generator for Battleboats training.

Generates N semi-random starting positions using only the large maps.
Each scenario picks a random budget (between --min-budget and --max-budget,
rounded to nearest 25). Players buy ships completely at random until they
cannot afford any more. Ships are placed on random legal water tiles
adjacent to any of their owned ports.

Output: ./Battleboats/runs/scenarios/scenarios_N.json (paths resolved relative
to this script file) containing a list of scenario dicts.
Each scenario can be loaded via gameEngine.reset_from_scenario() later.

This is intended to be used by harvest.py and other training scripts.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from battleboats.core.gameEngine import BASE_STATS, STARTING_CASH
from battleboats.core.map.Map import Map
from battleboats.core.shipyard.ship_type import ShipType

# Paths relative to this script file
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = SCRIPT_DIR.parent


def find_large_maps() -> List[Path]:
    """Return all map_large_*.json files."""
    maps_dir = REPO_ROOT / "battleboats" / "core" / "config" / "maps"
    return sorted(maps_dir.glob("map_large_*.json"))


def get_legal_spawn_positions(map_obj: Map, player_ports: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return all unique water tiles adjacent to any of the player's ports."""
    legal = set()
    for port in player_ports:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pos = (port[0] + dx, port[1] + dy)
            if map_obj.in_bounds(pos) and map_obj.is_water(pos) and not map_obj.is_occupied(pos):
                legal.add(pos)
    return list(legal)


def generate_random_fleet(budget: int, rng: random.Random) -> Tuple[List[Dict[str, Any]], int]:
    """Buy ships completely at random until budget is exhausted.

    Returns (fleet, remaining_cash). Picks only affordable ships each step
    (avoids wasteful retries). Allows degenerate fleets as requested.
    """
    fleet = []
    ship_types = list(ShipType)
    remaining = budget

    while True:
        affordable = [t for t in ship_types if BASE_STATS[t].cost <= remaining]
        if not affordable:
            break
        ship_type = rng.choice(affordable)
        cost = BASE_STATS[ship_type].cost
        fleet.append({"type": ship_type.name, "cost": cost})
        remaining -= cost

    return fleet, remaining


def generate_scenario(map_path: Path, budget: int, scenario_id: int, rng: random.Random) -> Dict[str, Any]:
    """Generate one complete starting scenario."""
    map_obj = Map()
    map_obj.load(str(map_path))

    # Get ports per player from the map
    player_ports = {0: [], 1: []}
    for pos in map_obj.port_positions:
        owner = int(map_obj.port_owner[pos])
        if owner in (0, 1):
            player_ports[owner].append(pos)

    scenario = {
        "id": scenario_id,
        "map_path": str(map_path),
        "seed": rng.randint(0, 2**32 - 1),
        "budget": budget,
        "player_0": {"cash": budget, "ships": []},
        "player_1": {"cash": budget, "ships": []},
    }

    # Generate fleets
    for p_id in (0, 1):
        player_key = f"player_{p_id}"
        fleet, remaining_cash = generate_random_fleet(budget, rng)
        scenario[player_key]["cash"] = remaining_cash
        scenario[player_key]["ships"] = fleet

    # Place ships
    for p_id in (0, 1):
        player_key = f"player_{p_id}"
        ports = player_ports[p_id]
        if not ports:
            continue

        legal_spawns = get_legal_spawn_positions(map_obj, ports)
        if not legal_spawns:
            continue  # rare edge case

        ships = scenario[player_key]["ships"]
        for ship in ships:
            if not legal_spawns:
                break
            position = rng.choice(legal_spawns)
            ship["position"] = list(position)
            # Remove this position so we don't stack ships
            legal_spawns.remove(position)

    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random starting scenarios using large maps."
    )
    parser.add_argument(
        "--num-scenarios",
        "-n",
        type=int,
        default=100,
        help="Number of scenarios to generate (default: 100)",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=1000,
        help="Minimum budget per player (default: 1000)",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=4000,
        help="Maximum budget per player (default: 4000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "scenarios",
        help="Output directory (default: ./Battleboats/runs/scenarios relative to script)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility",
    )
    args = parser.parse_args()

    if args.min_budget > args.max_budget:
        print("Error: --min-budget cannot be larger than --max-budget")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    large_maps = find_large_maps()
    if not large_maps:
        print("Error: No large maps found!")
        return

    print(f"Found {len(large_maps)} large maps.")
    print(f"Generating {args.num_scenarios} scenarios with budget in "
          f"[${args.min_budget}, ${args.max_budget}]...")

    scenarios = []
    for i in range(args.num_scenarios):
        # Pick random budget uniformly from range, rounded to nearest 25.
        # This produces varied remainders (0/25/50/75) after random buying.
        budget = rng.randint(args.min_budget, args.max_budget)
        budget = round(budget / 25) * 25

        map_path = rng.choice(large_maps)
        # Store relative path for portability (reset_from_scenario handles it)
        rel_map = map_path.relative_to(REPO_ROOT)
        scenario = generate_scenario(map_path, budget, i, rng)
        scenario["map_path"] = str(rel_map)
        scenarios.append(scenario)

        if (i + 1) % 20 == 0 or (i + 1) == args.num_scenarios:
            print(f"  Generated {i + 1}/{args.num_scenarios}...")

    output_path = args.output_dir / f"scenarios_{args.num_scenarios}.json"
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"Done. Wrote {len(scenarios)} scenarios to {output_path}")


if __name__ == "__main__":
    main()
