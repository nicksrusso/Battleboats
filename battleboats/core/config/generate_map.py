import yaml
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Port placement redraws per map before the constraint is called infeasible.
MAX_PORT_ATTEMPTS = 200


def _water_next_to(w: int, h: int, land: Set[Tuple[int, int]], p: Tuple[int, int]) -> List[Tuple[int, int]]:
    out = []
    for dx, dy in _CARDINALS:
        q = (p[0] + dx, p[1] + dy)
        if 0 <= q[0] < w and 0 <= q[1] < h and q not in land:
            out.append(q)
    return out


def port_enemy_home_water(
    width: int,
    height: int,
    land: Iterable[Iterable[int]],
    ports: Dict[str, List[List[int]]],
) -> Optional[int]:
    """Fewest water moves from any port's spawn tile to a tile next to the enemy home.

    Spawn tiles are every water tile next to a port (home or satellite), since
    the engine spawns a bought ship on one of them. The goal is any water tile
    next to the other player's home port (where a landing ship can capture it).
    BFS over water only. Returns the minimum over both players, or None when no
    spawn tile can reach the enemy home by water.
    """
    land_set = {(int(x), int(y)) for x, y in land}
    best: Optional[int] = None
    for pid, enemy in (("0", "1"), ("1", "0")):
        if not ports.get(pid) or not ports.get(enemy):
            continue
        eh = (int(ports[enemy][0][0]), int(ports[enemy][0][1]))
        dist = {q: 0 for q in _water_next_to(width, height, land_set, eh)}
        frontier = deque(dist)
        while frontier:
            p = frontier.popleft()
            for q in _water_next_to(width, height, land_set, p):
                if q not in dist:
                    dist[q] = dist[p] + 1
                    frontier.append(q)
        for port in ports[pid]:
            for spawn in _water_next_to(width, height, land_set, (int(port[0]), int(port[1]))):
                d = dist.get(spawn)
                if d is not None and (best is None or d < best):
                    best = d
    return best


def generate_map_json(
    yaml_path: str,
    output_path: str,
    seed: int | None = None,
    min_port_enemy_home_water: int | None = None,
) -> None:
    """YAML → minimal map.json (land + ports only; rest = water).

    Land/water assigned per tile by Bernoulli(landFraction). Each player owns
    one half of the map (player 0 left, player 1 right) and gets numPorts
    ports placed on land tiles in their half; one is selected at random as
    the home port and listed first so the engine's _infer_home_ports picks
    it up.

    ``min_port_enemy_home_water`` (or the YAML key ``minPortEnemyHomeWater``;
    the argument wins) redraws the port placement until every port's spawn
    tile is at least that many water moves from the enemy home (see
    port_enemy_home_water). 0 or unset keeps the old draw exactly. After
    MAX_PORT_ATTEMPTS redraws it raises ValueError instead of looping.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["Map"]
    w, h = m["sizeX"], m["sizeY"]
    land_fraction = m["landFraction"]
    num_ports = m["numPorts"]
    if min_port_enemy_home_water is None:
        min_port_enemy_home_water = int(m.get("minPortEnemyHomeWater", 0) or 0)

    land_set = {
        (x, y)
        for x in range(w)
        for y in range(h)
        if rng.random() < land_fraction
    }

    midline = w // 2

    def half_tiles(player: int) -> List[Tuple[int, int]]:
        x_range = range(0, midline) if player == 0 else range(midline, w)
        return [(x, y) for x in x_range for y in range(h)]

    base_land = set(land_set)

    def place_ports() -> Tuple[Set[Tuple[int, int]], Dict[str, List[List[int]]]]:
        land = set(base_land)
        placed: Dict[str, List[List[int]]] = {}
        for p in (0, 1):
            half = half_tiles(p)
            land_in_half = [pos for pos in half if pos in land]
            if len(land_in_half) < num_ports:
                non_land_in_half = [pos for pos in half if pos not in land]
                extra = rng.sample(non_land_in_half, num_ports - len(land_in_half))
                land.update(extra)
                land_in_half.extend(extra)

            selected = rng.sample(land_in_half, num_ports)
            home = selected.pop(rng.randrange(num_ports))
            placed[str(p)] = [list(home)] + [list(pos) for pos in sorted(selected)]
        return land, placed

    land_set, ports = place_ports()
    if min_port_enemy_home_water and min_port_enemy_home_water > 0:
        attempts = 1
        while True:
            d = port_enemy_home_water(w, h, land_set, ports)
            if d is None or d >= min_port_enemy_home_water:
                break
            if attempts >= MAX_PORT_ATTEMPTS:
                raise ValueError(
                    f"{w}x{h}: no port placement with every spawn >= {min_port_enemy_home_water} "
                    f"water moves from the enemy home after {attempts} draws (seed {seed})"
                )
            land_set, ports = place_ports()
            attempts += 1

    map_data = {
        "width": w,
        "height": h,
        "seed": seed,
        "land": sorted([[x, y] for x, y in land_set]),
        "ports": ports,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(map_data, f, indent=2)


if __name__ == "__main__":
    cfg_dir = Path(__file__).parent
    maps_dir = cfg_dir / "maps"
    # (yaml_filename, size_label, count)
    # Training substrate: 50 random 24x12 maps (map_training_NN.json) — the maps
    # scenario_generator.find_training_maps() globs. The old small/medium/large
    # batches are retired (their YAMLs remain on disk if ever needed again).
    batches = [
        ("mapConfigTraining.yaml", "training", 50),
    ]
    for yaml_name, label, count in batches:
        yaml_path = cfg_dir / yaml_name
        for i in range(count):
            out = maps_dir / f"map_{label}_{i:02d}.json"
            generate_map_json(str(yaml_path), str(out))
            print(f"wrote {out}")
