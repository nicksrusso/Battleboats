import yaml
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def generate_map_json(
    yaml_path: str,
    output_path: str,
    seed: int | None = None,
) -> None:
    """YAML → minimal map.json (land + ports only; rest = water).

    Land/water assigned per tile by Bernoulli(landFraction). Each player owns
    one half of the map (player 0 left, player 1 right) and gets numPorts
    ports placed on land tiles in their half; one is selected at random as
    the home port and listed first so the engine's _infer_home_ports picks
    it up.
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

    ports: Dict[str, List[List[int]]] = {}
    for p in (0, 1):
        half = half_tiles(p)
        land_in_half = [pos for pos in half if pos in land_set]
        if len(land_in_half) < num_ports:
            non_land_in_half = [pos for pos in half if pos not in land_set]
            extra = rng.sample(non_land_in_half, num_ports - len(land_in_half))
            land_set.update(extra)
            land_in_half.extend(extra)

        selected = rng.sample(land_in_half, num_ports)
        home = selected.pop(rng.randrange(num_ports))
        ports[str(p)] = [list(home)] + [list(pos) for pos in sorted(selected)]

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
    batches = [
        ("mapConfigSmall.yaml", "small", 10),
        ("mapConfigMedium.yaml", "medium", 25),
        ("mapConfigLarge.yaml", "large", 25),
    ]
    for yaml_name, label, count in batches:
        yaml_path = cfg_dir / yaml_name
        for i in range(count):
            out = maps_dir / f"map_{label}_{i:02d}.json"
            generate_map_json(str(yaml_path), str(out))
            print(f"wrote {out}")
