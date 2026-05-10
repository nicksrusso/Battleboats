import yaml
import json
import random
from typing import List, Dict, Tuple


def generate_map_json(
    yaml_path: str,
    output_path: str = "map.json",
    seed: int | None = None,
) -> None:
    """YAML → minimal map.json (land + ports only; rest = water)."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["Map"]
    w, h = m["sizeX"], m["sizeY"]
    num_land = max(1, int(w * h * m.get("landFraction", 0.01)))

    all_pos = [(x, y) for x in range(w) for y in range(h)]
    land = random.sample(all_pos, num_land)
    land_set = set(land)

    ports: Dict[str, List[List[int]]] = {}

    for p in [0, 1]:
        pc = cfg[f"Player{p}"]
        num_p = pc["numPorts"]
        # Handle your current YAML syntax bug
        xbounds = pc["areaXBounds"]
        ybounds = pc["areaYBounds"]
        if isinstance(xbounds, str) or len(xbounds) == 1:
            xbounds = [int(v) for v in str(xbounds[0] if isinstance(xbounds, list) else xbounds).split()]
            ybounds = [int(v) for v in str(ybounds[0] if isinstance(ybounds, list) else ybounds).split()]
        xmin, xmax = xbounds
        ymin, ymax = ybounds

        area_land = [pos for pos in land if xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax]

        if len(area_land) < num_p:
            # force extra land for ports
            extra = random.sample(
                [pos for pos in all_pos if pos not in land_set and xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax],
                num_p - len(area_land),
            )
            land.extend(extra)
            land_set.update(extra)
            area_land.extend(extra)

        selected = random.sample(area_land, num_p)
        ports[str(p)] = sorted(selected)

    map_data = {
        "width": w,
        "height": h,
        "seed": seed,
        "land": sorted([[x, y] for x, y in land]),
        "ports": ports,
    }

    with open(output_path, "w") as f:
        json.dump(map_data, f, indent=2)
