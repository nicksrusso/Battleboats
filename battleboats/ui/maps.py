"""Map pool discovery and resolution for the local play UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from battleboats.core.config.generate_map import generate_map_json
from battleboats.envs.map_pool import (
    PoolSize,
    check_map,
    episode_limits,
    write_size_yaml,
)
from battleboats.envs.skill_maps import write_full_map

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WXH_RE = re.compile(r"^(\d+)x(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class MapPreset:
    name: str
    width: int
    height: int
    ports: int
    land_fraction: float
    min_port_enemy_home_water: int
    open_water: bool
    description: str


MAP_PRESETS: Dict[str, MapPreset] = {
    "16x8": MapPreset("16x8", 16, 8, 2, 0.02, 7, False, "16x8 map pool (2 ports/side)"),
    "24x12": MapPreset("24x12", 24, 12, 4, 0.01, 7, False, "24x12 training pool (4 ports/side)"),
    "32x16": MapPreset("32x16", 32, 16, 3, 0.02, 7, False, "32x16 map pool (3 ports/side)"),
    "48x24": MapPreset("48x24", 48, 24, 4, 0.02, 7, False, "48x24 map pool (4 ports/side)"),
    "open-8x8": MapPreset("open-8x8", 8, 8, 2, 0.0, 0, True, "8x8 open water (2 ports/side)"),
    "open-16x8": MapPreset("open-16x8", 16, 8, 2, 0.0, 0, True, "16x8 open water (2 ports/side)"),
}

_ALIASES: Dict[str, str] = {
    "small": "16x8",
    "training": "24x12",
    "medium": "32x16",
    "large": "48x24",
    "open": "open-16x8",
}


@dataclass(frozen=True)
class ResolvedMap:
    name: str
    path: Path
    width: int
    height: int
    recommended_max_turns: int
    source: str


def discover_pool_maps(search_roots: Optional[List[Path]] = None) -> List[Path]:
    """Return any existing map_*.json files from config/maps or runs/*/maps."""
    if search_roots is None:
        search_roots = [
            _REPO_ROOT / "battleboats" / "core" / "config" / "maps",
            _REPO_ROOT / "runs",
        ]
    found: List[Path] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        if root.name == "maps":
            candidates = sorted(root.glob("map_*.json"))
        else:
            candidates = sorted(root.glob("*/maps/map_*.json"))
        for p in candidates:
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            # Skip tiny sandbox maps (e.g. map_landing_2x3.json)
            if any(tag in p.name for tag in ("_landing_", "_merchant_", "_builder_")):
                continue
            seen.add(rp)
            found.append(p)
    return found


def list_map_options(search_roots: Optional[List[Path]] = None) -> List[dict]:
    """UI-friendly list of preset and on-disk pool maps."""
    options: List[dict] = []
    for key, preset in MAP_PRESETS.items():
        options.append(
            {
                "value": key,
                "label": preset.description,
                "kind": "preset",
                "width": preset.width,
                "height": preset.height,
            }
        )
    for path in discover_pool_maps(search_roots):
        options.append(
            {
                "value": str(path),
                "label": f"Pool file: {path.stem}",
                "kind": "file",
            }
        )
    return options


def _recommended_turns(data: dict) -> int:
    turns, _ = episode_limits(
        data,
        max_turns=80,
        max_steps=8000,
        ref_home_distance=14,
    )
    return turns


def _generate_checked_map(
    out_path: Path,
    *,
    width: int,
    height: int,
    ports: int,
    land_fraction: float,
    min_port_enemy_home_water: int,
    seed: int,
    label: str,
) -> Path:
    yaml_path = out_path.parent / f"mapConfig_{label}.yaml"
    size = PoolSize(width=width, height=height, count=1, ports=ports)
    write_size_yaml(yaml_path, size, land_fraction, min_port_enemy_home_water)
    for offset in range(120):
        candidate_seed = abs(int(seed)) + offset
        try:
            generate_map_json(
                str(yaml_path),
                str(out_path),
                seed=candidate_seed,
                min_port_enemy_home_water=min_port_enemy_home_water,
            )
        except ValueError:
            continue
        data = json.loads(out_path.read_text())
        ok, _ = check_map(data, min_port_enemy_home_water)
        if ok:
            return out_path
    # Fallback to open-water layout if constrained generation failed on tiny sizes
    write_full_map(out_path, width, height, seed=seed)
    return out_path


def resolve_map(
    map_spec: Optional[str],
    *,
    seed: int,
    map_dir: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    search_roots: Optional[List[Path]] = None,
) -> ResolvedMap:
    """Resolve a preset name, WxH spec, pool map stem, or JSON path into a playable map."""
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    if map_spec:
        spec = map_spec.strip()
        # 1. Explicit file path
        cand = Path(spec)
        if cand.is_file():
            data = json.loads(cand.read_text())
            return ResolvedMap(
                name=cand.stem,
                path=cand,
                width=int(data["width"]),
                height=int(data["height"]),
                recommended_max_turns=_recommended_turns(data),
                source="file",
            )

        # 2. Match discovered pool map by stem or filename
        for pool_path in discover_pool_maps(search_roots):
            if spec == pool_path.stem or spec == pool_path.name:
                data = json.loads(pool_path.read_text())
                return ResolvedMap(
                    name=pool_path.stem,
                    path=pool_path,
                    width=int(data["width"]),
                    height=int(data["height"]),
                    recommended_max_turns=_recommended_turns(data),
                    source="pool",
                )

        # 3. Preset or alias
        key = _ALIASES.get(spec.lower(), spec.lower())
        if key in MAP_PRESETS:
            preset = MAP_PRESETS[key]
            out_path = map_dir / f"play_{key}_{seed}.json"
            if preset.open_water:
                write_full_map(out_path, preset.width, preset.height, seed=seed)
            else:
                _generate_checked_map(
                    out_path,
                    width=preset.width,
                    height=preset.height,
                    ports=preset.ports,
                    land_fraction=preset.land_fraction,
                    min_port_enemy_home_water=preset.min_port_enemy_home_water,
                    seed=seed,
                    label=key,
                )
            data = json.loads(out_path.read_text())
            return ResolvedMap(
                name=preset.name,
                path=out_path,
                width=int(data["width"]),
                height=int(data["height"]),
                recommended_max_turns=_recommended_turns(data),
                source="preset",
            )

        # 4. Arbitrary WxH
        m = _WXH_RE.match(spec)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            return _resolve_dimensions(w, h, seed=seed, map_dir=map_dir)

        raise ValueError(
            f"Unknown map {map_spec!r}. Choose a preset ({', '.join(MAP_PRESETS)}), "
            "WxH (e.g. 24x12), a pool map name, or a .json path."
        )

    if width is not None and height is not None:
        return _resolve_dimensions(int(width), int(height), seed=seed, map_dir=map_dir)

    return resolve_map("16x8", seed=seed, map_dir=map_dir, search_roots=search_roots)


def _resolve_dimensions(
    width: int, height: int, *, seed: int, map_dir: Path
) -> ResolvedMap:
    if width < 4 or height < 2:
        raise ValueError(f"Map dimensions must be at least 4x2, got {width}x{height}")
    key = f"{width}x{height}"
    if key in MAP_PRESETS:
        return resolve_map(key, seed=seed, map_dir=map_dir)
    out_path = map_dir / f"play_{key}_{seed}.json"
    if width >= 8 and height >= 4:
        ports = max(2, min(4, width // 10 + 1))
        min_water = min(7, max(0, width // 2 - 1))
        _generate_checked_map(
            out_path,
            width=width,
            height=height,
            ports=ports,
            land_fraction=0.02,
            min_port_enemy_home_water=min_water,
            seed=seed,
            label=key,
        )
    else:
        write_full_map(out_path, width, height, seed=seed)
    data = json.loads(out_path.read_text())
    return ResolvedMap(
        name=key,
        path=out_path,
        width=int(data["width"]),
        height=int(data["height"]),
        recommended_max_turns=_recommended_turns(data),
        source="generated",
    )
