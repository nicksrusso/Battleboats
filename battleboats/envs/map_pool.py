"""Seeded pools of random full-game maps from ``core/config/generate_map.py``.

The generator is used as-is: one YAML per board size, one seed per map. Maps
the generator can make but that would make bad training boards are skipped
and the next seed is tried, so a pool is reproducible from its seed. A map is
kept when:

- every port has a water neighbor (ships spawn there; landings attack from it),
- a water path joins the two home ports,
- the home ports are at least half the board width apart (Manhattan), and
- the two homes sit about as far from the midline as each other, and their
  satellite ports are about as far from home as the other side's, and
- (when ``min_port_enemy_home_water`` > 0) every port's spawn tile, home and
  satellites, is at least that many water moves from a tile next to the
  enemy home. At 7 with landing speed 3, a landing ship bought on turn 0
  cannot capture before turn 3 (it cannot act on its build turn). The
  generator redraws ports to meet it; a size where it cannot be met fails
  with RuntimeError instead of looping.

Seats alternate every episode, so any leftover asymmetry is played from both
sides.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from battleboats.core.config.generate_map import generate_map_json, port_enemy_home_water

Pos = Tuple[int, int]
_CARDINALS: Tuple[Pos, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Rejected seeds before giving up on a size. About 1 in 3 seeds is kept.
_MAX_TRIES_PER_MAP = 50


@dataclass
class PoolSize:
    width: int
    height: int
    count: int
    ports: int = 3


def _dist(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def home_ports(data: dict) -> Tuple[Pos, Pos]:
    """First port listed per player is home (the engine's convention)."""
    p0 = data["ports"]["0"][0]
    p1 = data["ports"]["1"][0]
    return (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))


def home_distance(data: dict) -> int:
    a, b = home_ports(data)
    return _dist(a, b)


def water_path_length(data: dict) -> Optional[int]:
    """Fewest water steps from a tile next to home 0 to a tile next to home 1."""
    w, h = int(data["width"]), int(data["height"])
    land = {(int(x), int(y)) for x, y in data["land"]}

    def water_next_to(p: Pos) -> List[Pos]:
        out = []
        for dx, dy in _CARDINALS:
            q = (p[0] + dx, p[1] + dy)
            if 0 <= q[0] < w and 0 <= q[1] < h and q not in land:
                out.append(q)
        return out

    h0, h1 = home_ports(data)
    goals = set(water_next_to(h1))
    frontier = deque((p, 0) for p in water_next_to(h0))
    seen = {p for p, _ in frontier}
    while frontier:
        p, d = frontier.popleft()
        if p in goals:
            return d
        for q in water_next_to(p):
            if q not in seen:
                seen.add(q)
                frontier.append((q, d + 1))
    return None


def spawn_enemy_home_water(data: dict) -> Optional[int]:
    """Min water moves from any port's spawn tile to a tile next to the enemy home."""
    return port_enemy_home_water(int(data["width"]), int(data["height"]), data["land"], data["ports"])


def earliest_capture_turn(moves: int, speed: int = 3) -> int:
    """Earliest game turn a landing ship bought on turn 0 can capture, given the
    water moves from its spawn tile to a tile next to the enemy home. A bought
    ship cannot act on its build turn, then moves ``speed`` tiles a turn;
    capturing is free once adjacent. So ceil(moves / speed), and at least 1."""
    return max(1, -(-int(moves) // int(speed)))


def check_map(data: dict, min_port_enemy_home_water: int = 0) -> Tuple[bool, str]:
    """(ok, reason). See the module docstring for the rules."""
    w, h = int(data["width"]), int(data["height"])
    land = {(int(x), int(y)) for x, y in data["land"]}
    for pid in ("0", "1"):
        if not data["ports"].get(pid):
            return False, f"player {pid} has no ports"
        for x, y in data["ports"][pid]:
            if not any(
                0 <= x + dx < w and 0 <= y + dy < h and (x + dx, y + dy) not in land
                for dx, dy in _CARDINALS
            ):
                return False, f"port {(x, y)} has no water neighbor"
    if water_path_length(data) is None:
        return False, "no water path between homes"
    if home_distance(data) < w // 2:
        return False, f"homes too close ({home_distance(data)} < {w // 2})"
    h0, h1 = home_ports(data)
    # Distance from each home to the midline between the halves.
    mid = w / 2
    if abs((mid - h0[0]) - (h1[0] + 1 - mid)) > max(2.0, w / 8):
        return False, "one home is much nearer the midline"
    def sat_spread(pid: str, home: Pos) -> float:
        sats = data["ports"][pid][1:]
        return sum(_dist(home, s) for s in sats) / max(1, len(sats))
    if abs(sat_spread("0", h0) - sat_spread("1", h1)) > (w + h) / 4:
        return False, "satellite ports much farther from one home"
    if min_port_enemy_home_water > 0:
        d = spawn_enemy_home_water(data)
        if d is None or d < min_port_enemy_home_water:
            return False, f"a port spawn is {d} water moves from the enemy home (< {min_port_enemy_home_water})"
    return True, "ok"


def write_size_yaml(path: Path, size: PoolSize, land_fraction: float, min_port_enemy_home_water: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Map:\n"
        f"  sizeX: {size.width}\n"
        f"  sizeY: {size.height}\n"
        f"  landFraction: {land_fraction}\n"
        f"  numPorts: {size.ports}\n"
        + (f"  minPortEnemyHomeWater: {int(min_port_enemy_home_water)}\n" if min_port_enemy_home_water > 0 else "")
    )
    return path


def build_map_pool(
    out_dir: Path | str,
    sizes: Sequence[PoolSize],
    *,
    seed: int = 0,
    land_fraction: float = 0.02,
    min_port_enemy_home_water: int = 0,
) -> List[str]:
    """Write ``count`` accepted maps per size into ``out_dir``; return their paths.

    Size ``i`` uses seeds starting at ``seed + 100_000 * i``, so changing one
    size's count never reshuffles another size's maps.
    """
    out_dir = Path(out_dir)
    paths: List[str] = []
    for i, size in enumerate(sizes):
        label = f"{size.width}x{size.height}"
        yaml_path = write_size_yaml(out_dir / f"mapConfig_{label}.yaml", size, land_fraction, min_port_enemy_home_water)
        s = seed + 100_000 * i
        tries = 0
        kept = 0
        while kept < size.count:
            if tries >= _MAX_TRIES_PER_MAP * size.count:
                extra = (
                    f" with min_port_enemy_home_water={min_port_enemy_home_water}"
                    if min_port_enemy_home_water > 0
                    else ""
                )
                raise RuntimeError(f"only {kept}/{size.count} usable {label} maps after {tries} seeds{extra}")
            out = out_dir / f"map_{label}_{kept:02d}.json"
            try:
                generate_map_json(str(yaml_path), str(out), seed=s)
            except ValueError:
                # Port constraint infeasible for this seed's land; try the next seed.
                s += 1
                tries += 1
                continue
            ok, _ = check_map(json.loads(out.read_text()), min_port_enemy_home_water)
            s += 1
            tries += 1
            if ok:
                paths.append(str(out))
                kept += 1
        # The last rejected candidate may have been left at the next index.
        stray = out_dir / f"map_{label}_{kept:02d}.json"
        if stray.exists():
            stray.unlink()
    return paths


def episode_limits(
    data: dict,
    *,
    max_turns: int,
    max_steps: int,
    ref_home_distance: int,
) -> Tuple[int, int]:
    """Scale (max_turns, max_steps) by home distance over the reference board's.

    The 8x8 self-play boards had homes 14 apart with an 80-turn / 3000-step cap.
    A map with homes 2x as far gets 2x both caps, rounded up to 10 turns. Never
    below the reference caps.
    """
    if ref_home_distance <= 0:
        return max_turns, max_steps
    scale = max(1.0, home_distance(data) / ref_home_distance)
    turns = int(-(-max_turns * scale // 10) * 10)
    steps = int(round(max_steps * turns / max_turns))
    return turns, steps


def render_ascii(data: dict) -> str:
    """Rows top to bottom. H/h home ports (player 0/1), P/p other ports, # land, . water."""
    w, h = int(data["width"]), int(data["height"])
    grid: Dict[Pos, str] = {(int(x), int(y)): "#" for x, y in data["land"]}
    for pid, (home_ch, port_ch) in (("0", ("H", "P")), ("1", ("h", "p"))):
        for j, (x, y) in enumerate(data["ports"][pid]):
            grid[(int(x), int(y))] = home_ch if j == 0 else port_ch
    return "\n".join("".join(grid.get((x, y), ".") for x in range(w)) for y in range(h))
