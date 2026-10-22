"""Small JSON maps for skill sandboxes and the full-game curriculum.

Maps are generated rather than committed: ``**.json`` is gitignored, and the
historical ``map.json`` lives only on Nick's machine. Layouts keep the engine's
home-port convention (the first port listed for a player is home) and leave
every home with at least one water neighbor so a ship can spawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

Pos = Tuple[int, int]
_CARDINALS: Tuple[Pos, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dump(path: Path | str, data: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return str(path)


def _unique(points: Iterable[Pos]) -> List[Pos]:
    out: List[Pos] = []
    seen = set()
    for p in points:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _require_board(width: int, height: int) -> None:
    if width < 2 or height < 2:
        raise ValueError(f"map must be at least 2x2, got {width}x{height}")


def merchant_ports(width: int, height: int) -> Tuple[Pos, Pos, Pos]:
    """(learner home, satellite resource port, enemy home). All distinct."""
    _require_board(width, height)
    home = (0, 0)
    opp = (width - 1, height - 1)
    satellite = (width - 1, 0)
    if satellite in (home, opp):
        satellite = (0, height - 1)
    if len({home, opp, satellite}) < 3:
        raise ValueError(f"merchant map {width}x{height} has no room for 3 ports")
    return home, satellite, opp


def write_merchant_map(path: Path | str, width: int, height: int) -> str:
    home, satellite, opp = merchant_ports(width, height)
    return _dump(
        path,
        {
            "width": width,
            "height": height,
            "seed": 0,
            "land": [list(home), list(satellite), list(opp)],
            "ports": {
                "0": [list(home), list(satellite)],
                "1": [list(opp)],
            },
        },
    )


def write_two_home_map(path: Path | str, width: int, height: int) -> str:
    """Landing / combat board: one home port per player, everything else water."""
    _require_board(width, height)
    home = (0, 0)
    opp = (width - 1, height - 1)
    if home == opp:
        raise ValueError(f"two-home map {width}x{height} collapses to one tile")
    return _dump(
        path,
        {
            "width": width,
            "height": height,
            "seed": 0,
            "land": [list(home), list(opp)],
            "ports": {"0": [list(home)], "1": [list(opp)]},
        },
    )


def builder_target(width: int, height: int) -> Pos:
    """A non-port land tile with a water neighbor, far enough to require a move on bigger boards."""
    _require_board(width, height)
    home = (0, 0)
    opp = (width - 1, height - 1)
    target = (width - 1, 0)
    if target in (home, opp):
        target = (0, height - 1)
    if target in (home, opp):
        raise ValueError(f"builder map {width}x{height} has no tile for a build target")
    land = {home, opp, target}
    if not _water_neighbors(width, height, land, target):
        raise ValueError(
            f"builder target {target} on {width}x{height} has no water neighbor"
        )
    return target


def write_builder_map(path: Path | str, width: int, height: int) -> str:
    home = (0, 0)
    opp = (width - 1, height - 1)
    target = builder_target(width, height)
    return _dump(
        path,
        {
            "width": width,
            "height": height,
            "seed": 0,
            "land": [list(home), list(opp), list(target)],
            "ports": {"0": [list(home)], "1": [list(opp)]},
            "build_target": list(target),
        },
    )


def write_full_map(
    path: Path | str,
    width: int,
    height: int,
    *,
    seed: int = 0,
) -> str:
    """Open-water full game: two ports a side plus one empty land tile a side.

    ``seed`` only nudges the extra-port row so a curriculum stage can cycle a
    few distinct boards. Terrain stays mostly water so ship movement is the
    skill, not pathing around random blobs.
    """
    _require_board(width, height)
    shift = int(seed) % max(1, height - 1)
    home0 = (0, 0)
    home1 = (width - 1, height - 1)
    extra0 = (0, min(height - 1, 2 + (shift % max(1, height // 3 or 1))))
    extra1 = (width - 1, max(0, height - 1 - (2 + (shift % max(1, height // 3 or 1)))))
    ports0 = _unique([home0, extra0])
    ports1 = _unique([p for p in (home1, extra1) if p not in ports0])
    if home1 not in ports1:
        ports1.insert(0, home1)
    empty0 = (min(width - 1, 2), min(height - 1, 1))
    empty1 = (max(0, width - 1 - 2), max(0, height - 2))
    taken = set(ports0) | set(ports1)
    empties = [p for p in (empty0, empty1) if p not in taken]
    land = _unique([*ports0, *ports1, *empties])
    return _dump(
        path,
        {
            "width": width,
            "height": height,
            "seed": int(seed),
            "land": [list(p) for p in land],
            "ports": {
                "0": [list(p) for p in ports0],
                "1": [list(p) for p in ports1],
            },
        },
    )


def write_dev_map(path: Path | str, width: int = 48, height: int = 24) -> str:
    """Large open map for the existing engine tests.

    Ports sit in the corners (two per player, home listed first) so a water
    line through the middle is more than 17 Manhattan from every port — the
    fog tests reject lines inside port-scout range. A few non-port land tiles
    sit next to player 1 so port-to-port tests have somewhere to plant a port.
    """
    _require_board(width, height)
    ports0 = [(0, 0), (0, min(2, height - 1))]
    ports1 = [(width - 1, height - 1), (width - 1, max(0, height - 3))]
    ports0 = _unique(ports0)
    ports1 = _unique([p for p in ports1 if p not in ports0])
    extra_land = [
        (min(3, width - 1), 0),
        (max(0, width - 4), height - 1),
    ]
    taken = set(ports0) | set(ports1)
    extra_land = [p for p in extra_land if p not in taken]
    land = _unique([*ports0, *ports1, *extra_land])
    return _dump(
        path,
        {
            "width": width,
            "height": height,
            "seed": 0,
            "land": [list(p) for p in land],
            "ports": {
                "0": [list(p) for p in ports0],
                "1": [list(p) for p in ports1],
            },
        },
    )


def write_skill_map(path: Path | str, skill: str, width: int, height: int, *, seed: int = 0) -> str:
    """Dispatch on the curriculum skill name."""
    if skill == "merchant":
        return write_merchant_map(path, width, height)
    if skill in ("landing", "combat"):
        return write_two_home_map(path, width, height)
    if skill == "builder":
        return write_builder_map(path, width, height)
    if skill == "full":
        return write_full_map(path, width, height, seed=seed)
    raise ValueError(f"unknown skill {skill!r}")


def _water_neighbors(
    width: int, height: int, land: set, pos: Pos
) -> List[Pos]:
    out = []
    for dx, dy in _CARDINALS:
        q = (pos[0] + dx, pos[1] + dy)
        if 0 <= q[0] < width and 0 <= q[1] < height and q not in land:
            out.append(q)
    return out


def read_build_target(map_json_path: str) -> Optional[Pos]:
    """Return the builder sandbox target if the map recorded one."""
    with open(map_json_path) as f:
        data = json.load(f)
    raw: Optional[Sequence[int]] = data.get("build_target")
    if not raw:
        return None
    return int(raw[0]), int(raw[1])
