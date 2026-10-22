"""Generate the gitignored dev map once, if a clone does not already have one.

``**.json`` is gitignored, and several engine tests load
``battleboats/core/config/map.json``. Writing it here keeps ``pytest`` working
on a fresh checkout. An existing file (Nick's own map) is left alone.
"""

from pathlib import Path

MAP_JSON = (
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def pytest_configure(config) -> None:
    if MAP_JSON.is_file():
        return
    from battleboats.envs.skill_maps import write_dev_map

    write_dev_map(MAP_JSON)
