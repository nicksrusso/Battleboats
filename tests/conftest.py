"""Generate the gitignored dev map once, if a clone does not already have one.

``**.json`` is gitignored, and several engine tests load
``battleboats/core/config/map.json``. Writing it here keeps ``pytest`` working
on a fresh checkout. An existing file (Nick's own map) is left alone.
"""

import os
from pathlib import Path

MAP_JSON = (
    Path(__file__).resolve().parents[1]
    / "battleboats"
    / "core"
    / "config"
    / "map.json"
)


def pytest_configure(config) -> None:
    # Under pytest-xdist (-n N) every worker is its own process. One torch
    # thread each keeps N workers from oversubscribing the CPU.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        import torch

        torch.set_num_threads(1)
    if MAP_JSON.is_file():
        return
    from battleboats.envs.skill_maps import write_dev_map

    # Workers can race here on a fresh clone: write to a per-process temp
    # file and rename, so no worker ever reads a half-written map.
    tmp = MAP_JSON.with_name(f".map.{os.getpid()}.json")
    write_dev_map(tmp)
    os.replace(tmp, MAP_JSON)
