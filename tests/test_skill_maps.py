"""Generated skill maps stay legal on the sizes the curriculum uses."""

import json
from pathlib import Path

import pytest

from battleboats.envs.skill_maps import (
    builder_target,
    write_builder_map,
    write_dev_map,
    write_full_map,
    write_merchant_map,
    write_two_home_map,
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def test_merchant_2x3_has_three_ports(tmp_path: Path) -> None:
    data = _load(write_merchant_map(tmp_path / "m.json", 2, 3))
    ports = [tuple(p) for p in data["ports"]["0"]] + [tuple(p) for p in data["ports"]["1"]]
    assert len(ports) == 3
    assert len(set(ports)) == 3
    assert tuple(data["ports"]["0"][0]) == (0, 0)


def test_two_home_and_full_roundtrip(tmp_path: Path) -> None:
    two = _load(write_two_home_map(tmp_path / "t.json", 4, 4))
    assert two["ports"]["0"][0] == [0, 0]
    assert two["ports"]["1"][0] == [3, 3]
    full = _load(write_full_map(tmp_path / "f.json", 6, 6, seed=1))
    assert full["ports"]["0"][0] == [0, 0]
    assert full["ports"]["1"][0] == [5, 5]


def test_builder_2x2_has_no_water_neighbor() -> None:
    with pytest.raises(ValueError):
        builder_target(2, 2)


def test_builder_2x3_target_is_buildable(tmp_path: Path) -> None:
    data = _load(write_builder_map(tmp_path / "b.json", 2, 3))
    assert data["build_target"] == [1, 0]
    assert [1, 0] in data["land"]


def test_dev_map_is_large_enough_for_fog_tests(tmp_path: Path) -> None:
    data = _load(write_dev_map(tmp_path / "dev.json"))
    assert data["width"] == 48
    assert data["height"] == 24
    assert len(data["ports"]["0"]) >= 2
    assert len(data["ports"]["1"]) >= 2
    port_tiles = {tuple(p) for side in data["ports"].values() for p in side}
    land = {tuple(p) for p in data["land"]}
    assert land - port_tiles
