"""Shared pytest fixtures for the backend-agnostic golden test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.golden.backend import EngineBackend, available_backends


@pytest.fixture(params=available_backends(), ids=lambda b: b.name)
def backend(request: pytest.FixtureRequest) -> EngineBackend:
    """Parametrized engine backend fixture (plug Rust backend in here later)."""
    return request.param


def write_map_json(
    path: Path,
    width: int,
    height: int,
    p0_ports: List[List[int]],
    p1_ports: List[List[int]],
    extra_land: List[List[int]] | None = None,
    seed: int = 0,
) -> Path:
    """Write a minimal valid map.json file for golden tests."""
    land: List[List[int]] = []
    seen = set()
    for x, y in p0_ports + p1_ports + (extra_land or []):
        if (x, y) not in seen:
            seen.add((x, y))
            land.append([int(x), int(y)])
    payload: Dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "land": land,
        "ports": {"0": p0_ports, "1": p1_ports},
    }
    path.write_text(json.dumps(payload))
    return path
