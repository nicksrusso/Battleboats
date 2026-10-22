"""The hand prior must match the debugger's column keys when v5.json is gone."""

from battleboats.agents.heuristics import (
    DEFAULT_INTERCEPT,
    DEFAULT_WEIGHTS,
    FEATURE_KEYS,
    WEIGHTS_SOURCE,
)
from scripts.play_human_vs_random import FEATURE_LABEL


def test_feature_keys_match_debugger() -> None:
    assert list(FEATURE_KEYS) == list(FEATURE_LABEL)


def test_missing_v5_uses_zero_sum_prior() -> None:
    if WEIGHTS_SOURCE is not None:
        return
    assert DEFAULT_WEIGHTS["econ_value_self"] == 0.0
    assert DEFAULT_INTERCEPT == 0.0
    assert set(DEFAULT_WEIGHTS) == set(FEATURE_LABEL)
