"""Diversity score, cheap-type masking, and the buy-diversity sandbox."""

from pathlib import Path

import pytest

from battleboats.core.actions import BuildShipAction, EndTurnAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.sandboxes import (BuyDiversitySandbox, buydiv_layout,
                                        horizon, make_sandbox)
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.buydiv import (BUYDIV_HULLS, CHEAP_HULLS, HULL_COST,
                                         coverage_bonus, diversity_potential,
                                         diversity_raw, diversity_score,
                                         gold_spent_share, mask_prob,
                                         sum_counts)
from battleboats.training.sandbox_skills import SKILL_NAMES, skill_index
from battleboats.training.selfplay_purchases import (empty_purchase_counts,
                                                     hull_name)


def test_all_eight_types_are_scored() -> None:
    assert len(BUYDIV_HULLS) == 8
    assert set(BUYDIV_HULLS) == {hull_name(t) for t in ShipType}
    assert "landing" in BUYDIV_HULLS and "builder" in BUYDIV_HULLS
    assert "landing" not in CHEAP_HULLS
    assert set(CHEAP_HULLS) == {"destroyer", "submarine", "merchant"}
    assert HULL_COST["destroyer"] == 100
    assert HULL_COST["landing"] == 250
    assert "buy_diversity" in SKILL_NAMES
    assert skill_index("landing_build") == SKILL_NAMES.index("landing_build")


def test_diversity_score_equal_and_empty() -> None:
    empty = empty_purchase_counts()
    assert diversity_score(empty) == 0.0
    assert diversity_raw(empty) == 0.0
    even = {name: 2 for name in BUYDIV_HULLS}
    assert diversity_score(even) == pytest.approx(1.0)
    assert diversity_raw(even) == pytest.approx(1.0)
    # Four types at equal count: std/mean = 1, clamped score is 0.
    four = empty_purchase_counts()
    for name in ("destroyer", "submarine", "merchant", "cruiser"):
        four[name] = 3
    assert diversity_raw(four) == pytest.approx(0.0, abs=1e-9)
    assert diversity_score(four) == pytest.approx(0.0, abs=1e-9)
    # Five equal types: 1 - sqrt(3/5) > 0.
    five = empty_purchase_counts()
    for name in ("cruiser", "battleship", "carrier", "landing", "builder"):
        five[name] = 1
    expected = 1.0 - (3.0 / 5.0) ** 0.5
    assert diversity_score(five) == pytest.approx(expected)
    # Mono fleet is clamped to 0 and raw is negative.
    mono = empty_purchase_counts()
    mono["destroyer"] = 12
    assert diversity_raw(mono) < 0.0
    assert diversity_score(mono) == 0.0


def test_score_does_not_penalize_fleet_size() -> None:
    one = {name: 1 for name in BUYDIV_HULLS}
    three = {name: 3 for name in BUYDIV_HULLS}
    assert diversity_score(one) == pytest.approx(diversity_score(three))
    assert diversity_score(one) == pytest.approx(1.0)


def test_gold_share_and_coverage() -> None:
    counts = empty_purchase_counts()
    counts["destroyer"] = 2  # 200
    counts["carrier"] = 1  # 250
    share = gold_spent_share(counts)
    assert share["destroyer"] == pytest.approx(200 / 450)
    assert share["carrier"] == pytest.approx(250 / 450)
    assert share["landing"] == 0.0
    assert coverage_bonus(counts, 0.25) == pytest.approx(0.25 * 2 / 8)
    assert diversity_potential(counts, 0.25) == pytest.approx(
        diversity_score(counts) + 0.25 * 2 / 8
    )


def test_mask_prob_anneals_linearly() -> None:
    assert mask_prob(0, 5, 0.8, 0.0) == pytest.approx(0.8)
    assert mask_prob(4, 5, 0.8, 0.0) == pytest.approx(0.0)
    assert mask_prob(2, 5, 0.8, 0.0) == pytest.approx(0.4)
    assert 0.0 <= mask_prob(99, 10, 0.7, 0.1) <= 1.0


def test_buydiv_layout_and_horizon() -> None:
    small = buydiv_layout(3, 3)
    assert small["home"] == (0, 0)
    assert small["opp"] == (2, 2)
    assert small["ports0"] == [(0, 0)]
    wide = buydiv_layout(16, 8)
    assert wide["home"] != wide["opp"]
    assert len(wide["ports0"]) >= 2
    assert wide["home"] == wide["ports0"][0]
    # Inset ports so an isolated land tile has 4 water neighbors.
    for port in wide["ports0"] + wide["ports1"]:
        assert 0 < port[0] < 15 and 0 < port[1] < 7
    turns, steps = horizon("buy_diversity", 10)
    assert turns == 50
    assert steps == 50 * 80


def test_make_sandbox_buy_diversity(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "m.json", 3, 3)
    env = make_sandbox("buy_diversity", str(path), width=3, height=3, max_turns=6)
    env.reset(seed=0)
    assert env.skill_index == skill_index("buy_diversity")
    assert env.infos["player_0"]["legal_actions"]
    env.close()


def _buy(env, ship: ShipType) -> bool:
    legal = env.infos["player_0"]["legal_actions"]
    for action in legal:
        if isinstance(action, BuildShipAction) and action.ship_type is ship:
            env.step(action)
            return True
    return False


def _end_turn(env) -> None:
    legal = env.infos["player_0"]["legal_actions"]
    ends = [a for a in legal if isinstance(a, EndTurnAction)]
    assert ends, "endturn should stay legal"
    env.step(ends[0])


def test_mask_removes_cheap_types_never_landing(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "m.json", 6, 6)
    env = BuyDiversitySandbox(str(path), width=6, height=6, max_turns=8)
    env.reset(seed=1, options={"mask_cheap": True})
    legal = env.infos["player_0"]["legal_actions"]
    buys = [a for a in legal if isinstance(a, BuildShipAction)]
    names = {hull_name(a.ship_type) for a in buys}
    assert names & set(CHEAP_HULLS) == set()
    assert "landing" in names
    assert "builder" in names and "carrier" in names
    env.reset(seed=1, options={"mask_cheap": False})
    legal = env.infos["player_0"]["legal_actions"]
    names = {hull_name(a.ship_type) for a in legal if isinstance(a, BuildShipAction)}
    assert set(CHEAP_HULLS) <= names
    assert "landing" in names
    env.close()


def test_potential_deltas_sum_to_final_score(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "m.json", 8, 8)
    env = BuyDiversitySandbox(
        str(path), width=8, height=8, max_turns=30, coverage_coef=0.25
    )
    env.reset(seed=0, options={"mask_cheap": False, "coverage_coef": 0.25})
    # One of each type, one buy per turn (port lock).
    order = list(ShipType)
    bought = 0
    for _ in range(30):
        if bought >= len(order):
            _end_turn(env)
            if env.truncations["player_0"]:
                break
            continue
        if _buy(env, order[bought]):
            bought += 1
        _end_turn(env)
        if env.truncations["player_0"]:
            break
    assert bought == 8, f"only bought {bought} types: {env.purchase_counts}"
    expected = diversity_potential(env.purchase_counts, 0.25)
    assert env.training_potential == pytest.approx(expected)
    assert env.episode_reward == pytest.approx(expected)
    assert env.diversity == pytest.approx(1.0)
    # Coverage is training-only: reported score is 1.0, potential is 1.25.
    assert expected == pytest.approx(1.25)
    env.close()


def test_sum_counts() -> None:
    a = empty_purchase_counts()
    b = empty_purchase_counts()
    a["landing"] = 2
    b["landing"] = 1
    b["destroyer"] = 4
    merged = sum_counts([a, b])
    assert merged["landing"] == 3
    assert merged["destroyer"] == 4
    assert merged["carrier"] == 0
