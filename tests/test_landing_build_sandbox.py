"""landing_build sandbox: build a landing ship, then capture the enemy home."""

from pathlib import Path

import pytest

from battleboats.core.actions import BuildShipAction, CapturePortAction, EndTurnAction, MoveAction
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.sandboxes import LandingBuildSandbox, horizon, make_sandbox
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.sandbox_curriculum import SandboxRunConfig, _succeeded, drill_done, load_sandbox_config
from battleboats.training.sandbox_skills import SKILL_NAMES, skill_index

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path, size=5, hulls="all", seed=0):
    path = write_two_home_map(tmp_path / f"lb_{size}_{hulls}.json", size, size)
    turns, _ = horizon("landing_build", size)
    env = make_sandbox("landing_build", str(path), width=size, height=size, max_turns=turns, landing_build_hulls=hulls)
    env.reset(seed=seed)
    return env


def _legal(env):
    return env.infos["player_0"]["legal_actions"]


def _water_dist(env, goal):
    m = env.engine.map
    dist, frontier = {}, []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = (goal[0] + dx, goal[1] + dy)
        if m.in_bounds(q) and m.is_water(q):
            dist[q] = 1; frontier.append(q)
    while frontier:
        nxt = []
        for p in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + dx, p[1] + dy)
                if q not in dist and m.in_bounds(q) and m.is_water(q):
                    dist[q] = dist[p] + 1; nxt.append(q)
        frontier = nxt
    return dist


def _drive(env, hull=ShipType.LANDING, steps=400):
    """Buy ``hull`` if it is the first buy, then sail any landing ship at the enemy home."""
    goal = env.engine.players[1].home_port
    wd = _water_dist(env, goal)
    bought = False
    for _ in range(steps):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = _legal(env)
        caps = [a for a in legal if isinstance(a, CapturePortAction)]
        if caps:
            env.step(caps[0]); continue
        builds = [a for a in legal if isinstance(a, BuildShipAction)]
        want = hull if not bought else ShipType.LANDING
        pick = [a for a in builds if a.ship_type == want]
        if pick and not any(env.engine.ships[s].type == ShipType.LANDING for s in env.engine.players[0].owned_ship_ids):
            env.step(pick[0]); bought = True; continue
        lands = {s for s in env.engine.players[0].owned_ship_ids if env.engine.ships[s].type == ShipType.LANDING}
        moves = [a for a in legal if isinstance(a, MoveAction) and a.ship_id in lands]
        if moves:
            d = lambda a: wd.get(a.destination, 999)
            best = min(moves, key=d)
            here = env.engine.ships[best.ship_id].position
            if d(best) < wd.get(here, 999):
                env.step(best); continue
        env.step(next(a for a in legal if isinstance(a, EndTurnAction)))


def test_registered_after_build_slots():
    assert SKILL_NAMES[-1] == "landing_build"
    assert skill_index("landing") == 2  # earlier slots unchanged
    cfg = load_sandbox_config(ROOT / "configs" / "sandboxes.yaml")
    assert cfg.skills.index("landing_build") == cfg.skills.index("landing") + 1
    with pytest.raises(ValueError):
        SandboxRunConfig(run_dir=Path("x"), skills=["landing_build"], sizes=[3], landing_build_hulls="carrier")


@pytest.mark.parametrize("size", [3, 4, 5, 6, 8, 10])
def test_start_state(tmp_path, size):
    env = _env(tmp_path, size)
    eng = env.engine
    assert eng.players[0].cash == BASE_STATS[ShipType.LANDING].cost
    assert not eng.players[0].owned_ship_ids and not eng.players[1].owned_ship_ids
    assert len(eng.players[0].owned_port_positions) >= 1
    legal = _legal(env)
    assert {type(a) for a in legal} <= {BuildShipAction, MoveAction, CapturePortAction, EndTurnAction}
    types = {a.ship_type for a in legal if isinstance(a, BuildShipAction)}
    assert ShipType.LANDING in types and ShipType.DESTROYER in types  # all hulls buyable
    env.close()


def test_landing_only_variant_filters_builds(tmp_path):
    env = _env(tmp_path, 5, hulls="landing")
    types = {a.ship_type for a in _legal(env) if isinstance(a, BuildShipAction)}
    assert types == {ShipType.LANDING}
    env.close()
    with pytest.raises(ValueError):
        LandingBuildSandbox(str(write_two_home_map(tmp_path / "x.json", 3, 3)), width=3, height=3, hulls="nope")


@pytest.mark.parametrize("size", [3, 5, 10])
def test_build_then_capture_succeeds_and_pays(tmp_path, size):
    for seed in range(3):
        env = _env(tmp_path, size, seed=seed)
        _drive(env)
        assert env.built_landing and env.captured and _succeeded("landing_build", env)
        assert env.terminations["player_0"]
        assert 0.0 < env.rewards["player_0"] <= 1.0
        env.close()


def test_wrong_hull_pays_nothing_and_delays(tmp_path):
    env = _env(tmp_path, 5)
    legal = _legal(env)
    env.step(next(a for a in legal if isinstance(a, BuildShipAction) and a.ship_type == ShipType.DESTROYER))
    assert env.rewards["player_0"] == 0.0 and not env.built_landing
    assert env.engine.players[0].cash < BASE_STATS[ShipType.LANDING].cost
    assert not any(isinstance(a, BuildShipAction) and a.ship_type == ShipType.LANDING for a in _legal(env))
    env.close()
    # Still solvable within the horizon once income refills, but it pays less.
    fast = _env(tmp_path, 5, seed=1); _drive(fast); r_fast = fast.rewards["player_0"]; fast.close()
    slow = _env(tmp_path, 5, seed=1); _drive(slow, hull=ShipType.DESTROYER)
    assert slow.captured and 0.0 < slow.rewards["player_0"] < r_fast
    slow.close()


def test_no_build_no_capture_times_out(tmp_path):
    env = _env(tmp_path, 4)
    for _ in range(200):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        env.step(EndTurnAction())
    assert env.truncations["player_0"] and not env.captured and not _succeeded("landing_build", env)
    env.close()


def test_gate_is_the_success_window():
    common = dict(updates=10, episodes=100, reward_trace=[1.0] * 20, value_trace=[0.0] * 20,
                  success_threshold=0.8, need=64, min_updates=3, plateau_updates=10,
                  plateau_threshold=0.02, combat_min_episodes=32)
    assert drill_done("landing_build", window=[1.0] * 52 + [0.0] * 12, **common)
    assert not drill_done("landing_build", window=[1.0] * 50 + [0.0] * 14, **common)
    assert not drill_done("landing_build", window=[1.0] * 40, **common)  # window not full
