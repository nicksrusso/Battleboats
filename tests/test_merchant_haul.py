"""Fixed-length merchant drill: reward = gold hauled home / max achievable."""

from pathlib import Path

import pytest
import yaml

from battleboats.core.actions import (
    BuildShipAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.envs.sandboxes import (
    MerchantSandbox,
    horizon,
    make_sandbox,
    merchant_max_deliveries,
    water_leg,
)
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.sandbox_curriculum import _succeeded, load_sandbox_config

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path, size=5, seed=0, frac=0.5):
    path = write_two_home_map(tmp_path / f"m_{size}.json", size, size)
    turns, _ = horizon("merchant", size)
    env = make_sandbox("merchant", str(path), width=size, height=size, max_turns=turns, merchant_success_frac=frac)
    env.reset(seed=seed)
    return env


def _legal(env):
    return env.infos["player_0"]["legal_actions"]


def _done(env):
    return env.terminations["player_0"] or env.truncations["player_0"]


def _water_dist(env, port):
    m = env.engine.map
    dist, frontier = {}, []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = (port[0] + dx, port[1] + dy)
        if m.in_bounds(q) and m.is_water(q):
            dist[q] = 0
            frontier.append(q)
    while frontier:
        nxt = []
        for p in frontier:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (p[0] + dx, p[1] + dy)
                if q not in dist and m.in_bounds(q) and m.is_water(q):
                    dist[q] = dist[p] + 1
                    nxt.append(q)
        frontier = nxt
    return dist


def _drive(env, max_hauls=None, steps=2000):
    """Shortest-path hauler: fill up at the satellite, unload at home, repeat."""
    home = env.engine.players[0].home_port
    sat = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    to_home, to_sat = _water_dist(env, home), _water_dist(env, sat)
    total = 0.0
    for _ in range(steps):
        if _done(env):
            break
        legal = _legal(env)
        assert not any(isinstance(a, BuildShipAction) for a in legal)
        ship = next(env.engine.ships[s] for s in env.engine.players[0].owned_ship_ids)
        unloads = [a for a in legal if isinstance(a, MerchantUnloadAction)]
        loads = [a for a in legal if isinstance(a, MerchantLoadAction)]
        stop = max_hauls is not None and env.deliveries >= max_hauls
        if unloads and not stop:
            env.step(unloads[0])
        elif loads and ship.cargo == 0 and not stop:
            env.step(loads[0])
        else:
            dist = to_home if ship.cargo > 0 else to_sat
            here = dist.get(ship.position, 99)
            moves = [a for a in legal if isinstance(a, MoveAction) and dist.get(a.destination, 99) < here] if not stop else []
            env.step(moves[0] if moves else next(a for a in legal if isinstance(a, EndTurnAction)))
        total += env.rewards["player_0"]
    return total


def test_max_deliveries_bound():
    assert merchant_max_deliveries(4, 40, 3) == 15
    assert merchant_max_deliveries(0, 12, 3) == 18  # leg 0 counted as 1
    assert merchant_max_deliveries(30, 12, 3) == 1


def test_horizon_fixed_length():
    assert horizon("merchant", 3) == (12, 120)
    assert horizon("merchant", 10) == (40, 400)


@pytest.mark.parametrize("size", [3, 5, 8, 10])
def test_optimal_hauler_near_bound_and_no_early_end(tmp_path, size):
    fracs = []
    for seed in range(4):
        env = _env(tmp_path, size=size, seed=seed)
        assert env.leg == water_leg(env.engine, env.engine.players[0].home_port,
                                    next(p for p in env.engine.players[0].owned_port_positions
                                         if p != env.engine.players[0].home_port))
        assert env.leg >= 1  # no tile touches both ports
        total = _drive(env)
        # runs to the turn limit; a delivery never ends the episode
        assert env.truncations["player_0"] and not env.terminations["player_0"]
        assert env.engine.turn >= env.max_turns
        assert total == pytest.approx(env.gold_frac, abs=1e-9)
        assert env.deliveries >= 2 and 0 < env.gold_frac <= 1.0
        assert env.gold == env.deliveries * 200
        fracs.append(env.gold_frac)
        assert env.success
    assert min(fracs) >= 0.6


def test_more_round_trips_pay_more(tmp_path):
    one = _env(tmp_path, size=6, seed=2)
    r1 = _drive(one, max_hauls=1)
    many = _env(tmp_path, size=6, seed=2)
    rn = _drive(many)
    assert one.deliveries == 1 and r1 == pytest.approx(1 / one.max_deliveries)
    assert many.deliveries > 1 and rn > r1
    assert not one.success or one.max_deliveries <= 2


def test_passive_income_not_counted_and_no_builds(tmp_path):
    env = _env(tmp_path, size=5, seed=1)
    cash0 = env.engine.players[0].cash
    total = 0.0
    while not _done(env):
        legal = _legal(env)
        assert all(isinstance(a, (MoveAction, MerchantLoadAction, MerchantUnloadAction, EndTurnAction)) for a in legal)
        env.step(next(a for a in legal if isinstance(a, EndTurnAction)))
        total += env.rewards["player_0"]
    assert env.engine.players[0].cash > cash0  # home income still ticks
    assert total == 0.0 and env.gold == 0 and env.deliveries == 0 and not env.success


def test_success_threshold_and_curriculum_gate(tmp_path):
    env = _env(tmp_path, size=6, seed=2, frac=0.99)
    _drive(env, max_hauls=1)
    assert not env.success and not _succeeded("merchant", env)
    env = _env(tmp_path, size=6, seed=2, frac=0.01)
    _drive(env, max_hauls=1)
    assert env.success and _succeeded("merchant", env)


def test_satellite_stock_covers_every_delivery(tmp_path):
    env = _env(tmp_path, size=10, seed=3)
    home = env.engine.players[0].home_port
    sat = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    assert env.engine.ports[sat].stockpile == env.max_deliveries * 100
    assert env.max_gold == env.max_deliveries * 200


def test_config_key(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "sandboxes.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "r")
    raw["merchant_success_frac"] = 0.6
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_sandbox_config(p).merchant_success_frac == 0.6
    raw["merchant_success_frac"] = 0.0
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_sandbox_config(p)


# ---------------------------------------------------------------- gate
from battleboats.training.sandbox_curriculum import SandboxRunConfig, _drill_done, merchant_plateau_done

_GATE = dict(min_gold=0.4, min_episodes=32, plateau_updates=10, plateau_threshold=0.02)


def test_plateau_gate_needs_episodes_floor_and_flat_curve():
    flat = [0.6] * 40
    assert merchant_plateau_done(episodes=64, gold_window=[0.6] * 64, gold_trace=flat, **_GATE)
    assert not merchant_plateau_done(episodes=20, gold_window=[0.6] * 20, gold_trace=flat, **_GATE)
    # single-haul policy: flat but far below the floor
    assert not merchant_plateau_done(episodes=64, gold_window=[0.08] * 64, gold_trace=[0.08] * 40, **_GATE)
    rising = [0.4 + 0.01 * i for i in range(40)]
    assert not merchant_plateau_done(episodes=64, gold_window=[0.7] * 64, gold_trace=rising, **_GATE)


def test_drill_done_routes_merchant_by_gate(tmp_path):
    cfg = SandboxRunConfig(run_dir=tmp_path, skills=["merchant"], sizes=[3])
    assert cfg.merchant_gate == "plateau"
    state = {"updates_this_size": 50, "episodes": {"merchant": 60}, "combat_passed": {},
             "gold_windows": {"merchant": [0.7] * 64}, "gold_trace": {"merchant": [0.7] * 40},
             "reward_trace": {}, "value_trace": {}}
    # success window is empty, the plateau gate still passes and latches
    assert _drill_done(cfg, state, "merchant", window=[])
    assert state["combat_passed"]["merchant"] is True
    state["gold_windows"]["merchant"] = [0.0] * 64
    assert _drill_done(cfg, state, "merchant", window=[])  # latched for this size
    cfg2 = SandboxRunConfig(run_dir=tmp_path, skills=["merchant"], sizes=[3], merchant_gate="success")
    state2 = dict(state, combat_passed={})
    assert not _drill_done(cfg2, state2, "merchant", window=[0.0] * 64)
    assert _drill_done(cfg2, state2, "merchant", window=[1.0] * 64)
    with pytest.raises(ValueError):
        SandboxRunConfig(run_dir=tmp_path, skills=["merchant"], sizes=[3], merchant_gate="x")
