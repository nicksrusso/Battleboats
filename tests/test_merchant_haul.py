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


def test_satellite_stock_starts_in_range_and_tops_up(tmp_path):
    env = _env(tmp_path, size=10, seed=3)
    home = env.engine.players[0].home_port
    sat = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    assert env.engine.ports[sat].stockpile == MerchantSandbox.START_STOCKPILE
    seen = []
    raw = env.step

    def step(a):
        raw(a)
        seen.append(env.engine.ports[sat].stockpile)

    env.step = step
    _drive(env)
    # every load is full (top-up), and the observed stockpile stays near the
    # warm-start scale instead of max_deliveries * 100 (up to 6000)
    assert env.gold == env.deliveries * 200 and env.deliveries > 5
    assert min(seen) >= 100 and max(seen) <= 500 + 25 * env.max_turns


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


# ---------------------------------------------------------------- progress shaping
def _progress_env(tmp_path, progress, size=6, seed=2):
    path = write_two_home_map(tmp_path / f"p_{size}.json", size, size)
    turns, _ = horizon("merchant", size)
    env = make_sandbox("merchant", str(path), width=size, height=size, max_turns=turns, merchant_progress=progress)
    env.reset(seed=seed)
    return env


def test_progress_shaping_is_potential_based(tmp_path):
    env = _progress_env(tmp_path, 0.25)
    coef = 0.25 / (3 * env.max_turns)
    d0 = env.to_next_delivery()
    total = _drive(env)
    d1 = env.to_next_delivery()
    # telescopes: return = gold fraction + coef * (D_start - D_end)
    assert total == pytest.approx(env.gold_frac + coef * (d0 - d1), abs=1e-9)
    assert env.progress_total == pytest.approx(coef * (d0 - d1), abs=1e-9)
    assert abs(env.progress_total) <= coef * (2 * env.leg + 3 * 8)
    off = _progress_env(tmp_path, 0.0)
    assert _drive(off) == pytest.approx(off.gold_frac, abs=1e-12) and off.progress_total == 0.0


def test_progress_toward_pays_away_costs_and_loops_net_zero(tmp_path):
    env = _progress_env(tmp_path, 0.25, size=8, seed=4)
    coef = 0.25 / (3 * env.max_turns)
    home = env.engine.players[0].home_port
    sat = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    to_sat = _water_dist(env, sat)
    ship = next(env.engine.ships[s] for s in env.engine.players[0].owned_ship_ids)
    legal = _legal(env)
    here = to_sat[ship.position]
    toward = [a for a in legal if isinstance(a, MoveAction) and to_sat.get(a.destination, 99) < here]
    away = [a for a in legal if isinstance(a, MoveAction) and to_sat.get(a.destination, -1) > here]
    assert toward
    start = ship.position
    env.step(toward[0])
    assert env.rewards["player_0"] == pytest.approx(coef)
    back = next(a for a in _legal(env) if isinstance(a, MoveAction) and a.destination == start)
    env.step(back)
    assert env.rewards["player_0"] == pytest.approx(-coef)  # the loop nets 0
    if away:
        env.step(away[0])
        assert env.rewards["player_0"] == pytest.approx(-coef)


def test_delivery_step_stays_clearly_positive_with_shaping(tmp_path):
    env = _progress_env(tmp_path, 0.25, size=8, seed=4)
    home = env.engine.players[0].home_port
    rewards = []
    raw = env.step

    def step(a):
        raw(a)
        if isinstance(a, MerchantUnloadAction):
            rewards.append(env.rewards["player_0"])

    env.step = step
    _drive(env)
    per_delivery = 1.0 / env.max_deliveries
    # The unload step pays the delivery minus the potential jump to the next
    # delivery (larger when unloading on a home tile far from the satellite),
    # so it stays positive but below a full delivery.
    assert rewards and all(0.0 < r < per_delivery for r in rewards)


# ---------------------------------------------------------------- greedy + gate
def test_greedy_policy_is_deterministic_argmax(tmp_path):
    import random as _random
    import torch
    from battleboats.envs import observation as obs
    from battleboats.training.policy import PolicyNetwork
    from battleboats.training.rollout import collect_trajectory

    torch.manual_seed(0)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    net.greedy = True
    runs = []
    for k in range(2):
        env = _env(tmp_path, size=4, seed=1)
        torch.manual_seed(100 + k)  # different RNG; greedy must not care
        traj, _ = collect_trajectory(net, None, env, 0, _random.Random(k), max_steps=200, use_env_reward=True)
        runs.append([(t.asset_idx, t.verb_idx, t.target_idx) for t in traj])
    assert runs[0] == runs[1]


def test_greedy_gate_fills_window_not_training_episodes(tmp_path):
    from battleboats.training.sandbox_curriculum import run_sandboxes

    raw = yaml.safe_load((ROOT / "configs" / "sandboxes_smoke.yaml").read_text())
    raw.update(run_dir=str(tmp_path / "sb"), init_checkpoint=None, skills=["merchant"], sizes=[3],
               max_updates_per_size=2, min_updates=1, allow_unmet=True, decisions_per_update=64,
               eval_episodes=1, wandb=False, merchant_greedy_gate=True, merchant_gate_episodes=3,
               merchant_progress=0.25)
    (tmp_path / "sb.yaml").write_text(yaml.safe_dump(raw))
    cfg = load_sandbox_config(tmp_path / "sb.yaml")
    assert cfg.merchant_greedy_gate and cfg.merchant_gate_episodes == 3 and cfg.merchant_progress == 0.25
    run_sandboxes(cfg)
    import json as _json
    rows = [_json.loads(l) for l in (tmp_path / "sb" / "history.jsonl").read_text().splitlines()]
    assert rows and all("merchant/greedy_success" in r for r in rows)
    state = _json.loads((tmp_path / "sb" / "state.json").read_text())
    assert state["greedy_counter"] == 3 * len(rows)
