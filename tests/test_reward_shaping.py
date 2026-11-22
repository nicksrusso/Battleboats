"""Per-event shaped rewards: event semantics, caps, config, and collector
attribution (reward on the learner transition whose action earned it)."""

import json
from pathlib import Path

import pytest
import torch

from battleboats.core.actions import (
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
)
from battleboats.core.port import Port
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs import observation as obs
from battleboats.envs.action_masks import SHIP_TYPE_ORDER
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.training.checkpoints import model_config_of
from battleboats.training.policy import VERB_TO_IDX, PolicyNetwork
from battleboats.training.reward_shaping import COMPONENTS, RewardShaping, ShapingTracker
from battleboats.training.selfplay import load_selfplay_config
from battleboats.training.vec_collect import GameCollector

ROOT = Path(__file__).resolve().parents[1]
W = dict(build=0.01, load=0.01, unload=0.02, port_build=0.05, port_capture=0.1)


def _pool(tmp_path, sizes=(PoolSize(16, 8, 2, ports=2),), land=0.02):
    maps = build_map_pool(tmp_path / "maps", list(sizes), seed=3, land_fraction=land)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        turns, steps = episode_limits(data, max_turns=6, max_steps=60, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": turns, "max_steps": steps})
    return maps, info


def _engine(tmp_path):
    maps, _ = _pool(tmp_path, land=0.2)
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    return env.engine


def _free_water_next_to(eng, pos):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (pos[0] + dx, pos[1] + dy)
        if eng.map.in_bounds(nb) and eng.map.is_water(nb) and not eng.map.is_occupied(nb):
            return nb
    return None


def _coast_with_water(eng, skip=()):
    for tile in obs._coastline_tiles(eng):
        if tile in skip:
            continue
        w = _free_water_next_to(eng, tile)
        if w is not None:
            return tile, w
    raise AssertionError("no free coastline tile")


def _add_port(eng, pos, owner):
    eng.map.add_port(pos, owner)
    eng.players[owner].owned_port_positions.add(pos)
    eng.ports[pos] = Port(position=pos, owner=owner, is_home=False, stockpile=0)


def _build(tr, eng, ship_type):
    """Buy ``ship_type`` at the home port, score it, then remove the new ship
    so the spawn tile is free for the next build."""
    pid = eng.current_player
    home = eng.players[pid].home_port
    spawn = _free_water_next_to(eng, home)
    before = set(eng.players[pid].owned_ship_ids)
    got = _do(tr, eng, BuildShipAction(home, spawn, ship_type))
    for sid in set(eng.players[pid].owned_ship_ids) - before:
        eng._destroy_ship(sid)
    return got


def _shaping(**kw):
    return RewardShaping(enabled=True, **{**W, **kw})


def _do(tracker, eng, action, pid=None):
    pid = eng.current_player if pid is None else pid
    pre = tracker.snapshot(eng, pid, action)
    eng.step(action)
    return tracker.score(eng, pid, action, pre)


def test_build_reward_is_proportional_to_hull_cost(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    tr = ShapingTracker(_shaping())
    eng.players[pid].cash = 10_000
    for t in (ShipType.CARRIER, ShipType.DESTROYER, ShipType.BATTLESHIP):
        assert _build(tr, eng, t) == {"build": pytest.approx(0.01 * BASE_STATS[t].cost / 250)}
    # A build the engine rejects (no cash) pays nothing.
    eng.players[pid].cash = 0
    assert _build(tr, eng, ShipType.CARRIER) == {}
    assert tr.totals["build"] == pytest.approx(0.01 * (250 + 100 + 200) / 250)


def test_port_build_load_unload_rewards_and_loop_guard(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    tr = ShapingTracker(_shaping())
    tile, water = _coast_with_water(eng)
    builder = eng._spawn_ship(pid, ShipType.BUILDER, water)
    assert _do(tr, eng, BuildPortAction(builder.id, tile)) == {"port_build": pytest.approx(0.05)}
    assert tile in eng.players[pid].owned_port_positions

    # Load only pays for cargo actually taken on.
    eng.ports[tile].stockpile = 60
    merchant = eng._spawn_ship(pid, ShipType.MERCHANT, water)
    assert _do(tr, eng, MerchantLoadAction(merchant.id, tile)) == {"load": pytest.approx(0.01 * 0.6)}
    assert merchant.cargo == 60
    assert _do(tr, eng, MerchantLoadAction(merchant.id, tile)) == {}  # empty stockpile

    # Unload pays per unit delivered at home; a second unload has nothing to pay.
    home = eng.players[pid].home_port
    m2 = eng._spawn_ship(pid, ShipType.MERCHANT, _free_water_next_to(eng, home))
    m2.cargo = 40
    cash = eng.players[pid].cash
    assert _do(tr, eng, MerchantUnloadAction(m2.id, home)) == {"unload": pytest.approx(0.02 * 0.4)}
    assert eng.players[pid].cash == cash + 80
    assert _do(tr, eng, MerchantUnloadAction(m2.id, home)) == {}
    # Non-home unload is illegal in the engine and pays nothing.
    assert _do(tr, eng, MerchantUnloadAction(merchant.id, tile)) == {}
    assert merchant.cargo == 60


def test_capture_rewards_once_per_port_and_not_on_winning_capture(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    enemy = 1 - pid
    tr = ShapingTracker(_shaping())
    tile, water = _coast_with_water(eng)
    _add_port(eng, tile, enemy)
    lander = eng._spawn_ship(pid, ShipType.LANDING, water)
    assert _do(tr, eng, CapturePortAction(lander.id, tile)) == {"port_capture": pytest.approx(0.1)}
    # Enemy takes it back; recapturing the same port pays nothing more.
    eng.map.set_port_owner(tile, enemy)
    eng.players[pid].owned_port_positions.discard(tile)
    eng.players[enemy].owned_port_positions.add(tile)
    eng.ports[tile].owner = enemy
    assert _do(tr, eng, CapturePortAction(lander.id, tile)) == {}
    assert tile in eng.players[pid].owned_port_positions
    # Capturing the enemy home port wins: terminal reward only, no shaping.
    ehome = eng.players[enemy].home_port
    w = _free_water_next_to(eng, ehome)
    l2 = eng._spawn_ship(pid, ShipType.LANDING, w)
    assert _do(tr, eng, CapturePortAction(l2.id, ehome)) == {}
    assert eng.winner == pid
    assert tr.totals["port_capture"] == pytest.approx(0.1)


def test_rewards_go_to_the_acting_player_only(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    mine, theirs = ShapingTracker(_shaping()), ShapingTracker(_shaping())
    home = eng.players[pid].home_port
    eng.players[pid].cash = 1000
    action = BuildShipAction(home, _free_water_next_to(eng, home), ShipType.CARRIER)
    pre_m, pre_t = mine.snapshot(eng, pid, action), theirs.snapshot(eng, 1 - pid, action)
    eng.step(action)
    assert mine.score(eng, pid, action, pre_m) == {"build": pytest.approx(0.01)}
    assert theirs.score(eng, 1 - pid, action, pre_t) == {}


def test_caps_and_total_cap(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    tr = ShapingTracker(_shaping(build=0.1, caps={"build": 0.25}, total_cap=0.3))
    eng.players[pid].cash = 10_000
    got = [_build(tr, eng, ShipType.CARRIER).get("build", 0.0) for _ in range(4)]
    assert got == pytest.approx([0.1, 0.1, 0.05, 0.0])
    assert tr.totals["build"] == pytest.approx(0.25)
    tile, water = _coast_with_water(eng)
    b = eng._spawn_ship(pid, ShipType.BUILDER, water)
    assert _do(tr, eng, BuildPortAction(b.id, tile)) == {"port_build": pytest.approx(0.05)}  # total_cap 0.3


def test_disabled_shaping_pays_nothing(tmp_path):
    eng = _engine(tmp_path)
    pid = eng.current_player
    home = eng.players[pid].home_port
    tr = ShapingTracker(RewardShaping(enabled=False, **W))
    eng.players[pid].cash = 1000
    assert _do(tr, eng, BuildShipAction(home, _free_water_next_to(eng, home), ShipType.CARRIER)) == {}


def test_config_parsing_and_defaults():
    base = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps.yaml")
    assert base.reward_shaping.enabled is False
    plain = load_selfplay_config(ROOT / "configs" / "selfplay.yaml")
    assert plain.reward_shaping.enabled is False
    shaped = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped.yaml")
    rs = shaped.reward_shaping
    assert rs.enabled and rs.total_cap is not None and rs.total_cap < 1.0
    old = [k for k in COMPONENTS if k != "kill"]
    assert all(rs.weight(k) > 0 for k in old)
    assert rs.kill == 0.0  # the kill component lives in selfplay_bigmaps_shaped_fleets.yaml
    assert sum(rs.caps[k] for k in old) > rs.total_cap
    assert shaped.init_checkpoint == base.init_checkpoint
    with pytest.raises(ValueError):
        RewardShaping.from_dict({"enabled": True, "weights": {"bogus": 1.0}})


def _policy(seed):
    torch.manual_seed(seed)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    return net


def _collect(maps, info, net, shaping, budget=300, seed=0):
    col = GameCollector(
        maps, info, model_config=model_config_of(net), seed=seed, games=3, random_opponent_frac=1.0, shaping=shaping
    )
    col.set_weights(net.state_dict(), {}, [])
    torch.manual_seed(42)
    return col.collect(budget)


def test_shaping_off_matches_default_collector(tmp_path):
    maps, info = _pool(tmp_path)
    net = _policy(0)
    a = _collect(maps, info, net, None)
    b = _collect(maps, info, net, RewardShaping(enabled=False, **W))
    assert len(a.finished) > 0
    ta = [t for s in a.segments for t in s]
    tb = [t for s in b.segments for t in s]
    assert len(ta) == len(tb)
    for x, y in zip(ta, tb):
        assert (x.asset_idx, x.verb_idx, x.target_idx, x.reward) == (y.asset_idx, y.verb_idx, y.target_idx, y.reward)
        assert x.logp == y.logp and x.advantage == y.advantage
    # Off: only terminal +1/-1/0 on the last decision of finished games.
    assert all(t.reward == 0.0 for s in a.segments for t in s[:-1])
    assert all(v == 0.0 for v in a.shaping.values())


def test_collector_puts_build_reward_on_the_buying_transition(tmp_path):
    """Only build_ship decisions carry shaping, each exactly its hull's
    reward; the terminal outcome is added (not substituted) on a finished
    game's last decision."""
    maps, info = _pool(tmp_path)
    net = _policy(1)
    res = _collect(maps, info, net, RewardShaping(enabled=True, build=0.01))
    build_v = VERB_TO_IDX["build_ship"]
    finished_outcomes = {}
    for f in res.finished:
        finished_outcomes.setdefault(f["n_decisions"], []).append(f["outcome"])
    total, bought = 0.0, 0
    for seg in res.segments:
        for i, t in enumerate(seg):
            expected = 0.0
            if t.verb_idx == build_v:
                expected = 0.01 * BASE_STATS[SHIP_TYPE_ORDER[t.target_idx]].cost / 250
                bought += 1
                total += expected
            extra = t.reward - expected
            if i < len(seg) - 1:
                assert extra == pytest.approx(0.0, abs=1e-12)
            else:  # cut segment (0) or finished game's outcome (+1/-1/0)
                assert min(abs(extra - o) for o in (-1.0, 0.0, 1.0)) < 1e-9
    assert bought > 0
    assert res.shaping["build"] == pytest.approx(total)
    assert sum(res.shaping.values()) == pytest.approx(total)
    assert res.finished and all(set(f["shaping"]) == set(COMPONENTS) for f in res.finished)


def test_terminal_outcome_adds_to_last_shaped_step(tmp_path):
    """Sum of a finished segment's rewards = its outcome + its shaping."""
    maps, info = _pool(tmp_path)
    net = _policy(2)
    col = GameCollector(
        maps, info, model_config=model_config_of(net), seed=0, games=1, random_opponent_frac=1.0,
        shaping=RewardShaping(enabled=True, build=0.05),
    )
    col.set_weights(net.state_dict(), {}, [])
    torch.manual_seed(7)
    res = col.collect(400)
    assert res.finished
    # games=1: segments come in game order; the first finished game is segment 0.
    seg, fin = res.segments[0], res.finished[0]
    assert sum(t.reward for t in seg) == pytest.approx(fin["outcome"] + sum(fin["shaping"].values()))


def test_parallel_collector_carries_shaping(tmp_path):
    from battleboats.training.vec_collect import ParallelCollector

    maps, info = _pool(tmp_path)
    net = _policy(3)
    col = ParallelCollector(
        maps, info, num_workers=2, games_per_worker=2, collect_threads=1, episode_offset=0,
        model_config=model_config_of(net), seed=0, random_opponent_frac=1.0,
        shaping=RewardShaping(enabled=True, build=0.01),
    )
    try:
        res = col.collect(net, {}, 200)
    finally:
        col.close()
    build_v = VERB_TO_IDX["build_ship"]
    total = sum(
        0.01 * BASE_STATS[SHIP_TYPE_ORDER[t.target_idx]].cost / 250
        for s in res.segments for t in s if t.verb_idx == build_v
    )
    assert total > 0
    assert res.shaping["build"] == pytest.approx(total)


def test_selfplay_logs_shaping_components(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, run_selfplay

    cfg = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=16),
        map_pool=[PoolSize(16, 8, 2, ports=2)], updates=2, rollout_decisions=60,
        max_turns=6, max_steps=80, num_workers=0, games_per_worker=2, ppo_threads=2,
        reward_shaping=RewardShaping(enabled=True, **W),
    )
    assert run_selfplay(cfg)["ok"]
    rows = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert all(f"shape_{k}" in r for r in rows for k in COMPONENTS)
    assert " shape=build:" in capsys.readouterr().out
