"""Potential-based landing-ship progress shaping and forward landing starts."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs import observation as obs
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.fleet_starts import FleetStarts, apply_fleet_start, draw_fleet, place_fleets, sea_distances
from battleboats.training.policy import PolicyNetwork
from battleboats.training.reward_shaping import RewardShaping, ShapingTracker
from battleboats.training.vec_collect import GameCollector, ParallelCollector

ROOT = Path(__file__).resolve().parents[1]
COEF = 0.2


def _pool(tmp_path, sizes=(PoolSize(16, 8, 2, ports=2),), turns=6, steps=80):
    maps = build_map_pool(tmp_path / "maps", list(sizes), seed=3)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        t, s = episode_limits(data, max_turns=turns, max_steps=steps, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    return maps, info


def _engine(tmp_path, size=PoolSize(16, 8, 2, ports=2)):
    maps, _ = _pool(tmp_path, sizes=(size,))
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    return env.engine


def _cfg(**kw):
    return RewardShaping(enabled=True, **{"landing_progress": COEF, **kw})


def _teleport(eng, ship, pos):
    eng.map.relocate_ship(ship.id, ship.position, pos)
    ship.position = pos


def _path_to_enemy(eng, tr, pid):
    """Free water tiles at each distance 1..norm from the enemy home."""
    dist, norm = tr._distances(eng, pid)
    by_d = {}
    for pos, d in dist.items():
        if not eng.map.is_occupied(pos):
            by_d.setdefault(d, pos)
    return by_d, norm


def test_potential_rewards_telescope_and_oscillation_nets_zero(tmp_path):
    eng = _engine(tmp_path)
    me = eng.current_player
    tr = ShapingTracker(_cfg())
    by_d, norm = _path_to_enemy(eng, tr, me)
    assert tr.potential(eng, me) == pytest.approx(-COEF)  # no landing ship = at own home
    far = max(d for d in by_d if d <= norm)
    ship = eng._spawn_ship(me, ShipType.LANDING, by_d[far])
    tr.start(eng, me)
    assert tr.phi == pytest.approx(-COEF * far / norm)
    # Advance one step at a time: each step pays exactly coef/norm.
    got = []
    for d in range(far - 1, 0, -1):
        if d not in by_d:
            continue
        before = tr.phi
        _teleport(eng, ship, by_d[d])
        r = tr.settle(eng, me).get("landing_progress", 0.0)
        assert r == pytest.approx(tr.phi - before)
        got.append(r)
    assert sum(got) == pytest.approx(COEF * (far - 1) / norm)
    assert tr.totals["landing_progress"] == pytest.approx(tr.phi - tr.phi0)
    # Oscillating between two tiles nets zero.
    a, b = by_d[1], by_d[2]
    total = tr.totals["landing_progress"]
    for _ in range(25):
        _teleport(eng, ship, b)
        tr.settle(eng, me)
        _teleport(eng, ship, a)
        tr.settle(eng, me)
    assert tr.totals["landing_progress"] == pytest.approx(total)
    # Retreating past the home-to-home distance is clipped (phi never below -coef).
    assert tr.potential(eng, me) >= -COEF - 1e-12


def test_sinking_takes_progress_back_and_rebuild_pays_nothing(tmp_path):
    eng = _engine(tmp_path)
    me = eng.current_player
    tr = ShapingTracker(_cfg())
    by_d, norm = _path_to_enemy(eng, tr, me)
    ship = eng._spawn_ship(me, ShipType.LANDING, by_d[max(d for d in by_d if d <= norm)])
    tr.start(eng, me)
    start_total = tr.totals["landing_progress"]
    _teleport(eng, ship, by_d[2])
    gained = tr.settle(eng, me)["landing_progress"]
    assert gained > 0
    eng._destroy_ship(ship.id)
    lost = tr.settle(eng, me)["landing_progress"]
    assert tr.phi == pytest.approx(-COEF)
    # Net over the life of a landing ship started at d0: phi(no ship) - phi(d0) <= 0.
    assert tr.totals["landing_progress"] - start_total == pytest.approx(-COEF - tr.phi0)
    assert lost == pytest.approx(-COEF - (tr.phi0 + gained))
    # Building a new landing ship next to the home pays at most one step's worth.
    home = eng.players[me].home_port
    spawn = next(
        (home[0] + dx, home[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if eng.map.in_bounds((home[0] + dx, home[1] + dy)) and eng.map.is_water((home[0] + dx, home[1] + dy))
        and not eng.map.is_occupied((home[0] + dx, home[1] + dy))
    )
    eng._spawn_ship(me, ShipType.LANDING, spawn)
    r = tr.settle(eng, me).get("landing_progress", 0.0)
    assert abs(r) <= COEF / norm + 1e-12


def test_progress_cap_and_off_by_default(tmp_path):
    eng = _engine(tmp_path)
    me = eng.current_player
    tr = ShapingTracker(_cfg(landing_progress=1.0, caps={"landing_progress": 0.3}))
    by_d, norm = _path_to_enemy(eng, tr, me)
    ship = eng._spawn_ship(me, ShipType.LANDING, by_d[max(d for d in by_d if d <= norm)])
    tr.start(eng, me)
    _teleport(eng, ship, by_d[1])
    assert tr.settle(eng, me)["landing_progress"] == pytest.approx(min(0.3, tr.phi - tr.phi0))
    assert tr.totals["landing_progress"] <= 0.3 + 1e-12
    off = ShapingTracker(RewardShaping(enabled=True, build=0.01))
    off.start(eng, me)
    _teleport(eng, ship, by_d[2])
    assert off.settle(eng, me) == {}
    assert off.potential(eng, me) == 0.0


def test_only_own_landing_ships_count(tmp_path):
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    tr = ShapingTracker(_cfg())
    by_d, norm = _path_to_enemy(eng, tr, me)
    eng._spawn_ship(foe, ShipType.LANDING, by_d[1])  # enemy landing ship by its own home
    eng._spawn_ship(me, ShipType.DESTROYER, by_d[2])  # not a landing ship
    tr.start(eng, me)
    assert tr.phi == pytest.approx(-COEF)
    d, _, count = tr.landing_distance(eng, me)
    assert d is None and count == 0


def _policy(seed):
    torch.manual_seed(seed)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    return net


FLEETS = FleetStarts(prob=1.0, budget=1200, landing_prob=1.0, radius=6, landing_forward_prob=0.5)


def _segment_sum_matches(res):
    total = sum(t.reward for seg in res.segments for t in seg) - sum(f["outcome"] for f in res.finished)
    assert total == pytest.approx(res.shaping["landing_progress"], abs=1e-9)


def test_collector_attribution_single_process(tmp_path):
    maps, info = _pool(tmp_path, turns=30, steps=900)
    net = _policy(1)
    col = GameCollector(
        maps, info, model_config=model_config_of(net), seed=0, games=4, worker_id=0, num_workers=2,
        random_opponent_frac=1.0, shaping=_cfg(), fleet_starts=FLEETS,
    )
    col.set_weights(net.state_dict(), {}, [])
    torch.manual_seed(0)
    res = col.collect(1200)
    _segment_sum_matches(res)
    assert any(abs(t.reward) > 0 for seg in res.segments for t in seg)
    # Learner is always seat 0 here, so each game's paid progress telescopes.
    for f in res.finished:
        assert -COEF - 1e-9 <= f["shaping"]["landing_progress"] <= COEF + 1e-9
    for slot in col.slots:
        assert slot.tracker.totals["landing_progress"] == pytest.approx(slot.tracker.phi - slot.tracker.phi0, abs=1e-9)
    assert res.landing["samples"] == res.decisions
    assert res.landing["alive_sum"] > 0 and res.landing["min_dist"] is not None


def test_collector_attribution_parallel(tmp_path):
    maps, info = _pool(tmp_path, turns=30, steps=900)
    net = _policy(2)
    col = ParallelCollector(
        maps, info, num_workers=2, games_per_worker=3, collect_threads=1, episode_offset=0,
        model_config=model_config_of(net), seed=0, random_opponent_frac=1.0, shaping=_cfg(), fleet_starts=FLEETS,
    )
    try:
        res = col.collect(net, {}, 900)
    finally:
        col.close()
    _segment_sum_matches(res)
    assert res.landing["samples"] == res.decisions and res.landing["alive_sum"] > 0
    assert res.shaping["landing_progress"] != 0.0


@pytest.mark.parametrize("size", [PoolSize(16, 8, 2, ports=2), PoolSize(32, 16, 2, ports=3), PoolSize(48, 24, 2, ports=4)])
def test_forward_landing_placement_is_mirrored(tmp_path, size):
    maps, _ = _pool(tmp_path, sizes=(size,))
    for k, path in enumerate(maps):
        env = BattleboatsAEC(path, max_turns=50)
        env.reset(seed=k)
        eng = env.engine
        rng = random.Random(k)
        fleet = draw_fleet(rng, 1200, 1.0)
        ids = place_fleets(eng, fleet, rng, radius=6, forward_landing=True, forward_dist=(5, 7))
        types = [Counter(eng.ships[i].type for i in ids[p]) for p in (0, 1)]
        assert types[0] == types[1] == Counter(fleet)
        dists = []
        for p in (0, 1):
            ring = sea_distances(eng, eng.players[1 - p].home_port, 7)
            where = {t: d for d, tiles in ring.items() for t in tiles}
            landing = [eng.ships[i] for i in ids[p] if eng.ships[i].type == ShipType.LANDING]
            assert len(landing) == 1 and landing[0].owner == p
            assert landing[0].position in where  # near the ENEMY home
            dists.append(where[landing[0].position])
        assert dists[0] == dists[1] and 5 <= dists[0] <= 7


def test_forward_prob_off_keeps_landing_home(tmp_path):
    maps, _ = _pool(tmp_path)
    for i in range(30):
        env = BattleboatsAEC(maps[0], max_turns=50)
        env.reset(seed=i)
        eng = env.engine
        fleet = apply_fleet_start(eng, FleetStarts(prob=1.0, landing_prob=1.0), random.Random(i))
        assert ShipType.LANDING in fleet
        assert eng.fleet_start_forward is False
        for p in (0, 1):
            ring = sea_distances(eng, eng.players[p].home_port, 6)
            near = {t for tiles in ring.values() for t in tiles}
            for sid in eng.players[p].owned_ship_ids:
                assert eng.ships[sid].position in near


def test_landing_config_and_logging(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, load_selfplay_config, run_selfplay

    cfg = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets_landing.yaml")
    assert cfg.reward_shaping.landing_progress == pytest.approx(0.2)
    assert cfg.reward_shaping.caps["landing_progress"] == pytest.approx(0.3)
    assert cfg.fleet_starts.landing_forward_prob == pytest.approx(0.5)
    assert cfg.fleet_starts.landing_prob == pytest.approx(1.0)
    assert cfg.fleet_starts.landing_forward_dist == (5, 7)
    old = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets.yaml")
    assert old.reward_shaping.landing_progress == 0.0 and old.fleet_starts.landing_forward_prob == 0.0
    assert cfg.reward_shaping.landing_progress_distance_scale is False
    assert old.reward_shaping.landing_progress_distance_scale is False

    run = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=32),
        map_pool=[PoolSize(16, 8, 2, ports=2)], updates=2, rollout_decisions=150,
        max_turns=6, max_steps=120, num_workers=0, games_per_worker=3, ppo_threads=2,
        reward_shaping=_cfg(), fleet_starts=FLEETS,
    )
    assert run_selfplay(run)["ok"]
    rows = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    for r in rows:
        assert "shape_landing_progress" in r and "landing" in r and "landing_alive_mean" in r
    out = capsys.readouterr().out
    assert ",landing_progress:" in out and " land=bought:" in out and ",alive:" in out and "landing_forward_prob=0.5" in out


def test_forward_start_is_tagged(tmp_path):
    maps, _ = _pool(tmp_path)
    tags = []
    for i in range(40):
        env = BattleboatsAEC(maps[0], max_turns=50)
        env.reset(seed=i)
        fleet = apply_fleet_start(env.engine, FleetStarts(prob=1.0, landing_prob=1.0, landing_forward_prob=0.5), random.Random(i))
        assert fleet is not None
        tags.append(env.engine.fleet_start_forward)
    assert any(tags) and not all(tags)
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    assert apply_fleet_start(env.engine, FleetStarts(prob=0.0), random.Random(0)) is None
    assert env.engine.fleet_start_forward is False


def test_distance_scale_defaults_off_and_parses():
    assert RewardShaping().landing_progress_distance_scale is False
    off = RewardShaping.from_dict({"enabled": True, "weights": {"landing_progress": 0.2}})
    assert off.landing_progress_distance_scale is False
    on = RewardShaping.from_dict({
        "enabled": True,
        "weights": {"landing_progress": 0.2},
        "landing_progress_distance_scale": True,
    })
    assert on.landing_progress_distance_scale is True
    with pytest.raises(ValueError, match="unknown"):
        RewardShaping.from_dict({"enabled": True, "landing_progress_scale": True})


def test_distance_scale_full_trip_pays_coef_on_different_map_sizes(tmp_path):
    """With the flag on, walking a landing ship from home-distance to adjacent
    pays exactly landing_progress, on both a short and a longer map. With the
    flag off, the older formula pays coef * (start - 1) / norm, which shrinks
    on shorter paths.
    """
    sizes = (
        PoolSize(16, 8, 4, ports=2),
        PoolSize(20, 10, 4, ports=2),
    )
    paid_on, paid_off, norms = [], [], []
    for i, size in enumerate(sizes):
        maps, _ = _pool(tmp_path / f"s{i}", sizes=(size,))
        # pick a map whose home-to-home water path is long enough to walk
        eng = None
        by_d = norm = me = None
        for path in maps:
            env = BattleboatsAEC(path, max_turns=50)
            env.reset(seed=0)
            cand = env.engine
            tr = ShapingTracker(_cfg(landing_progress_distance_scale=True))
            bd, n = _path_to_enemy(cand, tr, cand.current_player)
            if n >= 4 and 1 in bd and n in bd:
                eng, by_d, norm, me = cand, bd, n, cand.current_player
                break
        assert eng is not None, f"no usable {size.width}x{size.height} map"
        norms.append(norm)

        def _walk(scale: bool) -> float:
            tr = ShapingTracker(_cfg(landing_progress_distance_scale=scale))
            ship = eng._spawn_ship(me, ShipType.LANDING, by_d[norm])
            tr.start(eng, me)
            if scale:
                assert tr.potential(eng, me) == pytest.approx(-COEF)
            _teleport(eng, ship, by_d[1])
            total = tr.settle(eng, me).get("landing_progress", 0.0)
            eng._destroy_ship(ship.id)
            return total

        on = _walk(True)
        off = _walk(False)
        paid_on.append(on)
        paid_off.append(off)
        assert on == pytest.approx(COEF)
        assert off == pytest.approx(COEF * (norm - 1) / norm)

    assert norms[0] != norms[1], f"need different path lengths, got {norms}"
    assert paid_on[0] == pytest.approx(paid_on[1])
    # The unscaled totals differ once the path lengths do.
    assert paid_off[0] != pytest.approx(paid_off[1])


def test_distance_scale_oscillation_still_nets_zero(tmp_path):
    eng = _engine(tmp_path)
    me = eng.current_player
    tr = ShapingTracker(_cfg(landing_progress_distance_scale=True))
    by_d, norm = _path_to_enemy(eng, tr, me)
    far = max(d for d in by_d if d <= norm)
    assert far >= 2 and (far - 1) in by_d
    ship = eng._spawn_ship(me, ShipType.LANDING, by_d[far])
    tr.start(eng, me)
    start = tr.totals["landing_progress"]
    _teleport(eng, ship, by_d[far - 1])
    tr.settle(eng, me)
    _teleport(eng, ship, by_d[far])
    tr.settle(eng, me)
    assert tr.totals["landing_progress"] == pytest.approx(start)

