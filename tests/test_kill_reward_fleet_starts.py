"""Zero-sum kill shaping (both sides, caps, collector attribution) and random
equal-cost fleet starts (cost, placement, fraction)."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from battleboats.core.actions import AttackAction, BuildPortAction, BuildShipAction
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs import observation as obs
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.fleet_starts import (
    COMBAT_HULLS,
    FleetStarts,
    apply_fleet_start,
    draw_fleet,
    fleet_cost,
    place_fleets,
    sea_distances,
)
from battleboats.training.policy import VERB_TO_IDX, PolicyNetwork
from battleboats.training.reward_shaping import RewardShaping, ShapingTracker
from battleboats.training.selfplay_purchases import hull_name
from battleboats.training.vec_collect import GameCollector, ParallelCollector

ROOT = Path(__file__).resolve().parents[1]


def _pool(tmp_path, sizes=(PoolSize(16, 8, 2, ports=2),), land=0.02, turns=6, steps=80):
    maps = build_map_pool(tmp_path / "maps", list(sizes), seed=3, land_fraction=land)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        t, s = episode_limits(data, max_turns=turns, max_steps=steps, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    return maps, info


def _engine(tmp_path, land=0.02):
    maps, _ = _pool(tmp_path, land=land)
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    return env.engine


def _water_pair(eng, n=1):
    """n disjoint pairs of adjacent free water tiles."""
    used, out = set(), []
    for x in range(eng.map.width):
        for y in range(eng.map.height):
            a, b = (x, y), (x + 1, y)
            if any(p in used for p in (a, b)):
                continue
            if all(eng.map.in_bounds(p) and eng.map.is_water(p) and not eng.map.is_occupied(p) for p in (a, b)):
                out.append((a, b))
                used.update((a, b, (x - 1, y), (x + 2, y)))
                if len(out) == n:
                    return out
    raise AssertionError("not enough water")


def _sink(eng, attacker_type, victim_type, attacker_pos, victim_pos):
    me = eng.current_player
    atk = eng._spawn_ship(me, attacker_type, attacker_pos)
    vic = eng._spawn_ship(1 - me, victim_type, victim_pos)
    return atk, vic


def _kill_cfg(**kw):
    return RewardShaping(enabled=True, **{"kill": 0.02, **kw})


def test_kill_is_zero_sum_between_players(tmp_path):
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    t_me, t_foe = ShapingTracker(_kill_cfg()), ShapingTracker(_kill_cfg())
    (a, b), = _water_pair(eng)
    atk, vic = _sink(eng, ShipType.CARRIER, ShipType.MERCHANT, a, b)  # strength 0 always dies
    t_me.mark_fleet(eng, me)
    t_foe.mark_fleet(eng, foe)
    action = AttackAction(atk.id, vic.id)
    pre = t_me.snapshot(eng, me, action)
    eng.step(action)
    assert vic.id not in eng.ships
    gain = t_me.score(eng, me, action, pre)
    assert gain == {"kill": pytest.approx(0.02 * 100 / 250)}
    loss = t_foe.settle_losses(eng, foe)
    assert loss == pytest.approx(-0.02 * 100 / 250)
    assert t_me.totals["kill"] + t_foe.totals["kill"] == pytest.approx(0.0)
    # Settling again finds nothing new; the attacker's own fleet lost nothing.
    assert t_foe.settle_losses(eng, foe) == 0.0
    assert t_me.settle_losses(eng, me) == 0.0


def test_missed_or_illegal_attack_pays_nothing(tmp_path):
    eng = _engine(tmp_path)
    me = eng.current_player
    tr = ShapingTracker(_kill_cfg())
    (a, b), = _water_pair(eng)
    atk, vic = _sink(eng, ShipType.CARRIER, ShipType.MERCHANT, a, b)
    own = eng._spawn_ship(me, ShipType.DESTROYER, _water_pair(eng)[0][0])
    action = AttackAction(atk.id, own.id)  # own ship: engine rejects
    pre = tr.snapshot(eng, me, action)
    eng.step(action)
    assert tr.score(eng, me, action, pre) == {}
    assert own.id in eng.ships


def test_builder_spent_on_a_port_is_not_a_loss(tmp_path):
    eng = _engine(tmp_path, land=0.2)
    me = eng.current_player
    tr = ShapingTracker(_kill_cfg(port_build=0.05))
    for tile in obs._coastline_tiles(eng):
        water = [
            (tile[0] + dx, tile[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if eng.map.in_bounds((tile[0] + dx, tile[1] + dy)) and eng.map.is_water((tile[0] + dx, tile[1] + dy))
            and not eng.map.is_occupied((tile[0] + dx, tile[1] + dy))
        ]
        if water:
            break
    builder = eng._spawn_ship(me, ShipType.BUILDER, water[0])
    tr.mark_fleet(eng, me)
    action = BuildPortAction(builder.id, tile)
    pre = tr.snapshot(eng, me, action)
    eng.step(action)
    assert tr.score(eng, me, action, pre) == {"port_build": pytest.approx(0.05)}
    tr.mark_fleet(eng, me)  # the collector re-marks after the learner's own step
    assert tr.settle_losses(eng, me) == 0.0
    assert tr.totals["kill"] == 0.0


def test_kill_caps_are_symmetric_and_separate_from_total_cap(tmp_path):
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    cfg = _kill_cfg(kill=0.1, build=0.05, caps={"kill": 0.15}, total_cap=0.05)
    t_me, t_foe = ShapingTracker(cfg), ShapingTracker(cfg)
    pairs = _water_pair(eng, 3)
    ships = [_sink(eng, ShipType.DESTROYER, ShipType.BUILDER, a, b) for a, b in pairs]  # builder: cost 250, strength 0
    t_me.mark_fleet(eng, me)
    t_foe.mark_fleet(eng, foe)
    gains, losses = [], []
    for atk, vic in ships:
        action = AttackAction(atk.id, vic.id)
        pre = t_me.snapshot(eng, me, action)
        eng.step(action)
        gains.append(t_me.score(eng, me, action, pre).get("kill", 0.0))
        losses.append(t_foe.settle_losses(eng, foe))
    assert gains == pytest.approx([0.1, 0.05, 0.0])
    assert losses == pytest.approx([-0.1, -0.05, 0.0])
    assert t_me.totals["kill"] == pytest.approx(0.15)
    assert t_foe.totals["kill"] == pytest.approx(-0.15)
    # total_cap (0.05) applies to the non-kill components only.
    eng.players[me].cash = 1000
    home = eng.players[me].home_port
    spawn = next(
        (home[0] + dx, home[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if eng.map.in_bounds((home[0] + dx, home[1] + dy)) and eng.map.is_water((home[0] + dx, home[1] + dy))
        and not eng.map.is_occupied((home[0] + dx, home[1] + dy))
    )
    action = BuildShipAction(home, spawn, ShipType.CARRIER)
    pre = t_me.snapshot(eng, me, action)
    eng.step(action)
    assert t_me.score(eng, me, action, pre) == {"build": pytest.approx(0.05)}


def _policy(seed):
    torch.manual_seed(seed)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    return net


def _expected(book, coef):
    return sum(coef * BASE_STATS[ShipType[h.upper()]].cost / 250 for h, n in book.items() for _ in range(n))


def _check_attribution(res, coef, exact):
    attack_v = VERB_TO_IDX["attack"]
    outcomes = {}
    seg_total = 0.0
    for seg in res.segments:
        for t in seg[:-1]:
            if t.verb_idx != attack_v:
                assert t.reward <= 1e-12  # non-attack steps can only carry settled losses
        seg_total += sum(t.reward for t in seg)
    seg_total -= sum(f["outcome"] for f in res.finished)
    assert seg_total == pytest.approx(res.shaping["kill"], abs=1e-9)
    gain = _expected(res.combat["sunk_by_learner"], coef)
    loss = _expected(res.combat["lost_by_learner"], coef)
    if exact:
        assert res.shaping["kill"] == pytest.approx(gain - loss, abs=1e-9)
    else:  # losses before a learner's first decision are dropped
        assert gain - loss - 1e-9 <= res.shaping["kill"] <= gain + 1e-9
    return gain, loss


def test_collector_kill_attribution_single_process(tmp_path):
    maps, info = _pool(tmp_path, turns=40, steps=1500)
    net = _policy(4)
    # num_workers=2, worker_id=0, offset 0: every episode index is even, so the
    # learner is seat 0 and always moves first (no loss before its first decision).
    col = GameCollector(
        maps, info, model_config=model_config_of(net), seed=0, games=4, worker_id=0, num_workers=2,
        random_opponent_frac=1.0, shaping=RewardShaping(enabled=True, kill=1.0),
        fleet_starts=FleetStarts(prob=1.0, budget=1500, landing_prob=0.5, radius=6),
    )
    col.set_weights(net.state_dict(), {}, [])
    torch.manual_seed(0)
    res = col.collect(1500)
    gain, loss = _check_attribution(res, 1.0, exact=True)
    assert res.combat["learner_sinks"] > 0 and res.combat["opp_sinks"] > 0
    assert gain > 0 and loss > 0
    assert res.combat["learner_attacks"] >= res.combat["learner_sinks"]


def test_collector_kill_attribution_parallel(tmp_path):
    maps, info = _pool(tmp_path, turns=40, steps=1500)
    net = _policy(5)
    col = ParallelCollector(
        maps, info, num_workers=2, games_per_worker=3, collect_threads=1, episode_offset=0,
        model_config=model_config_of(net), seed=0, random_opponent_frac=1.0,
        shaping=RewardShaping(enabled=True, kill=1.0),
        fleet_starts=FleetStarts(prob=1.0, budget=1500, landing_prob=0.5, radius=6),
    )
    try:
        res = col.collect(net, {}, 1200)
    finally:
        col.close()
    gain, loss = _check_attribution(res, 1.0, exact=False)
    assert res.combat["learner_sinks"] + res.combat["opp_sinks"] > 0


def test_draw_fleet_cost_and_landing_rate():
    rng = random.Random(0)
    landing = 0
    for _ in range(2000):
        fleet = draw_fleet(rng, 1200, 0.5)
        cost = fleet_cost(fleet)
        assert 1200 - 100 < cost <= 1200
        assert all(t in COMBAT_HULLS or t == ShipType.LANDING for t in fleet)
        assert fleet.count(ShipType.LANDING) <= 1
        landing += ShipType.LANDING in fleet
    assert 0.45 < landing / 2000 < 0.55


@pytest.mark.parametrize("size", [PoolSize(16, 8, 2, ports=2), PoolSize(32, 16, 2, ports=3), PoolSize(48, 24, 2, ports=4)])
def test_place_fleets_equal_cost_legal_and_mirrored(tmp_path, size):
    maps, _ = _pool(tmp_path, sizes=(size,))
    for k, path in enumerate(maps):
        env = BattleboatsAEC(path, max_turns=50)
        env.reset(seed=k)
        eng = env.engine
        rng = random.Random(k)
        fleet = draw_fleet(rng, 1500, 0.5)
        ids = place_fleets(eng, fleet, rng, radius=6)
        types = [Counter(eng.ships[i].type for i in ids[p]) for p in (0, 1)]
        assert types[0] == types[1] == Counter(fleet)
        assert fleet_cost([eng.ships[i].type for i in ids[0]]) == fleet_cost([eng.ships[i].type for i in ids[1]])
        positions = [eng.ships[i].position for p in (0, 1) for i in ids[p]]
        assert len(set(positions)) == len(positions)
        for p in (0, 1):
            ring = sea_distances(eng, eng.players[p].home_port, 6)
            near = {t for tiles in ring.values() for t in tiles}
            for i in ids[p]:
                ship = eng.ships[i]
                assert ship.owner == p and eng.map.is_water(ship.position) and ship.position in near
                assert eng.map.ship_at[ship.position] == i
            assert set(ids[p]) == set(eng.players[p].owned_ship_ids)
        # Mirrored: the sorted sea distances from each home match.
        dist = []
        for p in (0, 1):
            ring = sea_distances(eng, eng.players[p].home_port, 6)
            where = {t: d for d, tiles in ring.items() for t in tiles}
            dist.append(sorted(where[eng.ships[i].position] for i in ids[p]))
        assert sum(abs(a - b) for a, b in zip(*dist)) <= 2 * len(fleet)
        assert eng.players[0].cash == eng.players[1].cash


def test_fleet_start_fraction_and_off_by_default(tmp_path):
    maps, info = _pool(tmp_path)
    hits = sum(apply_fleet_start(_fresh(maps), FleetStarts(prob=0.333), random.Random(i)) is not None for i in range(600))
    assert 0.333 - 0.06 < hits / 600 < 0.333 + 0.06
    assert all(apply_fleet_start(_fresh(maps), FleetStarts(), random.Random(i)) is None for i in range(20))
    net = _policy(6)
    col = GameCollector(
        maps, info, model_config=model_config_of(net), seed=0, games=1,
        fleet_starts=FleetStarts(prob=0.333),
    )
    col._make_slots()
    slot = col.slots[0]
    starts = []
    for _ in range(240):
        col._start(slot)
        eng = slot.env.engine
        n0, n1 = len(eng.players[0].owned_ship_ids), len(eng.players[1].owned_ship_ids)
        assert n0 == n1
        assert (n0 > 0) == slot.fleet_start
        starts.append(slot.fleet_start)
    assert 0.333 - 0.09 < sum(starts) / len(starts) < 0.333 + 0.09


def _fresh(maps):
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    return env.engine


def test_fleets_config_and_logging(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, load_selfplay_config, run_selfplay

    cfg = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets.yaml")
    assert cfg.reward_shaping.enabled and cfg.reward_shaping.kill == pytest.approx(0.02)
    assert cfg.reward_shaping.caps["kill"] == pytest.approx(0.3)
    assert cfg.fleet_starts.prob == pytest.approx(0.333)
    assert load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps.yaml").fleet_starts.prob == 0.0

    run = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=32),
        map_pool=[PoolSize(16, 8, 2, ports=2)], updates=2, rollout_decisions=150,
        max_turns=6, max_steps=120, num_workers=0, games_per_worker=3, ppo_threads=2,
        reward_shaping=RewardShaping(enabled=True, kill=0.02, build=0.01),
        fleet_starts=FleetStarts(prob=1.0),
    )
    assert run_selfplay(run)["ok"]
    rows = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    for r in rows:
        assert "shape_kill" in r and "combat" in r and "by_start" in r and "mean_turn" in r
        assert set(r["combat"]) >= {"learner_attacks", "learner_sinks", "opp_attacks", "opp_sinks"}
    out = capsys.readouterr().out
    assert "fleet starts ON" in out and ",kill:" in out and " atk=L" in out


def test_ship_lost_defaults_to_zero_and_kill_stays_zero_sum(tmp_path):
    """Old configs: ship_lost is 0, so a sink is still a zero-sum kill only."""
    cfg = RewardShaping.from_dict({"enabled": True, "weights": {"kill": 0.02}})
    assert cfg.ship_lost == 0.0
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    t_me, t_foe = ShapingTracker(cfg), ShapingTracker(cfg)
    (a, b), = _water_pair(eng)
    atk, vic = _sink(eng, ShipType.CARRIER, ShipType.MERCHANT, a, b)
    t_me.mark_fleet(eng, me)
    t_foe.mark_fleet(eng, foe)
    action = AttackAction(atk.id, vic.id)
    pre = t_me.snapshot(eng, me, action)
    eng.step(action)
    assert t_me.score(eng, me, action, pre) == {"kill": pytest.approx(0.02 * 100 / 250)}
    assert t_foe.settle(eng, foe) == {"kill": pytest.approx(-0.02 * 100 / 250)}
    assert t_foe.totals["ship_lost"] == 0.0
    assert t_me.totals["kill"] + t_foe.totals["kill"] == pytest.approx(0.0)


def test_ship_lost_is_extra_cost_scaled_penalty(tmp_path):
    """Losing a merchant costs -ship_lost * 100/250, on top of the kill loss.
    The attacker does not receive ship_lost, so a 1-for-1 trade is net negative.
    """
    cfg = RewardShaping(enabled=True, kill=0.02, ship_lost=0.02)
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    t_me, t_foe = ShapingTracker(cfg), ShapingTracker(cfg)
    (a, b), = _water_pair(eng)
    atk, vic = _sink(eng, ShipType.CARRIER, ShipType.MERCHANT, a, b)
    t_me.mark_fleet(eng, me)
    t_foe.mark_fleet(eng, foe)
    action = AttackAction(atk.id, vic.id)
    pre = t_me.snapshot(eng, me, action)
    eng.step(action)
    gain = t_me.score(eng, me, action, pre)
    lost = t_foe.settle(eng, foe)
    unit = 100 / 250
    assert gain == {"kill": pytest.approx(0.02 * unit)}
    assert lost == {
        "kill": pytest.approx(-0.02 * unit),
        "ship_lost": pytest.approx(-0.02 * unit),
    }
    # Attacker's net for this sink: +kill. Victim: -kill -ship_lost.
    assert t_me.totals["kill"] + t_foe.totals["kill"] == pytest.approx(0.0)
    assert t_me.totals["ship_lost"] == 0.0
    assert t_foe.totals["ship_lost"] == pytest.approx(-0.02 * unit)


def test_ship_lost_works_without_kill(tmp_path):
    cfg = RewardShaping(enabled=True, kill=0.0, ship_lost=0.03)
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    tr = ShapingTracker(cfg)
    (a, b), = _water_pair(eng)
    atk, vic = _sink(eng, ShipType.CARRIER, ShipType.DESTROYER, a, b)
    tr.mark_fleet(eng, foe)
    eng.step(AttackAction(atk.id, vic.id))
    assert vic.id not in eng.ships
    got = tr.settle(eng, foe)
    assert got == {"ship_lost": pytest.approx(-0.03 * 100 / 250)}
    assert tr.totals["kill"] == 0.0


def test_ship_lost_scales_with_hull_cost(tmp_path):
    cfg = RewardShaping(enabled=True, ship_lost=0.02)
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    tr = ShapingTracker(cfg)
    penalties = []
    for victim, pair in ((ShipType.MERCHANT, 0), (ShipType.LANDING, 1)):
        pairs = _water_pair(eng, n=2)
        atk, vic = _sink(eng, ShipType.CARRIER, victim, *pairs[pair])
        # Victim owner is whoever is not the attacker at spawn time.
        owner = vic.owner
        tr.mark_fleet(eng, owner)
        eng.step(AttackAction(atk.id, vic.id))
        penalties.append(tr.settle(eng, owner)["ship_lost"])
    assert penalties[0] == pytest.approx(-0.02 * 100 / 250)
    assert penalties[1] == pytest.approx(-0.02 * 250 / 250)
    assert abs(penalties[1]) == pytest.approx(abs(penalties[0]) * 2.5)


def test_ship_lost_cap_and_not_in_total_cap(tmp_path):
    cfg = RewardShaping(
        enabled=True, ship_lost=0.1, build=0.05,
        caps={"ship_lost": 0.15}, total_cap=0.05,
    )
    eng = _engine(tmp_path)
    me, foe = eng.current_player, 1 - eng.current_player
    tr = ShapingTracker(cfg)
    # Three landing losses at 0.1 each would be 0.3; cap is 0.15.
    penalties = []
    for _ in range(3):
        (a, b), = _water_pair(eng, n=1)
        atk, vic = _sink(eng, ShipType.CARRIER, ShipType.LANDING, a, b)
        tr.mark_fleet(eng, vic.owner)
        eng.step(AttackAction(atk.id, vic.id))
        penalties.append(tr.settle(eng, vic.owner).get("ship_lost", 0.0))
    assert penalties == pytest.approx([-0.1, -0.05, 0.0])
    assert tr.totals["ship_lost"] == pytest.approx(-0.15)
    # total_cap ignores ship_lost: with build already at 0.05, a further 0.05
    # build is clipped to 0. If ship_lost (-0.15) were included in the
    # positive sum, there would be room left and _capped would pay 0.05.
    tr.totals["build"] = 0.05
    assert tr._capped("build", 0.05) == 0.0


def test_builder_spent_on_a_port_is_not_ship_lost(tmp_path):
    eng = _engine(tmp_path, land=0.2)
    me = eng.current_player
    tr = ShapingTracker(RewardShaping(enabled=True, ship_lost=0.02, port_build=0.05))
    for tile in obs._coastline_tiles(eng):
        water = [
            (tile[0] + dx, tile[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if eng.map.in_bounds((tile[0] + dx, tile[1] + dy)) and eng.map.is_water((tile[0] + dx, tile[1] + dy))
            and not eng.map.is_occupied((tile[0] + dx, tile[1] + dy))
        ]
        if water:
            break
    builder = eng._spawn_ship(me, ShipType.BUILDER, water[0])
    tr.mark_fleet(eng, me)
    action = BuildPortAction(builder.id, tile)
    pre = tr.snapshot(eng, me, action)
    eng.step(action)
    assert tr.score(eng, me, action, pre) == {"port_build": pytest.approx(0.05)}
    tr.mark_fleet(eng, me)
    assert tr.settle(eng, me) == {}
    assert tr.totals["ship_lost"] == 0.0
