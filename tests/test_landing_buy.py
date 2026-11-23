"""landing_buy: diminishing-returns reward for buying landing ships."""

import json
from pathlib import Path

import pytest

from battleboats.core.actions import BuildShipAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool
from battleboats.training.reward_shaping import COMPONENTS, RewardShaping, ShapingTracker
from battleboats.training.selfplay import load_selfplay_config

ROOT = Path(__file__).resolve().parents[1]


def _engine(tmp_path):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(16, 8, 1, ports=2)], seed=3, land_fraction=0.02)
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=0)
    eng = env.engine
    eng.players[eng.current_player].cash = 100_000
    return eng


def _free_next_to(eng, pos):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = (pos[0] + dx, pos[1] + dy)
        if eng.map.in_bounds(nb) and eng.map.is_water(nb) and not eng.map.is_occupied(nb):
            return nb
    return None


def _buy(tr, eng, ship_type, park=True):
    """Buy at home, score it, then move the new ship off the spawn tile (kept alive)."""
    pid = eng.current_player
    home = eng.players[pid].home_port
    spawn = _free_next_to(eng, home)
    assert spawn is not None
    before = set(eng.players[pid].owned_ship_ids)
    action = BuildShipAction(home, spawn, ship_type)
    pre = tr.snapshot(eng, pid, action)
    eng.step(action)
    got = tr.score(eng, pid, action, pre)
    (new,) = set(eng.players[pid].owned_ship_ids) - before
    if park:
        far = next(
            (x, y) for y in range(eng.map.height) for x in range(eng.map.width)
            if eng.map.is_water((x, y)) and not eng.map.is_occupied((x, y)) and abs(x - home[0]) + abs(y - home[1]) > 3
        )
        eng.map.remove_ship(eng.ships[new].position)
        eng.ships[new].position = far
        eng.map.place_ship(new, far)
    return got, new


def _rs(**kw):
    return RewardShaping(enabled=True, **{"build": 0.01, "landing_buy": 0.1, **kw})


def test_decays_with_landing_ships_alive(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs())
    got = [_buy(tr, eng, ShipType.LANDING)[0].get("landing_buy", 0.0) for _ in range(5)]
    assert got == pytest.approx([0.1, 0.05, 0.025, 0.0125, 0.00625])
    assert tr.totals["landing_buy"] == pytest.approx(sum(got))


def test_tenth_ship_is_tiny_and_decay_configurable(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs(landing_buy_decay=0.25))
    got = [_buy(tr, eng, ShipType.LANDING)[0]["landing_buy"] for _ in range(3)]
    assert got == pytest.approx([0.1, 0.025, 0.00625])
    assert 0.1 * 0.5 ** 9 == pytest.approx(0.0001953125)


def test_replacing_a_sunk_ship_pays_full_again(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs())
    first, sid = _buy(tr, eng, ShipType.LANDING)
    second, sid2 = _buy(tr, eng, ShipType.LANDING)
    assert (first["landing_buy"], second["landing_buy"]) == pytest.approx((0.1, 0.05))
    eng._destroy_ship(sid)
    assert _buy(tr, eng, ShipType.LANDING)[0]["landing_buy"] == pytest.approx(0.05)  # one still alive
    eng._destroy_ship(sid2)
    for s in [s for s in eng.players[eng.current_player].owned_ship_ids if eng.ships[s].type == ShipType.LANDING]:
        eng._destroy_ship(s)
    assert _buy(tr, eng, ShipType.LANDING)[0]["landing_buy"] == pytest.approx(0.1)  # none alive


def test_other_hulls_and_failed_buys_pay_nothing(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs())
    for hull in (ShipType.MERCHANT, ShipType.DESTROYER, ShipType.CARRIER, ShipType.BUILDER):
        got, _ = _buy(tr, eng, hull)
        assert "landing_buy" not in got and got["build"] > 0
    pid = eng.current_player
    eng.players[pid].cash = 0
    home = eng.players[pid].home_port
    action = BuildShipAction(home, _free_next_to(eng, home), ShipType.LANDING)
    pre = tr.snapshot(eng, pid, action)
    try:
        eng.step(action)
    except Exception:
        pass
    assert "landing_buy" not in tr.score(eng, pid, action, pre)
    assert tr.totals["landing_buy"] == 0.0


def test_cap_bounds_buy_and_suicide(tmp_path):
    # Buy, lose it, buy again: each replacement pays full price, so only the
    # per-game cap bounds the loop (0.3 = three full buys).
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs(caps={"landing_buy": 0.3}))
    paid = []
    for _ in range(6):
        got, sid = _buy(tr, eng, ShipType.LANDING)
        paid.append(got.get("landing_buy", 0.0))
        eng._destroy_ship(sid)
    assert paid == pytest.approx([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
    assert tr.totals["landing_buy"] == pytest.approx(0.3)


def test_counts_toward_total_cap(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(_rs(build=0.1, total_cap=0.25))
    got, _ = _buy(tr, eng, ShipType.CARRIER)  # build 0.1
    got, _ = _buy(tr, eng, ShipType.LANDING)  # build 0.1, then landing_buy gets 0.05 left
    assert got == {"build": pytest.approx(0.1), "landing_buy": pytest.approx(0.05)}
    got, _ = _buy(tr, eng, ShipType.LANDING)
    assert got == {}


def test_off_by_default_and_config():
    assert RewardShaping().landing_buy == 0.0
    tr_cfg = RewardShaping.from_dict({"enabled": True, "weights": {"build": 0.01}})
    assert tr_cfg.landing_buy == 0.0 and tr_cfg.landing_buy_decay == 0.5
    with pytest.raises(ValueError):
        RewardShaping.from_dict({"enabled": True, "landing_buy_decay": 1.5})
    cfg = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets_landing.yaml").reward_shaping
    assert cfg.landing_buy == pytest.approx(0.1) and cfg.landing_buy_decay == pytest.approx(0.5)
    assert cfg.caps["landing_buy"] == pytest.approx(0.3)
    old = load_selfplay_config(ROOT / "configs" / "selfplay_bigmaps_shaped_fleets.yaml").reward_shaping
    assert old.landing_buy == 0.0
    assert "landing_buy" in COMPONENTS


def test_off_pays_nothing(tmp_path):
    eng = _engine(tmp_path)
    tr = ShapingTracker(RewardShaping(enabled=True, build=0.01))
    assert "landing_buy" not in _buy(tr, eng, ShipType.LANDING)[0]


def test_selfplay_logs_landing_buy(tmp_path, capsys):
    from battleboats.training.curriculum import ArchSpec
    from battleboats.training.loop import PPOHyper
    from battleboats.training.selfplay import SelfPlayConfig, run_selfplay

    cfg = SelfPlayConfig(
        run_dir=tmp_path / "run", device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=16),
        map_pool=[PoolSize(16, 8, 2, ports=2)], updates=2, rollout_decisions=60,
        max_turns=6, max_steps=80, num_workers=0, games_per_worker=2, ppo_threads=2,
        reward_shaping=RewardShaping(enabled=True, build=0.01, landing_buy=0.1),
    )
    assert run_selfplay(cfg)["ok"]
    rows = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    for r in rows:
        assert "shape_landing_buy" in r and isinstance(r["landing_bought"], int)
        assert r["landing_bought"] == r["purchases"].get("landing", 0)
    out = capsys.readouterr().out
    assert ",landing_buy:" in out and " land=bought:" in out and "landing_buy_decay=0.5" in out
