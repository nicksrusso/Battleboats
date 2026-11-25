"""Rollout speedups: cached observation pieces, batched acting, exact-cap
collection across worker processes, and the self-play knobs that use them."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from battleboats.envs import observation as obs
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.agents.random_agent import random_action
from battleboats.training.checkpoints import load_checkpoint, model_config_of, save_checkpoint
from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.policy import PolicyNetwork, set_dropout
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.rollout import Transition, compute_gae
from battleboats.training.selfplay import SelfPlayConfig, run_selfplay
from battleboats.training.vec_collect import GameCollector, ParallelCollector, split_budget


def _old_fourier_axis(coord, wavelengths, row, offset):
    for i, wavelength in enumerate(wavelengths):
        angle = 2.0 * math.pi * (coord / wavelength)
        row[offset + 2 * i] = math.sin(angle)
        row[offset + 2 * i + 1] = math.cos(angle)


def _old_coastline(engine):
    m = engine.map
    out = []
    for x in range(m.width):
        for y in range(m.height):
            pos = (x, y)
            if not m.is_land(pos) or m.is_port(pos):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if m.in_bounds(nb) and m.is_water(nb):
                    out.append(pos)
                    break
    return out


def _pool(tmp_path, sizes=(PoolSize(16, 8, 2, ports=2),)):
    maps = build_map_pool(tmp_path / "maps", list(sizes), seed=3)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        turns, steps = episode_limits(data, max_turns=4, max_steps=24, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": turns, "max_steps": steps})
    return maps, info


def _played_env(maps, steps=40, seed=0):
    env = BattleboatsAEC(maps[0], max_turns=50)
    env.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        env.step(random_action(env.engine, env._player_id(agent), rng))
    return env


def _small_policy(seed=0):
    torch.manual_seed(seed)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    return net


def test_fourier_axis_matches_old_loop_exactly() -> None:
    sets = [obs.POSITION_WAVELENGTHS, obs.HOME_REL_WAVELENGTHS]
    for wavelengths in sets:
        for coord in list(range(-300, 301)) + [0.5, -7.25, 1e4]:
            a = np.zeros(2 * len(wavelengths) + 3, dtype=np.float32)
            b = np.zeros_like(a)
            _old_fourier_axis(coord, wavelengths, a, 3)
            obs._fourier_axis(coord, wavelengths, b, 3)
            assert np.array_equal(a, b)


def test_coastline_cache_matches_scan_and_drops_new_ports(tmp_path: Path) -> None:
    maps, _ = _pool(tmp_path)
    env = _played_env(maps, steps=30)
    engine = env.engine
    assert obs._coastline_tiles(engine) == _old_coastline(engine)
    assert engine.map._coastline_cache is not None
    coast = obs._coastline_tiles(engine)
    if coast:
        engine.map.add_port(coast[0], 0)
        assert engine.map._coastline_cache is None
        assert coast[0] not in obs._coastline_tiles(engine)
        assert obs._coastline_tiles(engine) == _old_coastline(engine)
    engine.map.load(maps[1])
    assert engine.map._coastline_cache is None


def test_tokens_and_globals_built_once_and_unchanged(tmp_path: Path) -> None:
    maps, _ = _pool(tmp_path)
    env = _played_env(maps)
    masks = ActionMasks(env.engine, 0)
    assert masks.tokens is masks.tokens
    assert masks.globals is masks.globals
    assert np.array_equal(masks.tokens[0].numpy(), obs.build_entity_tokens(env.engine, 0))
    assert np.array_equal(masks.globals[0].numpy(), obs.build_globals(env.engine, 0))


def test_priming_can_be_switched_off(tmp_path: Path) -> None:
    maps, _ = _pool(tmp_path)
    env = BattleboatsAEC(maps[0], max_turns=10)
    env.reset(seed=0)
    assert env.infos[env.agent_selection]["legal_actions"]
    env.prime_legal_actions = False
    env.reset(seed=0)
    assert "legal_actions" not in env.infos[env.agent_selection]


def test_act_batch_b1_matches_act(tmp_path: Path) -> None:
    maps, _ = _pool(tmp_path)
    net = _small_policy()
    for steps in (0, 15, 35):
        env = _played_env(maps, steps=steps, seed=steps)
        pid = env._player_id(env.agent_selection)
        masks = ActionMasks(env.engine, pid)
        torch.manual_seed(11)
        a, v, t, lp, val = net.act(masks.tokens, masks.pad_mask, masks, skill_idx=0)
        torch.manual_seed(11)
        out = net.act_batch(masks.tokens, masks.pad_mask, [masks], skill_idx=0, globals=masks.globals)
        assert int(out["asset_idx"][0]) == int(a)
        assert int(out["verb_idx"][0]) == int(v)
        assert int(out["target_idx"][0]) == (-1 if t is None else int(t))
        assert torch.allclose(out["logp"], lp)
        assert torch.allclose(out["value"], val)


def test_act_batch_padded_rows_match_single_state_scores(tmp_path: Path) -> None:
    maps, _ = _pool(tmp_path)
    net = _small_policy(1)
    masks_list = []
    for steps in (0, 9, 21, 37, 55):
        env = _played_env(maps, steps=steps, seed=steps + 1)
        masks_list.append(ActionMasks(env.engine, env._player_id(env.agent_selection)))
    toks = [m.tokens[0] for m in masks_list]
    lengths = torch.tensor([x.shape[0] for x in toks])
    tokens = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True)
    pad = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
    glob = torch.cat([m.globals for m in masks_list])
    torch.manual_seed(5)
    out = net.act_batch(tokens, pad, masks_list, skill_idx=0, globals=glob)
    for i, m in enumerate(masks_list):
        a = out["asset_idx"][i : i + 1]
        v = out["verb_idx"][i : i + 1]
        t = out["target_idx"][i : i + 1].clamp(min=0) if int(out["target_idx"][i]) >= 0 else out["target_idx"][i : i + 1]
        with torch.no_grad():
            lp, _ent, val = net.evaluate_actions(m.tokens, m.pad_mask, a, v, t, m, skill_idx=0, globals=m.globals)
        assert torch.allclose(lp, out["logp"][i : i + 1], atol=1e-5)
        assert torch.allclose(val, out["value"][i : i + 1], atol=1e-5)


def test_compute_gae_bootstraps_cut_segments() -> None:
    def seg():
        return [Transition(None, None, None, None, 0, 0, -1, 0.0, value=0.5) for _ in range(3)]

    done, cut = seg(), seg()
    compute_gae(done, gamma=0.9, lam=1.0)
    compute_gae(cut, gamma=0.9, lam=1.0, last_value=0.8)
    assert done[-1].advantage == pytest.approx(-0.5)
    assert cut[-1].advantage == pytest.approx(0.9 * 0.8 - 0.5)


def test_dropout_override_keeps_checkpoint_loadable(tmp_path: Path) -> None:
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.1)
    path = save_checkpoint(net, tmp_path / "old.pt")
    loaded, cfg = load_checkpoint(path, torch.device("cpu"))
    assert cfg["dropout"] == pytest.approx(0.1)
    set_dropout(loaded, 0.0)
    assert all(m.p == 0.0 for m in loaded.modules() if isinstance(m, torch.nn.Dropout))
    assert model_config_of(loaded)["dropout"] == 0.0
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    loaded.train()
    x = torch.randn(2, 5, obs.TOKEN_DIM)
    pad = torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(loaded.encode(x, pad)[1], loaded.encode(x, pad)[1])
    again, _ = load_checkpoint(save_checkpoint(loaded, tmp_path / "new.pt"), torch.device("cpu"))
    assert model_config_of(again)["dropout"] == 0.0


def _collector(maps, info, net, **kw):
    col = GameCollector(maps, info, model_config=model_config_of(net), seed=0, random_opponent_frac=0.5, **kw)
    opp = _small_policy(9)
    col.set_weights(net.state_dict(), {"update_00000": opp.state_dict()}, ["update_00000"])
    return col


def test_collector_caps_exactly_and_logps_replay(tmp_path: Path) -> None:
    maps, info = _pool(tmp_path)
    net = _small_policy(2)
    col = _collector(maps, info, net, games=3)
    total_games = 0
    for budget in (25, 40, 7):
        res = col.collect(budget)
        assert res.decisions == budget
        assert sum(len(s) for s in res.segments) == budget
        total_games += len(res.finished)
        for info_row in res.finished:
            assert info_row["outcome"] in (-1.0, 0.0, 1.0)
            assert info_row["map_size"] == "16x8"
        buf = RolloutBuffer()
        for s in res.segments:
            buf.add(s)
        for batch, masks in buf.iter_minibatches(16, shuffle=False):
            with torch.no_grad():
                lp, _e, val = net.evaluate_actions(
                    batch["tokens"], batch["pad_mask"], batch["asset_idx"], batch["verb_idx"],
                    batch["target_idx"], masks, skill_idx=batch["skill_idx"], globals=batch["globals"],
                )
            assert torch.allclose(lp, batch["logp_old"], atol=1e-4)
            assert torch.allclose(val, batch["value_old"], atol=1e-4)
    assert total_games > 0
    assert col.episodes_started == 3 + total_games


def test_split_budget_sums() -> None:
    assert split_budget(2048, 7) == [293] * 4 + [292] * 3
    assert sum(split_budget(10, 3)) == 10


def test_parallel_collector_two_workers(tmp_path: Path) -> None:
    maps, info = _pool(tmp_path)
    net = _small_policy(3)
    col = ParallelCollector(
        maps, info, num_workers=2, games_per_worker=2, collect_threads=1, episode_offset=10,
        model_config=model_config_of(net), seed=0, random_opponent_frac=0.5,
    )
    try:
        res = col.collect(net, {"update_00000": _small_policy(4)}, 31)
        assert res.decisions == 31
        assert sum(len(s) for s in res.segments) == 31
        tr = res.segments[0][0]
        assert isinstance(tr.tokens, torch.Tensor) and isinstance(tr.asset_mask, torch.Tensor)
        assert col.next_episode() >= 10 + 2 * 2
        res2 = col.collect(net, {"update_00000": _small_policy(4)}, 12)
        assert res2.decisions == 12
    finally:
        col.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_selfplay_update_with_workers(tmp_path: Path, workers: int) -> None:
    cfg = SelfPlayConfig(
        run_dir=tmp_path / "run",
        device="cpu",
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.1),
        ppo=PPOHyper(epochs=1, minibatch_size=8),
        map_pool=[PoolSize(16, 8, 2, ports=2)],
        updates=2,
        rollout_decisions=20,
        max_turns=4,
        max_steps=40,
        snapshot_every=1,
        num_workers=workers,
        games_per_worker=2,
        ppo_threads=2,
    )
    assert run_selfplay(cfg)["ok"]
    rows = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert [r["decisions"] for r in rows] == [20, 20]
    assert all("collect_s" in r and "ppo_s" in r for r in rows)
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["episode"] > 0
    loaded, cfg_out = load_checkpoint(tmp_path / "run" / "latest.pt", torch.device("cpu"))
    assert cfg_out["dropout"] == 0.0


def test_bigmaps_config_collection_knobs() -> None:
    from battleboats.training.selfplay import load_selfplay_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_selfplay_config(root / "configs" / "selfplay_bigmaps.yaml")
    assert (cfg.num_workers, cfg.games_per_worker, cfg.collect_threads, cfg.ppo_threads) == (7, 8, 1, 8)
    assert cfg.dropout == 0.0
    assert cfg.rollout_decisions == 2048
    assert cfg.init_checkpoint is None


def _record_dists(monkeypatch):
    """Wrap policy.masked_categorical so every distribution built is kept."""
    from battleboats.training import policy as policy_mod

    seen = []
    real = policy_mod.masked_categorical

    def wrapped(logits, mask=None):
        dist = real(logits, mask)
        seen.append(dist.probs.detach().clone())
        return dist

    monkeypatch.setattr(policy_mod, "masked_categorical", wrapped)
    return seen


def test_act_batch_full_distributions_match_serial(tmp_path: Path, monkeypatch) -> None:
    """Every factor's full probability vector in a padded batch equals the
    single-state (serial) path's vector for the same state and prefix."""
    maps, _ = _pool(tmp_path, sizes=(PoolSize(16, 8, 1, ports=2), PoolSize(32, 16, 1, ports=3)))
    net = _small_policy(6)
    masks_list = []
    for k, steps in enumerate((0, 7, 18, 33, 47, 61, 80, 95)):
        env = _played_env([maps[k % len(maps)]], steps=steps, seed=steps + 2)
        masks_list.append(ActionMasks(env.engine, env._player_id(env.agent_selection)))
    from battleboats.training.vec_collect import _batch_inputs

    tokens, pad, glob = _batch_inputs(masks_list)
    seen = _record_dists(monkeypatch)
    torch.manual_seed(3)
    out = net.act_batch(tokens, pad, masks_list, skill_idx=0, globals=glob)
    batch_asset, batch_verb, batch_targets = seen[0], seen[1], seen[2:]
    verbs = out["verb_idx"].tolist()
    # Target groups come back in VERBS_WITH_TARGET order; map row -> its probs.
    from battleboats.training.policy import VERB_TO_IDX, VERBS_WITH_TARGET

    row_target = {}
    g = 0
    for name in VERBS_WITH_TARGET:
        rows = [i for i, v in enumerate(verbs) if v == VERB_TO_IDX[name]]
        if rows:
            for j, i in enumerate(rows):
                row_target[i] = batch_targets[g][j]
            g += 1
    assert g == len(batch_targets)
    for i, m in enumerate(masks_list):
        seen.clear()
        t = out["target_idx"][i : i + 1].clamp(min=0)
        with torch.no_grad():
            net.evaluate_actions(
                m.tokens, m.pad_mask, out["asset_idx"][i : i + 1], out["verb_idx"][i : i + 1], t, m,
                skill_idx=0, globals=m.globals,
            )
        n = m.tokens.shape[1]
        assert torch.allclose(seen[0][0], batch_asset[i, :n], atol=1e-5)
        assert float(batch_asset[i, n:].sum()) == 0.0  # padding never sampled
        assert torch.allclose(seen[1][0], batch_verb[i], atol=1e-5)
        if i in row_target:
            assert torch.allclose(seen[2][0], row_target[i][: seen[2][0].shape[0]], atol=1e-5)
        else:
            assert len(seen) == 2


def test_parallel_collector_logps_replay_serially(tmp_path: Path) -> None:
    """Worker-collected transitions re-score to the same logp/value under the
    serial evaluate_actions path (what PPO uses)."""
    maps, info = _pool(tmp_path)
    net = _small_policy(7)
    col = ParallelCollector(
        maps, info, num_workers=2, games_per_worker=3, collect_threads=1, episode_offset=0,
        model_config=model_config_of(net), seed=1, random_opponent_frac=0.5,
    )
    try:
        res = col.collect(net, {"update_00000": _small_policy(8)}, 40)
    finally:
        col.close()
    assert res.decisions == 40
    buf = RolloutBuffer()
    for s in res.segments:
        buf.add(s)
    for batch, masks in buf.iter_minibatches(16, shuffle=False):
        with torch.no_grad():
            lp, _e, val = net.evaluate_actions(
                batch["tokens"], batch["pad_mask"], batch["asset_idx"], batch["verb_idx"],
                batch["target_idx"], masks, skill_idx=batch["skill_idx"], globals=batch["globals"],
            )
        assert torch.allclose(lp, batch["logp_old"], atol=1e-4)
        assert torch.allclose(val, batch["value_old"], atol=1e-4)


def test_v001_checkpoint_loads_with_dropout_zero() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "runs" / "checkpoints" / "v0.0.1_f4f7bef.pt"
    if not path.exists():
        pytest.skip("v0.0.1 checkpoint not present")
    net, cfg = load_checkpoint(path, torch.device("cpu"))
    before = {k: v.clone() for k, v in net.state_dict().items()}
    set_dropout(net, 0.0)
    assert model_config_of(net)["dropout"] == 0.0
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k])
    x = torch.randn(2, 6, cfg.get("token_dim", obs.TOKEN_DIM))
    pad = torch.ones(2, 6, dtype=torch.bool)
    g = torch.randn(2, obs.GLOBALS_DIM)
    net.train()
    with torch.no_grad():
        a = net.encode(x, pad, g)[1]
        b = net.encode(x, pad, g)[1]
    net.eval()
    with torch.no_grad():
        c = net.encode(x, pad, g)[1]
    assert torch.equal(a, b)
    assert torch.allclose(a, c, atol=1e-5)
