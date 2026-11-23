"""Epsilon-uniform exploration on the build hull (battleboats/training/hull_explore.py)."""

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from battleboats.envs import observation as obs
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.hull_explore import (
    HullExplore,
    empty_explore_stats,
    format_explore,
    merge_explore_stats,
    record_build,
)
from battleboats.training.policy import PolicyNetwork, masked_categorical
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.vec_collect import GameCollector, ParallelCollector


def _policy(seed=0):
    torch.manual_seed(seed)
    net = PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    net.eval()
    return net


def _pool(tmp_path):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(12, 6, 2, ports=2)], seed=3)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        t, s = episode_limits(data, max_turns=20, max_steps=400, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    return maps, info


# ---------------------------------------------------------------- config
def test_config_defaults_off_and_validation():
    off = HullExplore.from_dict(None)
    assert off.eps == 0.0 and not off.enabled and off.logprob == "policy"
    assert off.eps_at(100) == 0.0
    on = HullExplore.from_dict({"eps": 0.15})
    assert on.enabled and on.eps_at(0) == on.eps_at(999) == 0.15
    for bad in ({"eps": 1.5}, {"eps": 0.1, "eps_end": -0.1}, {"eps": 0.1, "logprob": "x"},
                {"eps": 0.1, "anneal_updates": -1}, {"eps": 0.1, "typo": 1}):
        with pytest.raises(ValueError):
            HullExplore.from_dict(bad)


def test_linear_anneal():
    h = HullExplore.from_dict({"eps": 0.2, "eps_end": 0.0, "anneal_updates": 100})
    assert h.eps_at(0) == pytest.approx(0.2)
    assert h.eps_at(50) == pytest.approx(0.1)
    assert h.eps_at(100) == pytest.approx(0.0)
    assert h.eps_at(500) == pytest.approx(0.0)


def test_stats_merge_and_format():
    a, b = empty_explore_stats(), empty_explore_stats()
    record_build(a, "landing", True)
    record_build(a, "battleship", False)
    record_build(b, "landing", False)
    merge_explore_stats(a, b)
    assert a["build"] == 3 and a["explored"] == 1
    assert a["explored_hulls"] == {"landing": 1}
    assert a["policy_hulls"] == {"battleship": 1, "landing": 1}
    assert format_explore(a, 0.15) == "explore=eps:0.150,1/3,landing:e1/p1"


# ---------------------------------------------------------------- policy
def _dist(collapsed=True):
    logits = torch.tensor([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0]] * 400)
    legal = torch.ones(400, 8, dtype=torch.bool)
    legal[:, 6] = False  # one illegal hull
    return masked_categorical(logits.masked_fill(~legal, float("-inf"))), legal


def test_eps_zero_is_identity_and_draws_no_rng():
    net = _policy()
    dist, legal = _dist()
    t = dist.sample()
    torch.manual_seed(5)
    before = torch.rand(1)
    torch.manual_seed(5)
    t2, lp, explored = net._explore_hull(dist, legal, t)
    after = torch.rand(1)
    assert torch.equal(t2, t) and not explored.any()
    assert torch.allclose(lp, dist.log_prob(t))
    assert torch.equal(before, after)  # no random numbers consumed


def test_eps_one_uniform_over_legal_and_policy_logprob():
    net = _policy()
    net.hull_explore_eps = 1.0
    dist, legal = _dist()
    torch.manual_seed(0)
    t, lp, explored = net._explore_hull(dist, legal, dist.sample())
    assert explored.all()
    assert not (t == 6).any()  # never illegal
    counts = torch.bincount(t, minlength=8)
    assert (counts[legal[0]] > 20).all()  # roughly uniform over the 7 legal hulls
    # policy mode: the stored log-prob is pi's own log-prob of the chosen hull
    assert torch.allclose(lp, dist.log_prob(t))
    assert lp.min() < -25  # collapsed head: explored hulls have tiny pi


def test_mixture_logprob():
    net = _policy()
    net.hull_explore_eps, net.hull_explore_mode = 0.15, "mixture"
    dist, legal = _dist()
    torch.manual_seed(1)
    t, lp, explored = net._explore_hull(dist, legal, dist.sample())
    assert 0 < explored.float().mean() < 0.35
    want = torch.log(0.85 * dist.log_prob(t).exp() + 0.15 / 7)
    assert torch.allclose(lp, want)
    # explored hull 1: behaviour prob ~ eps/n, pi ~ e^-30 -> ratio ~ 1e-12
    row = int(((t != 0)).nonzero()[0])
    assert lp[row].item() == pytest.approx(math.log(0.15 / 7), rel=1e-4)


# ---------------------------------------------------------------- collectors
def _assert_logp_matches(net, segments):
    """policy mode: stored logp == evaluate_actions logp under the same weights (ratio 1)."""
    buf = RolloutBuffer()
    for seg in segments:
        buf.add(seg)
    for batch, masks in buf.iter_minibatches(len(buf), device=torch.device("cpu")):
        with torch.no_grad():
            lp, _ent, _v = net.evaluate_actions(
                batch["tokens"], batch["pad_mask"], batch["asset_idx"], batch["verb_idx"],
                batch["target_idx"], masks, skill_idx=batch.get("skill_idx"), globals=batch.get("globals"),
            )
        assert torch.allclose(lp, batch["logp_old"], atol=1e-4)


def test_game_collector_learner_only_and_counts(tmp_path):
    maps, info = _pool(tmp_path)
    net = _policy(1)
    opp = _policy(2)
    col = GameCollector(maps, info, model_config=model_config_of(net), seed=0, games=4, worker_id=0,
                        num_workers=1, random_opponent_frac=0.0)
    col.set_weights(net.state_dict(), {"update_00000": opp.state_dict()}, ["update_00000"], explore=(1.0, "policy"))
    assert col.policy.hull_explore_eps == 1.0
    assert all(o.hull_explore_eps == 0.0 for o in col.opponents.values())
    torch.manual_seed(0)
    res = col.collect(600)
    he = res.hull_explore
    assert he["build"] > 0
    assert he["build"] == sum(res.purchases.values())  # one record per learner build
    assert he["explored"] == he["build"]  # eps 1: every learner build explored
    assert he["policy_hulls"] == {}
    assert len(he["explored_hulls"]) >= 3  # several hulls tried
    _assert_logp_matches(col.policy, res.segments)
    # default set_weights turns exploration off again
    col.set_weights(net.state_dict(), {"update_00000": opp.state_dict()}, ["update_00000"])
    assert col.policy.hull_explore_eps == 0.0
    res2 = col.collect(300)
    assert res2.hull_explore["explored"] == 0


def test_parallel_collector_forwards_eps(tmp_path):
    maps, info = _pool(tmp_path)
    net = _policy(3)
    net.hull_explore_eps, net.hull_explore_mode = 1.0, "policy"
    col = ParallelCollector(maps, info, num_workers=2, games_per_worker=2, collect_threads=1, episode_offset=0,
                            model_config=model_config_of(net), seed=0, random_opponent_frac=1.0)
    try:
        res = col.collect(net, {}, 600)
    finally:
        col.close()
    assert res.hull_explore["build"] > 0
    assert res.hull_explore["explored"] == res.hull_explore["build"]


def test_act_counts_stats():
    from battleboats.envs.action_masks import ActionMasks
    from battleboats.envs.sandboxes import horizon, make_sandbox
    from battleboats.envs.skill_maps import write_two_home_map
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = write_two_home_map(Path(d) / "lb.json", 5, 5)
        turns, _ = horizon("landing_build", 5)
        env = make_sandbox("landing_build", str(path), width=5, height=5, max_turns=turns, landing_build_hulls="all")
        env.reset(seed=0)
        net = _policy(4)
        net.hull_explore_eps = 1.0
        net.hull_explore_stats = empty_explore_stats()
        from battleboats.training.rollout import collect_trajectory
        import random

        _horizon_turns, steps = horizon("landing_build", 5)
        torch.manual_seed(0)
        for seed in range(6):
            env.reset(seed=seed)
            collect_trajectory(net, None, env, 0, random.Random(seed), max_steps=steps, use_env_reward=True)
        st = net.hull_explore_stats
        assert st["build"] > 0 and st["explored"] == st["build"]


# ---------------------------------------------------------------- configs
def test_selfplay_and_sandbox_configs_load(tmp_path):
    from battleboats.training.sandbox_curriculum import load_sandbox_config
    from battleboats.training.selfplay import load_selfplay_config

    sp = tmp_path / "sp.yaml"
    sp.write_text(yaml.safe_dump({"run_dir": str(tmp_path / "r"), "hull_explore": {"eps": 0.15, "logprob": "mixture"}}))
    cfg = load_selfplay_config(sp)
    assert cfg.hull_explore.eps == 0.15 and cfg.hull_explore.logprob == "mixture"
    sp.write_text(yaml.safe_dump({"run_dir": str(tmp_path / "r")}))
    assert not load_selfplay_config(sp).hull_explore.enabled

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "sandboxes.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "sb")
    raw["hull_explore"] = {"eps": 0.3, "eps_end": 0.05, "anneal_updates": 200}
    sb = tmp_path / "sb.yaml"
    sb.write_text(yaml.safe_dump(raw))
    scfg = load_sandbox_config(sb)
    assert scfg.hull_explore.eps_at(100) == pytest.approx(0.175)


def test_sandbox_update_prints_explore_line(tmp_path, capsys):
    from battleboats.training.sandbox_curriculum import load_sandbox_config, run_sandboxes

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "sandboxes_smoke.yaml").read_text())
    raw.update(run_dir=str(tmp_path / "sb"), init_checkpoint=None, skills=["landing_build"], sizes=[3],
               max_updates_per_size=1, min_updates=1, allow_unmet=True, decisions_per_update=64,
               eval_episodes=1, wandb=False, hull_explore={"eps": 1.0})
    (tmp_path / "sb.yaml").write_text(yaml.safe_dump(raw))
    run_sandboxes(load_sandbox_config(tmp_path / "sb.yaml"))
    out = capsys.readouterr().out
    assert "hull exploration ON eps=1.0" in out
    assert "explore=eps:1.000," in out
    rows = [json.loads(l) for l in (tmp_path / "sb" / "history.jsonl").read_text().splitlines()]
    assert rows and rows[0]["hull_explore/eps"] == 1.0
    assert rows[0]["hull_explore/explored"] == rows[0]["hull_explore/builds"]
    assert "_explore_line" not in rows[0]
