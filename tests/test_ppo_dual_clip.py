"""Dual-clip PPO (ppo.dual_clip) guards policy-mode hull exploration against ratio blow-ups."""

import copy
import json
from pathlib import Path

import pytest
import torch

from battleboats.envs import observation as obs
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.hull_explore import HullExplore, ensure_dual_clip
from battleboats.training.loop import PPOHyper
from battleboats.training.policy import VERB_TO_IDX, PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.vec_collect import GameCollector


def _policy(seed=0):
    torch.manual_seed(seed)
    return PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)


def _segments(tmp_path, net):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(12, 6, 2, ports=2)], seed=3)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        t, s = episode_limits(data, max_turns=20, max_steps=400, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    col = GameCollector(maps, info, model_config=model_config_of(net), seed=0, games=4, worker_id=0,
                        num_workers=1, random_opponent_frac=1.0)
    col.set_weights(net.state_dict(), {}, [], explore=(1.0, "policy"))
    torch.manual_seed(0)
    res = col.collect(400)
    assert res.hull_explore["explored"] > 0
    return res.segments


def _tiny_old_logp(segments):
    """Mimic an explored hull the policy gives ~e^-150: stored log pi_old is tiny."""
    segs = copy.deepcopy(segments)
    n = 0
    for seg in segs:
        for tr in seg:
            if tr.verb_idx == VERB_TO_IDX["build_ship"]:
                tr.logp -= 150.0
                tr.advantage = -1.0 if n % 2 else 1.0
                n += 1
    assert n > 0
    return segs


def _run(net, segs, dual_clip):
    buf = RolloutBuffer()
    for s in segs:
        buf.add(s)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    return ppo_update(net, opt, buf, epochs=2, minibatch_size=64, dual_clip=dual_clip)


def test_dual_clip_bounds_exploded_ratio(tmp_path):
    net = _policy(1)
    segs = _tiny_old_logp(_segments(tmp_path, net))
    # Plain clip: the smoke_v5 failure (NaN weights -> "row with no legal option").
    plain_net = copy.deepcopy(net)
    try:
        plain = _run(plain_net, segs, None)
        broke = plain["approx_kl"] > 1e10 or not all(torch.isfinite(p).all() for p in plain_net.parameters())
    except AssertionError:
        broke = True
    assert broke
    dual_net = copy.deepcopy(net)
    m = _run(dual_net, segs, 3.0)
    assert m["approx_kl"] < 1e10 and all(torch.isfinite(p).all() for p in dual_net.parameters())
    assert abs(m["policy_loss"]) < 10


def test_dual_clip_matches_plain_on_on_policy_data(tmp_path):
    net = _policy(2)
    segs = _segments(tmp_path, net)
    a, b = copy.deepcopy(net), copy.deepcopy(net)
    torch.manual_seed(7)
    ma = _run(a, segs, None)
    torch.manual_seed(7)
    mb = _run(b, segs, 3.0)
    # ratios stay near 1 on fresh data, so neither cap binds
    assert ma["policy_loss"] == pytest.approx(mb["policy_loss"], abs=1e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_ensure_dual_clip_and_validation():
    ppo = PPOHyper()
    assert ensure_dual_clip(HullExplore.from_dict({"eps": 0.15}), ppo) is not None and ppo.dual_clip == 3.0
    ppo = PPOHyper(dual_clip=5.0)
    assert ensure_dual_clip(HullExplore.from_dict({"eps": 0.15}), ppo) is None and ppo.dual_clip == 5.0
    ppo = PPOHyper()
    assert ensure_dual_clip(HullExplore.from_dict({"eps": 0.15, "logprob": "mixture"}), ppo) is None
    assert ensure_dual_clip(HullExplore.from_dict(None), ppo) is None and ppo.dual_clip is None
    with pytest.raises(ValueError):
        PPOHyper(dual_clip=1.0)
