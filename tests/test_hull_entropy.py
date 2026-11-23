"""Per-head entropy bonus on the build_ship (hull) head (ppo.hull_ent_coef)."""

import copy
import json
from pathlib import Path

import pytest
import torch

from battleboats.envs import observation as obs
from battleboats.envs.map_pool import PoolSize, build_map_pool, episode_limits
from battleboats.training.checkpoints import model_config_of
from battleboats.training.loop import PPOHyper, apply_ppo
from battleboats.training.policy import NUM_SHIP_TYPES, VERB_TO_IDX, PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.vec_collect import GameCollector


def _policy(seed=0):
    torch.manual_seed(seed)
    return PolicyNetwork(token_dim=obs.TOKEN_DIM, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)


def _segments(tmp_path, net, decisions=400, seed=0):
    maps = build_map_pool(tmp_path / "maps", [PoolSize(12, 6, 2, ports=2)], seed=3)
    info = []
    for path in maps:
        data = json.loads(Path(path).read_text())
        t, s = episode_limits(data, max_turns=20, max_steps=400, ref_home_distance=14)
        info.append({"size": f"{data['width']}x{data['height']}", "max_turns": t, "max_steps": s})
    col = GameCollector(
        maps,
        info,
        model_config=model_config_of(net),
        seed=seed,
        games=4,
        worker_id=0,
        num_workers=1,
        random_opponent_frac=1.0,
    )
    col.set_weights(net.state_dict(), {}, [], explore=(0.0, "policy"))
    torch.manual_seed(seed)
    res = col.collect(decisions)
    return res.segments


def _buffer(segments):
    buf = RolloutBuffer()
    for s in segments:
        buf.add(s)
    return buf


def _peak_hull_head(net, hull_idx: int = 4, bias: float = 8.0):
    """Make build_ship nearly deterministic on one hull (default: submarine)."""
    last = net.head_target.build_ship.mlp[-1]
    with torch.no_grad():
        last.bias.zero_()
        last.bias[hull_idx] = bias


def test_hull_entropy_is_build_ship_target_entropy(tmp_path):
    net = _policy(1)
    segs = _segments(tmp_path, net)
    buf = _buffer(segs)
    n_build = sum(1 for s in segs for tr in s if tr.verb_idx == VERB_TO_IDX["build_ship"])
    assert n_build > 0
    opt = torch.optim.Adam(net.parameters(), lr=0.0)  # no step effect needed; we still step
    # Peek at factor entropies on the whole buffer via one evaluate pass after ppo_update metric.
    metrics = ppo_update(net, opt, buf, epochs=1, minibatch_size=64, hull_ent_coef=0.0)
    assert "hull_entropy" in metrics
    assert 0.0 <= metrics["hull_entropy"] <= torch.log(torch.tensor(NUM_SHIP_TYPES)).item() + 1e-5


def test_hull_ent_coef_zero_matches_plain_update(tmp_path):
    net = _policy(2)
    segs = _segments(tmp_path, net)
    a, b = copy.deepcopy(net), copy.deepcopy(net)
    buf_a, buf_b = _buffer(copy.deepcopy(segs)), _buffer(copy.deepcopy(segs))
    opt_a = torch.optim.Adam(a.parameters(), lr=3e-4)
    opt_b = torch.optim.Adam(b.parameters(), lr=3e-4)
    torch.manual_seed(7)
    ma = ppo_update(a, opt_a, buf_a, epochs=2, minibatch_size=64, hull_ent_coef=0.0)
    torch.manual_seed(7)
    mb = ppo_update(b, opt_b, buf_b, epochs=2, minibatch_size=64)
    assert ma["policy_loss"] == pytest.approx(mb["policy_loss"], abs=1e-6)
    assert ma["entropy"] == pytest.approx(mb["entropy"], abs=1e-6)
    assert ma["hull_entropy"] == pytest.approx(mb["hull_entropy"], abs=1e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_hull_ent_coef_raises_peaked_hull_entropy(tmp_path):
    """A peaked hull head plus a hull-only bonus should increase hull entropy."""
    net = _policy(3)
    _peak_hull_head(net, hull_idx=4, bias=8.0)
    segs = _segments(tmp_path, net, decisions=600, seed=3)
    n_build = sum(1 for s in segs for tr in s if tr.verb_idx == VERB_TO_IDX["build_ship"])
    assert n_build >= 8

    def _run(coef):
        policy = copy.deepcopy(net)
        buf = _buffer(copy.deepcopy(segs))
        opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # Several updates on the same frozen batch (advantages already stored).
        last = None
        for _ in range(6):
            last = ppo_update(
                policy,
                opt,
                buf,
                epochs=2,
                minibatch_size=64,
                hull_ent_coef=coef,
                ent_coef=0.0,  # isolate the hull-head term
                vf_coef=0.0,
            )
        return last["hull_entropy"]

    peaked = _run(0.0)
    boosted = _run(0.2)
    # Random-init peaked head should start well below uniform (log 8 ~ 2.08).
    assert peaked < 1.5
    assert boosted > peaked + 0.05


def test_ppohyper_accepts_hull_ent_coef():
    h = PPOHyper(hull_ent_coef=0.05)
    assert h.hull_ent_coef == 0.05
    # apply_ppo on an empty buffer still returns the metric.
    net = _policy(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    m = apply_ppo(net, opt, RolloutBuffer(), h, torch.device("cpu"))
    assert m["hull_entropy"] == 0.0
