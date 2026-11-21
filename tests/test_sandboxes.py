"""Sparse sandboxes, the mixed warm-up, and the skill-free pointer net."""

import json
import random
from pathlib import Path

import torch

from battleboats.core.actions import (
    AttackAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_stats import ShipStats
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.base_aec import BattleboatsBaseAEC
from battleboats.envs.observation import (
    OWNER_OFFSET,
    SHIP_TYPE_INDEX,
    STATS_OFFSET,
    COST_SCALE,
)
from battleboats.envs.sandboxes import (
    CombatSandbox,
    HOME_TURN_INCOME,
    LandingSandbox,
    MerchantSandbox,
    ScoutingSandbox,
    ShipBuildSandbox,
    TypedBuildSandbox,
    combat_outcome,
    discounted_combat_reward,
    enemy_strength_scale,
    make_sandbox,
    opening_cash,
    scouting_find_reward,
    ship_detection_range,
    side_power,
    turn_decay,
    typed_build_reward,
)
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.policy import PolicyNetwork
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.curriculum import _new_policy
from battleboats.training.sandbox_curriculum import (
    DecaySpec,
    SandboxRunConfig,
    _drill_done,
    _size_cleared,
    choose_episode,
    choose_mixed_episode,
    drill_done,
    format_stop_reason,
    full_smooth,
    load_sandbox_config,
    plateaued,
    return_scale,
    run_sandboxes,
    scale_trajectory,
    smooth_series,
    window_cleared,
)
from battleboats.training.wandb_log import init_wandb
from battleboats.training.sandbox_skills import SKILL_NAMES, skill_index


def _step_toward(env, goal):
    legal = env.infos["player_0"]["legal_actions"]
    caps = [a for a in legal if isinstance(a, (CapturePortAction, AttackAction))]
    if caps and not isinstance(caps[0], AttackAction):
        env.step(caps[0])
        return
    attacks = [a for a in legal if isinstance(a, AttackAction)]
    if attacks and getattr(env, "killed", False) is False and goal is None:
        env.step(attacks[0])
        return
    moves = [a for a in legal if isinstance(a, MoveAction)]
    if goal is not None and moves:
        def dist(action: MoveAction) -> int:
            return abs(action.destination[0] - goal[0]) + abs(action.destination[1] - goal[1])
        env.step(min(moves, key=dist))
        return
    if attacks:
        env.step(attacks[0])
        return
    env.step(next(a for a in legal if isinstance(a, EndTurnAction)))


def test_merchant_rewards_only_a_completed_haul(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "m.json", 3, 3)
    env = MerchantSandbox(str(path), width=3, height=3, max_turns=20)
    env.reset(seed=1)
    assert env.hauled is False
    home = env.engine.players[0].home_port
    satellite = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    loaded = False
    for _ in range(40):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        loads = [a for a in legal if isinstance(a, MerchantLoadAction)]
        unloads = [a for a in legal if isinstance(a, MerchantUnloadAction)]
        if loads and not loaded:
            env.step(loads[0])
            loaded = True
            assert env.hauled is False
            assert env.terminations["player_0"] is False
            continue
        if unloads and loaded:
            env.step(unloads[0])
            break
        _step_toward(env, satellite if not loaded else home)
    assert loaded
    assert env.hauled
    paid = env.rewards["player_0"]
    assert paid == env._turn_scale(env.engine.turn)
    assert 0.0 < paid <= 1.0
    assert env.terminations["player_0"] is True
    env.close()


def test_builder_needs_a_haul_before_the_buy(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "b.json", 3, 3)
    env = ShipBuildSandbox(str(path), width=3, height=3, max_turns=12)
    env.reset(seed=0)
    assert env.engine.players[0].cash == 0
    assert not any(isinstance(a, BuildShipAction) for a in env.infos["player_0"]["legal_actions"])
    env.step(EndTurnAction())
    assert env.engine.players[0].cash == 0
    assert env.built is False
    home = env.engine.players[0].home_port
    satellite = next(p for p in env.engine.players[0].owned_port_positions if p != home)
    loaded = False
    for _ in range(40):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        builds = [a for a in legal if isinstance(a, BuildShipAction)]
        loads = [a for a in legal if isinstance(a, MerchantLoadAction)]
        unloads = [a for a in legal if isinstance(a, MerchantUnloadAction)]
        if builds:
            env.step(builds[0])
            break
        if loads and not loaded:
            env.step(loads[0])
            loaded = True
            assert env.built is False
            continue
        if unloads and loaded:
            env.step(unloads[0])
            assert env.engine.players[0].cash > 0
            assert env.built is False
            continue
        _step_toward(env, satellite if not loaded else home)
    assert env.built
    paid = env.rewards["player_0"]
    assert paid == env._turn_scale(env.engine.turn)
    assert 0.0 < paid <= 1.0
    assert env.terminations["player_0"] is True
    env.close()


def test_landing_capture_is_the_only_reward(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "l.json", 3, 3)
    env = LandingSandbox(str(path), width=3, height=3, max_turns=12)
    env.reset(seed=2)
    goal = env.engine.players[1].home_port
    for _ in range(30):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        caps = [a for a in legal if isinstance(a, CapturePortAction)]
        if caps:
            env.step(caps[0])
            break
        _step_toward(env, goal)
    assert env.captured
    paid = env.rewards["player_0"]
    assert paid == env._turn_scale(env.engine.turn)
    assert 0.0 < paid <= 1.0
    env.close()
    from battleboats.envs.base_aec import BattleboatsBaseAEC

    far_path = write_two_home_map(tmp_path / "far.json", 10, 10)
    far_env = LandingSandbox(str(far_path), width=10, height=10, max_turns=8)
    far_env._write_map([(0, 0), (9, 9)], [(0, 0)], [(9, 9)])
    BattleboatsBaseAEC.reset(far_env, seed=0)
    assert (9, 9) not in far_env.engine.players[0].port_sightings
    far_env.close()


def test_combat_reward_is_the_power_change_at_the_horizon(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "c.json", 3, 3)
    env = CombatSandbox(str(path), ship="destroyer", width=3, height=3, max_turns=3)
    env.reset(seed=0)
    target = env.engine.ships[env._target_id]
    assert target.type is ShipType.DESTROYER
    assert target.stats.strength < BASE_STATS[ShipType.DESTROYER].strength
    assert env.engine.players[1].cash == 0
    assert abs(enemy_strength_scale(3) - 0.15) < 1e-9
    assert enemy_strength_scale(10) == 1.0
    env.step(EndTurnAction())
    assert env.engine.current_player == 0
    assert env.engine.players[1].cash == 0
    assert env.terminations["player_0"] is False
    assert env.rewards["player_0"] == 0.0
    for _ in range(30):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        env.step(EndTurnAction())
    assert env.terminations["player_0"] is False
    assert env.truncations["player_0"] is True
    assert env.engine.turn >= 3
    expected = discounted_combat_reward(env._power_samples, env.decay_horizon, env.decay_floor)
    assert abs(env.episode_reward - expected) < 1e-9
    assert abs(env.episode_reward) <= 1.0
    assert env.rewards["player_0"] == env.episode_reward
    # The raw swing is still what the weights scale. A win is not added.
    raw = combat_outcome(env._my0, env._en0, side_power(env.engine, 0), side_power(env.engine, 1))
    assert abs(raw) <= 1.0
    env.close()


def test_make_sandbox_covers_every_skill(tmp_path: Path) -> None:
    for name in SKILL_NAMES:
        path = write_two_home_map(tmp_path / f"{name}.json", 3, 3)
        env = make_sandbox(name, str(path), width=3, height=3, max_turns=6)
        env.reset(seed=0)
        assert env.skill_index == skill_index(name)
        assert env.infos["player_0"]["legal_actions"]
        env.close()


def test_skill_one_hot_changes_the_context() -> None:
    net = PolicyNetwork(
        token_dim=8, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0, skill_count=8
    )
    ctx = torch.zeros(2, 32)
    merchant = net._with_skill(ctx, skill_index("merchant"), 2)
    combat = net._with_skill(ctx, skill_index("combat_carrier"), 2)
    assert not torch.allclose(merchant, combat)
    plain = PolicyNetwork(token_dim=8, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)
    assert torch.equal(plain._with_skill(ctx, 3, 2), ctx)


def test_window_and_episode_mix() -> None:
    assert window_cleared([1, 1, 0], need=4, threshold=0.5) is False
    assert window_cleared([1, 0, 1, 1], need=4, threshold=0.7) is True
    from battleboats.training.sandbox_curriculum import size_advances

    full = [1.0] * 64
    assert size_advances(2, full, 64, 0.8, 3) is False
    assert size_advances(3, full, 64, 0.8, 3) is True
    rng = random.Random(0)
    roles = [
        choose_episode(
            rng,
            focus_skill="landing",
            focus_size=6,
            smaller_sizes=[3, 4],
            earlier=[("merchant", 3), ("builder", 3)],
            rehearsal_frac=0.15,
            size_mix_frac=0.2,
        )[2]
        for _ in range(400)
    ]
    assert roles.count("rehearsal") > 30
    assert roles.count("mix") > 30
    assert roles.count("focus") > 150


def test_mixed_sampling_is_even_across_sandboxes() -> None:
    rng = random.Random(0)
    skills = ["merchant", "builder", "landing"]
    counts = {name: 0 for name in skills}
    roles = []
    for i in range(300):
        skill = skills[i % len(skills)]
        picked, size, role = choose_mixed_episode(
            rng,
            skills=skills,
            focus_size=6,
            smaller_sizes=[3, 4],
            rehearsal_frac=0.15,
            size_mix_frac=0.2,
            skill=skill,
        )
        assert picked == skill
        assert size in (6, 3, 4)
        counts[picked] += 1
        roles.append(role)
    assert counts == {"merchant": 100, "builder": 100, "landing": 100}
    assert roles.count("focus") > roles.count("rehearsal")
    assert roles.count("rehearsal") > 10
    focus_only, _, role = choose_mixed_episode(
        rng,
        skills=skills,
        focus_size=10,
        smaller_sizes=[],
        rehearsal_frac=1.0,
        size_mix_frac=1.0,
        skill="merchant",
    )
    assert (focus_only, role) == ("merchant", "focus")


def test_return_scale_shrinks_only_the_larger_sandbox() -> None:
    assert return_scale(0.0, 0.0, 1) == 1.0

    class _Step:
        def __init__(self, reward: float) -> None:
            self.reward = reward

    small = {"mean": 0.0, "m2": 0.0, "count": 0}
    large = {"mean": 0.0, "m2": 0.0, "count": 0}
    last_small = _Step(1.0)
    last_large = _Step(50.0)
    for _ in range(6):
        last_small = _Step(1.0)
        last_large = _Step(50.0)
        assert scale_trajectory([last_small], small, gamma=1.0) == 1.0
        assert scale_trajectory([last_large], large, gamma=1.0) == 50.0
    assert last_small.reward == 1.0
    assert abs(last_large.reward - 1.0) < 1e-6
    assert small["mean"] == 1.0
    assert large["mean"] == 50.0


def test_combat_plateau_uses_a_moving_average_and_stays_passed(tmp_path: Path) -> None:
    flat = [0.2] * 20
    assert plateaued(flat, 10, 0.02, higher_is_better=True)
    assert plateaued([0.4] * 20, 10, 0.02, higher_is_better=False)
    rising = [0.01 * (i + 1) for i in range(30)]
    assert not plateaued(full_smooth(rising, method="mean", span=8), 10, 0.02, higher_is_better=True)
    assert not plateaued([1.0, 0.5], 2, 0.02, higher_is_better=False)
    # The per-update swing that kept a raw 2% test from firing.
    swing = [0.08, 0.30] * 20
    assert not plateaued(swing, 10, 0.02, higher_is_better=True)
    smoothed = full_smooth(swing, method="mean", span=8)
    assert smoothed
    assert max(smoothed) - min(smoothed) < 1e-9
    assert abs(smoothed[-1] - 0.19) < 1e-6
    assert plateaued(smoothed, 10, 0.02, higher_is_better=True)
    assert abs(smooth_series([1.0, 3.0], method="ema", span=1)[-1] - 3.0) < 1e-9
    common = dict(
        updates=10,
        window=[0.0] * 64,
        episodes=40,
        success_threshold=0.8,
        need=64,
        min_updates=3,
        plateau_updates=10,
        plateau_threshold=0.02,
        plateau_smooth="mean",
        plateau_smooth_span=8,
        combat_min_episodes=32,
    )
    assert drill_done(
        "combat_destroyer",
        reward_trace=swing,
        value_trace=[0.5, 0.2] * 20,
        **common,
    )
    assert not drill_done(
        "combat_destroyer",
        reward_trace=rising,
        value_trace=[0.4] * len(rising),
        **common,
    )
    assert not drill_done(
        "combat_destroyer",
        reward_trace=flat,
        value_trace=[0.4] * len(flat),
        **{**common, "episodes": 10},
    )
    assert not drill_done("merchant", reward_trace=flat, value_trace=[0.4] * len(flat), **common)
    # A later climb would reopen the gate. already_passed keeps this size shut.
    climbed = flat + [0.2 + 0.05 * i for i in range(1, 15)]
    assert not drill_done(
        "combat_destroyer",
        reward_trace=climbed,
        value_trace=[0.4] * len(climbed),
        **common,
    )
    assert drill_done(
        "combat_destroyer",
        reward_trace=climbed,
        value_trace=[0.4] * len(climbed),
        already_passed=True,
        **common,
    )
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "latch",
        skills=["combat_destroyer"],
        sizes=[3],
        window=64,
        min_updates=3,
    )
    state = {
        "updates_this_size": 10,
        "episodes": {"combat_destroyer": 40},
        "reward_trace": {"combat_destroyer": list(swing)},
        "value_trace": {"combat_destroyer": [0.5, 0.2] * 20},
        "combat_passed": {},
    }
    assert _drill_done(cfg, state, "combat_destroyer", [0.0])
    assert state["combat_passed"]["combat_destroyer"] is True
    state["reward_trace"]["combat_destroyer"] = list(climbed)
    assert _drill_done(cfg, state, "combat_destroyer", [0.0])

    cfg = SandboxRunConfig(
        run_dir=tmp_path / "cap",
        skills=["merchant"],
        sizes=[3],
        device="cpu",
        success_threshold=0.99,
        window=50,
        max_episodes_per_size=100,
        max_updates_per_size=1,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=False,
        min_updates=5,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"] is False
    assert "stopped" in result["reason"]
    summary = json.loads((tmp_path / "cap" / "summary.json").read_text())
    size = summary["skills"]["merchant"]["sizes"]["3"]
    assert size["gate_met"] is False
    assert size["capped"] is True


def test_episode_cap_stops_when_allow_unmet_is_false(tmp_path: Path) -> None:
    # Short episodes can burn max_episodes_per_size in a few updates.
    # With allow_unmet false, that cap must stop the run, not advance.
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "epcap",
        skills=["merchant"],
        sizes=[3],
        device="cpu",
        success_threshold=0.99,
        window=50,
        max_episodes_per_size=4,
        max_updates_per_size=50,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=False,
        min_updates=1,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"] is False
    assert "merchant 3x3 stopped" in result["reason"]
    assert "max_episodes_per_size=4" in result["reason"]
    summary = json.loads((tmp_path / "epcap" / "summary.json").read_text())
    size = summary["skills"]["merchant"]["sizes"]["3"]
    assert size["gate_met"] is False
    assert size["capped"] is True



def test_goal_gate_reopens_when_the_live_window_drops(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "live",
        skills=["build_landing", "merchant"],
        sizes=[3],
        success_threshold=0.80,
        window=4,
        min_updates=1,
    )
    state = {
        "updates_this_size": 5,
        "episodes": {"build_landing": 20, "merchant": 20},
        "reward_trace": {"build_landing": [], "merchant": []},
        "value_trace": {"build_landing": [], "merchant": []},
        "combat_passed": {},
    }
    windows = {
        "build_landing": [1.0, 1.0, 1.0, 1.0],
        "merchant": [1.0, 1.0, 1.0, 1.0],
    }
    assert _drill_done(cfg, state, "build_landing", windows["build_landing"])
    assert _size_cleared(cfg, state, windows)
    # New on-size failures must be able to reopen a drill that already passed,
    # and the size must not advance until every live window clears again.
    windows["build_landing"] = [1.0, 1.0, 0.0, 0.0]
    assert not _drill_done(cfg, state, "build_landing", windows["build_landing"])
    assert not _size_cleared(cfg, state, windows)
    windows["build_landing"] = [1.0, 1.0, 1.0, 1.0]
    assert _size_cleared(cfg, state, windows)


def test_stop_reason_names_the_skill_that_hit_the_episode_cap(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "stop",
        skills=["merchant", "build_landing", "build_cruiser"],
        sizes=[5],
        success_threshold=0.80,
        window=64,
        max_episodes_per_size=2000,
        max_updates_per_size=750,
    )
    # The 5x5 globals run stopped with merchant first in the unmet list, but
    # the cap was the expensive build_* drills at 2000 episodes.
    unmet = [
        ("merchant", 0.50, 137),
        ("build_landing", 0.00, 2007),
        ("build_cruiser", 0.00, 2012),
    ]
    capped = [("build_landing", 0.00, 2007), ("build_cruiser", 0.00, 2012)]
    reason = format_stop_reason(cfg, 5, unmet, cap_kind="episodes", capped_skills=capped)
    assert reason.startswith("build_landing 5x5 stopped at window 0.00")
    assert "after 2007 episodes" in reason
    assert "build_cruiser window 0.00 after 2012 episodes" in reason
    assert "max_episodes_per_size=2000" in reason
    assert not reason.startswith("merchant")
    update_reason = format_stop_reason(cfg, 5, unmet[:1], cap_kind="updates")
    assert "merchant 5x5 stopped at window 0.50" in update_reason
    assert "max_updates_per_size=750" in update_reason



def test_episode_cap_stop_names_the_capped_skill_not_the_first_unmet(tmp_path: Path) -> None:
    # build_* episodes are short, so they hit max_episodes_per_size while
    # merchant still has a short unmet window. The stop text must name the
    # capped build drill, not just merchant (first in the skills list).
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "capname",
        skills=["merchant", "build_landing"],
        sizes=[3],
        device="cpu",
        success_threshold=0.99,
        window=50,
        max_episodes_per_size=8,
        max_updates_per_size=50,
        decisions_per_update=8,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=False,
        min_updates=1,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"] is False
    reason = result["reason"]
    assert "build_landing 3x3 stopped" in reason
    assert "max_episodes_per_size=8" in reason
    # merchant may also be unmet, but it must not be reported as the cap.
    assert not reason.startswith("merchant")

def test_episode_cap_advances_when_allow_unmet_is_true(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "epcap_ok",
        skills=["merchant"],
        sizes=[3],
        device="cpu",
        success_threshold=0.99,
        window=50,
        max_episodes_per_size=4,
        max_updates_per_size=50,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=True,
        min_updates=1,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"]
    summary = json.loads((tmp_path / "epcap_ok" / "summary.json").read_text())
    size = summary["skills"]["merchant"]["sizes"]["3"]
    assert size["gate_met"] is False
    assert size["capped"] is True


def test_unknown_smooth_rejected(tmp_path: Path) -> None:
    try:
        SandboxRunConfig(
            run_dir=tmp_path / "bad",
            skills=["merchant"],
            sizes=[3],
            plateau_smooth="kalman",
        )
    except ValueError as exc:
        assert "plateau_smooth" in str(exc)
    else:
        raise AssertionError("unknown smooth method should be rejected")


def test_unknown_skill_rejected() -> None:
    try:
        skill_index("league")
    except ValueError as exc:
        assert "league" in str(exc)
    else:
        raise AssertionError("unknown skill should be rejected")


def test_load_real_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_sandbox_config(root / "configs" / "sandboxes.yaml")
    assert cfg.skills[0] == "merchant"
    assert cfg.skills[2] == "landing"
    assert cfg.sizes[-1] == 10
    assert cfg.rehearsal_frac == 0.15
    assert cfg.success_threshold == 0.8
    assert cfg.window == 64
    assert cfg.min_updates == 3
    assert cfg.skills[3] == "scouting"
    # The registry still names a typed-build drill per hull, but the
    # training curriculum omits the expensive ones that collapsed on 4x4.
    cheap_builds = ["build_merchant", "build_destroyer", "build_submarine"]
    expensive_builds = [
        "build_landing",
        "build_cruiser",
        "build_battleship",
        "build_carrier",
        "build_builder",
    ]
    assert "builder" in cfg.skills
    assert [name for name in cfg.skills if name.startswith("build_")] == cheap_builds
    for name in cheap_builds:
        assert name in SKILL_NAMES
    for name in expensive_builds:
        assert name in SKILL_NAMES
        assert name not in cfg.skills
    all_builds = [name for name in SKILL_NAMES if name.startswith("build_")]
    assert len(all_builds) == len(ShipType)
    smoke = load_sandbox_config(root / "configs" / "sandboxes_smoke.yaml")
    assert [name for name in smoke.skills if name.startswith("build_")] == all_builds
    assert cfg.plateau_threshold == 0.02
    assert cfg.plateau_updates == 10
    assert cfg.plateau_smooth == "mean"
    assert cfg.plateau_smooth_span == 8
    assert cfg.combat_min_episodes == 32
    assert cfg.max_updates_per_size == 750
    assert cfg.max_episodes_per_size == 2000
    assert cfg.allow_unmet is False
    assert cfg.ppo.ent_coef == 0.01
    assert cfg.device == "cpu"
    assert cfg.wandb is False
    assert cfg.wandb_project == "battleboats-skill-pipeline"
    assert cfg.reward_decay == DecaySpec(0, 0.0)
    assert cfg.reward_decay_drills == {}
    assert smoke.reward_decay == DecaySpec()
    assert smoke.decay_for("merchant") == DecaySpec()
    assert smoke.decay_for("combat_carrier") == DecaySpec()


def test_one_update_trains_every_sandbox(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "mix",
        skills=["merchant", "landing"],
        sizes=[3],
        device="cpu",
        success_threshold=0.0,
        window=1,
        max_episodes_per_size=4,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=True,
        min_updates=1,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"]
    row = json.loads((tmp_path / "mix" / "history.jsonl").read_text().splitlines()[0])
    assert "merchant/success_rate" in row and "landing/success_rate" in row
    assert "merchant/mean_reward" in row and "landing/mean_reward" in row
    loaded, model_config = load_checkpoint(tmp_path / "mix" / "latest.pt", torch.device("cpu"))
    assert model_config["skill_count"] == 0


def test_tiny_builder_curriculum_writes_summary(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "run",
        skills=["builder"],
        sizes=[3],
        device="cpu",
        success_threshold=0.0,
        window=2,
        max_episodes_per_size=8,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=True,
        min_updates=1,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=4, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"]
    assert (tmp_path / "run" / "summary.json").is_file()
    assert (tmp_path / "run" / "builder" / "size_3.pt").is_file()
    assert (tmp_path / "run" / "builder" / "final.pt").is_file()
    latest = tmp_path / "run" / "latest.pt"
    assert latest.is_file()
    loaded, model_config = load_checkpoint(latest, torch.device("cpu"))
    assert model_config["skill_count"] == 0
    assert loaded.skill_count == 0
    assert "skill_embed.weight" not in loaded.state_dict()
    fresh = PolicyNetwork(**model_config)
    game = _new_policy(cfg.arch, int(model_config["token_dim"]), torch.device("cpu"))
    assert game.skill_count == 0
    assert set(loaded.state_dict()) == set(fresh.state_dict()) == set(game.state_dict())
    for key, value in loaded.state_dict().items():
        assert tuple(value.shape) == tuple(game.state_dict()[key].shape)
    ctx = torch.zeros(1, loaded.d_model)
    assert torch.equal(loaded._with_skill(ctx, 0, 1), loaded._with_skill(ctx, 7, 1))


def test_scouting_starts_hidden_and_pays_sooner(tmp_path: Path) -> None:
    assert scouting_find_reward(0, 4) == 1.0
    assert abs(scouting_find_reward(1, 4) - 0.75) < 1e-9
    assert scouting_find_reward(4, 4) == 0.0
    assert ship_detection_range(ShipType.SUBMARINE, ShipType.CARRIER) == 4.0
    from battleboats.envs.sandboxes import horizon

    assert horizon("scouting", 3) == (12, 96)
    assert horizon("scouting", 10) == (40, 320)

    path = write_two_home_map(tmp_path / "s.json", 8, 8)
    layout = {
        "learner_ship": (0, 1),
        "enemy_ship": (0, 6),
        "home": (0, 0),
        "opp": (7, 7),
    }

    def open_fixed(turns: int) -> ScoutingSandbox:
        env = ScoutingSandbox(str(path), width=8, height=8, max_turns=turns)
        env._learner_type = ShipType.SUBMARINE
        env._enemy_type = ShipType.CARRIER
        env._layout = layout
        env._write_map([layout["home"], layout["opp"]], [layout["home"]], [layout["opp"]])
        BattleboatsBaseAEC.reset(env, seed=0)
        return env

    env = open_fixed(4)
    assert env.found is False
    assert env._sees_enemy() is False
    # The home port would see this carrier. Only the submarine's own range counts.
    assert not env.engine.players[0].owned_port_positions
    legal = env.infos["player_0"]["legal_actions"]
    assert legal
    assert all(isinstance(a, (MoveAction, EndTurnAction)) for a in legal)
    env.step(next(a for a in legal if isinstance(a, MoveAction) and a.destination == (0, 2)))
    assert env.found is True
    assert env.rewards["player_0"] == 1.0
    assert env.terminations["player_0"] is True
    env.close()

    late = open_fixed(4)
    late.step(EndTurnAction())
    assert late.found is False
    assert late.engine.turn == 1
    legal = late.infos["player_0"]["legal_actions"]
    late.step(next(a for a in legal if isinstance(a, MoveAction) and a.destination == (0, 2)))
    assert late.found is True
    assert abs(late.episode_reward - 0.75) < 1e-9
    late.close()

    expired = open_fixed(1)
    expired.step(EndTurnAction())
    assert expired.found is False
    assert expired.truncations["player_0"] is True
    assert expired.rewards["player_0"] == 0.0
    expired.close()

    seen = set()
    path3 = write_two_home_map(tmp_path / "s3.json", 3, 3)
    for seed in range(12):
        probe = ScoutingSandbox(str(path3), width=3, height=3, max_turns=6)
        probe.reset(seed=seed)
        own = probe.engine.ships[probe._hunter_id]
        tgt = probe.engine.ships[probe._target_id]
        dist = abs(own.position[0] - tgt.position[0]) + abs(own.position[1] - tgt.position[1])
        assert dist > ship_detection_range(own.type, tgt.type)
        assert probe.found is False
        assert probe._sees_enemy() is False
        seen.add((own.type, tgt.type))
        probe.close()
    assert len(seen) >= 2


def test_scouting_gate_is_detection_not_a_plateau() -> None:
    common = dict(
        updates=3,
        episodes=64,
        reward_trace=[0.2] * 20,
        value_trace=[0.4] * 20,
        success_threshold=0.8,
        need=64,
        min_updates=3,
        plateau_updates=10,
        plateau_threshold=0.02,
        combat_min_episodes=32,
    )
    assert drill_done("scouting", window=[1.0] * 64, **common)
    assert not drill_done("scouting", window=[0.0] * 64, **common)
    assert not drill_done("merchant", window=[1.0] * 10, **common)


def test_scouting_update_logs_the_drill_and_wandb_state(tmp_path: Path, capsys) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "scout",
        skills=["scouting"],
        sizes=[3],
        device="cpu",
        success_threshold=0.99,
        window=8,
        max_updates_per_size=1,
        max_episodes_per_size=8,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=True,
        min_updates=5,
        wandb=False,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"]
    logged = capsys.readouterr().out
    assert "[wandb] off (config wandb=false, BATTLEBOATS_WANDB unset)" in logged
    assert "wandb=off" in logged
    row = json.loads((tmp_path / "scout" / "history.jsonl").read_text().splitlines()[0])
    for key in ("success_rate", "mean_reward", "gate", "return_scale", "reward_smooth"):
        assert f"scouting/{key}" in row
    assert init_wandb(False, project="battleboats-skill-pipeline", config={}).enabled is False


def _builds(env, ship_type=None):
    legal = env.infos["player_0"]["legal_actions"]
    found = [a for a in legal if isinstance(a, BuildShipAction)]
    if ship_type is None:
        return found
    return [a for a in found if a.ship_type is ship_type]


def _cue_token(env, ship_type):
    tokens = env.observe("player_0")["entity_tokens"]
    hot = SHIP_TYPE_INDEX[ship_type]
    rows = [
        row
        for row in tokens
        if row[OWNER_OFFSET + 1] == 1.0 and row[hot] == 1.0
    ]
    assert len(rows) == 1
    return rows[0]


def test_typed_build_waits_then_pays_only_the_cued_hull(tmp_path: Path) -> None:
    assert typed_build_reward(0, 6) == 1.0
    assert abs(typed_build_reward(1, 6) - 5.0 / 6.0) < 1e-9
    assert typed_build_reward(6, 6) == 0.0
    income = HOME_TURN_INCOME
    assert income == 50
    for ship in ShipType:
        cash = opening_cash(BASE_STATS[ship].cost)
        assert 0 <= cash < BASE_STATS[ship].cost
        assert cash == max(0, BASE_STATS[ship].cost - income)

    path = write_two_home_map(tmp_path / "build.json", 3, 3)
    for ship in ShipType:
        env = TypedBuildSandbox(str(path), ship=ship.name.lower(), width=3, height=3, max_turns=6)
        env.reset(seed=0)
        cost = BASE_STATS[ship].cost
        assert env.engine.players[0].cash == opening_cash(cost)
        assert env.engine.players[0].cash < cost
        assert _builds(env, ship) == []
        cue = _cue_token(env, ship)
        assert abs(float(cue[STATS_OFFSET + 5]) - cost / COST_SCALE) < 1e-6
        assert env.skill_index == skill_index(f"build_{ship.name.lower()}")
        env.step(EndTurnAction())
        assert env.engine.turn == 1
        assert env.engine.players[0].cash == cost
        assert _builds(env, ship)
        # A second income tick must not unlock a dearer hull.
        env.step(EndTurnAction())
        assert env.engine.players[0].cash == cost
        assert all(BASE_STATS[a.ship_type].cost <= cost for a in _builds(env))
        env.close()

    # Same price as a destroyer. Buying the destroyer is legal and pays nothing.
    merchant = TypedBuildSandbox(str(path), ship="merchant", width=3, height=3, max_turns=6)
    merchant.reset(seed=1)
    merchant.step(EndTurnAction())
    wrong = _builds(merchant, ShipType.DESTROYER)
    assert wrong and not merchant.built_target
    merchant.step(wrong[0])
    assert merchant.built_target is False
    assert merchant.rewards["player_0"] == 0.0
    assert merchant.terminations["player_0"] is True
    merchant.close()

    # Immediate buy of the cued hull, on the turn it becomes affordable.
    prompt = TypedBuildSandbox(str(path), ship="merchant", width=3, height=3, max_turns=6)
    prompt.reset(seed=2)
    prompt.step(EndTurnAction())
    prompt.step(_builds(prompt, ShipType.MERCHANT)[0])
    assert prompt.built_target is True
    assert prompt.rewards["player_0"] == 1.0
    assert prompt.terminations["player_0"] is True
    prompt.close()

    # One extra EndTurn after it is affordable pays strictly less.
    late = TypedBuildSandbox(str(path), ship="merchant", width=3, height=3, max_turns=6)
    late.reset(seed=2)
    late.step(EndTurnAction())
    late.step(EndTurnAction())
    late.step(_builds(late, ShipType.MERCHANT)[0])
    assert late.built_target is True
    assert abs(late.episode_reward - 5.0 / 6.0) < 1e-9
    late.close()

    # Cheaper hulls are legal before a battleship is. Buying one does not pay,
    # and waiting never makes a carrier legal.
    early = TypedBuildSandbox(str(path), ship="battleship", width=3, height=3, max_turns=6)
    early.reset(seed=3)
    assert early.engine.players[0].cash < BASE_STATS[ShipType.BATTLESHIP].cost
    cheaper = _builds(early, ShipType.CRUISER)
    assert cheaper and _builds(early, ShipType.BATTLESHIP) == []
    early.step(cheaper[0])
    assert early.built_target is False
    assert early.rewards["player_0"] == 0.0
    assert early.terminations["player_0"] is True
    early.close()

    capped = TypedBuildSandbox(str(path), ship="battleship", width=3, height=3, max_turns=8)
    capped.reset(seed=4)
    for _ in range(4):
        if capped.terminations["player_0"] or capped.truncations["player_0"]:
            break
        capped.step(EndTurnAction())
    assert capped.engine.players[0].cash == BASE_STATS[ShipType.BATTLESHIP].cost
    assert _builds(capped, ShipType.BATTLESHIP)
    assert _builds(capped, ShipType.CARRIER) == []
    capped.close()

    # Home-port sight of a submarine is 4 tiles. This cue sits 17 away, so the
    # token is there only because the drill keeps the sighting. That token is
    # the same enemy-ship channel the full game already uses.
    wide = write_two_home_map(tmp_path / "wide.json", 10, 10)
    far = TypedBuildSandbox(str(wide), ship="submarine", width=10, height=10, max_turns=6)
    far._layout = {"home": (0, 0), "opp": (9, 9), "cue": (9, 8)}
    far._write_map([(0, 0), (9, 9)], [(0, 0)], [(9, 9)])
    BattleboatsBaseAEC.reset(far, seed=0)
    cue_ship = far.engine.ships[far._cue_id]
    dist = abs(cue_ship.position[0]) + abs(cue_ship.position[1])
    assert dist > 4
    _cue_token(far, ShipType.SUBMARINE)
    far.step(EndTurnAction())
    _cue_token(far, ShipType.SUBMARINE)
    far.close()

    common = dict(
        updates=3,
        episodes=64,
        reward_trace=[0.2] * 20,
        value_trace=[0.4] * 20,
        success_threshold=0.8,
        need=64,
        min_updates=3,
        plateau_updates=10,
        plateau_threshold=0.02,
        combat_min_episodes=32,
    )
    assert drill_done("build_landing", window=[1.0] * 64, **common)
    assert not drill_done("build_landing", window=[0.0] * 64, **common)
    assert not drill_done("build_merchant", window=[], **common)

    cfg = SandboxRunConfig(
        run_dir=tmp_path / "buy",
        skills=["build_landing"],
        sizes=[3],
        device="cpu",
        success_threshold=0.0,
        window=1,
        max_updates_per_size=1,
        max_episodes_per_size=4,
        decisions_per_update=4,
        rehearsal_frac=0.0,
        size_mix_frac=0.0,
        eval_episodes=0,
        allow_unmet=True,
        min_updates=1,
        wandb=False,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=2, ent_coef=0.01),
    )
    result = run_sandboxes(cfg, resume=False)
    assert result["ok"]
    row = json.loads((tmp_path / "buy" / "history.jsonl").read_text().splitlines()[0])
    for key in ("success_rate", "mean_reward", "gate", "return_scale"):
        assert f"build_landing/{key}" in row
    loaded, model_config = load_checkpoint(tmp_path / "buy" / "latest.pt", torch.device("cpu"))
    assert model_config["skill_count"] == 0
    assert loaded.skill_count == 0


def test_turn_decay_prefers_a_fast_finish(tmp_path: Path) -> None:
    assert turn_decay(0, 8) == 1.0
    assert turn_decay(0, 8) > turn_decay(1, 8) > turn_decay(7, 8) > 0.0
    assert turn_decay(8, 8) == 0.0
    assert turn_decay(3, 8, floor=0.2) > turn_decay(8, 8, floor=0.2) == 0.2
    assert scouting_find_reward(1, 4) == turn_decay(1, 4)
    assert typed_build_reward(1, 6) == turn_decay(1, 6)
    early = discounted_combat_reward([(0, 0.0), (0, 0.5)], 6)
    late_swing = discounted_combat_reward([(0, 0.0), (3, 0.5)], 6)
    stall = discounted_combat_reward([(0, 0.0), (5, 0.0)], 6)
    assert early == 0.5
    assert abs(late_swing - 0.5 * turn_decay(3, 6)) < 1e-9
    assert early > late_swing > stall == 0.0
    died_now = discounted_combat_reward([(0, 0.2), (0, -0.3)], 6)
    died_late = discounted_combat_reward([(0, 0.2), (4, -0.3)], 6)
    assert died_now < died_late < 0.0
    assert abs(died_now) <= 1.0 and abs(died_late) <= 1.0

    path = tmp_path / "decay.yaml"
    path.write_text(
        "run_dir: runs/x\nskills: [merchant, landing]\nsizes: [3]\n"
        "reward_decay:\n  horizon: 10\n  floor: 0.1\n  drills:\n"
        "    merchant: {horizon: 4, floor: 0.2}\n"
        "    landing: {floor: 0.05}\n"
    )
    cfg = load_sandbox_config(path)
    assert cfg.decay_for("merchant") == DecaySpec(4, 0.2)
    assert cfg.decay_for("landing") == DecaySpec(10, 0.05)
    assert cfg.decay_for("builder") == DecaySpec(10, 0.1)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "run_dir: runs/x\nskills: [merchant]\nsizes: [3]\nreward_decay: {horizon: 0, floor: 1}\n"
    )
    try:
        load_sandbox_config(bad)
    except ValueError as exc:
        assert "floor" in str(exc)
    else:
        raise AssertionError("a floor of 1 removes the reason to finish sooner")

    board = write_two_home_map(tmp_path / "m.json", 3, 3)

    def haul(extra: int) -> float:
        env = MerchantSandbox(str(board), width=3, height=3, max_turns=30)
        env.reset(seed=1)
        for _ in range(extra):
            env.step(EndTurnAction())
            assert env.rewards["player_0"] == 0.0
            assert env.hauled is False
        home = env.engine.players[0].home_port
        satellite = next(p for p in env.engine.players[0].owned_port_positions if p != home)
        loaded = False
        for _ in range(40):
            if env.terminations["player_0"] or env.truncations["player_0"]:
                break
            legal = env.infos["player_0"]["legal_actions"]
            loads = [a for a in legal if isinstance(a, MerchantLoadAction)]
            unloads = [a for a in legal if isinstance(a, MerchantUnloadAction)]
            if loads and not loaded:
                env.step(loads[0])
                loaded = True
                continue
            if unloads and loaded:
                env.step(unloads[0])
                break
            _step_toward(env, satellite if not loaded else home)
        paid = env.rewards["player_0"]
        assert env.hauled and paid == env._turn_scale(env.engine.turn)
        env.close()
        return paid

    assert 1.0 >= haul(0) > haul(1) > 0.0

    stalled = MerchantSandbox(str(board), width=3, height=3, max_turns=4)
    stalled.reset(seed=1)
    for _ in range(8):
        if stalled.truncations["player_0"]:
            break
        stalled.step(EndTurnAction())
    assert stalled.hauled is False
    assert stalled.rewards["player_0"] == 0.0
    stalled.close()

    def capture(extra: int) -> float:
        env = LandingSandbox(str(board), width=3, height=3, max_turns=20)
        env.reset(seed=2)
        for _ in range(extra):
            env.step(EndTurnAction())
            assert env.rewards["player_0"] == 0.0
        goal = env.engine.players[1].home_port
        for _ in range(30):
            if env.terminations["player_0"] or env.truncations["player_0"]:
                break
            legal = env.infos["player_0"]["legal_actions"]
            caps = [a for a in legal if isinstance(a, CapturePortAction)]
            if caps:
                env.step(caps[0])
                break
            _step_toward(env, goal)
        paid = env.rewards["player_0"]
        assert env.captured and paid == env._turn_scale(env.engine.turn)
        env.close()
        return paid

    assert 1.0 >= capture(0) > capture(1) > 0.0

    def buy(extra: int) -> float:
        env = ShipBuildSandbox(str(board), width=3, height=3, max_turns=30)
        env.reset(seed=0)
        for _ in range(extra):
            env.step(EndTurnAction())
            assert env.rewards["player_0"] == 0.0
            assert env.built is False
        home = env.engine.players[0].home_port
        satellite = next(p for p in env.engine.players[0].owned_port_positions if p != home)
        loaded = False
        for _ in range(40):
            if env.terminations["player_0"] or env.truncations["player_0"]:
                break
            legal = env.infos["player_0"]["legal_actions"]
            builds = [a for a in legal if isinstance(a, BuildShipAction)]
            loads = [a for a in legal if isinstance(a, MerchantLoadAction)]
            unloads = [a for a in legal if isinstance(a, MerchantUnloadAction)]
            if builds:
                env.step(builds[0])
                break
            if loads and not loaded:
                env.step(loads[0])
                loaded = True
                continue
            if unloads and loaded:
                env.step(unloads[0])
                continue
            _step_toward(env, satellite if not loaded else home)
        paid = env.rewards["player_0"]
        assert env.built and paid == env._turn_scale(env.engine.turn)
        env.close()
        return paid

    assert 1.0 >= buy(0) > buy(1) > 0.0

    def freeze(env) -> None:
        ship = env.engine.ships[env._target_id]
        stats = ship.stats
        ship.stats = ShipStats(
            speed=0,
            cost=stats.cost,
            strength=0.0,
            attack_range=0,
            visibility=stats.visibility,
            scouting=stats.scouting,
        )

    def fight(delay: int) -> float:
        env = CombatSandbox(str(board), ship="destroyer", width=3, height=3, max_turns=6)
        env.reset(seed=0)
        freeze(env)
        for _ in range(delay):
            env.step(EndTurnAction())
            assert env.rewards["player_0"] == 0.0
        attacks = [a for a in env.infos["player_0"]["legal_actions"] if isinstance(a, AttackAction)]
        assert attacks
        env.step(attacks[0])
        assert env.killed
        assert env.rewards["player_0"] == 0.0
        assert env.terminations["player_0"] is False
        for _ in range(20):
            if env.truncations["player_0"]:
                break
            env.step(EndTurnAction())
        assert env.terminations["player_0"] is False
        assert env.truncations["player_0"] is True
        paid = env.episode_reward
        weighted = discounted_combat_reward(env._power_samples, env.decay_horizon, env.decay_floor)
        assert abs(paid - weighted) < 1e-9
        assert abs(paid) <= 1.0
        env.close()
        return paid

    assert fight(0) > fight(2) > 0.0
