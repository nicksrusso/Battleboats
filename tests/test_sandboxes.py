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
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.sandboxes import (
    CombatSandbox,
    LandingSandbox,
    MerchantSandbox,
    ScoutingSandbox,
    ShipBuildSandbox,
    combat_outcome,
    detection_range,
    detection_reward,
    enemy_strength_scale,
    make_sandbox,
    scout_layout,
    side_power,
)
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.policy import PolicyNetwork
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.curriculum import _new_policy
from battleboats.training.sandbox_curriculum import (
    SandboxRunConfig,
    _drill_done,
    _ensure_runtime_state,
    _fresh_state,
    _reset_size_progress,
    choose_episode,
    choose_mixed_episode,
    drill_done,
    load_sandbox_config,
    plateaued,
    return_scale,
    run_sandboxes,
    scale_trajectory,
    smooth_series,
    window_cleared,
)
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
    assert env.rewards["player_0"] == 1.0
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
    assert env.rewards["player_0"] == 1.0
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
    assert env.rewards["player_0"] == 1.0
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
    expected = combat_outcome(
        env._my0, env._en0, side_power(env.engine, 0), side_power(env.engine, 1)
    )
    assert env.episode_reward == expected
    assert env.rewards["player_0"] == expected
    env.close()


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def test_detection_reward_is_higher_when_the_sighting_is_sooner() -> None:
    assert detection_reward(0, 9) == 1.0
    assert detection_reward(8, 9) == 1.0 / 9
    assert detection_reward(0, 9) > detection_reward(4, 9) > detection_reward(9, 9)
    assert detection_reward(9, 9) == 0.0


def test_scouting_starts_blind_and_pays_for_the_first_sighting(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "s.json", 3, 3)
    env = ScoutingSandbox(str(path), width=3, height=3, max_turns=9)
    seen = set()
    for seed in range(8):
        env.reset(seed=seed)
        seen.add(env.learner_type)
        seen.add(env.enemy_type)
        assert env.found is False
        assert env.engine.visible_enemy_ships(0) == []
        hunter = env.engine.ships[env._hunter_id]
        target = env.engine.ships[env._target_id]
        reach = detection_range(hunter.type, target.type)
        assert _manhattan(hunter.position, target.position) > reach
        legal = env.infos["player_0"]["legal_actions"]
        assert legal
        assert all(isinstance(a, (MoveAction, EndTurnAction)) for a in legal)
        held = target.position
        env.step(EndTurnAction())
        assert env.engine.ships[env._target_id].position == held
        assert env.found is False
        assert env.rewards["player_0"] == 0.0
    assert len(seen) >= 3
    assert any(t in seen for t in (ShipType.MERCHANT, ShipType.LANDING, ShipType.BUILDER))
    env.reset(seed=1)
    goal = env.engine.ships[env._target_id].position
    for _ in range(40):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        moves = [a for a in legal if isinstance(a, MoveAction)]
        if not moves:
            env.step(EndTurnAction())
            continue
        env.step(min(moves, key=lambda a: _manhattan(a.destination, goal)))
    assert env.found
    assert env.terminations["player_0"] is True
    assert env.episode_reward == detection_reward(env.engine.turn, env.max_turns)
    assert env.rewards["player_0"] == env.episode_reward
    assert 0.0 < env.episode_reward <= 1.0
    env.close()

    far = write_two_home_map(tmp_path / "s10.json", 10, 10)
    wide = ScoutingSandbox(str(far), width=10, height=10, max_turns=4)
    wide.reset(seed=2)
    assert wide.engine.visible_enemy_ships(0) == []
    for _ in range(6):
        if wide.terminations["player_0"] or wide.truncations["player_0"]:
            break
        wide.step(EndTurnAction())
    assert wide.found is False
    assert wide.episode_reward == 0.0
    assert wide.truncations["player_0"] is True
    wide.close()
    assert scout_layout(random.Random(0), 3, 3, reach=20) is None


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


def test_combat_plateau_ignores_wins_and_the_update_cap_advances(tmp_path: Path) -> None:
    flat = [0.2] * 10
    assert plateaued(flat, 10, 0.02, higher_is_better=True)
    assert plateaued([0.4] * 10, 10, 0.02, higher_is_better=False)
    assert not plateaued([0.01 * i for i in range(1, 11)], 10, 0.02, higher_is_better=True)
    assert not plateaued([1.0, 0.5], 2, 0.02, higher_is_better=False)
    common = dict(
        updates=10,
        window=[0.0] * 64,
        episodes=40,
        success_threshold=0.8,
        need=64,
        min_updates=3,
        plateau_updates=10,
        plateau_threshold=0.02,
        combat_min_episodes=32,
    )
    assert drill_done(
        "combat_destroyer",
        reward_trace=flat,
        value_trace=[0.4] * 10,
        **common,
    )
    assert not drill_done(
        "combat_destroyer",
        reward_trace=[0.01 * i for i in range(1, 11)],
        value_trace=[0.4] * 10,
        **common,
    )
    assert not drill_done(
        "combat_destroyer",
        reward_trace=flat,
        value_trace=[0.4] * 10,
        **{**common, "episodes": 10},
    )
    assert not drill_done("merchant", reward_trace=flat, value_trace=[0.4] * 10, **common)
    assert smooth_series([1.0, 3.0], 2, "mean") == [1.0, 2.0]
    assert smooth_series([0.0, 0.0, 10.0], 1, "ema") == [0.0, 0.0, 10.0]
    noisy = [0.05, 0.35] * 12
    assert not plateaued(noisy, 10, 0.02, higher_is_better=True)
    assert plateaued(noisy, 10, 0.02, higher_is_better=True, smooth_window=10, smooth="mean")
    assert plateaued([0.4] * 20, 10, 0.02, higher_is_better=False, smooth_window=10, smooth="ema")
    # Scouting uses the goal bar, not the combat plateau.
    assert drill_done("scouting", reward_trace=noisy, value_trace=[0.4] * 10, **{**common, "window": [1.0] * 64})
    assert not drill_done("scouting", reward_trace=flat, value_trace=[0.4] * 10, **{**common, "window": []})

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
    assert result["ok"]
    summary = json.loads((tmp_path / "cap" / "summary.json").read_text())
    size = summary["skills"]["merchant"]["sizes"]["3"]
    assert size["gate_met"] is False
    assert size["capped"] is True


def test_combat_pass_stays_passed_for_that_size(tmp_path: Path) -> None:
    cfg = SandboxRunConfig(
        run_dir=tmp_path / "latch",
        skills=["combat_destroyer", "merchant"],
        sizes=[3, 4],
    )
    assert cfg.plateau_smooth == "mean"
    assert cfg.plateau_smooth_window == 10
    state = _fresh_state()
    _ensure_runtime_state(state, cfg)
    state["updates_this_size"] = 20
    state["episodes"]["combat_destroyer"] = 40
    state["reward_trace"]["combat_destroyer"] = [0.2] * 20
    state["value_trace"]["combat_destroyer"] = [0.4] * 20
    assert _drill_done(cfg, state, "combat_destroyer", [])
    state["reward_trace"]["combat_destroyer"] = [0.05, 0.35] * 16
    state["value_trace"]["combat_destroyer"] = [0.9, 0.1] * 16
    assert _drill_done(cfg, state, "combat_destroyer", [])
    assert state["combat_passed"]["combat_destroyer"] is True
    assert not _drill_done(cfg, state, "merchant", [])
    _reset_size_progress(state, cfg)
    state["updates_this_size"] = 20
    state["episodes"]["combat_destroyer"] = 40
    state["reward_trace"]["combat_destroyer"] = [0.01 * i for i in range(1, 21)]
    state["value_trace"]["combat_destroyer"] = [0.4] * 20
    assert state["combat_passed"]["combat_destroyer"] is False
    assert not _drill_done(cfg, state, "combat_destroyer", [])


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
    assert cfg.plateau_threshold == 0.02
    assert cfg.plateau_updates == 10
    assert cfg.plateau_smooth == "mean"
    assert cfg.plateau_smooth_window == 10
    assert cfg.combat_min_episodes == 32
    assert cfg.max_updates_per_size == 150
    assert cfg.skills[3] == "scouting"
    assert cfg.device == "cpu"
    assert cfg.wandb is False
    assert cfg.wandb_project == "battleboats-skill-pipeline"


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
