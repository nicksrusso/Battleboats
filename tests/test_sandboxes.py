"""Sparse sandboxes, the size gate, and the skill one-hot."""

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
    ShipBuildSandbox,
    enemy_strength_scale,
    make_sandbox,
)
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.policy import PolicyNetwork
from battleboats.training.sandbox_curriculum import (
    SandboxRunConfig,
    choose_episode,
    load_sandbox_config,
    run_sandboxes,
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


def test_combat_scripted_enemy_is_weakened_and_killable(tmp_path: Path) -> None:
    path = write_two_home_map(tmp_path / "c.json", 3, 3)
    env = CombatSandbox(str(path), ship="destroyer", width=3, height=3, max_turns=8)
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
    env.reset(seed=1)
    killed = False
    for _ in range(24):
        if env.terminations["player_0"] or env.truncations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        attacks = [a for a in legal if isinstance(a, AttackAction)]
        if attacks:
            env.step(attacks[0])
            if env.killed:
                killed = True
                break
            continue
        tgt = env.engine.ships.get(env._target_id)
        if tgt is None:
            break
        _step_toward(env, tgt.position)
    assert killed or env.killed
    assert env.rewards["player_0"] == 1.0
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
    assert cfg.device == "cpu"
    assert cfg.wandb is False
    assert cfg.wandb_project == "battleboats-skill-pipeline"


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
    assert (tmp_path / "run" / "latest.pt").is_file()
