"""Combat and builder sandboxes, plus the scripted baseline staying legal."""

import random
from pathlib import Path

from battleboats.agents.scripted import scripted_action
from battleboats.core.actions import AttackAction, BuildPortAction, MoveAction
from battleboats.core.gameEngine import gameEngine
from battleboats.envs.builder_skill_aec import BuilderSkillAEC
from battleboats.envs.combat_skill_aec import CombatSkillAEC
from battleboats.envs.skill_maps import write_builder_map, write_full_map, write_two_home_map


def test_combat_attack_in_range_kills(tmp_path: Path) -> None:
    env = CombatSkillAEC(write_two_home_map(tmp_path / "c.json", 2, 3), max_turns=10)
    env.reset(seed=0)
    legal = env.infos["player_0"]["legal_actions"]
    attacks = [a for a in legal if isinstance(a, AttackAction)]
    assert attacks, "destroyer should start in range of the merchant on 2x3"
    env.step(attacks[0])
    assert env.killed
    assert env.rewards["player_0"] == 1.0
    assert env.terminations["player_0"] is True


def test_builder_reaches_target_and_builds(tmp_path: Path) -> None:
    env = BuilderSkillAEC(write_builder_map(tmp_path / "b.json", 2, 3), max_turns=10)
    env.reset(seed=0)
    target = env.build_target
    assert target == (1, 0)
    for _ in range(8):
        if env.terminations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        builds = [a for a in legal if isinstance(a, BuildPortAction) and a.target == target]
        if builds:
            env.step(builds[0])
            break
        moves = [a for a in legal if isinstance(a, MoveAction)]
        assert moves

        def _dist(action: MoveAction) -> int:
            return abs(action.destination[0] - target[0]) + abs(action.destination[1] - target[1])

        env.step(min(moves, key=_dist))
    assert env.built
    assert env.rewards["player_0"] == 1.0
    assert env.terminations["player_0"] is True


def test_scripted_actions_are_legal(tmp_path: Path) -> None:
    engine = gameEngine(write_full_map(tmp_path / "f.json", 4, 4, seed=1))
    engine.reset(seed=1)
    rng = random.Random(0)
    taken = 0
    while engine.winner is None and engine.turn < 6 and taken < 40:
        pid = engine.current_player
        legal = engine.enumerate_legal(pid)
        action = scripted_action(engine, pid, rng)
        assert action in legal
        engine.step(action)
        taken += 1
    assert taken > 0
