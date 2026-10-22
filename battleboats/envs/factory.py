"""Construct the env for a curriculum skill name."""

from __future__ import annotations

from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.builder_skill_aec import BuilderSkillAEC
from battleboats.envs.combat_skill_aec import CombatSkillAEC
from battleboats.envs.landing_skill_aec import LandingSkillAEC
from battleboats.envs.merchant_skill_aec import MerchantSkillAEC


def make_env(
    skill: str,
    map_path: str,
    *,
    max_turns: int,
    randomize: bool = False,
    approach_coef: float = 0.0,
):
    """Return a fresh env. ``randomize`` / ``approach_coef`` apply only where the skill has them."""
    if skill == "merchant":
        return MerchantSkillAEC(
            map_path,
            max_turns=max_turns,
            randomize_ports=randomize,
        )
    if skill == "landing":
        return LandingSkillAEC(
            map_path,
            max_turns=max_turns,
            randomize_target=randomize,
            approach_coef=approach_coef,
        )
    if skill == "combat":
        return CombatSkillAEC(
            map_path,
            max_turns=max_turns,
            randomize_target=randomize,
            approach_coef=approach_coef,
        )
    if skill == "builder":
        return BuilderSkillAEC(
            map_path,
            max_turns=max_turns,
            approach_coef=approach_coef,
        )
    if skill == "full":
        return BattleboatsAEC(map_path, max_turns=max_turns)
    raise ValueError(f"unknown skill {skill!r}")
