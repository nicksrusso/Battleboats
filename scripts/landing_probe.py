"""Go/no-go probe for the landing sandbox.

Runs a uniform-random learner in LandingSkillAEC for many episodes and reports
how often it captures the enemy home port within max_turns, and how many engine
steps that takes. If capture-rate is near zero, sparse reward gives no gradient
and shaping is needed before training.

Usage:
    python scripts/landing_probe.py [--episodes N] [--max-turns T]
"""

from __future__ import annotations

import argparse
import random
import statistics

from battleboats.agents.random_agent import random_action
from battleboats.envs.landing_skill_aec import LandingSkillAEC

MAP_PATH = "battleboats/core/config/maps/map_landing_2x3.json"
LEARNER = "player_0"


def run(episodes: int, max_turns: int) -> None:
    rng = random.Random(0)
    env = LandingSkillAEC(MAP_PATH, max_turns=max_turns)

    captures = 0
    steps_to_capture: list[int] = []
    for ep in range(episodes):
        env.reset(seed=ep)
        steps = 0
        while not (env.terminations[LEARNER] or env.truncations[LEARNER]):
            env.step(random_action(env.engine, 0, rng))
            steps += 1
        if env.terminations[LEARNER]:  # capture (only terminal in this env)
            captures += 1
            steps_to_capture.append(steps)

    rate = captures / episodes
    print(f"episodes={episodes}  max_turns={max_turns}")
    print(f"capture_rate={rate:.3f}  ({captures}/{episodes})")
    if steps_to_capture:
        print(
            f"steps_to_capture: mean={statistics.mean(steps_to_capture):.1f} "
            f"min={min(steps_to_capture)} max={max(steps_to_capture)}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max-turns", type=int, default=50)
    args = ap.parse_args()
    run(args.episodes, args.max_turns)
