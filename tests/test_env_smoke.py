"""End-to-end smoke test for the PettingZoo AEC wrapper.

Plays random-vs-random games and checks the env's lifecycle invariants:
    - Every game ends (terminated by capture or truncated by turn limit).
    - Terminal rewards are zero-sum across both agents.
    - No exception is raised during the agent_iter loop (covers illegal
      action handling, dead-agent advancement, dict-key conventions, etc.).

Run with: poetry run pytest tests/test_env_smoke.py -v
"""

import random

import pytest

from battleboats.agents.random_agent import random_action
from battleboats.envs.battleboats_aec import BattleboatsAEC

MAP_JSON = "/home/nick/Desktop/repos/NavalCivGame/battleboats/core/config/map.json"
NUM_GAMES = 5
MAX_TURNS = 1500  # smoke test only needs to exercise the lifecycle, not play out full games
STEP_BUDGET = 200_000  # safety net so a runaway loop fails fast


def _play_one_game(env: BattleboatsAEC, rng: random.Random) -> int:
    """Drive one full episode with random policies on both seats.

    Returns the number of env steps taken.
    """
    steps = 0
    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            action = None
        else:
            action = random_action(env.engine, env._player_id(agent), rng)
        env.step(action)
        steps += 1
        if steps > STEP_BUDGET:
            pytest.fail(f"step budget exceeded ({STEP_BUDGET}); env probably not terminating")
    return steps


@pytest.mark.parametrize("seed", range(NUM_GAMES))
def test_random_vs_random_terminates_zero_sum(seed: int) -> None:
    env = BattleboatsAEC(map_json_path=MAP_JSON, max_turns=MAX_TURNS)
    env.reset(seed=seed)
    rng = random.Random(seed)

    steps = _play_one_game(env, rng)

    terminated_by_win = env.engine.winner is not None
    truncated_by_turns = env.engine.turn >= env.max_turns
    assert terminated_by_win or truncated_by_turns, (
        f"seed={seed}: game neither terminated nor truncated after {steps} steps "
        f"(winner={env.engine.winner}, turn={env.engine.turn})"
    )

    reward_sum = sum(env._cumulative_rewards.values())
    assert reward_sum == 0.0, f"seed={seed}: rewards not zero-sum: {env._cumulative_rewards}"

    if terminated_by_win:
        assert all(env.terminations.values()), f"seed={seed}: winner set but terminations not all True: {env.terminations}"
    elif truncated_by_turns:
        assert all(env.truncations.values()), f"seed={seed}: turn limit hit but truncations not all True: {env.truncations}"
