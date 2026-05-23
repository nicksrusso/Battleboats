"""Tests for the god-mode MCTS agent.

Coverage:
  - Returns a legal action from initial state (basic smoke).
  - Works for either side (player 0 or player 1).
  - Deterministic under seeded RNG.
  - Beats the random baseline at modest iteration budgets (the real check;
    marked slow because it plays full games).

Run with:
    poetry run pytest tests/test_godmode_mcts.py -v
    poetry run pytest tests/test_godmode_mcts.py -v -m slow   # include benchmark
"""

import random

import pytest

from battleboats.agents.godmode_mcts import godmode_mcts_action
from battleboats.agents.random_agent import random_action
from battleboats.core.gameEngine import gameEngine
from battleboats.envs.battleboats_aec import BattleboatsAEC

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"

# Smoke-test budget: small, just enough to exercise selection / expansion / backprop.
SMOKE_ITERATIONS = 20

# Benchmark budget: modest, balances signal vs wall-time.
BENCH_ITERATIONS = 250
BENCH_NUM_GAMES = 1  # alternating sides → 3 as player 0, 3 as player 1
BENCH_MAX_TURNS = 250
BENCH_STEP_BUDGET = 500_000


@pytest.fixture
def fresh_engine() -> gameEngine:
    engine = gameEngine(map_json_path=MAP_JSON)
    engine.reset(seed=0)
    return engine


# -------------------------------------------------------------- smoke tests


def test_returns_legal_action(fresh_engine: gameEngine) -> None:
    """The chosen action must be in enumerate_legal for the current player."""
    rng = random.Random(0)
    legal = fresh_engine.enumerate_legal(0)
    action = godmode_mcts_action(fresh_engine, 0, rng, iterations=SMOKE_ITERATIONS)
    assert action in legal, f"MCTS returned an action not in legal set: {action}"


def test_works_for_player_1(fresh_engine: gameEngine) -> None:
    """Step a few random moves so it becomes player 1's turn, then run MCTS for them."""
    rng = random.Random(0)
    # Advance until it's player 1's turn (random play, short loop).
    safety = 0
    while fresh_engine.current_player != 1:
        action = random_action(fresh_engine, fresh_engine.current_player, rng)
        fresh_engine.step(action)
        safety += 1
        if safety > 1000:
            pytest.skip("could not reach player 1's turn within budget")
    action = godmode_mcts_action(fresh_engine, 1, rng, iterations=SMOKE_ITERATIONS)
    assert action in fresh_engine.enumerate_legal(1)


def test_deterministic_with_seeded_rng() -> None:
    """Same seed (engine + rng) → same chosen action."""
    engine_a = gameEngine(map_json_path=MAP_JSON)
    engine_a.reset(seed=0)
    engine_b = gameEngine(map_json_path=MAP_JSON)
    engine_b.reset(seed=0)
    action_a = godmode_mcts_action(engine_a, 0, random.Random(42), iterations=SMOKE_ITERATIONS)
    action_b = godmode_mcts_action(engine_b, 0, random.Random(42), iterations=SMOKE_ITERATIONS)
    assert action_a == action_b, f"MCTS not deterministic under fixed seed: {action_a} vs {action_b}"


def test_engine_not_mutated_by_mcts(fresh_engine: gameEngine) -> None:
    """MCTS must not leak state into the caller's engine — it operates on clones."""
    rng = random.Random(0)
    turn_before = fresh_engine.turn
    current_before = fresh_engine.current_player
    winner_before = fresh_engine.winner
    _ = godmode_mcts_action(fresh_engine, 0, rng, iterations=SMOKE_ITERATIONS)
    assert fresh_engine.turn == turn_before
    assert fresh_engine.current_player == current_before
    assert fresh_engine.winner == winner_before


# ------------------------------------------------------------ benchmark


def _play_one_game(
    env: BattleboatsAEC,
    mcts_player_id: int,
    rng: random.Random,
    iterations: int,
) -> int:
    """Drive one game: MCTS plays as ``mcts_player_id``, random plays the other side.

    Returns env.engine.winner (0/1) on capture, or -1 if truncated/budgeted out.
    """
    steps = 0
    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            action = None
        elif env._player_id(agent) == mcts_player_id:
            action = godmode_mcts_action(env.engine, mcts_player_id, rng, iterations=iterations)
        else:
            action = random_action(env.engine, env._player_id(agent), rng)
        env.step(action)
        steps += 1
        if steps > BENCH_STEP_BUDGET:
            break
    return env.engine.winner if env.engine.winner is not None else -1


@pytest.mark.slow
def test_mcts_beats_random() -> None:
    """MCTS should win at least as often as random does, over a small set of games.

    Random-vs-random in battleboats almost never terminates; even with MCTS on
    one side, most games will truncate. The signal is: among games that DO
    end in a capture, MCTS should be the one capturing the enemy home port.

    Threshold is conservative — MCTS must out-win random strictly. If random
    happens to win as often as MCTS at this iteration budget, something is
    wrong (the heuristic, the backup, or the search isn't paying off).
    """
    mcts_wins = 0
    random_wins = 0
    truncated = 0

    for game_idx in range(BENCH_NUM_GAMES):
        env = BattleboatsAEC(map_json_path=MAP_JSON, max_turns=BENCH_MAX_TURNS)
        env.reset(seed=game_idx)
        rng = random.Random(game_idx)
        mcts_player = game_idx % 2  # alternate sides

        winner = _play_one_game(env, mcts_player, rng, BENCH_ITERATIONS)
        if winner == mcts_player:
            mcts_wins += 1
        elif winner == (1 - mcts_player):
            random_wins += 1
        else:
            truncated += 1

    assert mcts_wins > random_wins, (
        f"MCTS not outperforming random over {BENCH_NUM_GAMES} games: "
        f"mcts={mcts_wins}, random={random_wins}, truncated={truncated}"
    )
