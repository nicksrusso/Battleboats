"""Pre-flight probes for MCTS implementation.

Two checks before committing to the MCTS implementation:

  1. Action dataclass equality + hashing. The tree expansion path uses
     `untried_actions.remove(action)` (or pop-by-index) and may also use
     dict-keyed children lookups in future variants. Confirms that the
     frozen+slots dataclasses behave as expected.

  2. Engine clone + random-rollout perf. Establishes how many MCTS
     iterations are feasible per move at acceptable wall time.

Run with:
    poetry run python scripts/mcts_probes.py
"""

import random
import time

from battleboats.agents.random_agent import random_action
from battleboats.core.actions import EndTurnAction, MoveAction
from battleboats.core.gameEngine import gameEngine

MAP_JSON = "/home/nick/Desktop/repos/Battleboats/battleboats/core/config/map.json"
ROLLOUT_STEP_BUDGET = 100  # engine *steps*, not turns — random play late-game is unboundedly slow
N_ROLLOUTS = 20
CLONE_TRIALS = 1000


def check_action_equality() -> None:
    """Verify Action dataclass equality and hashability."""
    print("=== Probe 1: Action equality + hashing ===")

    # Fieldless action: all instances must compare equal.
    e1 = EndTurnAction()
    e2 = EndTurnAction()
    fieldless_eq = e1 == e2
    fieldless_hash = hash(e1) == hash(e2)

    # Fielded action: equal iff fields equal.
    m1 = MoveAction(ship_id=1, destination=(0, 0))
    m2 = MoveAction(ship_id=1, destination=(0, 0))
    m3 = MoveAction(ship_id=2, destination=(0, 0))
    fielded_eq = m1 == m2
    fielded_neq = m1 == m3
    fielded_hash = hash(m1) == hash(m2)

    # Practical: pop by value out of an untried-actions list.
    untried = [m1, m3]
    untried.remove(m2)
    remove_by_value = untried == [m3]

    # Cross-type distinct.
    cross_type = e1 == m1

    in_set = m2 in {m1, m3}

    print(f"  EndTurnAction() == EndTurnAction():             {fieldless_eq}  (expected True)")
    print(f"  hash(EndTurn) consistent:                       {fieldless_hash}  (expected True)")
    print(f"  MoveAction same fields equal:                   {fielded_eq}  (expected True)")
    print(f"  MoveAction diff fields equal:                   {fielded_neq}  (expected False)")
    print(f"  hash(MoveAction) consistent:                    {fielded_hash}  (expected True)")
    print(f"  list.remove() matches by value:                 {remove_by_value}  (expected True)")
    print(f"  cross-type equality (Move == EndTurn):          {cross_type}  (expected False)")
    print(f"  set membership matches by value:                {in_set}  (expected True)")

    all_ok = (
        fieldless_eq and fieldless_hash
        and fielded_eq and (not fielded_neq) and fielded_hash
        and remove_by_value and (not cross_type) and in_set
    )
    print(f"  ALL PASS: {all_ok}")
    print()


def check_engine_perf() -> None:
    """Time engine.clone() and full random rollouts to terminal."""
    print("=== Probe 2: Clone + rollout perf ===")

    engine = gameEngine(map_json_path=MAP_JSON)
    engine.reset(seed=0)
    rng = random.Random(0)

    # --- Clone alone ---
    t0 = time.perf_counter()
    for _ in range(CLONE_TRIALS):
        _ = engine.clone()
    clone_us = (time.perf_counter() - t0) / CLONE_TRIALS * 1_000_000
    print(f"  engine.clone() avg:                  {clone_us:7.1f} us  ({CLONE_TRIALS} trials)")

    # --- Step-capped random rollouts from initial state ---
    rollout_times = []
    rollout_steps = []
    outcomes = []  # winner_id or None (None = cap hit before terminal)

    for _ in range(N_ROLLOUTS):
        sim = engine.clone()
        t0 = time.perf_counter()
        steps = 0
        while sim.winner is None and steps < ROLLOUT_STEP_BUDGET:
            action = random_action(sim, sim.current_player, rng)
            sim.step(action)
            steps += 1
        rollout_times.append(time.perf_counter() - t0)
        rollout_steps.append(steps)
        outcomes.append(sim.winner)

    avg_ms = sum(rollout_times) / len(rollout_times) * 1000
    avg_steps = sum(rollout_steps) / len(rollout_steps)
    us_per_step = avg_ms * 1000 / max(avg_steps, 1)
    terminated = sum(1 for w in outcomes if w is not None)
    truncated = N_ROLLOUTS - terminated

    print(f"  step-capped rollout avg:             {avg_ms:7.1f} ms  ({avg_steps:.0f} steps avg, cap={ROLLOUT_STEP_BUDGET})")
    print(f"  per env step:                        {us_per_step:7.1f} us")
    print(f"  terminated by step {ROLLOUT_STEP_BUDGET}:           {terminated}/{N_ROLLOUTS}  (rest need heuristic eval at depth-out)")
    print(f"  hit step cap:                        {truncated}/{N_ROLLOUTS}")

    print()
    print("  --- MCTS budget projection (single move, step-capped rollouts) ---")
    for iters in (100, 500, 1000, 5000, 10_000):
        projected_s = iters * avg_ms / 1000
        print(f"    {iters:>5} iterations  ~=  {projected_s:7.1f} s / move")
    print()
    print(f"  If terminated/{N_ROLLOUTS} is low (say <3), random play rarely produces a")
    print("  decisive ending within the step budget — MCTS will lean on the heuristic")
    print("  eval at depth-out for most leaf values. That's fine, just informative.")


if __name__ == "__main__":
    check_action_equality()
    check_engine_perf()
