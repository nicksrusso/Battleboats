"""Play a checkpoint against random or the scripted baseline.

Used by ``scripts/eval_agent.py``. Not a league evaluator: one policy, one
opponent kind, a win/loss/draw count.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

from battleboats.agents.scripted import scripted_action
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.skill_maps import write_full_map
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.device import resolve_device
from battleboats.training.rollout import collect_trajectory


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    opponent: str = "random",
    games: int = 20,
    width: int = 8,
    height: int = 8,
    map_count: int = 2,
    max_turns: int = 40,
    max_steps: int = 200,
    seed: int = 0,
    device: str = "auto",
    map_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Return win/loss/draw rates for ``checkpoint`` over ``games`` matches.

    ``opponent`` is ``random`` or ``scripted``. The learner alternates seats.
    """
    if opponent not in ("random", "scripted"):
        raise ValueError(f"opponent must be random or scripted, got {opponent!r}")
    dev = resolve_device(device)
    policy, _ = load_checkpoint(checkpoint, dev)
    policy.eval()
    map_dir = Path(map_dir) if map_dir else Path(checkpoint).parent / "eval_maps"
    maps = [
        write_full_map(map_dir / f"eval_{i}.json", width, height, seed=seed + i)
        for i in range(max(1, map_count))
    ]
    env = BattleboatsAEC(maps[0], max_turns=max_turns)
    rng = random.Random(seed)
    wins = losses = draws = 0
    steps = 0
    action_fn = scripted_action if opponent == "scripted" else None
    for g in range(games):
        env.map_json_path = maps[g % len(maps)]
        env.reset(seed=seed + g)
        _traj, info = collect_trajectory(
            policy,
            None,
            env,
            agent_pid=g % 2,
            rng=rng,
            max_steps=max_steps,
            use_env_reward=False,
            opponent_action=action_fn,
        )
        outcome = info["outcome"]
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1
        steps += int(info["steps"])
    n = max(1, games)
    try:
        env.close()
    except Exception:
        pass
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "mean_steps": steps / n,
    }
