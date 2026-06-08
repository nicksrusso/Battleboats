"""PPO rollout driver and Generalized Advantage Estimation.

Plays games with the current (learning) policy in one seat against a frozen
opponent policy in the other, recording only the learning seat's decisions as
``Transition``s. ``compute_gae`` then fills their advantages and value targets.

Each ``Transition`` stores the legality masks along the chosen (asset, verb,
target) path, not just the action. The PPO clipped objective needs the ratio
``pi_new / pi_old``, and ``log pi`` depends on the legality mask applied before
the softmax. At rollout time the true ``ActionMasks`` are available; at update
time only the tokens remain (which do not encode legality), so the masks travel
with the transition and let ``evaluate_actions`` reproduce ``log pi_old`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from battleboats.agents.heuristics import heuristic_eval
from battleboats.agents.random_agent import random_action
from battleboats.envs.action_masks import ActionMasks
from battleboats.training.policy import VERB_NAMES, VERBS_WITH_TARGET, PolicyNetwork


@dataclass
class Transition:
    """One policy decision in a rollout.

    Tensors are detached, on CPU, with the batch axis stripped (the buffer's
    collate re-adds it). ``reward``, ``advantage``, and ``ret`` are filled in
    after the game by the terminal outcome and GAE.
    """

    tokens: torch.Tensor  # (N, TOKEN_DIM) — encoder input at this state
    asset_mask: torch.Tensor  # (N,) bool   — legal assets (asset pointer head)
    verb_mask: torch.Tensor  # (NUM_VERBS,) bool — legal verbs given the chosen asset
    target_mask: Optional[torch.Tensor]  # (K,) bool — legal targets given asset+verb; None for no-target verbs
    asset_idx: int
    verb_idx: int
    target_idx: int  # -1 sentinel for no-target verbs
    logp: float  # log pi_old(a,v,t) — the behavior-policy log-prob (frozen for the ratio)
    value: float  # V_old(s) — for the GAE baseline / value-target bootstrap
    reward: float = 0.0  # sparse: 0 except the terminal decision (±1), set post-game
    advantage: float = 0.0  # filled by GAE
    ret: float = 0.0  # value target = advantage + value, filled by GAE


def heuristic_action(engine, pid: int, rng, eps: float = 1e-9):
    """Eval-only 1-ply greedy heuristic policy (not a training opponent).

    Clones the engine for each legal action, applies it, scores the resulting
    state with ``heuristic_eval`` from ``pid``'s point of view, and returns the
    highest-scoring action (ties broken randomly via ``rng``). Used to
    benchmark a trained policy against a fixed yardstick; too slow to serve as
    a training opponent.

    Args:
        engine: live game engine; cloned per candidate, never mutated.
        pid: player id whose point of view scores the resulting states.
        rng: random source used to break score ties.
        eps: tolerance for treating two scores as tied.

    Returns:
        The chosen Action, ready for ``engine.step()``.
    """
    legal = engine.enumerate_legal(pid)
    best = []
    best_score = -math.inf
    for a in legal:
        sim = engine.clone()
        sim.step(a)
        score = heuristic_eval(sim, pid)
        if score > best_score + eps:
            best = [a]
            best_score = score
        elif abs(score - best_score) <= eps:
            best.append(a)
    return rng.choice(best)


@torch.no_grad()
def collect_trajectory(
    policy: PolicyNetwork,
    opponent_policy: PolicyNetwork,
    env,
    agent_pid: int,
    rng,
    max_steps: int = 400,
) -> Tuple[List[Transition], Dict[str, Any]]:
    """Play one game and return the learning seat's decisions and outcome.

    ``policy`` controls seat ``agent_pid`` and its decisions are recorded as
    ``Transition``s; ``opponent_policy`` is a frozen snapshot controlling the
    other seat and is not recorded. Both run under ``no_grad``. Each recorded
    transition stores the legality masks along the chosen (asset, verb, target)
    path so the PPO update can reproduce ``log pi_old``. Reward is sparse
    terminal (+1 win / -1 loss / 0 draw), attached to the last recorded
    decision.

    The env must already be reset by the caller (the scenario/seed are the
    caller's choice). ``max_steps`` caps total engine steps, not turns;
    ``agent_pid`` lets the caller alternate the learning seat across games so
    the policy learns both sides of the first-player advantage.

    Returns:
        ``(transitions, info)`` where ``info`` carries "winner", "outcome",
        "steps", "n_decisions", and "truncated".
    """
    transitions = []
    steps = 0
    for agent in env.agent_iter():
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        pid = env._player_id(agent)
        engine = env.engine
        masks = ActionMasks(engine, pid)
        if pid == agent_pid:
            net = policy
        else:
            net = opponent_policy

        # sample an action
        a_idx, v_idx, t_idx, logp, value = net.act(masks.tokens, masks.pad_mask, masks)

        if pid == agent_pid:
            verb_name = VERB_NAMES[int(v_idx)]
            has_target = verb_name in VERBS_WITH_TARGET
            target_mask = masks.target_for(verb_name, a_idx).squeeze(0).clone() if has_target else None
            transitions.append(
                Transition(
                    tokens=masks.tokens.squeeze(0).clone(),  # (N, TOKEN_DIM), drop batch axis
                    asset_mask=masks.asset.squeeze(0).clone(),  # (N,)
                    verb_mask=masks.verbs_for(a_idx).squeeze(0).clone(),  # (NUM_VERBS,)
                    target_mask=target_mask,  # (K,) or None
                    asset_idx=int(a_idx),
                    verb_idx=int(v_idx),
                    target_idx=-1 if t_idx is None else int(t_idx),  # -1 sentinel = no target
                    logp=float(logp),
                    value=float(value),
                )
            )

        # apply action to env
        env.step(masks.to_action(asset_idx=a_idx, verb_idx=v_idx, target_idx=t_idx))
        steps += 1
        if steps > max_steps:
            break

    winner = env.engine.winner
    outcome = 1.0 if winner == agent_pid else (-1.0 if winner is not None else 0.0)
    if transitions:
        transitions[-1].reward = outcome  # sparse terminal reward on the last decision
    return transitions, {
        "winner": winner,
        "outcome": outcome,
        "steps": steps,
        "n_decisions": len(transitions),
        "truncated": winner is None,
    }


def compute_gae(transitions: List[Transition], gamma: float = 0.99, lam: float = 0.95) -> None:
    """Fill each Transition's ``advantage`` and ``ret`` in place via GAE.

    Generalized Advantage Estimation over one trajectory: a backward,
    exponentially-weighted (``lam``) sum of the TD residuals
    ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)``, bootstrapping the terminal
    next-state value with 0. ``lam`` trades bias (toward 0, pure 1-step TD) for
    variance (toward 1, pure Monte-Carlo). ``ret = advantage + value`` is the
    value-head regression target, so a value target falls out for free.

    Operates per trajectory only; advantage normalization happens later in the
    PPO update, not here.
    """
    gae_next = 0.0
    for t in reversed(range(len(transitions))):
        this_t = transitions[t]
        if t + 1 < len(transitions):
            next_val = transitions[t + 1].value
        else:
            next_val = 0.0
        delta = this_t.reward + gamma * next_val - this_t.value
        gae = delta + gamma * lam * gae_next
        this_t.advantage = gae
        this_t.ret = gae + this_t.value
        gae_next = gae
