"""God-mode MCTS agent — perfect-information UCT over the engine.

Used as a fixed benchmark opponent, not as a trainable agent. "God mode"
means the search operates on the true game state with no fog-of-war
filtering on either side. This is appropriate because the agent under
test still plays under its own fog; this opponent's job is to be a
strong, fixed yardstick whose quality depends only on its iteration
budget. See docs/sota_research_2026-05-20.md for the design rationale.

Vanilla UCT (Kocsis & Szepesvári 2006):
    Each iteration walks the tree from the root via UCB1, expands one
    untried action at the reached leaf, rolls out to terminal with a
    uniform-random default policy, and backpropagates the outcome.
    Final move = child of root with the most visits.

Negamax backup convention:
    Node.total_value is stored from the perspective of node.side_to_move.
    When backpropagating, sign-flip at each step up the tree.
"""

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from battleboats.core.actions import Action
    from battleboats.core.gameEngine import gameEngine

from battleboats.agents.heuristics import heuristic_eval
from battleboats.agents.random_agent import random_action

DEFAULT_C = math.sqrt(2)
DEFAULT_ITERATIONS = 10000
ROLLOUT_STEP_BUDGET = 100  # engine steps; random play rarely terminates in battleboats


@dataclass
class Node:
    """One node in the MCTS search tree.

    Tree state only — no game state. The engine state corresponding to
    this node is reconstructed at iteration time by cloning the root
    engine and replaying actions down the path. We never store engines
    on nodes (would balloon memory).
    """

    parent: Optional["Node"]
    action_in: Optional["Action"]  # action that produced this node from parent; None at root
    side_to_move: int  # whose turn it is AT this node
    untried_actions: List["Action"]  # legal actions not yet expanded into children
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0  # from the perspective of side_to_move (negamax)


def godmode_mcts_action(
    engine: "gameEngine",
    player_id: int,
    rng: random.Random,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    c: float = DEFAULT_C,
) -> "Action":
    """Run UCT from `engine`'s current state and return the best action for `player_id`.

    Preconditions:
        engine.current_player == player_id
        engine.is_terminal() is False
        engine.enumerate_legal(player_id) is non-empty

    Notes:
        - Tree is discarded after each call (no subtree reuse in v1).
        - rng is used for rollout action selection and tie-breaking.
    """
    root = Node(
        parent=None, action_in=None, side_to_move=player_id, untried_actions=list(engine.enumerate_legal(player_id=player_id))
    )

    for _ in range(iterations):
        sim = engine.clone()
        leaf = _select(root, sim, c)
        if not sim.is_terminal() and leaf.untried_actions:
            leaf = _expand(leaf, sim, rng)
        value = _evaluate(sim, leaf.side_to_move, rng)
        _backprop(leaf, value)
    return _best_child_by_visits(root, rng).action_in


def _select(node: Node, engine: "gameEngine", c: float) -> Node:
    """Descend the tree via UCB1, applying actions to `engine` as we go.

    Stops at a node that is either terminal in `engine` or has untried
    actions. Mutates `engine` along the path. Caller passes a clone.
    """
    while not engine.is_terminal() and not node.untried_actions and node.children:
        best_child = max(node.children, key=lambda chld: _ucb1(chld, node.visits, c))
        engine.step(best_child.action_in)
        node = best_child
    return node


def _expand(node: Node, engine: "gameEngine", rng: random.Random) -> Node:
    """Pop one untried action from `node`, apply to `engine`, attach a new child.

    Returns the new child node. If `node` has no untried actions (i.e.,
    it's terminal), returns `node` unchanged.
    """
    if len(node.untried_actions) == 0:
        return node

    act = node.untried_actions.pop()
    engine.step(action=act)
    untried_actions = engine.enumerate_legal(engine.current_player) if not engine.is_terminal() else []
    new_child = Node(
        parent=node,
        action_in=act,
        side_to_move=engine.current_player,
        untried_actions=untried_actions,
    )
    node.children.append(new_child)
    return new_child


def _evaluate(engine: "gameEngine", side_at_leaf: int, rng: random.Random) -> float:
    """Step-capped random rollout then heuristic eval at depth-out.

    Plays uniform-random actions for up to ROLLOUT_STEP_BUDGET engine
    steps or until terminal, then returns heuristic_eval from
    side_at_leaf's perspective. The hybrid signal is automatic because
    heuristic_eval short-circuits to ±1 for terminal states.

    Mutates `engine`; caller must already hold the clone they care about.
    """
    step_count = 0
    while not engine.is_terminal() and step_count < ROLLOUT_STEP_BUDGET:
        act = random_action(engine=engine, player_id=engine.current_player, rng=rng)
        engine.step(act)
        step_count += 1
    return heuristic_eval(engine=engine, me=side_at_leaf)


def _backprop(leaf: Node, value_at_leaf: float) -> None:
    """Propagate `value_at_leaf` from the leaf up to the root.

    `value_at_leaf` is in [-1, +1] from `leaf.side_to_move`'s perspective
    (as returned by _evaluate). Walk up the tree from `leaf`; at each
    node, increment visits and add the value translated to that node's
    side-to-move (same sign if sides match, negated if they differ).

    Sides only flip across EndTurn boundaries, so the per-node `==` check
    correctly handles multi-action turns.
    """

    node = leaf
    while node is not None:
        node.visits += 1
        if node.side_to_move == leaf.side_to_move:
            node.total_value += value_at_leaf
        else:
            node.total_value -= value_at_leaf
        node = node.parent


def _ucb1(child: Node, parent_visits: int, c: float) -> float:
    """UCB1 score for `child` evaluated from its parent.

    Returns +inf for unvisited children to force initial exploration.
    """
    if child.visits == 0:
        return float("inf")

    raw_exploit = child.total_value / child.visits
    exploit = raw_exploit if child.side_to_move == child.parent.side_to_move else -1 * raw_exploit
    explore = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploit + explore


def _best_child_by_visits(node: Node, rng: random.Random) -> Node:
    """Return the most-visited child of `node`, breaking ties with `rng`.

    Visit count is more stable than mean value for final move selection —
    the most-visited child is the one UCT kept committing to.
    """
    visit_count = [child.visits for child in node.children]

    max_visits = max(visit_count)
    top = [child for child in node.children if child.visits == max_visits]
    return rng.choice(top)
