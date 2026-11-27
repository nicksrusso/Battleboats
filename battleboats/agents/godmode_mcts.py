"""God-mode MCTS agent — perfect-information UCT over the engine.

Used as a fixed benchmark opponent, not as a trainable agent. "God mode"
means the search operates on the true game state with no fog-of-war
filtering on either side. Leaf evaluation is the value net applied to the
fogged observation of ``side_at_leaf`` (the training distribution). Search
expansion still sees the true board.

Vanilla UCT (Kocsis & Szepesvári 2006):
    Each iteration walks the tree from the root via UCB1, expands one
    untried action at the reached leaf, evaluates the leaf with the value
    net (or a short optional random rollout first), and backpropagates
    the value. Final move = child of root with the most visits.

Negamax backup convention:
    Node.total_value is stored from the perspective of node.side_to_move.
    When backpropagating, sign-flip at each step up the tree.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from battleboats.core.actions import Action
    from battleboats.core.gameEngine import gameEngine
    from battleboats.training.value_net import ValueNet

from battleboats.agents.random_agent import random_action

DEFAULT_C = math.sqrt(2)
DEFAULT_ITERATIONS = 10000
ROLLOUT_STEP_BUDGET = 0  # engine steps before the value-net leaf; 0 = evaluate immediately

# Cache loaded nets per process so harvest/benchmark workers load once.
_NET_CACHE: dict[tuple[str, str], "ValueNet"] = {}


@dataclass
class Node:
    """One node in the MCTS search tree."""

    parent: Optional["Node"]
    action_in: Optional["Action"]  # action that led to this node from parent
    side_to_move: int  # whose turn it is AT this node
    untried_actions: List["Action"]  # legal actions not yet expanded into children
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0  # from the perspective of side_to_move (negamax)


def load_mcts_value_net(checkpoint: str | Path, device: str = "auto") -> "ValueNet":
    """Load a value-net checkpoint once per (path, device) in this process.

    ``device`` is ``cpu``, ``cuda``, ``cuda:N``, or ``auto`` (see
    ``battleboats.training.device.resolve_device``). The model is returned
    in eval mode on that device.
    """
    from battleboats.training.device import resolve_device
    from battleboats.training.value_net import load_value_checkpoint

    path = str(Path(checkpoint).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(f"value-net checkpoint not found: {path}")
    dev = resolve_device(device)
    key = (path, str(dev))
    model = _NET_CACHE.get(key)
    if model is None:
        model, _ = load_value_checkpoint(path, map_location=dev)
        model.to(dev)
        model.eval()
        _NET_CACHE[key] = model
    return model


def _run_search(
    engine: "gameEngine",
    player_id: int,
    rng: random.Random,
    iterations: int,
    c: float,
    value_net: "ValueNet",
) -> Node:
    """Run UCT for `iterations` and return the populated root node.

    Shared core of `godmode_mcts_action` and `godmode_mcts_search`. Returning
    the root lets callers extract either the best action, the mean root value
    (negamax-corrected to root's POV), the visit distribution, or all three.
    """
    root = Node(
        parent=None, action_in=None, side_to_move=player_id, untried_actions=list(engine.enumerate_legal(player_id=player_id))
    )

    for _ in range(iterations):
        sim = engine.clone()
        leaf = _select(root, sim, c)
        if not sim.is_terminal() and leaf.untried_actions:
            leaf = _expand(leaf, sim, rng)
        value = _evaluate(sim, leaf.side_to_move, rng, value_net)
        _backprop(leaf, value)

    return root


def godmode_mcts_action(
    engine: "gameEngine",
    player_id: int,
    rng: random.Random,
    value_net: "ValueNet",
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
        - rng is used for optional rollout action selection and tie-breaking.
        - value_net is the only leaf evaluator (load with load_mcts_value_net).
    """
    root = _run_search(engine, player_id, rng, iterations, c, value_net)
    return _best_child_by_visits(root, rng).action_in


def godmode_mcts_search(
    engine: "gameEngine",
    player_id: int,
    rng: random.Random,
    value_net: "ValueNet",
    *,
    iterations: int = DEFAULT_ITERATIONS,
    c: float = DEFAULT_C,
) -> tuple["Action", float]:
    """Run UCT and return (best_action, root_value).

    `root_value = root.total_value / root.visits` is the mean backed-up value
    at the root, in [-1, +1] from `player_id`'s POV (negamax convention —
    root.total_value is already stored from root.side_to_move's perspective,
    which equals `player_id`). This is the higher-quality, lower-variance
    label for value-function regression compared to the terminal MC return,
    bounded by current MCTS strength.
    """
    root = _run_search(engine, player_id, rng, iterations, c, value_net)
    root_value = root.total_value / root.visits if root.visits > 0 else 0.0
    return _best_child_by_visits(root, rng).action_in, root_value


def godmode_mcts_search_debug(
    engine: "gameEngine",
    player_id: int,
    rng: random.Random,
    value_net: "ValueNet",
    *,
    iterations: int = DEFAULT_ITERATIONS,
    c: float = DEFAULT_C,
) -> tuple["Action", float, List[dict]]:
    """Like `godmode_mcts_search`, but also returns the root child stats.

    The third element is `root_child_stats(root)` — the per-move (N, Q) table
    for offline inspection. Selection is unchanged (most-visited child), so the
    chosen action still equals the argmax-visits row. Only call this on the
    flag-gated debug path; building the table is cheap but logging it is not.
    """
    root = _run_search(engine, player_id, rng, iterations, c, value_net)
    root_value = root.total_value / root.visits if root.visits > 0 else 0.0
    return _best_child_by_visits(root, rng).action_in, root_value, root_child_stats(root)


def root_child_stats(root: Node) -> List[dict]:
    """Per-root-child search summary, visit-sorted (the search's own ranking).

    The standard MCTS-debugging artifact: one row per root move with the
    visit count N (the policy — argmax-N is the move actually selected) and
    the mean backed-up value Q from the ROOT's POV (the value — why). Read
    them together: N is the decision, Q is the justification, and a gap
    between argmax-N and argmax-Q is itself diagnostic. The raw `Action`
    object is returned so the caller can label/serialize it however it
    already does (we don't reach into action formatting here).

    `q_root_pov` negates Q for children whose side-to-move differs from the
    root's (negamax), so every Q is comparable on one +favours-root scale.
    """
    out: List[dict] = []
    for child in sorted(root.children, key=lambda c: -c.visits):
        q_child_pov = child.total_value / child.visits if child.visits else 0.0
        sign = 1 if child.side_to_move == root.side_to_move else -1
        out.append({"action": child.action_in, "visits": child.visits, "q_root_pov": sign * q_child_pov})
    return out


def _debug_print_root(root: Node) -> None:
    """Print top-10 root children by visit count, with Q from root's POV.

    Used to diagnose the search at the root. If a class of actions (e.g.,
    all BuildLanding spawn variants) gets fragmented across many children,
    per-child visits stay low even when the class collectively dominates.
    """
    if not root.children:
        return
    stats = root_child_stats(root)
    print(f"    [mcts root] {len(root.children)} children, {sum(c.visits for c in root.children)} total visits")
    for s in stats[:10]:
        print(f"    [mcts root]   visits={s['visits']:4d}  Q={s['q_root_pov']:+.3f}  {s['action']}")


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


def _evaluate(engine: "gameEngine", side_at_leaf: int, rng: random.Random, value_net: "ValueNet") -> float:
    """Optional step-capped random rollout, then the value-net leaf.

    Plays uniform-random actions for up to ROLLOUT_STEP_BUDGET engine
    steps or until terminal. Terminal positions short-circuit to ±1 / 0
    from ``side_at_leaf``'s point of view and never call the net.
    Otherwise the leaf is the value net on the fogged observation of
    ``side_at_leaf``.

    Mutates `engine`; caller must already hold the clone they care about.
    """
    from battleboats.envs.observation import build_observation
    from battleboats.training.value_net import value_of_observation

    step_count = 0
    while not engine.is_terminal() and step_count < ROLLOUT_STEP_BUDGET:
        act = random_action(engine=engine, player_id=engine.current_player, rng=rng)
        engine.step(act)
        step_count += 1
    if engine.is_terminal():
        winner = engine.winner
        if winner is None:
            return 0.0
        return 1.0 if winner == side_at_leaf else -1.0
    obs = build_observation(engine, side_at_leaf)
    return value_of_observation(value_net, obs["entity_tokens"], obs["globals"])


def _backprop(leaf: Node, value_at_leaf: float) -> None:
    """Propagate `value_at_leaf` from the leaf up to the root.

    `value_at_leaf` is in [-1, +1] from `leaf.side_to_move`'s perspective
    (as returned by _evaluate). Walk up the tree from `leaf`; at each
    node, increment visits and add the value translated to that node's
    side-to-move (same sign if sides match, negated if they differ).

    Sides only flip across EndTurn boundaries, so the per-node `==` check
    correctly handles multi-action turns.
    """
    node: Optional[Node] = leaf
    while node is not None:
        node.visits += 1
        sign = 1.0 if node.side_to_move == leaf.side_to_move else -1.0
        node.total_value += sign * value_at_leaf
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
    """Return the most-visited child of `node`; break visit ties on mean Q.

    Visit count is more stable than mean value for final move selection —
    the most-visited child is the one UCT kept committing to. But when the
    action space is large enough that vanilla UCT can't break ties via
    exploration (i.e. branching factor ≥ iteration budget so every child
    gets the same number of visits), falling back to mean Q lets the
    value net's leaf signal decide instead of a coin flip.
    """
    max_visits = max(child.visits for child in node.children)
    top = [child for child in node.children if child.visits == max_visits]
    if len(top) == 1:
        return top[0]

    def q_from_root_pov(child: Node) -> float:
        q = child.total_value / child.visits
        return q if child.side_to_move == node.side_to_move else -q

    max_q = max(q_from_root_pov(child) for child in top)
    best = [child for child in top if q_from_root_pov(child) >= max_q - 1e-12]
    return rng.choice(best)
