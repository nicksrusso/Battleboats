"""Hand-coded heuristic eval for use as the MCTS leaf evaluator.

Full design rationale: docs/heuristic_design.md.

Four terms, each computed symmetrically (me − opponent), each tanh-normalized
to roughly [-1, +1], combined via convex weighted sum to keep the final
heuristic bounded in [-1, +1]:

    T_HOME    — home-port pressure, both directions
    T_COMBAT  — pairwise tactical advantage with asymmetric PKs
    T_MAT     — aggregate material differential
    T_ECON    — liquid value (cash + stockpiles + cargo)

Terminal states bypass the heuristic and return ±1 directly.
"""

import math
from typing import TYPE_CHECKING, Iterable, Tuple

from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.shipyard.ship_data import attack_modifier
from battleboats.core.shipyard.ship_type import ShipType

if TYPE_CHECKING:
    from battleboats.core.gameEngine import gameEngine
    from battleboats.core.shipyard.ship import Ship

# ----------------------------------------------------------------- weights
# Convex weighted sum: H = (Σ w_k · T_k) / Σ w_k.
W_HOME = 10.0  # win-condition primacy
W_COMBAT = 1.0
W_MAT = 1.0
W_ECON = 5.0  # bumped to make merchant positional gradient meaningful in dH

# ----------------------------------------------------------- tanh scales
# Each term's raw value is divided by its SCALE before tanh. Tune later by
# histogramming raw values across real game states (see doc, "Tuning").
SCALE_HOME = 500.0
SCALE_COMBAT = 500.0
SCALE_MAT = 2000.0
SCALE_ECON = 1500.0  # widened to accommodate merchant logistics shaping bonus

# ------------------------------------------------------------ ship buckets
# Types that contribute to the pairwise combat term.
COMBAT_TYPES = frozenset(
    {
        ShipType.CARRIER,
        ShipType.BATTLESHIP,
        ShipType.CRUISER,
        ShipType.DESTROYER,
        ShipType.SUBMARINE,
    }
)

# Types that contribute zero threat to a home port (no offensive role).
HOME_THREAT_NON_COMBAT_TYPES = frozenset(
    {
        ShipType.MERCHANT,
        ShipType.BUILDER,
    }
)

# Home-threat weight overrides per type (otherwise: ship.stats.strength).
HOME_THREAT_WEIGHT_LANDING = 250  # only ship type that can actually capture

# Distance kernels — the two distance-using terms model different things
# and therefore use different kernel shapes:
#
# T_HOME (navigation / pressure incentive). Spatial kernel
#     proximity = HOME_KERNEL_BASE ** (d / char_dist)
# uses a flatter base (0.8) so the gradient remains usable far from the
# target; with 0.5 the signal vanishes beyond ~2 characteristic distances
# and MCTS can't see any benefit to closing in from the other side of
# the map. char_dist is map-scaled: full diagonal = HOME_K_PER_DIAGONAL
# units of k, so kernel value at the diagonal = HOME_KERNEL_BASE ** HOME_K_PER_DIAGONAL.
HOME_KERNEL_BASE = 0.8
HOME_K_PER_DIAGONAL = 4.0

# T_COMBAT (engagement physics). Temporal kernel
#     proximity = 0.5 ** (turns_to_engage / COMBAT_K_TURNS)
# where turns_to_engage = d / (friend.speed + foe.speed). This makes the
# kernel an honest per-turn discount factor — the shape value functions
# in MDPs have — and correctly captures that fast ships project threat
# farther in space than slow ones. Half-life is COMBAT_K_TURNS turns of
# pair-closing travel.
COMBAT_K_TURNS = 20.0

# ------------------------------------------------------------ mat term bonuses
MAT_LANDING_BONUS = 500.0  # premium for win-enabling units
MAT_BUILDER_BONUS = 100.0  # mild premium for economy enablers
MAT_MERCHANT_BONUS = 300.0  # ≈ one ferry round trip's net cash value; offsets
# the 100-cash build cost so MCTS sees a positive build-time delta even when
# no port has stockpile yet (turn-0 state where V(empty merchant) ≈ 0).
MAT_PORT_VALUE = 400.0  # per-port flat credit

# ------------------------------------------------------------ econ logistics shaping
# Per-merchant positional shaping bonus. Linear proximity kernel (no decay)
# so MCTS sees uniform per-tile gradient toward the next useful action across
# the whole map — no plateau like an exponential kernel would create.
#
#   V(merchant) = f_empty  * ECON_W_OUT * p_to_nearest_loading_port
#               + f_loaded * ECON_W_IN  * p_to_home_port
#
# ECON_W_IN > ECON_W_OUT makes the delivery leg dominate (full-merchant
# movement matters more than empty-merchant movement). ECON_W_IN must stay
# *below* the conversion gain at unload (CASH_PER_MATERIAL × CAPACITY ≈ 100)
# so that "unload at home" is always the highest-delta single action.
ECON_W_OUT = 10.0
ECON_W_IN = 50.0

# Cargo carries higher weight than stockpile in T_ECON_raw — stockpile sitting
# at a non-home port is locked (can't be cash without a merchant), while cargo
# on a merchant is in transit toward the 2× home-port conversion. Making
# W_CARGO > 1 ensures a Load action shows a net positive ΔT_ECON instead of
# the cargo↔stockpile swap netting to zero. Sized so a 25-cargo load beats
# the +50 cash EndTurn delivers via the income tick.
W_CARGO = 4.0


def pk(attacker: "Ship", defender: "Ship", kill_curve_k: float) -> float:
    """Probability that ``attacker`` destroys ``defender`` in one attack.

    Mirrors gameEngine._resolve_attack's formula without consuming RNG:
    ``x = (atk.strength * matchup_modifier) / def.strength``,
    ``pk = x**k / (1 + x**k)``. Returns 1.0 when defender strength <= 0,
    matching the engine's "always-destroyed" edge case.
    """
    if defender.stats.strength <= 0:
        return 1.0  # mirror engine._resolve_attack: strength-0 defender always destroyed

    x = attacker.stats.strength * attack_modifier(attacker=attacker.type, defender=defender.type) / defender.stats.strength
    return x**kill_curve_k / (1 + x**kill_curve_k)


def _home_threat(
    home_pos: Tuple[int, int],
    attackers: Iterable["Ship"],
    manhattan,
    char_dist: float,
) -> float:
    """Sum of distance-weighted threat applied to ``home_pos`` by ``attackers``.

    Ships whose type is in HOME_THREAT_NON_COMBAT_TYPES contribute zero.
    The remaining ships use ``HOME_THREAT_WEIGHT_LANDING`` for Landings
    (the only type that can actually capture) and ``stats.strength``
    otherwise; contribution is scaled by the spatial proximity kernel
    ``HOME_KERNEL_BASE ** (distance / char_dist)``. The flatter base
    preserves a usable gradient at long range so MCTS can plan toward
    a distant home even when no ship is close.
    """
    val = 0.0
    for a in attackers:
        if a.type in HOME_THREAT_NON_COMBAT_TYPES:
            continue
        if a.type is ShipType.LANDING:
            w = HOME_THREAT_WEIGHT_LANDING
        else:
            w = a.stats.strength
        proximity = HOME_KERNEL_BASE ** (manhattan(a.position, home_pos) / char_dist)
        val += w * proximity
    return val


def _combat_balance(
    my_combat: Iterable["Ship"],
    opp_combat: Iterable["Ship"],
    kill_curve_k: float,
    manhattan,
    k_turns: float,
) -> float:
    """Sum of proximity-weighted *advantageous* expected exchanges.

    For each (i, j) in ``my_combat × opp_combat``:
        pkDiff = pk(i, j) * i.strength - pk(j, i) * j.strength
        proximity = 0.5 ** (turns_to_engage / k_turns)
        turns_to_engage = distance / (i.speed + j.speed)
    Contribution to the term is ``max(pkDiff, 0) * proximity`` — i.e.,
    only matchups where ``i`` has the favorable trade contribute.

    Why the floor? With raw ``pkDiff * proximity``, an outmatched fleet
    is rewarded for retreating (negative pkDiff muted by low proximity).
    That produces a stable-but-passive equilibrium far from the enemy.
    Flooring at zero means distance only modulates engagements you'd
    win; losing trades are not "improved" by backing off. Defensive
    awareness still lives in ``T_MAT`` (raw material differential).

    Tradeoff — the heuristic is no longer strictly zero-sum across
    players: both sides can simultaneously report positive ``T_COMBAT``
    (each counting its own favorable matchups). godmode_mcts backs up
    with negamax (assumes h(s, p0) = -h(s, p1)), so the backup is
    slightly biased — each side overestimates the other's leaf value
    by roughly the sum of mutual favorable matchups. Bounded by tanh,
    so the error is small in absolute terms; flagged here because it's
    a deliberate break.

    The exponential plays the role of a per-turn discount factor on
    expected exchange value — the shape MDP value functions have.
    O(|my| * |opp|).
    """
    combatExpectation = 0.0
    for friend in my_combat:
        for foe in opp_combat:
            pkDiff = pk(friend, foe, kill_curve_k) * friend.stats.strength - pk(foe, friend, kill_curve_k) * foe.stats.strength
            if pkDiff <= 0:
                continue
            combined_speed = friend.stats.speed + foe.stats.speed
            turns_to_engage = manhattan(friend.position, foe.position) / combined_speed
            proximity = 0.5 ** (turns_to_engage / k_turns)
            combatExpectation += pkDiff * proximity
    return combatExpectation


def _ship_value(ship: "Ship") -> float:
    """Material value of one ship: ``strength * (1 + range)`` plus type bonus.

    Range acts as a force multiplier (engage first, control more space).
    The ``1 +`` floor prevents range-0 transports from collapsing to zero
    so the LANDING / BUILDER / MERCHANT bonuses layer on top meaningfully.
    """
    base = ship.stats.strength * (1 + ship.stats.attack_range)
    if ship.type is ShipType.LANDING:
        base += MAT_LANDING_BONUS
    elif ship.type is ShipType.BUILDER:
        base += MAT_BUILDER_BONUS
    elif ship.type is ShipType.MERCHANT:
        base += MAT_MERCHANT_BONUS
    return base


def _merchant_logistics_value(
    my_ships: Iterable["Ship"],
    owned_port_positions,
    home_port: Tuple[int, int],
    manhattan,
    map_diag: float,
) -> float:
    """Sum over my merchants of per-merchant positional shaping value.

    Hard phase switch on cargo level (no smooth blend):
        if cargo < MERCHANT_CAPACITY:  V(m) = ECON_W_OUT * p_load
        else (cargo == CAPACITY):      V(m) = ECON_W_IN  * p_home

    The blend version was mathematically incapable of "stay at port until
    full" — at any finite weight ratio, sufficiently-loaded merchants
    always saw positive ΔV by leaving. The hard switch makes leaving a
    partial-cargo state strictly worse (gradient still points at the
    nearest loading port) and creates a discrete V jump at the moment
    of becoming full — when the merchant should change phase.

    Loading targets include ALL owned non-home ports (no stockpile
    filter). A drained port still pulls an empty merchant because the
    port will re-produce 25 stockpile per turn — the merchant should
    park there, not wander away.

    p_home = max(0, 1 − d(m, home_port) / map_diag)
    p_load = max over owned non-home ports of max(0, 1 − d(m, port) / map_diag)

    Non-merchant ships contribute zero.
    """
    if not owned_port_positions:
        return 0.0

    loading_ports = [pos for pos in owned_port_positions if pos != home_port]

    total = 0.0
    for m in my_ships:
        if m.type is not ShipType.MERCHANT:
            continue

        if m.cargo < MERCHANT_CAPACITY:
            # Outbound phase: pull toward any owned non-home port.
            if loading_ports:
                best_d = min(manhattan(m.position, p) for p in loading_ports)
                p_load = max(0.0, 1.0 - best_d / map_diag)
                total += ECON_W_OUT * p_load
        else:
            # Inbound phase: deliver to home.
            d_home = manhattan(m.position, home_port)
            p_home = max(0.0, 1.0 - d_home / map_diag)
            total += ECON_W_IN * p_home

    return total


def heuristic_eval(engine: "gameEngine", me: int) -> float:
    """Zero-sum game-state evaluation in ``[-1, +1]`` from ``me``'s perspective.

    Returns +1.0 / -1.0 directly for terminal states. Otherwise composes
    four ``tanh``-normalized terms via convex weighted sum:

        T_HOME    home-port pressure (mine on theirs minus theirs on mine)
        T_COMBAT  pairwise expected exchange value over combat ships
        T_MAT     aggregate ship + port material differential
        T_ECON    one-sided liquid value + merchant logistics shaping bonus
                  (cash + stockpiles + cargo + per-merchant positional V)

        H = (W_HOME * T_HOME + W_COMBAT * T_COMBAT
             + W_MAT  * T_MAT  + W_ECON   * T_ECON)
            / (W_HOME + W_COMBAT + W_MAT + W_ECON)

    See docs/heuristic_design.md for the design rationale and tuning notes.
    """

    if engine.is_terminal():
        if engine.winner == me:
            return 1.0
        else:
            return -1.0

    opp_player = engine.players[1 - me]
    my_player = engine.players[me]

    opp_ships = [engine.ships[sid] for sid in opp_player.owned_ship_ids]
    my_ships = [engine.ships[sid] for sid in my_player.owned_ship_ids]

    opp_combat = [s for s in opp_ships if s.type in COMBAT_TYPES]
    my_combat = [s for s in my_ships if s.type in COMBAT_TYPES]

    # Map-scaled characteristic distance for the spatial home-pressure kernel.
    home_char_dist = (engine.map.width + engine.map.height) / HOME_K_PER_DIAGONAL

    pressure_on_op_home = _home_threat(
        home_pos=opp_player.home_port,
        attackers=my_ships,
        manhattan=engine.map.manhattan,
        char_dist=home_char_dist,
    )
    pressure_on_my_home = _home_threat(
        home_pos=my_player.home_port,
        attackers=opp_ships,
        manhattan=engine.map.manhattan,
        char_dist=home_char_dist,
    )
    T_HOME = math.tanh((pressure_on_op_home - pressure_on_my_home) / SCALE_HOME)

    T_COMBAT = math.tanh(
        _combat_balance(my_combat, opp_combat, engine.kill_curve_k, engine.map.manhattan, COMBAT_K_TURNS) / SCALE_COMBAT
    )

    opp_val = sum(_ship_value(s) for s in opp_ships) + MAT_PORT_VALUE * len(opp_player.owned_port_positions)
    friendly_val = sum(_ship_value(s) for s in my_ships) + MAT_PORT_VALUE * len(my_player.owned_port_positions)
    T_MAT = math.tanh((friendly_val - opp_val) / SCALE_MAT)

    # T_ECON is one-sided (does not subtract opponent liquidity) and includes
    # a per-merchant positional shaping bonus so the merchant logistics
    # pipeline (build → outbound → load → inbound → unload) has gradient
    # at every step instead of only at the final unload. Same negamax
    # bias caveat as T_COMBAT applies — h(s, p0) + h(s, p1) is no longer
    # exactly zero, but the magnitude is small.
    map_diag = engine.map.width + engine.map.height
    econ_raw = (
        my_player.cash
        + sum(engine.ports[pos].stockpile for pos in my_player.owned_port_positions)
        + W_CARGO * sum(s.cargo for s in my_ships if s.type is ShipType.MERCHANT)
        + _merchant_logistics_value(
            my_ships=my_ships,
            owned_port_positions=my_player.owned_port_positions,
            home_port=my_player.home_port,
            manhattan=engine.map.manhattan,
            map_diag=map_diag,
        )
    )
    T_ECON = math.tanh(econ_raw / SCALE_ECON)

    return (W_HOME * T_HOME + W_COMBAT * T_COMBAT + W_MAT * T_MAT + W_ECON * T_ECON) / (W_HOME + W_COMBAT + W_MAT + W_ECON)
