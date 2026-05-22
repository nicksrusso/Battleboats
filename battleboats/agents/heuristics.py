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

from battleboats.core.shipyard.ship_data import attack_modifier
from battleboats.core.shipyard.ship_type import ShipType

if TYPE_CHECKING:
    from battleboats.core.gameEngine import gameEngine
    from battleboats.core.shipyard.ship import Ship

# ----------------------------------------------------------------- weights
# Convex weighted sum: H = (Σ w_k · T_k) / Σ w_k.
W_HOME = 3.0  # win-condition primacy
W_COMBAT = 1.0
W_MAT = 1.0
W_ECON = 0.5

# ----------------------------------------------------------- tanh scales
# Each term's raw value is divided by its SCALE before tanh. Tune later by
# histogramming raw values across real game states (see doc, "Tuning").
SCALE_HOME = 150.0
SCALE_COMBAT = 500.0
SCALE_MAT = 2000.0
SCALE_ECON = 500.0

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

# ------------------------------------------------------------ mat term bonuses
MAT_LANDING_BONUS = 500.0  # premium for win-enabling units
MAT_BUILDER_BONUS = 100.0  # mild premium for economy enablers
MAT_PORT_VALUE = 400.0  # per-port flat credit


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


def _home_threat(home_pos: Tuple[int, int], attackers: Iterable["Ship"], manhattan) -> float:
    """Sum of distance-weighted threat applied to ``home_pos`` by ``attackers``.

    Ships whose type is in HOME_THREAT_NON_COMBAT_TYPES contribute zero.
    The remaining ships use ``HOME_THREAT_WEIGHT_LANDING`` for Landings
    (the only type that can actually capture) and ``stats.strength``
    otherwise; each contribution is divided by ``1 + manhattan(ship, home)``.
    """
    val = 0
    for a in attackers:
        if a.type in HOME_THREAT_NON_COMBAT_TYPES:
            continue
        if a.type is ShipType.LANDING:
            w = HOME_THREAT_WEIGHT_LANDING
        else:
            w = a.stats.strength
        val += w / (1 + manhattan(a.position, home_pos))
    return val


def _combat_balance(
    my_combat: Iterable["Ship"],
    opp_combat: Iterable["Ship"],
    kill_curve_k: float,
    manhattan,
) -> float:
    """Pairwise expected exchange value across combat ships.

    For each (i, j) in ``my_combat × opp_combat`` adds
    ``[pk(i, j) * i.strength - pk(j, i) * j.strength]
    / (1 + manhattan(i.position, j.position))``. Positive means net
    advantage to ``my_combat``. O(|my| * |opp|).
    """
    combatExpectation = 0
    for friend in my_combat:
        for foe in opp_combat:
            pkDiff = pk(friend, foe, kill_curve_k) * friend.stats.strength - pk(foe, friend, kill_curve_k) * foe.stats.strength
            combatExpectation = combatExpectation + pkDiff / (1 + manhattan(friend.position, foe.position))
    return combatExpectation


def _ship_value(ship: "Ship") -> float:
    """Material value of one ship: ``strength * (1 + range)`` plus type bonus.

    Range acts as a force multiplier (engage first, control more space).
    The ``1 +`` floor prevents range-0 transports from collapsing to zero
    so the LANDING / BUILDER bonuses layer on top meaningfully.
    """
    base = ship.stats.strength * (1 + ship.stats.attack_range)
    if ship.type is ShipType.LANDING:
        base += MAT_LANDING_BONUS
    elif ship.type is ShipType.BUILDER:
        base += MAT_BUILDER_BONUS
    return base


def heuristic_eval(engine: "gameEngine", me: int) -> float:
    """Zero-sum game-state evaluation in ``[-1, +1]`` from ``me``'s perspective.

    Returns +1.0 / -1.0 directly for terminal states. Otherwise composes
    four ``tanh``-normalized terms via convex weighted sum:

        T_HOME    home-port pressure (mine on theirs minus theirs on mine)
        T_COMBAT  pairwise expected exchange value over combat ships
        T_MAT     aggregate ship + port material differential
        T_ECON    liquid value: cash + port stockpiles + merchant cargo

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

    pressure_on_op_home = _home_threat(
        home_pos=opp_player.home_port,
        attackers=[engine.ships[sid] for sid in my_player.owned_ship_ids],
        manhattan=engine.map.manhattan,
    )
    pressure_on_my_home = _home_threat(
        home_pos=my_player.home_port,
        attackers=[engine.ships[sid] for sid in opp_player.owned_ship_ids],
        manhattan=engine.map.manhattan,
    )
    T_HOME = math.tanh((pressure_on_op_home - pressure_on_my_home) / SCALE_HOME)

    T_COMBAT = math.tanh(_combat_balance(my_combat, opp_combat, engine.kill_curve_k, engine.map.manhattan) / SCALE_COMBAT)

    opp_val = sum(_ship_value(s) for s in opp_ships) + MAT_PORT_VALUE * len(opp_player.owned_port_positions)
    friendly_val = sum(_ship_value(s) for s in my_ships) + MAT_PORT_VALUE * len(my_player.owned_port_positions)
    T_MAT = math.tanh((friendly_val - opp_val) / SCALE_MAT)

    my_liquid = (
        my_player.cash
        + sum(engine.ports[pos].stockpile for pos in my_player.owned_port_positions)
        + sum(s.cargo for s in my_ships if s.type is ShipType.MERCHANT)
    )
    opp_liquid = (
        opp_player.cash
        + sum(engine.ports[pos].stockpile for pos in opp_player.owned_port_positions)
        + sum(s.cargo for s in opp_ships if s.type is ShipType.MERCHANT)
    )
    T_ECON = math.tanh((my_liquid - opp_liquid) / SCALE_ECON)

    return (W_HOME * T_HOME + W_COMBAT * T_COMBAT + W_MAT * T_MAT + W_ECON * T_ECON) / (W_HOME + W_COMBAT + W_MAT + W_ECON)
