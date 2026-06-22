"""Hand-coded heuristic eval — clean feature-vector formulation.

Designed as the bootstrap evaluator for the value-network learning loop.
Each feature is an independently-monotonic scalar describing one aspect of
state from `me`'s perspective. The heuristic is the weighted sum of features,
wrapped in tanh to stay in [-1, +1].

Design choices:
  - Linear proximity kernel everywhere (no exponential decay). Whatever
    curvature actually matters can be recovered by the learner.
  - Asymmetric features where natural (combat win-only, merchant cargo on
    my side). Negamax backup will be slightly biased; acceptable for a
    bootstrap heuristic.
  - DEFAULT_WEIGHTS are first-cut values. The next phase replaces them
    with learned weights from regression on MCTS root values.

Terminal states bypass `features()` and return ±1 directly.
"""

import math
from typing import TYPE_CHECKING, Dict, Tuple

from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.shipyard.ship_data import attack_modifier
from battleboats.core.shipyard.ship_type import ShipType
import json

if TYPE_CHECKING:
    from battleboats.core.gameEngine import gameEngine
    from battleboats.core.shipyard.ship import Ship


# ----------------------------------------------------------- material bonuses
MAT_LANDING_BONUS = 500.0
MAT_BUILDER_BONUS = 100.0
MAT_MERCHANT_BONUS = 300.0
MAT_PORT_VALUE = 400.0

# Diminishing-returns bonus for owning merchants. Stacks on top of the flat
# per-merchant MAT_MERCHANT_BONUS, but applied as K × √n so the 1st merchant
# is worth K, the 2nd adds K(√2−1) ≈ 0.41K, the 3rd adds ≈ 0.32K, etc.
# Concave by construction so the heuristic prefers "1–3 merchants" over
# "build 10 of them and ignore everything else."
MERCHANT_COUNT_VALUE_K = 1000.0

# ------------------------------------------------------------- ship buckets
COMBAT_TYPES = frozenset(
    {
        ShipType.CARRIER,
        ShipType.BATTLESHIP,
        ShipType.CRUISER,
        ShipType.DESTROYER,
        ShipType.SUBMARINE,
    }
)

HOME_THREAT_NON_COMBAT_TYPES = frozenset(
    {
        ShipType.MERCHANT,
        ShipType.BUILDER,
    }
)

HOME_THREAT_WEIGHT_LANDING = 250  # Landings are the only win-enabling unit

# Combat ships within this Manhattan distance of a home port add NO home pressure.
# Bordering a hostile home with combat ships neither captures it (only a Landing
# can) nor helps — and it can wall off your own Landing's approach. Zeroing the
# adjacent contribution discourages crowding the doorstep while still rewarding
# closing in (a ship at distance 2 now scores more pressure than one at distance 1).
# Landings are exempt: adjacency is exactly where the capture unit wants to be.
HOME_PRESSURE_ADJ_CUTOFF = 1

# ----------------------------------------------------------- per-type schema
# Stable ordering for per-type features (`own_<name>` / `opp_<name>`).
# Order is fixed so the column layout in φ stays consistent across runs —
# regression weights are keyed by name, but downstream tooling that flattens
# φ dicts to arrays still benefits from a deterministic iteration order.
SHIP_TYPE_ORDER: Tuple[ShipType, ...] = (
    ShipType.CARRIER,
    ShipType.BATTLESHIP,
    ShipType.CRUISER,
    ShipType.DESTROYER,
    ShipType.SUBMARINE,
    ShipType.LANDING,
    ShipType.BUILDER,
    ShipType.MERCHANT,
)
SHIP_TYPE_FEATURE_NAME: Dict[ShipType, str] = {
    ShipType.CARRIER: "carriers",
    ShipType.BATTLESHIP: "battleships",
    ShipType.CRUISER: "cruisers",
    ShipType.DESTROYER: "destroyers",
    ShipType.SUBMARINE: "submarines",
    ShipType.LANDING: "landings",
    ShipType.BUILDER: "builders",
    ShipType.MERCHANT: "merchants",
}

# ----------------------------------------------------------- econ_value_self params
# `econ_value_self` consolidates cash + stockpile + cargo + loading-capacity
# into a single coherent economic feature. Each form of raw material is
# counted exactly once, at a multiplier reflecting how close it is to being
# convertible cash:
#     cash:            value 1× (it IS cash)
#     stockpile:       value 1× (locked at non-home, but recoverable)
#     cargo:           value 1 + ECON_ALPHA × p_home (premium for proximity to delivery)
#     loading capacity: value ECON_BETA × p_load     (small bonus for "can soon become cargo")
# ECON_ALPHA must be < CASH_PER_MATERIAL − 1 = 1.0 so unload-at-home stays
# positive (cargo at home worth < 2 cash so converting yields net gain).
# ECON_BETA is small — it exists only to give empty merchants a gradient
# toward ports, not to dominate the material-value accounting.
ECON_ALPHA = 0.5
ECON_BETA = 0.1

# ----------------------------------------------------------- default weights
# Each value is `weight / per-feature-scale` collapsed into one number, so
# DEFAULT_WEIGHTS[k] * features[k] is a term in the pre-tanh sum directly.
# The learning loop will replace this dict with the output of regression
# on (features, mcts_root_value) pairs; don't over-tune these defaults.
with open("/home/nick/Desktop/repos/Battleboats/runs/weights/v6_handfit.json", "r") as f:
    regression_data = json.load(f)
DEFAULT_WEIGHTS = regression_data["weights"]

# Per-state bias term learned by v3's intercept. Not currently applied in
# heuristic_eval (tanh(Σ w·φ) only). If fidelity matters, add it inside the
# tanh; magnitude (~0.34) shifts some states meaningfully. Acceptable for now.
DEFAULT_INTERCEPT = regression_data["intercept"]

FEATURE_KEYS: Tuple[str, ...] = tuple(DEFAULT_WEIGHTS.keys())


def pk(attacker: "Ship", defender: "Ship", kill_curve_k: float) -> float:
    """Probability that `attacker` destroys `defender` in one attack.

    Mirrors gameEngine._resolve_attack's formula without consuming RNG.
    Returns 1.0 when defender.strength <= 0 (engine treats a strength-0
    defender as always-destroyed).
    """
    if defender.stats.strength <= 0:
        return 1.0
    x = attacker.stats.strength * attack_modifier(attacker=attacker.type, defender=defender.type) / defender.stats.strength
    return x**kill_curve_k / (1 + x**kill_curve_k)


def _ship_value(ship: "Ship") -> float:
    """Material value of one ship: strength*(1+range) plus type bonus."""
    base = ship.stats.strength * (1 + ship.stats.attack_range)
    if ship.type is ShipType.LANDING:
        base += MAT_LANDING_BONUS
    elif ship.type is ShipType.BUILDER:
        base += MAT_BUILDER_BONUS
    elif ship.type is ShipType.MERCHANT:
        base += MAT_MERCHANT_BONUS
    return base


def features(engine: "gameEngine", me: int) -> Dict[str, float]:
    """φ(s) — fixed-key dict of scalar features from `me`'s perspective.

    Sign convention throughout: positive favors `me`. Should not be called
    on terminal states; callers short-circuit to ±1 from `engine.winner`.

    Features (see DEFAULT_WEIGHTS for the canonical key list):
        material_diff           ship + port material differential (zero-sum)
        home_pressure_diff      Σ_i strength_i × p_to_opp_home over my ships,
                                minus the symmetric term over opp ships (zero-sum)
        combat_balance          For each opp combat ship, the AVERAGE of
                                pkDiff × proximity across my combat ships
                                with favorable matchup against it; summed over
                                opp ships. Per-opp contribution is bounded by
                                a single ship's pkDiff so realizing a kill via
                                attack costs at most one ship's-worth of
                                forecast against the +_ship_value gain in
                                material_diff. Building a new combat ship
                                doesn't dilute existing combat_balance.
        econ_value_self         my.cash
                                + Σ my stockpile_at_owned_ports
                                + Σ over my merchants of [
                                    cargo × (1 + ECON_ALPHA × p_to_home)
                                    + (CAP - cargo) × p_to_nearest_non_home × ECON_BETA
                                  ]
                                Consolidated economic value — cash, stockpile, and
                                cargo are all in one feature with consistent units,
                                so load (stockpile → cargo) and unload (cargo → cash)
                                produce the correct directional deltas. One-sided.
        has_landing_self        1.0 if I own ≥1 Landing else 0.0. Discrete
                                capability gate — without a Landing the player
                                literally cannot capture the enemy home.
        landing_pressure_self   Σ over my Landings of p_to_opp_home (one-sided)

        --- iteration-2 features (added at weight=0; regression assigns real
        weights from harvest data) ---

        own_<type>              Count of `me`'s ships of each type (8 features:
        opp_<type>              carriers, battleships, cruisers, destroyers,
                                submarines, landings, builders, merchants).
                                Opponent's symmetric counts (8 features).
                                Together these let the regression learn
                                fleet-composition preferences that `material_diff`
                                aggregates away.

        combat_total_overmatch  Σ max(0, M[i,j]) over my×opp combat-ship pairs,
                                where M[i,j] = pk(i→j)*str(i) - pk(j→i)*str(j).
                                Raw matchup magnitude (no proximity weighting —
                                combat_balance already provides that).
        combat_coverage_min     min_j max_i M[i,j] — for each opp combat ship,
                                my best counter; take min across opp ships.
                                Negative / small = I'm vulnerable to at least
                                one opp ship type with no good answer.
        combat_uncovered_count  Number of opp combat ships for which I have
                                NO favorable counter (max_i M[i,j] ≤ 0).
                                Discrete coverage-gap indicator.
    """
    me_player = engine.players[me]
    opp_player = engine.players[1 - me]

    my_ships = [engine.ships[sid] for sid in me_player.owned_ship_ids]
    opp_ships = [engine.ships[sid] for sid in opp_player.owned_ship_ids]

    map_diag = engine.map.width + engine.map.height
    manhattan = engine.map.manhattan
    home = me_player.home_port
    opp_home = opp_player.home_port

    # material_diff
    my_mat = sum(_ship_value(s) for s in my_ships) + MAT_PORT_VALUE * len(me_player.owned_port_positions)
    opp_mat = sum(_ship_value(s) for s in opp_ships) + MAT_PORT_VALUE * len(opp_player.owned_port_positions)
    material_diff = my_mat - opp_mat

    # home_pressure_diff (linear proximity, mine on opp home minus opp on my home).
    def _pressure_on(target_pos, attackers):
        total = 0.0
        for s in attackers:
            if s.type in HOME_THREAT_NON_COMBAT_TYPES:
                continue
            is_landing = s.type is ShipType.LANDING
            d = manhattan(s.position, target_pos)
            # Combat ships bordering the home add nothing — don't reward crowding
            # the doorstep (it can't capture and may block the Landing). Landings
            # are exempt; adjacency is where they need to be.
            if not is_landing and d <= HOME_PRESSURE_ADJ_CUTOFF:
                continue
            w = HOME_THREAT_WEIGHT_LANDING if is_landing else s.stats.strength
            p = max(0.0, 1.0 - d / map_diag)
            total += w * p
        return total

    home_pressure_diff = _pressure_on(opp_home, my_ships) - _pressure_on(home, opp_ships)

    # combat_balance — win-only, linear proximity, averaged per opp ship.
    #
    # For each opp combat ship, take the AVERAGE pkDiff × proximity across
    # my combat ships that have a favorable matchup against it. Per-opp
    # contribution is bounded by max single-pair value (~ship strength)
    # instead of multiplied by fleet size. This makes "kill opp ship X"
    # roughly trade `+_ship_value(X)` in material_diff for `-avg_value(X)`
    # in combat_balance — a net positive when properly weighted, with no
    # ganging-up forecast inflation.
    #
    # Per-opp averaging (vs dividing by N_my_combat) means building a
    # new combat ship doesn't dilute existing combat_balance — only the
    # ships that *can favorably fight a given opp* count toward that
    # opp's average.
    my_combat = [s for s in my_ships if s.type in COMBAT_TYPES]
    opp_combat = [s for s in opp_ships if s.type in COMBAT_TYPES]
    combat_balance = 0.0
    for foe in opp_combat:
        favorable_sum = 0.0
        favorable_count = 0
        for friend in my_combat:
            pkDiff = (
                pk(friend, foe, engine.kill_curve_k) * friend.stats.strength
                - pk(foe, friend, engine.kill_curve_k) * foe.stats.strength
            )
            if pkDiff <= 0:
                continue
            p = max(0.0, 1.0 - manhattan(friend.position, foe.position) / map_diag)
            favorable_sum += pkDiff * p
            favorable_count += 1
        if favorable_count > 0:
            combat_balance += favorable_sum / favorable_count

    # econ_value_self — one consolidated economic feature spanning cash,
    # stockpile, cargo, and "near a port" capacity, with cargo getting a
    # proximity-to-home premium and capacity getting a small ECON_BETA boost.
    my_stockpile = sum(engine.ports[pos].stockpile for pos in me_player.owned_port_positions)
    my_merchants = [s for s in my_ships if s.type is ShipType.MERCHANT]
    non_home_ports = [pos for pos in me_player.owned_port_positions if pos != home]
    cargo_value = 0.0
    capacity_value = 0.0
    for m in my_merchants:
        p_home = max(0.0, 1.0 - manhattan(m.position, home) / map_diag)
        cargo_value += m.cargo * (1.0 + ECON_ALPHA * p_home)
        if non_home_ports:
            best_d = min(manhattan(m.position, p) for p in non_home_ports)
            p_load = max(0.0, 1.0 - best_d / map_diag)
            capacity_value += (MERCHANT_CAPACITY - m.cargo) * p_load * ECON_BETA
    econ_value_self = float(me_player.cash) + float(my_stockpile) + cargo_value + capacity_value

    # landing_pressure_self — Landings are the only ship type that can actually
    # win the game; reward proximity to opp home independently of their strength
    # contribution (which is zero — Landing has strength=0).
    my_landings = [s for s in my_ships if s.type is ShipType.LANDING]
    landing_pressure_self = sum(max(0.0, 1.0 - manhattan(s.position, opp_home) / map_diag) for s in my_landings)

    # landing_danger_self — penalize my Landings sitting within striking range of
    # enemy combat ships. Landings are defenseless (strength 0) and get intercepted
    # en route to the objective; this gives short-horizon MCTS a gradient to route
    # AROUND threats instead of marching into them. Reach-gated: only enemies that
    # could actually reach + fire this turn (move speed + attack_range) count, so a
    # Landing approaches freely until an interceptor is in striking distance — that
    # discourages suicide runs without making Landings too timid to ever close.
    # One-sided (positive = more danger to me); pair with a NEGATIVE weight.
    landing_danger_self = 0.0
    for L in my_landings:
        for e in opp_ships:
            if e.stats.attack_range <= 0:
                continue  # can't shoot (merchant / builder / landing)
            reach = e.stats.speed + e.stats.attack_range
            d = manhattan(e.position, L.position)
            if d <= reach:
                landing_danger_self += e.stats.strength * max(0.0, 1.0 - d / map_diag)

    # has_landing_self — discrete capability indicator. Separated from
    # landing_pressure_self because "do I own one" and "how close is it"
    # answer different questions and deserve independently-learnable weights.
    has_landing_self = 1.0 if my_landings else 0.0

    # merchant_count_value_self — concave diminishing-returns bonus so the
    # heuristic prefers building the first few merchants strongly, with each
    # subsequent merchant contributing less.
    merchant_count_value_self = MERCHANT_COUNT_VALUE_K * math.sqrt(len(my_merchants))

    # ----- per-type counts (16 features) ------------------------------------
    # Tally each side's ship inventory per type. Counts initialized to 0 for
    # every type in SHIP_TYPE_ORDER so the returned dict always has all keys
    # even when a side owns zero of a type.
    own_counts: Dict[ShipType, int] = {t: 0 for t in SHIP_TYPE_ORDER}
    for ship in my_ships:
        own_counts[ship.type] += 1
    opp_counts: Dict[ShipType, int] = {t: 0 for t in SHIP_TYPE_ORDER}
    for ship in opp_ships:
        opp_counts[ship.type] += 1

    # ----- matchup matrix features (3 features) -----------------------------
    # Raw pairwise matchup (no proximity weighting — distinct signal from
    # combat_balance which is proximity-weighted). M[i,j] is "expected damage
    # my ship i does to opp j minus expected damage j does back."
    #
    # combat_total_overmatch is the sum of positive M[i,j] entries — overall
    # favorability potential. combat_coverage_min is min over opp ships of my
    # best counter — flags worst-defended threat. combat_uncovered_count is
    # the count of opp ships I have no favorable counter for — diversification
    # gap indicator. All three are zero when either side has no combat ships.
    combat_total_overmatch = 0.0
    combat_coverage_min = 0.0
    combat_uncovered_count = 0.0
    if my_combat and opp_combat:
        per_opp_best: list[float] = []
        for foe in opp_combat:
            best_against_foe = -math.inf
            for friend in my_combat:
                m = (
                    pk(friend, foe, engine.kill_curve_k) * friend.stats.strength
                    - pk(foe, friend, engine.kill_curve_k) * foe.stats.strength
                )
                if m > 0:
                    combat_total_overmatch += m
                if m > best_against_foe:
                    best_against_foe = m
            per_opp_best.append(best_against_foe)
            if best_against_foe <= 0:
                combat_uncovered_count += 1.0
        combat_coverage_min = min(per_opp_best)
    elif opp_combat:
        # No friendly combat ships but enemy has some — every threat uncovered.
        combat_uncovered_count = float(len(opp_combat))

    result: Dict[str, float] = {
        "material_diff": float(material_diff),
        "home_pressure_diff": home_pressure_diff,
        "combat_balance": combat_balance,
        "econ_value_self": econ_value_self,
        "merchant_count_value_self": merchant_count_value_self,
        "has_landing_self": has_landing_self,
        "landing_pressure_self": landing_pressure_self,
        "landing_danger_self": landing_danger_self,
    }
    # Per-type counts, in canonical order (own first, then opp).
    for t in SHIP_TYPE_ORDER:
        result[f"own_{SHIP_TYPE_FEATURE_NAME[t]}"] = float(own_counts[t])
    for t in SHIP_TYPE_ORDER:
        result[f"opp_{SHIP_TYPE_FEATURE_NAME[t]}"] = float(opp_counts[t])
    # Matchup features.
    result["combat_total_overmatch"] = combat_total_overmatch
    result["combat_coverage_min"] = combat_coverage_min
    result["combat_uncovered_count"] = combat_uncovered_count
    return result


def heuristic_eval(engine: "gameEngine", me: int) -> float:
    """Game-state evaluation in [-1, +1] from `me`'s perspective.

    Terminal: ±1 from engine.winner.
    Non-terminal: tanh(Σ_k DEFAULT_WEIGHTS[k] * features(s)[k]).
    """
    if engine.is_terminal():
        return 1.0 if engine.winner == me else -1.0
    phi = features(engine, me)
    return math.tanh(sum(DEFAULT_WEIGHTS[k] * phi[k] for k in DEFAULT_WEIGHTS) + DEFAULT_INTERCEPT)


def decompose(engine: "gameEngine", me: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Diagnostic variant: return (H, features_dict, contributions_dict).

    `contributions[k] = DEFAULT_WEIGHTS[k] * features[k]` — the per-feature
    contribution to the pre-tanh sum. `H = tanh(sum(contributions.values()))`
    for non-terminal states, or ±1 for terminal states.

    The display scripts use this instead of reimplementing the heuristic body.
    """
    if engine.is_terminal():
        H = 1.0 if engine.winner == me else -1.0
        return H, {}, {}
    phi = features(engine, me)
    contributions = {k: DEFAULT_WEIGHTS[k] * phi[k] for k in DEFAULT_WEIGHTS}
    H = math.tanh(sum(contributions.values()) + DEFAULT_INTERCEPT)
    return H, phi, contributions
