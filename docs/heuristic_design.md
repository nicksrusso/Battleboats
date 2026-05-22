# MCTS Heuristic — Design

## Purpose

Random rollouts in battleboats almost never terminate within a reasonable
step budget (verified: 0/20 rollouts produced a winner within 100 steps).
So every MCTS leaf will be evaluated by this heuristic, not by a rollout
outcome. **The heuristic is the value function**, not a fallback.

Used by:
- God-mode MCTS leaf eval (full information).
- Later: any rollout-based agent, behavior cloning, PPO reward shaping.
- Eventually replaced (or composed with) a learned value net.

## Design principles

1. **Zero-sum perspective.** Every term is `f(me) − f(opponent)`. MCTS needs
   "who is winning," not "how am I doing."
2. **Normalize each term** to roughly `[-1, +1]` via `tanh`. Weights then
   represent *relative importance*, not scale absorption.
3. **No hard thresholds.** Continuous functions everywhere. Cliffs in the
   eval function create cliffs in the search.
4. **Asymmetric combat.** Use both directional PKs (`my_PK(i→j)` and
   `their_PK(j→i)`), never assume one implies the other (subs vs surface
   ships break the symmetry assumption).
5. **Win-condition primacy.** The home-port pressure term dominates;
   everything else is enabling.
6. **Computationally cheap.** Called at every MCTS leaf. Avoid O(n²) blowups
   where avoidable; precompute per-state quantities once.

## State quantities (available from `engine.get_state()`)

For each player:
- `ships` — list of `Ship(id, type, stats, owner, position, cargo, ...)`.
- `ports` — list of `Port(position, owner, is_home, stockpile)`.
- `cash` — int.
- `home_port` — position tuple.

Helpers needed (not all exist yet):
- `pk(attacker: Ship, defender: Ship) -> float` — engine's kill probability
  formula without consuming RNG:
  `x = (atk.strength * attack_modifier(atk.type, def.type)) / def.strength`,
  `pk = x^k / (1 + x^k)` with `k = engine.kill_curve_k`.
- `manhattan(a, b)` — already on `engine.map`.

## Term decomposition

Four terms total. Each computed symmetrically (`me` then `opponent`) and
subtracted; each normalized to `[-1, +1]` via `tanh`.

---

### Term 1 — Home pressure (T_HOME)

The win condition. Captures both *I am closing on their home* and *they
are closing on mine*. The single most important term.

For a given home position `H` and a set of enemy ships `E`:
```
threat(H, E) = Σ_{e ∈ E}  w_type(e) / (1 + manhattan(e.position, H))
```
Where `w_type(e)`:
- `Landing`: `250`  — the only ship type that can actually *capture* the port;
  worth ~2× the strongest combat ship since it represents the win condition.
- `Builder`: `0`   — no offensive role.
- `Merchant`: `0`  — no offensive role.
- otherwise: `e.stats.strength` — combat ships (25-125 in current config) threaten via attrition.

```
T_HOME_raw = threat(opp.home, my_ships) − threat(my.home, opp_ships)
T_HOME     = tanh(T_HOME_raw / SCALE_HOME)
```

`SCALE_HOME ≈ 150` — calibrated against actual ship strengths. Intuition:
a single Battleship (strength 125) adjacent to my home contributes
`125 / 2 = 62.5` raw → `tanh(0.42) ≈ 0.39`, meaningful but not pegged.
A Landing ship adjacent to enemy home contributes `250 / 2 = 125` raw
→ `tanh(0.83) ≈ 0.68`, dominant signal as intended.

**Asymmetry note:** uses straight-line Manhattan distance, not pathfinding.
The engine doesn't expose path cost cheaply; if blocked-by-terrain becomes
a noticeable failure mode, we can add a coarse passability check.

---

### Term 2 — Combat balance (T_COMBAT)

Pairwise tactical advantage. For every (friendly, enemy) combat-ship pair,
contribute the net expected exchange value, distance-weighted.

```
combat_pair(i, j) = [pk(i, j) · i.strength − pk(j, i) · j.strength]
                    / (1 + manhattan(i.pos, j.pos))

T_COMBAT_raw = Σ_{i ∈ my_combat, j ∈ opp_combat}  combat_pair(i, j)
T_COMBAT     = tanh(T_COMBAT_raw / SCALE_COMBAT)
```

`my_combat` = ships with `type ∉ {Merchant, Landing, Builder}`. (Optional:
include Landing as defenders only.) `SCALE_COMBAT ≈ 500` — calibrated for
strengths in the 25-125 range; a 5×5 fleet sum at moderate engagement
distance produces raw values in the hundreds.

**Why this shape:**
- Closer pairs dominate (proximity is what makes combat real).
- `pk · strength` is the expected damage delivered — accounts for both
  hit probability and consequence.
- Subtracting both directions captures asymmetric matchups (sub vs
  destroyer); positive means *I expect to come out ahead* in that pair.

**Cost note:** O(|my_combat| × |opp_combat|). At 10×10 = 100 pairs and
microseconds per `pk`, this is cheap. If entity counts grow, prune to
"pairs within engagement range" before evaluating.

---

### Term 3 — Material (T_MAT)

Aggregate material differential. Coarse but stable; survives the noise
in the combat term.

```
ship_value(s):
  base = s.stats.strength * (1 + s.stats.attack_range)
  if s.type is Landing: base += 3.0   # premium for win-enabling units
  if s.type is Builder: base += 1.0   # mild premium for economy enabler
  return base

# Why `strength * (1 + range)`: reach is a force multiplier — high-range
# ships engage first and control more space, so two ships of equal strength
# but different range are not equally valuable. The `1 +` floor prevents
# range-0 transports from collapsing to zero before the type bonuses are
# applied. Cost is deliberately NOT used as a value proxy: game cost may
# reflect designer intent or playtesting tweaks, not pure stat balance.

T_MAT_raw = (Σ_{s ∈ my_ships} ship_value(s) + 2 * |my_ports|)
          − (Σ_{s ∈ opp_ships} ship_value(s) + 2 * |opp_ports|)
T_MAT     = tanh(T_MAT_raw / SCALE_MAT)
```

`SCALE_MAT ≈ 2000` — calibrated for `strength × (1 + range)` values: a
Battleship is `125 × 5 = 625`, a Carrier is `100 × 13 = 1300`, so a
fleet differential of a few ships lives in the 1000-3000 range.

Bonuses calibrated to match: `MAT_LANDING_BONUS = 500` (> strongest
combat ship's value, since Landing is win-enabling), `MAT_BUILDER_BONUS
= 100` (modest economy enabler), `MAT_PORT_VALUE = 400` (roughly one
combat ship's worth — ports are sustained-warfare assets).

---

### Term 4 — Economy (T_ECON)

Liquid value differential. Cash + port stockpiles + merchant cargo in
flight. All convert to cash eventually, so sum them.

```
liquid(player):
  cash = player.cash
  stockpile = Σ_{p ∈ player.ports} p.stockpile
  cargo = Σ_{s ∈ player.ships, s.type=Merchant} s.cargo
  return cash + stockpile + cargo

T_ECON_raw = liquid(me) − liquid(opp)
T_ECON     = tanh(T_ECON_raw / SCALE_ECON)
```

`SCALE_ECON ≈ 500` — calibrated against ship costs (100-250 per ship);
a few-turn production differential should produce a meaningful but
non-saturating signal. Still needs empirical validation against actual
cash-trajectory histograms.

---

## Combination

Convex weighted sum:

```
weights = {
  T_HOME:   3.0,   # win-condition primacy
  T_COMBAT: 1.0,
  T_MAT:    1.0,
  T_ECON:   0.5,
}

H(state, me) = Σ_k  weights[k] · T_k  /  Σ_k  weights[k]
```

`H` is bounded in `[-1, +1]`. The denominator normalization is for
interpretability — easy to read "this position evaluates to +0.3 from
my perspective."

**Initial weight rationale:**
- `T_HOME` triples the others because the home port *is* the win condition.
- `T_COMBAT` and `T_MAT` are equal — short-term tactics and long-term
  material balance are complementary.
- `T_ECON` is half because cash is one or two moves of latency away from
  becoming ships; it's enabling, not decisive.

These are *opinionated guesses* and will need tuning against actual play.
See "Tuning" below.

## Terminal handling

If the state is terminal (`engine.winner` is set), bypass the heuristic:

```
if engine.is_terminal():
    return +1.0 if engine.winner == me else -1.0
```

This gives the search a sharp, correct signal at terminal nodes that the
soft heuristic can't be expected to match.

## Edge cases

- **Empty entity sets** (no ships, no enemies): terms involving sums over
  empty sets contribute 0, no division-by-zero. Handle via empty-sum convention.
- **Strength zero defenders:** the engine's `_resolve_attack` treats them
  as guaranteed kills (PK = 1). Our `pk()` helper must mirror this to
  avoid divide-by-zero.
- **Map symmetry:** Manhattan distance is symmetric, so no perspective
  issue in distance terms.
- **Game very early (no enemies sighted under fog):** N/A for god-mode;
  for the future fog-aware variant, use sightings rather than ground truth.

## Tuning

Weights and scales are starting guesses. The validation loop:

1. **Sanity check by inspection.** Drop into a few hand-crafted states
   (you have a free ship adjacent to their home; you have lost half your
   fleet; etc.) and confirm `H` reflects intuition.
2. **Calibrate scales by running 50-100 rollouts** through real game
   states (from existing random-vs-random games), histogram raw term
   values, set `SCALE_*` so terms span most of `[-1, +1]` without saturating.
3. **MCTS-vs-random win rate** at fixed iteration budget — the integration
   metric. If MCTS doesn't beat random ≥ 70% at 200 iterations, the
   heuristic isn't pulling its weight.
4. **Ablation** later: zero out one term at a time, see which contributes.

## Computational layout

`heuristic_eval(engine, me)` is called once per MCTS iteration. To stay
cheap:

1. Compute once per call: `my_ships`, `opp_ships`, `my_combat`,
   `opp_combat`, home positions. Don't re-list inside loops.
2. The pairwise combat sum is the only quadratic piece — fine at current
   entity scales (typically < 20 ships per side).
3. No allocations inside inner loops (use generator sums).

Target: < 100 µs per call. Well within budget given ~9.5 ms per MCTS
iteration is currently spent in the rollout.

## Out of scope for v1

Things worth thinking about *later* if the agent has visible failure
modes despite the above:

- **Builder progress term** — builders en route to constructible terrain,
  gated by `if num_my_ports < threshold`. Add if the agent neglects
  expansion.
- **Path-cost distances** — replace Manhattan with shortest path through
  terrain. Add if the agent walks ships into shores.
- **Threat anticipation** — value enemy ships' *next-turn* reachable set
  rather than current position. Tactical refinement.
- **Cargo-in-flight bonus** — already counted in `T_ECON`, but could
  receive a multiplier for "merchant near home with cargo" (about to
  convert).
- **Engagement-range pruning** in `T_COMBAT` — keep only pairs within
  some distance to cut O(n²) cost when ships proliferate.

## Future composition

When a learned value net `V(state)` exists (post-PPO), we have options:

1. **Replace** the heuristic entirely with the value net (standard
   AlphaZero leaf eval).
2. **Compose** — `H_final = α · H_handcoded + (1-α) · V_net`, anneal `α`
   from 1 → 0 as the net trains. Smoother handoff; the handcoded
   heuristic is a regularizer early in training.

The architecture supports either — only the leaf-eval callable changes.
