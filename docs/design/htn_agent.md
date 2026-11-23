# Design: HTN + influence-map scripted agent (v0.5.0 teacher)

Status: **design only, nothing implemented yet.** Base: `origin/main` @ 4afb242.

## 0. Why, and what "good" means

The plan is to replace `tactical_scripted_action` (`battleboats/agents/tactical.py`, #46) as the
behavior-cloning teacher with a stronger agent. It has a deliberate game plan and fleet mixes
that vary from game to game. We then clone it with BC, and later DAgger, instead of
post-training in sandboxes.

Requirements, in priority order:

1. **Diverse strategy.** Fleet composition varies from game to game and includes landing ships,
   port builds where the map allows them, satellite captures, and home captures. This matters
   more than raw strength.
2. **Decisive games.** The agent always commits to an invasion, so turn-cap draws are rare.
3. **Cloneable.** It uses exactly the information in the policy observation (fog of war) and
   makes decisions a pointer policy can imitate. It can also be queried as an expert on any
   state the student reaches.
4. **No search.** No MCTS, no rollouts, no value net. God-mode MCTS failed before: there was no
   good leaf heuristic, and games were not decisive enough to train a value net
   (`docs/bc_findings.md`, `docs/heuristic_design.md`). Every decision here is a closed-form
   score over influence maps.
5. **Cheap.** CPU only, roughly as fast per decision as tactical, so 2,600-game harvests stay
   practical.

Baselines this must beat. All are sampled play on the v0.2.2 16x8+20x10 pool:

| Agent | Result | Source |
|---|---|---|
| tactical (#46) self-play | 13/14/13 over 40 games, **32.5% draws** | `runs/eval/scripted_selfplay_smoke_anyport` |
| tactical vs v0.3.0 | 29/46/25, score 0.415 | `runs/eval/scripted_v_v0.3.0_anyport` |
| v0.4.0 (BC) vs tactical | 24/28/48, score 0.480; buys skew to destroyer, merchant, sub; landing about 3% | `docs/model_versions.md` |
| v0.4.1 vs tactical | 10/25/15, score 0.350; landing 5.2%; self-play 72% decisive | `docs/model_versions.md` |

## 1. Rule and code facts that constrain the design

All of these were checked against `origin/main`.

**Economy** (`core/gameEngine.py`)
- `STARTING_CASH` is 250.
- The home port pays `PORT_PRODUCTION * CASH_PER_MATERIAL` = **50 cash per own turn**, applied at
  the start of your turn (`_tick_port_income`).
- Every other owned port stockpiles **25 materials per turn**. Only a merchant can turn them into
  cash: `MerchantLoadAction` at the satellite (up to `MERCHANT_CAPACITY` 100), then
  `MerchantUnloadAction` at the **home port only**, at 2 cash per material.
- So one merchant on a satellite loop is worth about +50 cash/turn, doubling income. A full load
  is 200 cash, which repays the merchant's 100 cost in 2 turns of satellite production.
- A captured port loses its stockpile.

**Map pool** (`runs/selfplay_mixed_v0.2.2/maps`, 40 maps)
- Every map has **2 ports per side**: home plus one satellite. There are **no neutral ports**.
- Home-to-satellite distance averages 4.4 on 16x8 and 6.1 on 20x10.
- Home-to-home Manhattan distance is 9–19 on 16x8 (mean 12.9) and 10–25 on 20x10 (mean 17.0).
- `land_fraction` is 0.02. **26 of 40 maps have zero empty land tiles.** On those maps a builder
  (cost 250) can never build a port (`_do_build_port` needs empty land). Empty-land counts:
  16x8 has 17×0, 2×1, 1×2; 20x10 has 9×0, 6×1, 3×2, 1×3, 1×4.
- This is the main reason ports are rarely built. It is a map-pool fact, not an agent failing.

**Turn cap** (`envs/map_pool.episode_limits`)
- The cap is `max_turns = 80 × max(1, home_distance / 14)`, so 80 on most 16x8 maps and up to
  about 143 on 20x10.
- A turn-cap timeout is a draw (`base_aec._is_truncated`). The only win is capturing the enemy
  home (`_do_capture_port` sets `winner`).

**Ship stats** (`core/config/shipTypes.csv`, `attackModifiers.csv`)

| Type | Cost | Strength | Range | Speed | Visibility | Scouting |
|---|---|---|---|---|---|---|
| Carrier | 250 | 100 | 12 | 6 | 4 | 4 |
| Battleship | 200 | 125 | 4 | 3 | 4 | 2 |
| Cruiser | 150 | 75 | 3 | 5 | 3 | 3 |
| Destroyer | 100 | 25 | 3 | 5 | 3 | 3 |
| Submarine | 100 | 100 | 3 | 3 | 1 | 1 |
| Landing | 250 | 0 | 0 | 3 | 2 | 1 |
| Merchant | 100 | 0 | 0 | 3 | 2 | 1 |
| Builder | 250 | 0 | 0 | 3 | 2 | 1 |

**Combat** (`_resolve_attack`)
- Each attack is a single roll: P(kill) = x²/(1+x²), where x = strength × modifier / defender
  strength and `kill_curve_k` is 2. There is no HP.
- Strength-0 hulls (landing, merchant, builder) always die when hit.

P(kill), attacker in rows, defender in columns:

| | Car | Bat | Cru | Des | Sub |
|---|---|---|---|---|---|
| Car | .50 | .59 | .64 | .90 | .50 |
| Bat | .61 | .50 | .86 | .96 | .47 |
| Cru | .47 | .26 | .50 | .95 | .36 |
| Des | .09 | .04 | .10 | .50 | .12 |
| Sub | .80 | .72 | .64 | .80 | .50 |

Submarines are by far the most cost-efficient hull: 100 cost, at least .5 against everything,
and nearly invisible. **An agent that optimizes only for strength will collapse to subs**, so
fleet diversity has to be a deliberate strategy choice (§6) and cannot emerge from a value
function.

**Detection** (`_detection_distance`, `_do_attack`)
- Detection range = scout's scouting × target's visibility. Ports use 4 for both.
- To attack, the target must be within the **attacker's own** range **and** the attacker's own
  detection range. Vision is not shared for targeting.
- Moving and attacking are independent flags, so a ship can move and then shoot in the same turn.
- **Kill reach** of attacker *a* against a target of visibility *v* is
  `speed_a + min(range_a, scout_a × v)`:

| Attacker | vs vis 1 (sub) | vs vis 2 (landing, merchant, builder) | vs vis 3 | vs vis 4 |
|---|---|---|---|---|
| Carrier | 10 | **14** | 18 | 18 |
| Battleship | 5 | 7 | 7 | 7 |
| Cruiser | 8 | 8 | 8 | 8 |
| Destroyer | 8 | 8 | 8 | 8 |
| Submarine | 4 | 5 | 6 | 6 |

- The carrier is the anti-invasion hull. Subs are poor at hunting landings.
- Ports detect landing ships at 8 tiles but cannot attack.

**Landing and capture**
- A landing ship moves up to 3 tiles and can then `CapturePortAction` in the same turn, from any
  tile at Manhattan distance 1 from a non-own port. The port needs no garrison.
- So **a landing that starts our turn alive within 3 water steps of a free home approach tile
  wins that turn.** This "strike distance" is what the invasion logic is built around.
- Bought ships cannot act on the turn they are bought (#38). The earliest direct capture is
  turn 3.

**Blocking**
- Moves require an unoccupied water tile.
- A home port has at most 4 water approach tiles. A player who fills all of them with their own
  ships makes their home **impossible to capture**, and also blocks their own home spawns.
- This home-plugging stalemate is a draw generator the teacher must not demonstrate (§7).

**Fog and observation** (`envs/observation.py`)
- The policy sees friendly entities, enemy ships and ports only through `player.sightings` and
  `player.port_sightings` (fresh or stale), plus **coastline tokens, which exclude ports**
  (`_coastline_tiles`).
- On this pool almost all land *is* ports. Reading `engine.map.terrain`, `is_land`, or
  `port_positions` would therefore reveal the unseen enemy ports, information the student never
  gets. The HTN agent must treat unsighted non-coastline land as unknown (§4).
- Note: tactical's `_water_dist_map` leaks this slightly today.
- Globals carry turn and board size but **not** the per-map turn cap
  (`build_globals`).

**Agent interface**
- Every scripted agent has the signature `fn(engine, player_id, rng: random.Random) -> Action`.
  It is called once per primitive action until it returns `EndTurnAction`.
- Consumers: `training/evaluate.py`, `rollout.collect_trajectory(opponent_action=...)`,
  `scripts/eval_tactical.py`, `scripts/harvest_tactical.py`.
- Per-game state lives on the engine and is keyed by `id(engine.players[0])`
  (`tactical.build_queue_for`).
- Labels must be in `engine.enumerate_legal` and round-trip through `ActionMasks.factor` /
  `to_action` (`training/tactical_bc.encode_tactical_action`).
- `BuildShipAction` spawns are canonicalized to the free tile nearest the opponent's home, so the
  agent chooses only the port and the type.

**BC lesson from v0.4.0**
- Buy agreement was 17.4% because the teacher's random build queue is hidden state.
- **Every random choice the new teacher makes must either depend only on the observation, or be
  recorded and fed to the student as conditioning (§8, §10).**

## 2. Architecture

```
battleboats/agents/htn/
  __init__.py      htn_scripted_action(engine, pid, rng, cfg=None) -> Action   (drop-in signature)
  memory.py        AgentMemory: per-game, per-player state on the engine (marker = id(engine.players[0]))
  beliefs.py       fog-safe world model: known ships/ports, enemy-home belief, unseen-fleet belief
  influence.py     influence layers (numpy HxW float32) + water BFS/Dijkstra caches
  domain.py        HTN domain: compound tasks, methods, preconditions, utilities
  planner.py       total-order decomposition (SHOP-style), plan repair, replanning triggers
  roles.py         ship-to-role assignment (greedy, by influence-weighted path cost)
  orders.py        per-ship order executors -> primitive Actions (Move/Attack/Capture/Load/...)
  buy.py           composition-driven buy planner
  personality.py   Personality dataclass, seeded sampler, metadata export
```

Each call to `htn_scripted_action` does the following:

1. **Turn start.** This is detected by the pair `(engine.turn, engine.current_player)` changing
   from the value stored in `AgentMemory`. On a new turn the agent:
   1. calls `engine._refresh_sightings()`, as tactical does;
   2. updates its beliefs (§4);
   3. recomputes the influence layers (§3);
   4. checks the replanning triggers and, if one fires, repairs or replans (§5);
   5. assigns roles to ships;
   6. builds a **turn script**: the order in which ships act, for example captures first, then
      free kills, merchant transfers, screen moves, landing moves, attacks after the moves, and
      buys last.
2. **Within the turn.** The agent pops the next intent and turns it into a primitive action
   through that ship's order executor, then checks it against `enumerate_legal(pid)`.
   - Attack results are random, so after every `AttackAction` the agent reruns a cheap local
     update: it removes a killed sighting, adjusts threat in the surrounding cells, and re-scores
     only the affected ships.
   - If an intent is illegal or stale, it falls through to the next one. Final fallback:
     `tactical._choose`, then `EndTurnAction`.
3. **Always legal.** The agent never returns an action outside `enumerate_legal`, and it never
   reads `engine.ships`, `engine.ports`, or `players[1-pid]` for enemy information.
   - This gets a guard test that wraps the engine in a proxy and asserts that enemy state is
     accessed only through `known_enemy_*`.
   - It is a stricter version of the check in `tests/test_tactical_scripted.py`.

Cost target: under 2 ms per decision on 20x10. Maps are only 128–200 cells. All layers are
vectorized numpy, and the BFS from each enemy position is cached per turn.

## 3. Influence maps

All layers are `float32[H, W]` over water cells and are recomputed at turn start. Distances are
**water-path distances** (BFS on passable water, with own-ship occupancy ignored for planning).
They use the fog-safe passability mask from §4.

| Layer | Definition | Used by |
|---|---|---|
| `T_v` enemy threat, one per victim visibility v ∈ {1, 2, 3, 4} | Σ over enemies e of w_e · P(e reaches a vis-v victim at cell c next turn) · E[pk]. Reach is `dist(e, c) ≤ speed_e + min(range_e, scout_e·v)`, the table in §1. For a stale sighting, w_e = exp(−age/τ_mem); this replaces tactical's linear 4-turn decay. Belief mass (`B`, below) contributes the same way, using the class-weighted average stats of the expected unseen fleet. | Movement safety, landing paths, merchant routes, fight-or-flee |
| `S_v` own strike coverage | The same formula over own combat ships, i.e. "can we kill a vis-v intruder here on our next turn". | Defense ring check, escorts, screen placement |
| `C` control | `S_3 − T_3`, smoothed with one 4-neighbour blur. | Fleet positioning, choosing where to fight, wave go/no-go |
| `V` value | Enemy home (belief-weighted, §4): 10. Enemy satellite: 3. Known enemy landing: 6, builder: 3, merchant (scaled by expected cargo): 2. Own home: 8, own satellite: 3. Each value spreads with exp(−d/2) along water. | Targets for hunt and raid, defense priority |
| `D` enemy detection | 1 where any known enemy ship or **sighted** enemy port would see a vis-2 ship (port: 8; ship: scout·2); decays for stale sightings. | Stealthy landing approach (being unseen buys a turn) |
| `B_k` unseen-fleet belief, k ∈ {capital, light, sub, landing} | Probability mass of unseen enemy ships per class (§4). | Feeds `T_v`; drives scouting |
| `H` enemy-home belief | Probability over candidate home tiles (§4). | Invasion target, scouting goal |

How each behavior uses the layers:

- **Movement** (all hulls). Each order executor yields a goal set G plus a cost field. The ship
  steps to the neighbour that minimizes `cost_to_go(G)` under a Dijkstra cost per cell:
  `1 + λ_risk(hull) · T_vis(hull)(c) + λ_stealth · D(c) · [hull is landing] − λ_ctrl · C(c) · [escort or screen]`.
  - Ties and near-ties are broken with a softmax at the personality temperature, which keeps
    tactical's noise.
  - Ships move one tile per `MoveAction`, so the field is computed once per ship per turn and
    then followed step by step.
- **Target choice.** Among legal `AttackAction`s, score each as
  `pk(att, def) · (value(def) + V(def.pos)) − retaliation`.
  - Retaliation is `T_vis(att)(att.pos)` after the shot, weighted by the attacker's cost.
  - Strength-0 targets in range are always taken first, keeping tactical's free-kill rule.
  - If there is a known enemy landing ship within reach of any shooter, it outranks everything
    else, since it is a direct threat to the home port.
- **Fight or flee.** A combat ship holds or engages when
  `S_3(pos) + own_strength ≥ margin · T_3(pos)`, with `margin` a personality knob in [0.8, 1.6].
  Otherwise it retreats along −∇T toward high `C`.
- **Landing paths.**
  - Run Dijkstra from the landing to the approach tiles of the target port with the landing cost
    field: heavy `λ_risk` on `T_2`, plus `λ_stealth` on `D`.
  - For multi-landing waves, generate k diverse paths by adding a penalty on cells already used.
  - The path is cut at a **staging cell**: the first cell within strike distance (water distance
    ≤ 3 to a free approach tile). The landing waits just outside it, at distance 4–6, until the
    wave go condition holds (§5), then dashes and captures in a single turn.

## 4. Fog-safe beliefs

**Allowed inputs**
- Own ships, ports, and cash.
- `known_enemy_ships`, `known_enemy_ports`.
- Coastline tiles (`observation._coastline_tiles`).
- Board size and the turn number.

**Passability mask**
- Water, plus every tile that is not a coastline token, not an own port, and not a sighted
  enemy port, is treated as *possibly water*.
- A path that turns out to hit an unsighted port tile is repaired on the next turn. The cost is
  at most one wasted step, and the information stays what the student has.

**Enemy-home belief `H`**
- Prior over candidate tiles: board cells at least `min_port_enemy_home_water` (7) water steps
  from both our ports, weighted by exp(−|c − mirror(own_home)|₁ / σ).
  - mirror(p) = (W−1−x, H−1−y), the same as `tactical._explore_goal`.
  - The 16x8 maps are only roughly mirrored (for example homes (4,3) and (13,4)), so σ ≈ 3.
- When a port sighting with `is_home` appears, H collapses to that tile.
- A non-home enemy port sighting zeroes that tile and down-weights a radius of 2 around it.
- Cells we have had in port-detection range (own ship scout × 4 ≥ dist) with no sighting are
  zeroed. This is negative information.
- Scouting goal: argmax of H mass reachable per turn. Destroyers see ports at 12 and carriers at
  16, so one destroyer pass usually resolves H within 2–3 turns.

**Unseen-fleet belief `B_k`**
- Budget model: we know the enemy's income rule exactly. Enemy spend so far is bounded by
  250 + 50·(enemy turns) + 50·(turns they held their satellite with a merchant alive). The
  merchant term is uncertain and is estimated as 0.5 of that bound unless merchants were
  sighted.
- Subtract the cost of every enemy ship ever sighted and not witnessed killed. The remainder is
  the expected cash in unseen ships, which may still be unspent.
- Convert that cash to expected counts per class using a prior mix. The prior is the
  personality's `paranoia` blend of the uniform mix and the observed enemy mix.
- Spatial model: mass starts at H's spawn tiles. Each enemy turn it diffuses one speed step along
  water (a speed-weighted blur restricted to passable cells). It is zeroed wherever our current
  detection covers that class's visibility; subs (vis 1) are almost never ruled out, by design.
  The total is renormalized to the expected count.
- A stale sighting is moved from `T` into `B` once its weight drops below 0.2. Its mass is then
  diffused like any other unseen ship.

Every one of these inputs is in the observation, or is a deterministic function of the history
of observations. This is what lets DAgger replay the expert on student trajectories (§10).

## 5. HTN planner

### 5.1 Representation
- **Planning state** `PS` is a compact summary built from beliefs and the influence layers:
  turn, cash, income, own fleet by type, estimated enemy fleet by class, H entropy and argmax,
  port ownership as known, strike-ring coverage, material ratio, and the open waves with their
  status.
- **Compound tasks** have ordered **methods**. Each method has a precondition over `PS`, a
  utility `U(PS, personality)`, and an ordered subtask list. Primitive tasks are role
  assignments plus buy-planner targets, not engine actions.
- Decomposition is **total-order, forward** (SHOP2 style), depth at most 4, with no search over
  the world.
- **Method choice:** among the methods whose preconditions hold, sample with
  `softmax(U / τ_plan)` using the per-game seeded RNG. This is the second source of diversity
  after the strategy prior. The choice is logged.

### 5.2 Domain

```
WinGame
  ├─ method NavalDominance      pre: always            U ∝ prior[naval]
  ├─ method EconomicExpansion   pre: own satellite held U ∝ prior[econ] (+ if coastline tiles exist)
  ├─ method AmphibiousInvasion  pre: always            U ∝ prior[invasion]
  └─ method Rush                pre: turn ≤ 2 and home-to-H-argmax ≤ 14   U ∝ prior[rush]
```

Each strategy decomposes into **phases**. Every strategy's last phase is
`MountInvasion(home)`, so decisiveness is structural (§7).

| Strategy | Phase 1 | Phase 2 | Phase 3 (all) | Target composition (before personality jitter) |
|---|---|---|---|---|
| **NavalDominance** | `Scout(H)`, `DefendHome`, `SecureEconomy(min)` | `EstablishSeaControl`: capital fleet seeks the max-`C` contest region and hunts `B` mass; methods `CapitalLine` (battleship + cruiser) or `CarrierGroup` (carrier + destroyer pickets) or `Wolfpack` (subs + 1 destroyer as eyes) | `MountInvasion(home)` once the material ratio is ≥ 1.3 or the deadline passes | car .15, bat .2, cru .2, des .15, sub .1, landing .15, merchant .05 |
| **EconomicExpansion** | `SecureEconomy(full)`: 1–2 merchants hauling; `BuildPort` on the best coastline tile if any exists (score = water distance to home ≤ 6 and low `T`) | `RaidEconomy`: capture the enemy satellite (landing + escort), hunt enemy merchants, defend our satellite | `MountInvasion(home)` with the income edge, 2 waves | merchant .15, builder .1 (0 if no coastline), landing .25, des .15, cru .15, sub .1, car .1 |
| **AmphibiousInvasion** | `Scout(H)` with 2 destroyers, `DefendHome(light)` | `PrepareWaves`: buy landings toward ≥ 3 in flight, stage on k diverse paths; `Feint`: one landing on the satellite approach to pull defenders | `MountInvasion(home)` with **multi-prong** waves | landing .35, des .2, cru .15, car .15, sub .1, merchant .05 |
| **Rush** | Turn-0/1 buy: landing + destroyer. Destroyer scouts ahead along the H argmax path. Second landing on turn 2–3. | *(none: goes straight to phase 3)* | `MountInvasion(home)` immediately; on failure, revert to the sampled fallback strategy (NavalDominance or AmphibiousInvasion) | landing .4, des .3, sub .15, cru .15 |

**Shared compound tasks**
- `DefendHome(level)`: keep the strike ring covered. The ring is every water cell within 4
  water steps of a home approach tile. Each ring cell must have `S_2(c) ≥ 1` on our turn, which
  guarantees we can kill a landing that ended its turn there.
  - Method `CarrierAnchor`: one carrier within 6 of home covers the whole ring, with kill reach
    14 against vis 2.
  - Method `Pickets`: 2–3 destroyers or cruisers on the approach axes taken from `B_landing`
    and H.
  - Level `light` accepts coverage on the H-facing half only.
  - Constraint: **never occupy every home approach tile** (§7).
- `SecureEconomy(level)`: `min` keeps 1 merchant alive on the satellite loop (tactical's
  load → home → unload). `full` keeps 2 and adds an escort when `T_2` on the route exceeds a
  threshold.
- `Scout(target)`: a destroyer (or carrier) moves to maximize the H and B mass newly brought into
  detection per tile moved.
- `EstablishSeaControl`: the fleet goal is the cell maximizing `V · σ(C)`. Engage with the
  fight-or-flee rule. Hunt the B mass peak when there are no known enemies.
- `RaidEconomy`: attack the enemy satellite with a landing plus escort, and hunt merchants on
  the satellite-to-home route.
- `MountInvasion(target)` has the subtasks:
  1. `ResolveTarget`: if H's argmax probability is below 0.8, run `Scout` first.
  2. `FormWave(k landings, escorts)`: a wave of k ∈ {1, 2, 3} landings, each with 0–2 escorts,
     plus an optional decoy.
  3. `Stage`: move each landing to its staging cell, 4–6 water steps from approach.
  4. `Strike`: on the go condition, every landing in the wave dashes and captures in one turn.
  - **Go condition:** at least one landing at staging distance ≤ 3+1 has a path whose
    accumulated `T_2` over the next enemy turn is ≤ `risk_budget`, *or* the wave deadline has
    passed. Escorts move first along the landing paths to push `C` up.
  - Multi-prong waves overload the defender: one carrier can kill only one landing per turn
    (`has_attacked`).

### 5.3 Role assignment
- Each phase emits role demands, for example `{Escort(L7): 1, Picket(axis N): 2, Haul: 1, Scout: 1, Hunt: rest}`.
- Ships are assigned greedily by `suitability(hull, role) / (1 + path_cost)`. Suitability comes
  from the tables in §1: a carrier is best for Picket and CarrierAnchor, a destroyer for Scout
  and Escort against landings, a sub for Hunt against capitals.
- Assignments persist between turns unless the role disappears or the ship's cost to that role
  rises by more than 50%, to avoid jitter. Unassigned ships get `Hunt`.

### 5.4 Replanning triggers
These are checked at turn start. A **repair** re-decomposes only the affected subtree. A
**replan** re-decomposes from `WinGame`, keeps the current strategy with probability
`1 − switch_prob`, and pays a hysteresis cost.

| Trigger | Response |
|---|---|
| Enemy home first sighted, or H argmax jumps by more than 4 tiles | Repair `MountInvasion` / `Scout` |
| Known enemy landing within `T` reach of our strike ring, or belief `B_landing` mass in the ring > 0.3 | Preempt: `DefendHome(full)` gets top priority this turn |
| A wave lost ≥ 50% of its landings, or a wave deadline passed with no capture | Repair `MountInvasion`: new wave and a different path set (penalize the failed cells) |
| A port changes hands (ours or theirs) | Repair `SecureEconomy` / `RaidEconomy` |
| Material ratio (own ÷ estimated enemy strength) < 0.6 for 3 turns | Replan; methods with strength upside get a utility bonus |
| Phase deadline reached (§7) | Advance the phase (forced) |
| Every `K` = 6 turns | Soft replan (usually keeps the plan) |

## 6. Buy planner

The buy planner is driven by the active strategy's **target composition** `c*` (a fraction per
type, from the table in §5.2), adjusted as follows:

1. **Personality jitter.** `c* ← normalize(c* · exp(ε))`, with ε ~ N(0, σ_mix²) drawn once per
   game. There is also a capital-preference draw among carrier, battleship and cruiser.
2. **Counter-adaptation.** Weight `counter_w` shifts mass toward the types with high P(kill)
   against the observed and believed enemy mix, using the §1 table. Examples: enemy battleships
   → subs (.72) and battleships; enemy subs → battleships and carriers (.47–.5), because
   destroyers are bad against subs (.12); enemy landings → carrier or destroyer pickets.
3. **Map feasibility.** If the map has no coastline tile, set `c*_builder = 0`; this is 26 of 40
   maps. Once a builder exists or a port has been built, set `c*_builder = 0` until it is
   consumed. Renormalize.
4. **Hard reservations.** Phase-driven reservations override the composition (§7). Examples:
   "≥ 2 landings in flight by T_commit", and "a DefendHome deficit means buy the cheapest type
   that closes `S_2` ring coverage".

**Choosing what to buy**
- Score each type by deficit: `deficit_t = c*_t · (N_owned + N_planned + 1) − n_t`, where
  `N_planned` is the number of buys expected over the next 3 turns given income.
- Buy the argmax among types that a free, unlocked port can spawn.
- If the argmax is unaffordable, **save**, unless:
  - a reservation or emergency applies, or
  - waiting would take more than `patience` turns, in which case buy the best affordable type
    with a positive deficit.
- This replaces tactical's hidden uniform queue with a function of observable state plus
  `(strategy, personality)`. It is learnable when the clone is conditioned on those two (§10).
- Port choice: the home port for capitals and landings. The satellite for merchants (they
  spawn next to their loop) and for ships with a `Picket` role on the satellite side.

**Composition logging.** The planner tracks the realized per-type counts. If the realized mix
drifts more than 0.15 L1 from `c*` for 10 turns, it biases deficits harder.

## 7. Decisiveness mechanisms

These are hard rules that sit above the strategy. They target fewer than 10% draws in HTN
self-play.

1. **Commit deadline.**
   - The invasion phase must start by `T_commit = round(f_commit · cap_est)`.
   - `f_commit` is a personality draw from U[0.30, 0.55].
   - `cap_est = 80`, the observable lower bound of every map's cap, because the per-map cap
     depends on the home distance and is not in the observation.
   - So T_commit falls on turns 24–44. Rush commits by turn 3–6.
2. **Landing reservation.**
   - From `T_commit − 5`, the buy planner reserves 250 cash per missing landing until at least
     `k_wave` landings are alive or in production.
   - After a failed wave, the next wave must launch within `wave_gap` turns (personality draw
     from [4, 8]).
3. **Risk escalation.** The landing risk weight decays linearly as the turn approaches cap_est:
   `λ_risk(t) = λ_risk0 · max(0, 1 − (t − T_commit) / (cap_est − T_commit))`. Late-game
   landings accept suicide paths; a draw scores worse than a lost landing.
4. **No home plugging.** At most (number of free home approach tiles − 1) own ships may end a
   turn on home approach tiles. Defense comes from kill coverage (`S_2` on the ring), not
   blocking. The same rule applies to the satellite.
5. **Opportunistic captures are always taken.** Any capture that `enumerate_legal` offers on the
   enemy home is taken, regardless of threat. This changes tactical's `_choose_capture`, which
   skips threatened landings; the capture ends the game, so the threat is irrelevant. Satellite
   captures are taken unless the landing is reserved for a strike this turn.
6. **Draw-averse utilities.** Method utilities score an expected draw below an expected narrow
   loss of material. This matches the −0.15 draw penalty used in RL.
7. **Telemetry.** For each game, log the turn of the first invasion commit, the turns of each
   wave go and strike, and how each wave ended (capture, killed, timeout). Draw diagnoses then
   come from data.

## 8. Personality system

`Personality` is a frozen dataclass sampled once per (game, seat) from
`np.random.default_rng(hash((game_seed, pid, "htn")))`. It is independent of the engine RNG, so
combat rolls stay reproducible.

| Knob | Range or distribution | Effect |
|---|---|---|
| `strategy_prior` | Dirichlet(α = 1.0) over {naval, econ, invasion, rush}; rush is masked if home-to-H distance > 14 | Which strategy `WinGame` picks |
| `strategy` | Sampled from the prior at turn 0 (`τ_plan` 0.5) | Recorded as an integer 0–3 |
| `fallback_strategy` | Second sample | Used after a failed rush, or on a replan |
| `aggression` (fight margin) | U[0.8, 1.6] | Fight-or-flee threshold (tactical uses 1.2) |
| `λ_risk0` | log-U[0.5, 4] | How carefully landings route |
| `λ_stealth` | U[0, 2] | Preference for undetected approaches |
| `f_commit` | U[0.30, 0.55] | Commit deadline |
| `k_wave` | {1: .3, 2: .5, 3: .2} | Landings per wave |
| `wave_gap` | U{4..8} | Turns between waves |
| `patience` | U{1..4} | Turns to save for the target type |
| `σ_mix`, `capital_pref` | 0.3; Dirichlet(1) over car/bat/cru | Composition jitter |
| `counter_w` | U[0, 0.6] | Adaptation to the enemy mix |
| `econ_merchants` | {0: .2, 1: .5, 2: .3} (overridden by the strategy minimum) | Merchant count target |
| `paranoia` | U[0, 1] | Prior weight on the uniform mix for unseen enemies |
| `τ_mem` | U[2, 6] turns | Stale-sighting decay |
| `move_temp` | U[0.2, 0.8] | Move softmax temperature |
| `switch_prob` | U[0.1, 0.4] | Chance of switching strategy on a replan |

**Recording**
- `htn_scripted_action` exposes `episode_metadata(engine, pid)`, which returns:
  `{teacher: "htn", teacher_version, personality: {...}, personality_seed, strategy, strategy_trace: [(turn, strategy, phase, trigger)], waves: [...]}`.
- `harvest_*` stores it in each game's `meta_json` (`tactical_bc.write_game_npz`). Eval
  summaries store it per game.
- The harvest `run.json` (`training/manifest.py`) records `teacher_version`, plus the personality
  distribution as a config hash so that `scripts/runs_index.py` tracks teacher lineage.
- Per row, the harvest also stores `strategy_id` (0–3) and `phase_id`. These make conditioning
  possible (§10) and allow analysis by strategy.

## 9. Eval plan

**Harness**
- `scripts/eval_htn.py`, a fork of `scripts/eval_tactical.py` that reuses `_schedule`,
  `_limits`, `wilson`, `summarize` and the landing tracker.
- Pool: `runs/selfplay_mixed_v0.2.2/maps` (40 maps), per-map caps from `episode_limits`, seed 0.
- Learned opponents are sampled with `policy.act`.
- CPU: at most 4 workers while other training is running.

**Matchups**, 200 games each, seats alternating and split evenly by size:
1. HTN vs tactical (#46)
2. HTN vs v0.3.0
3. HTN vs v0.4.1
4. HTN self-play, random personalities on both sides
5. HTN vs random (sanity check, expect ≥ 98% wins)
6. Ablations, 100 games each vs tactical:
   - fixed median personality, to see what randomization costs in strength;
   - with and without the §7 rules, to measure what each rule contributes to decisiveness.

**Metrics** (per matchup, overall and by seat, by size, and by HTN strategy)
- W/L/D, win rate with Wilson 95% CI, and score (draw = ½).
- **Decisive rate**: the share of games ending in a home capture.
- Mean and median game length; turn of the first commit and the first strike.
- **Diversity score**: `1 − std/mean` over the 8 ship types' buy counts.
  - Reported pooled over all HTN games, and as a per-game mean.
  - Also over **buildable types only** (7 types on maps with no coastline). Builders are
    structurally excluded on 26 of 40 maps and would otherwise cap the 8-type score.
- **Landing share** of buys; landings bought, captures (satellite and home), landings killed.
- Port builds per game on maps that have coastline; merchant cash as a share of income.
- Strategy frequency: each strategy should occur in ≥ 12% of games.
- Seat gap: the difference in win rate between seat 0 and seat 1.
- Illegal actions (must be 0) and decision latency in ms (p50, p99).

**Acceptance gate** before we harvest. Ask Nick before relaxing any of these.

| Metric | Target | Reference |
|---|---|---|
| Self-play decisive rate | ≥ 90% | Tactical: 67.5% |
| vs tactical score | ≥ 0.60 | Tactical self-play: 0.49 |
| vs v0.3.0 score | ≥ 0.50 | Tactical: 0.415 |
| vs v0.4.1 score | ≥ 0.65 | |
| Diversity, pooled (8-type and buildable-only) | ≥ 0.55 | v0.4.1 sandbox: 0.48 |
| Landing share | 10–25% | |
| Seat gap | ≤ 10 pts | |
| Latency | ≤ 2× tactical | |

**After cloning**, the same matchups for the v0.5.0 BC and DAgger students, plus:
- held-out agreement by verb (target: buy ≥ 40%, vs 17.4% for v0.4.0);
- the student's own diversity and landing share, with the gap to the teacher.

## 10. BC and DAgger integration

**Harvest**
- Generalize `scripts/harvest_tactical.py` to `--teacher {tactical,htn}`. The schedule stays the
  same: self-play, vs v0.3.0, vs random, with v0.4.1 added as an opponent.
- Rows use the same `encode_tactical_action` path, so the label format, the pointer action space
  and the `.npz` packing do not change.
- New per-row fields: `strategy_id`, `phase_id`. New per-game field: personality metadata.
- Size: about 2,600 games as for v0.4.0. With ~570 labels per game that is about 1.5M rows.

**Conditioning** (recommended; needs Nick's decision)
- Train the clone with `skill_count = 4` (`PolicyNetwork.skill_embed` already exists and is
  added to the pooled context) and pass `skill_idx = strategy_id`.
- At play time, sample the strategy from the same prior per game. This gives the student the
  same controllable diversity, and the buy head becomes learnable.
- Option B: no conditioning. The student learns a mixture policy from observable cues, and buy
  agreement stays lower.
- Option C: add `phase_id` as a second embedding.

**DAgger** (`scripts/dagger_htn.py`, new)
- **The expert as an oracle on arbitrary states.** The HTN keeps memory: beliefs, plan, roles,
  personality. To label student states, the expert runs in **shadow mode** alongside the student
  in the same game.
  - At every student decision it runs the same turn-start and within-turn updates as if it were
    acting, emits a label, and the *student's* action is executed.
  - This works because every memory input is either an observation (§4) or the expert's own
    plan, and executors re-derive orders from state every turn. Divergence ("my escort didn't
    move") is just state, and role assignment repairs it.
  - Requirement that falls out of this: **no executor may assume its previous primitive action
    was executed.** Each executor is a function of (state, order).
- **Mixing.**
  - Iteration i rolls out with β_i·expert + (1−β_i)·student at **per-turn** granularity: one
    actor plays a whole turn, so turns stay coherent.
  - β = 1.0 (plain BC, the v0.5.0 student), then 0.5, 0.25, 0.
  - Each iteration harvests about 600 games (self-play student vs HTN with random personality,
    tactical, and v0.3.0), aggregates them into the dataset, and fine-tunes 2 epochs from the
    previous weights.
  - 3–4 iterations in total.
- **Label reproducibility.** Expert sampling uses `rng = random.Random(hash((game_seed, pid, turn, decision_idx)))`,
  so relabeling is deterministic.
- **Conditioning in DAgger.** The shadow expert's sampled strategy is passed to the student as
  `skill_idx` for that game.
- **Afterwards.** PPO from the DAgger student can use a **league of HTN personalities as fixed
  opponents**, alongside snapshots. This is the "diverse opponents" route for keeping the mix
  varied under RL.

## 11. Phased implementation plan

Effort is rough agent working time. Compute time is extra and runs at ≤ 4 workers.

| Phase | Scope | Effort | Exit check |
|---|---|---|---|
| P0 Scaffolding | `agents/htn/` package, `AgentMemory`, turn detection, legality fallback, fog guard test, `scripts/eval_htn.py` | 3–4 h | Plays legal games vs random, 0 illegal, fog test passes |
| P1 Beliefs + influence | §3 and §4 layers, BFS/Dijkstra caches, debug plots via `agents/debug_plot.py`, unit tests (reach table, mask, H collapse, B diffusion) | 6–8 h | Layer tests pass; ≤ 1 ms per layer refresh on 20x10 |
| P2 Executors | Haul, Escort, Picket, Scout, Hunt, Land (stage and strike), BuildPort; fight-or-flee and target scoring; one hard-coded strategy (AmphibiousInvasion) | 6–8 h | Beats tactical with score ≥ 0.55 over 100 games; decisive rate ≥ 80% |
| P3 HTN + buys | Domain, planner, role assignment, replanning triggers, 4 strategies, buy planner | 8–10 h | Each strategy wins ≥ 50% vs tactical; strategy mix logged |
| P4 Personality + decisiveness | §7 and §8, metadata, short tuning sweep (≤ 3 knobs) | 4–6 h | Acceptance gate (§9) on 200-game evals |
| P5 Eval report | Full §9 matrix, `docs/htn_agent_eval.md`, decision with Nick | 2 h + about 2–3 h compute | Nick signs off |
| P6 Harvest + BC v0.5.0 | `--teacher htn`, conditioned BC, tag | 3–4 h + harvest and training compute (v0.4.0 took about 30 min to train) | Clone beats v0.4.1, buy agreement ≥ 40% |
| P7 DAgger | `scripts/dagger_htn.py`, 3–4 iterations, eval each | 5–6 h + compute | Student score vs tactical ≥ 0.55, diversity ≥ 0.45 |

Total: about 40–50 hours of work before compute, with a go/no-go gate at P2 and at P5.

## 12. Risks
- **Hand-tuning burden.** This is the MCTS heuristic problem in a new form. Mitigations: fewer
  than about 20 knobs, all of them personality-sampled so they need no tuning; acceptance is on
  aggregate metrics, not perfect play.
- **Subs dominate.** Composition targets keep the mix diverse, but strategies with few subs may
  lose to Wolfpack personalities. This is acceptable given Nick's stated priority. Track win
  rate by strategy.
- **The student can't see H or B.** Belief state lives in the teacher's memory. It is
  reconstructable from the observation history, but the student is memoryless (a single
  observation). Stale sightings and their staleness are tokenized, so most of the signal is
  there. Watch scout-phase agreement.
- **Turn-based shadowing cost.** DAgger doubles per-decision compute (expert plus student),
  which is fine at about 2 ms.

## 13. Open questions for Nick
1. **Builders and ports.** 26 of 40 maps have no buildable land, so port builds are impossible
   there. Should we regenerate the pool with buildable coastline (for example `land_fraction`
   0.05 with a minimum of 2 coastline tiles per side)? And should the diversity score be over 8
   types or over buildable types?
2. **Home plugging.** Is a self-imposed "never plug home" rule in the teacher enough, or do you
   want a rules change so captures can't be fully blocked?
3. **Public port positions.** Should port *locations*, not owners, be public in the observation?
   Today the student can't see where ports are until it scouts them, so invasions start with a
   scouting phase.
4. **Conditioning.** Should the clone be strategy-conditioned through `skill_idx`, or learn the
   strategy mixture unconditioned?
5. **Strength bar.** Is "beat tactical at 0.60, match v0.3.0 at 0.50" the right bar, given that
   diversity comes first?
6. **Version number.** `docs/model_versions.md` says a minor bump means a rules change. A clone
   of this teacher under the same #38 rules would be v0.4.2, unless we pair it with a rule or map
   change (Q1–Q3). Keep v0.5.0?
7. **Turn cap.** Should the teacher get the true per-map `max_turns` from the harness (sharper
   deadlines, but the student can't see it), or keep the observable `cap_est = 80`?
