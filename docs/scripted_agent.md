# Scripted agent

`battleboats/agents/scripted_agent.py` is a fog-limited scripted agent built from
the Gemini 4 Carbon design in `docs/gemini_carbon_scripted_agent_pseudocode.md`.
It is an opponent in the play UI and a labeller for harvests:

```bash
poetry run python scripts/play.py --map 32x16 --opponent scripted_agent --seed 0
PYTHONPATH=. python scripts/eval_scripted_agent.py --maps 32x16 48x24 --games 100 --workers 12 \
    --output runs/eval_scripted_agent/summary.json
PYTHONPATH=. python scripts/harvest_tactical.py --labeler scripted_agent ...   # same flags as before
poetry run python scripts/eval_agent.py --checkpoint ... --opponent scripted_agent
```

`battleboats/agents/registry.py` maps `scripted_agent`, `tactical`, and
`random` to objects with `act(engine, pid, rng)`. Configs, CLIs, and
stored harvest metadata may still say `fog_commander`; that name is an alias
for `scripted_agent`. `battleboats.agents.fog_commander` re-exports this module.
The name `scripted` was a separate full-information baseline and is rejected.

## What it reads

Only the seat's own information: own ships, ports, cash and stockpiles, the
turn, `player.sightings` and `player.port_sightings`, the public rule tables
(`BASE_STATS`, the spot matrix, attack modifiers, `kill_curve_k`), the static
terrain mask, and `enumerate_legal(pid)` reduced to fog-safe facts.

* **Terrain.** Water means `terrain != LAND`. Land never changes during a game
  (builders can only found ports on tiles that are already land) and every
  generator lists port tiles as land, so the mask hides nothing. The play UI
  shows the human the same terrain, with unsighted ports drawn as plain land.
* **Not read:** enemy ships, ports, cash or stockpiles, the engine RNG,
  absolute ship-id order across owners, the order of the legal list, and the
  coastline tokens from `observation.py`.

## Leaks found and how they are handled

1. **Canonical spawn tile.** `enumerate_legal` lists one `BuildShipAction` per
   (port, type), spawning on the free tile closest to the *true* enemy home.
   The engine is unchanged, because policies and MCTS depend on that action
   space. The agent ignores the listed spawn tile and picks its own from the
   free water next to the port. Every tile next to an own port is always
   visible, since all spot ranges are at least 1. `engine_accepts()` checks
   such a buy the way `_do_build_ship` does. `canonical_build_action()` maps
   it back to the listed action for label encoders.
2. **Coastline tokens** (`observation.py`). These leave out *unsighted* enemy
   ports, so their holes show where hidden ports are, and a newly built enemy
   port changes the set. The pseudocode flood-filled water from those tokens,
   so unsighted ports became "water". This agent never uses them. A port tile
   is land, and it is a search candidate until a unit covers it at Port spot
   range. *Fixed:* `observation.py` now excludes only ports the observer owns or
   has sighted from coastline tokens, so unsighted enemy ports appear as
   plain coastal land to RL policies too.
3. **Ship ids** come from one global counter. They are used only as identity
   keys and to order ships within one owner.

## Deviations from the pseudocode

| Pseudocode | Here | Why |
|---|---|---|
| Water = flood fill of non-coastline tiles | Static `terrain` mask | Fixes unsighted ports treated as water; fog-invariant |
| Spawn tile from the legal action | Agent picks the tile | Canonical tile leaks the true enemy home |
| Home-search prior incl. y-flip and mirror | Static coastal land not yet covered at Port range; "enemy half" (x3 / x0.3), near-own-home (x0.5) and x-mirror bonus are soft weights; y-flip dropped | No generator guarantees the y relation |
| Threat-weighted Dijkstra per move | Cached BFS fields over static water + one-step threat penalty (`fd + risk*threat`, must reduce distance) | About 0.3-0.5 ms per decision on CPU |
| Threat reach = exact water BFS of enemy move + range | Manhattan `speed + effective range` | Speed. It can over-estimate around land |
| `legal_actions` order for ties | All candidates sorted by position, type, own id | Determinism and fog invariance |

Kept as specified: six doctrines (weights, aggression, builder mode, merchant
target, buy temperature), per-game type jitter U(0.85, 1.15), phantom pruning
of stale sightings, phase order (capture, merchant transfer, same-turn cash,
combat, body-block / lane clearing / kiting, unclog, builder, merchant,
landing, fleet roles, buy), and the 250-step turn cap.

## Tests (`tests/test_scripted_agent.py`, 41)

* **Fog invariance** (4 games, every decision). Clone the engine and change
  only hidden state: move unseen enemy ships to other unseen water, change
  their cargo and flags, remove one, add a hidden submarine, change enemy cash
  and stockpiles, reseed the engine RNG, and move the opponent's `home_port`
  (the canonical-spawn input). The test asserts the seat's view is unchanged,
  then checks that the action, the rng state and the memory are identical.
  Two mutations were confirmed to fail it: reading enemy cash, and using the
  canonical spawn tile.
* **Determinism.** Same seed gives the same action log; a different seed
  gives a different one. The functional wrapper matches the class and resets
  with `engine.reset`.
* **No illegal actions.** 26 games on 8x8 open, 16x8, 24x12, 32x16 and 48x24
  against itself, scripted, random and tactical. Every action is accepted by
  the engine, and every non-buy action is in `enumerate_legal`.
* **Doctrine sampling.** All 6 doctrines appear over 300 seeds; sampling is
  seeded, jitter stays in range, forcing a doctrine works, and the two seats
  sample independently.
* Buys span at least 6 hull types. Unsighted ports are land. The harvest
  labels scripted_agent decisions with 0 encoding errors. The UI full-game
  fog-safety test runs with `scripted_agent`.

## Evaluation (2026-11-25, 100 games per matchup, seats alternate)

`scripts/eval_scripted_agent.py`, fresh preset maps per game, UI turn caps
(100-260 turns on 32x16). 600 games took 70 s on 14 workers. There were 0
crashes, 0 rejected actions and 0 forced end turns.

| Map | Matchup | Decisive | Wins | FC win by seat (0 / 1) | Avg turns (decisive) |
|---|---|---|---|---|---|
| 32x16 | FC vs FC | 68% | seat 0: 36, seat 1: 32 | - | 74.1 (34.6) |
| 32x16 | FC vs scripted | 82% | FC 74, scripted 8, draw 18 | 34/50, 40/50 | 43.5 (22.2) |
| 32x16 | FC vs random | 94% | FC 94, draw 6 | 46/50, 48/50 | 26.9 (16.3) |
| 48x24 | FC vs FC | 80% | seat 0: 35, seat 1: 45 | - | 78.4 (42.0) |
| 48x24 | FC vs scripted | 79% | FC 70, scripted 9, draw 21 | 34/50, 36/50 | 66.0 (28.1) |
| 48x24 | FC vs random | 97% | FC 97, draw 3 | 49/50, 48/50 | 31.1 (23.9) |

Buy mix, FC self-play on 32x16 (9,738 buys):

| Hull | Share |
|---|---|
| Destroyer | 41.0% |
| Submarine | 13.6% |
| Cruiser | 10.1% |
| Battleship | 9.5% |
| Carrier | 9.4% |
| Landing | 7.3% |
| Merchant | 7.1% |
| Builder | 1.9% |

Destroyers lead because of the +1.1 scouting bonus while the enemy home is
unsighted, plus the counter to submarines.

Doctrine W-L-D in self-play (both seats counted):

| Doctrine | 32x16 | 48x24 |
|---|---|---|
| ECONOMIC_BOOM | 17-9-11 | 9-9-10 |
| RAIDER_BLITZ | 16-8-14 | 21-20-5 |
| ISLAND_HOPPER | 13-21-6 | 10-17-4 |
| CAPITAL_ARMADA | 9-9-15 | 14-11-6 |
| WOLFPACK_STEALTH | 9-12-10 | 11-11-9 |
| BALANCED_ADAPTIVE | 4-9-8 | 15-12-6 |

Against scripted, every doctrine is net positive. The worst is ECONOMIC_BOOM
at 9-2-3 on 32x16. WOLFPACK_STEALTH has most of the draws against random
(4 of 6 on 32x16).

ms/decision, measured in one process with no other load:

| Map | Mean | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|
| 32x16 | 0.33 | 0.21 | 1.12 | 1.48 | 2.5 |
| 48x24 | 0.47 | 0.32 | 1.47 | 1.81 | 6.7 |

Under 14 parallel workers the mean is 0.55-0.99 ms and the max 66 ms.
