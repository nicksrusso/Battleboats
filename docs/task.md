# Battleboats: Scripted Harvest → Value Net → MCTS

**Date:** 25 November 2026 (America/Chicago)
**Repo:** [nicksrusso/Battleboats](https://github.com/nicksrusso/Battleboats), branch tip `origin/main` (`8eccf1f`, PR #62).
**Purpose:** A self-contained brief of the game rules and the training plan. An outside LLM should be able to reason about the rules and write a scripted agent from this document alone. Keep this file as the reusable rules-and-plan source; retune numbers only from the cited CSVs and engine constants, never by editing this file first.

All numbers below are taken from the engine and config on `origin/main` as of that commit. Nothing here is invented.

---

## 1. Plan

Goal: harvest games from scripted (and mixed) agents, train a value network on fog-of-war observations, then run Monte Carlo tree search guided by that value net. Do not build search until the value net beats a coin-flip / simple baseline on held-out outcomes.

This is a revival of an earlier MCTS idea that was shelved because there was no good heuristic and too few games were decisive to train a value net. The rules and the scripted baseline have both changed since then. A 50-game scripted-vs-scripted check on 32×16 maps after PR #62 looked promising (see §1.1). A related longer-term track (BAT-26) is an HTN + influence-map teacher with DAgger behavior cloning; this value net would also be useful later for scoring HTN plans, and that teacher would be a better data source than today's scripted agent. The work below does not depend on the HTN landing first.

### 1.1 Harvest games from scripted agents

Run on the order of **1,000 cheap CPU games**, primarily on **32×16** maps (the size ship stats are built for), using the map-pool turn cap described in §9.

**Opponents and mix.** The full-information baseline in `battleboats/agents/scripted.py` (PR #62) has been removed. Harvest and eval use the fog-limited **scripted agent** (`battleboats/agents/scripted_agent.py`) and the fog-limited **tactical** agent (`battleboats/agents/tactical.py`). Passing the old name `scripted` is an error, not an alias. Therefore:

- Play a mix of matchups: scripted agent vs itself, vs tactical, and vs the uniform random agent in `battleboats/agents/random_agent.py`.
- Regardless of how an agent chooses its action, **record each seat's fogged observation** from `battleboats/envs/observation.py` (`build_observation(engine, player_id)`), plus the legal-action list, the action taken, the true board state (for replay), and the eventual game outcome.
- Keep enough metadata to replay a game from seed + map + action sequence.

**50-game sanity check already in hand** (scripted vs scripted, 32×16, same seeds as the pre-#62 fixed-order baseline):

| | Old fixed buy order | New random queue (PR #62) |
|---|---:|---:|
| Decisive | 60% | **76%** |
| Seat 0 / seat 1 wins | 20 / 10 | **19 / 19** |
| Average length | 59.7 turns | 51.5 turns |
| Merchant share of buys | 89% | 12.5% |

Each ship type is now about 12–13% of buys. That is a much healthier fleet mix. The 76% decisive rate and even seat split are the reason this plan is worth trying. A 1,000-game harvest should report decisive rate, win split by seat and by matchup, and game-length distribution before anyone trains.

**Known limitations of the current scripted teacher (do not hide these):**

- It cheats with full information. Positions MCTS will query under fog can be off-distribution if we only imitate its information set. Recording fogged observations, and mixing in the fog-limited tactical agent, is the mitigation.
- It is still a shallow greedy policy: capture, then attack, then merchant load/unload, then build a port, then buy the queue head, then a move that shortens Manhattan distance to a type-specific objective, else end the turn. It does not path around land, does not scout deliberately, and does not save a landing for a visible home if a closer satellite is also legal to capture.
- Because the queue is a full permutation and it never skips a type, a seat that rolls Carrier / Battleship / Builder first can sit on thin income for a while (no guaranteed early merchant). Leave that for diversity unless the 1,000-game check shows it collapses into stalls.

### 1.2 Train a value net on fogged observations

**Input.** The existing **76-dimensional entity-token** observation from `battleboats/envs/observation.py`, plus the 8-dimensional globals vector from the same module. Tokens are a variable-length set: friendly ships, friendly ports, enemy ship sightings (fresh and stale), enemy port sightings (fresh and stale), and coastline tiles. See §10.

**Architecture.** Reuse the transformer encoder already in the repo:

- `TransformerValueModel` in `battleboats/training/transformer.py` projects each 76-dim token to `d_model=64`, runs a 2-layer 4-head Transformer encoder with padding masks, mean-pools real tokens, then a small MLP value head with `tanh` so the output is in `[-1, 1]`.
- That class currently **does not ingest globals**. Own cash is also written onto the friendly home-port token (`port_state[2]`), so a tokens-only net can still see cash. Prefer extending the encoder the way `PolicyNetwork` already does: project the 8-dim globals vector to `d_model` and prepend it as a globals token, then pool / read it for the value head.

**Labels.** From the observing seat's point of view, on **decisive** games: `+1` if that seat eventually wins, `-1` if it loses. Every decision (or a regular subsample of decisions) in a finished game is a training row, labelled with the same terminal outcome. Truncated / drawn games (hit the turn cap with no winner) should be reported, then either dropped or labelled `0`; the last value-net attempt died on too many draws, so the first run should **train on decisive games only** and keep the draw rate as a harvest diagnostic.

**Gate (must pass before any search is built).** On a held-out set of decisive games, the net must beat:

1. A coin flip (accuracy > 50%, and MSE / Brier clearly better than predicting 0 or a 50/50 probability).
2. A simple baseline that does not use learned features — for example a logistic or linear model on a handful of counts (own vs sighted enemy combat strength, cash, port count, turn fraction).

Report accuracy, MSE or Brier score, and a calibration curve. If it cannot beat those baselines, stop and revisit the data mix or representation; do not start MCTS.

### 1.3 MCTS guided by the value net

Only after the gate in §1.2.

The repo already has **perfect-information** UCT in `battleboats/agents/godmode_mcts.py` (clone the engine, UCB1, expand, rollout or heuristic eval, negamax backup). That search is not usable as a fog-of-war player: it sees the true board, and its random rollouts rarely finish a game. The new search should reuse the tree / UCT / `engine.clone()` machinery where it helps, but change two things:

1. **Leaf evaluation** comes from the value net, not from a long random rollout. Evaluate the leaf's fogged observation (or an observation consistent with the current determinization) and back up that scalar in `[-1, 1]`.
2. **Determinization for fog.** The searcher does not know where unseen enemy ships are, and stale sightings may be wrong. At the root, sample one or more complete worlds consistent with what the acting seat has actually seen: place unseen enemy ships in unobserved water, and treat stale last-known positions as noisy rather than certain. Run UCT in each sampled world (or average root values / visits across worlds) and pick an action that is legal in the real, fogged state. Do not let the search take an action that requires knowledge the player does not have.

Engine combat already uses a seeded RNG, and `clone()` deep-copies it, so search does not consume the live game's randomness.

### 1.4 Success metrics

- **Harvest:** decisive rate (target: comparable to the 76% 50-game check, reported with a confidence interval), even-ish win split by seat, buy diversity across all eight types, game-length distribution, and a breakdown by matchup (scripted vs scripted, scripted vs tactical, etc.).
- **Value net:** held-out accuracy and Brier/MSE vs coin flip and the simple baseline; calibration; no large seat bias.
- **MCTS (after the gate):** vs the scripted agent and vs the tactical agent, on 32×16, report win rate, decisive rate, and average length. A first success is a clearly better-than-even record against the scripted agent under fog, at a thinking budget that still runs on CPU.

### 1.5 Open questions

- How many determinizations per root move are enough, and should stale sightings be resampled or pinned?
- Should the value net also be trained on the opponent's fogged view of the same state (both seats already recorded) even when that seat did not act?
- Is 32×16 the only harvest size, or should a slice of 16×8 / 48×24 be included so the net generalizes across board size (positions are Fourier features of raw coordinates; board width/height are also in the token and in globals)?
- The current scripted agent may still be too weak or too narrow (straight-line Manhattan, full information). If harvest diversity or the value-net gate disappoints, a stronger fog-limited scripted policy (the intended use of the pseudocode this brief will be given to) should replace or sit alongside it. The BAT-26 HTN teacher remains the longer-term data source.
- Placeholder spotting values in the Landing / Merchant / Builder / Port rows and columns of `visibilityMatrix.csv` are noted in `docs/combat_and_scouting.md` as carried over from an older formula and still tunable. This plan uses the matrix as it stands.

### 1.6 Existing code to reuse (and what not to reuse blindly)

| Piece | Path | Notes |
|---|---|---|
| Engine / rules | `battleboats/core/gameEngine.py` | Sole owner of rules. |
| Ship / combat / spotting tables | `battleboats/core/config/*.csv`, `shipyard/ship_data.py` | Source of truth for numbers. |
| Fogged observation | `battleboats/envs/observation.py` | 76-dim tokens + 8-dim globals. |
| Scripted agent | `battleboats/agents/scripted_agent.py` | Fog-limited doctrine agent. The old full-information baseline was removed. |
| Tactical baseline | `battleboats/agents/tactical.py` | Fog-limited, also a random build queue. |
| Value transformer | `battleboats/training/transformer.py` | Encoder + value head; add globals. |
| Policy encoder | `battleboats/training/policy.py` | Already prepends a globals token. |
| God-mode MCTS | `battleboats/agents/godmode_mcts.py` | Perfect-info UCT; reuse tree code, not the fog model. |
| Older harvest / train scripts | `scripts/harvest.py`, `scripts/train_transformer.py` | Built for **MCTS root-value** labels from god-mode search, often on truncated games. Do not feed those labels into this plan. The JSONL-shard + `TransformerValueModel` training loop is reusable if the rows are fogged observations labelled with terminal outcomes. |

---

## 2. Game overview

Battleboats is a two-player, zero-sum, imperfect-information, turn-based naval wargame on a rectangular grid. Tiles are water or land. Ships occupy water tiles, at most one ship per tile, and move in the four cardinal directions. Ports sit on land. Each player starts with a **home port** and usually some **satellite ports** on their half of the map, **250 cash**, and **no ships**.

The only way to win is to **capture the enemy home port** with a landing ship. Hitting the turn cap with no winner is a draw (the environment truncates; the engine itself leaves `winner is None`). There is no "eliminate all ships" victory, and cash or ship count never ends the game by itself.

Players alternate. Player 0 moves first. A "turn" in the engine is one player's burst of **atomic actions** (move one tile, fire once, buy a ship, load, unload, build or capture a port, or end the turn). The integer `engine.turn` is a **round counter**: it starts at 0 and increments after player 1 finishes their turn.

---

## 3. Turn structure and legal actions

The engine is an action-by-action state machine (`gameEngine.step`). On a player's turn they may issue any sequence of legal actions, involving any mix of their ships and ports, and must eventually play `EndTurnAction`. `enumerate_legal(player_id)` always includes `EndTurnAction` for the current player when the game is not over.

Movement and combat are spent on the acting ship, then reset when that player ends their turn:

- A ship may move up to `speed` water tiles per turn, **one cardinal tile per `MoveAction`**. The destination must be in bounds, water, and unoccupied. There is no diagonal move and no moving onto land.
- A combat ship may make **at most one attack** per turn (`has_attacked`).
- **A ship bought this turn cannot take any action** until its owner's next turn (`bought_this_turn`). The engine also marks its move and attack as already spent so observations stay consistent.
- **A port that was captured or built this turn cannot build ships** until the owner's next turn (`locked_this_turn`). That lock clears when the owner ends a turn.

Action types (`battleboats/core/actions.py`):

| Action | Who | Rule in brief |
|---|---|---|
| `MoveAction(ship_id, destination)` | Any own ship with remaining speed | One cardinal step onto empty water. |
| `AttackAction(attacker_id, target_id)` | Own combat ship that has not attacked | Target is an enemy ship; Manhattan distance ≤ the attacker's `attack_range` **and** ≤ the attacker's spotting distance against that target type. Misses still spend the attack. |
| `BuildShipAction(port, spawn_position, ship_type)` | Owned, unlocked port | Spawn on a water tile adjacent to the port; pay the type's cost. `enumerate_legal` picks a single canonical spawn per (port, type): the adjacent open water tile closest to the enemy home, ties broken lexicographically. |
| `BuildPortAction(builder_ship_id, target)` | Own builder | Target is an adjacent land tile that is not already a port. The builder is **consumed**. The new port is not a home port, starts with stockpile 0, and is locked this turn. |
| `CapturePortAction(landing_ship_id, target)` | Own landing ship | Target is an adjacent port the player does not already own. The landing ship **is not consumed**. The port's stockpile is **lost**. The port is locked this turn. If the target is the previous owner's **home port**, the acting player wins immediately. |
| `MerchantLoadAction(merchant_id, port)` | Own merchant | Adjacent **owned non-home** port with a stockpile. Transfer `min(stockpile, 100 - cargo)` materials onto the ship. |
| `MerchantUnloadAction(merchant_id, port)` | Own merchant with cargo > 0 | Adjacent **home port only**. Cash increases by `cargo * 2`; cargo goes to 0. Unloading at a non-home port is illegal (it would only dump cargo back into a stockpile and create load/unload loops). |
| `EndTurnAction()` | Current player, always legal | Reset that player's ship flags and port locks, increment the round if player 1 just finished, hand control to the opponent, then pay the **incoming** player their start-of-turn income. |

Illegal actions are silent no-ops in the engine; agents should only play actions from `enumerate_legal`.

**Income timing matters.** `reset()` does **not** pay income. Player 0's first turn starts with only the 250 starting cash. When player 0 ends their turn, player 1 receives income and then acts. When player 1 ends their turn, `engine.turn` becomes 1 and player 0 receives income. So player 1 is paid once before their first action, and player 0 is not.

---

## 4. Economy and ports

Constants from `battleboats/core/gameEngine.py`:

| Constant | Value | Meaning |
|---|---:|---|
| `STARTING_CASH` | 250 | Each seat, at `reset()`. |
| `PORT_PRODUCTION` | 25 | Materials produced by each owned port at the start of that owner's turn. |
| `CASH_PER_MATERIAL` | 2 | Conversion rate at the home port (production and merchant deliveries). |
| `MERCHANT_CAPACITY` | 100 | Max materials a merchant can carry. Cargo persists across turns until unloaded. |

- The **home port** converts its 25 materials immediately into **50 cash** at the start of the owner's turn.
- Each **other owned port** adds 25 to its **stockpile**. That stockpile is useless until a merchant loads it and unloads at home: a full 100-material delivery is worth **200 cash**.
- Capturing a port zeros its stockpile.
- A newly built port is never a home port. The `is_home` flag is assigned at game start and does not change when a port is captured. (Capturing the enemy home ends the game, so a captured former home is mainly an observation feature.)
- Standard generated maps give every port to player 0 or player 1 at kickoff; there are no neutral ports in the current generator.

Cash is private. The opponent's cash is never in a fogged observation.

---

## 5. Ship types

Source of truth: `battleboats/core/config/shipTypes.csv`, loaded by `battleboats/core/shipyard/ship_data.py`. Detection is **not** a per-ship stat; it comes from the spotting matrix in §7.

| Type | Cost | Strength | Attack range | Speed | Role / special rules |
|---|---:|---:|---:|---:|---|
| Carrier | 250 | 100 | 12 | 6 | Combat. Longest attack range and (among ships) very long port spotting. Pays the whole starting bank. |
| Battleship | 200 | 125 | 4 | 3 | Combat. Highest strength, slow. |
| Cruiser | 150 | 75 | 3 | 5 | Combat. Mid-weight, fast. |
| Destroyer | 100 | 50 | 3 | 5 | Combat. Dedicated sub-hunter (attack modifier 2.0 vs Submarine). Cheap and fast. |
| Submarine | 125 | 50 | 3 | 3 | Combat. Hard to see (most hulls spot it at 1; Destroyer at 3). Strong attack modifiers vs Carrier (3.0) and Battleship (4.0); weak vs Destroyer (0.5). |
| Landing | 250 | 0 | 0 | 3 | **Cannot attack.** Always destroyed if attacked (strength 0). The only unit that can **capture** a port. Not consumed on capture. |
| Merchant | 100 | 0 | 0 | 3 | **Cannot attack.** Always destroyed if attacked. Carries up to 100 materials from owned satellite ports to the home port. |
| Builder | 250 | 0 | 0 | 3 | **Cannot attack.** Always destroyed if attacked. Consumed to **found a new port** on an adjacent empty land tile. |

There are exactly these eight buyable types. `enumerate_legal` offers a type whenever the player can afford it and has an unlocked port with a free adjacent water tile.

---

## 6. Combat

An attack is a single Bernoulli trial. There is **no hit-point track**: the defender is either destroyed or it is not, and a miss does nothing else. Combat is the only stochastic engine rule. It uses the engine's seeded RNG, so a fixed seed makes the same attacks (and therefore the same game, given the same actions) reproducible. `clone()` deep-copies the RNG.

Default kill-curve exponent `kill_curve_k = 2.0`.

\[
P(\text{kill}) = \frac{x^{k}}{1 + x^{k}}, \qquad
x = \frac{S_{\text{attacker}} \cdot \mu(t_a, t_d)}{S_{\text{defender}}}, \qquad k = 2.
\]

- If the defender's strength is 0 (Landing, Merchant, Builder), \(P(\text{kill}) = 1\).
- Equal effective strength (\(x = 1\)) is a coin flip.
- Larger \(k\) makes small strength edges more decisive. The default is 2.

Attack modifiers \(\mu\) from `battleboats/core/config/attackModifiers.csv` (attacker = row, defender = column). Missing entries would default to 1.0; every combat-vs-combat pair is present. Non-combat types have no row or column; they always die when attacked.

| atk ↓ \ def → | Carrier | Battleship | Cruiser | Destroyer | Submarine |
|---|---:|---:|---:|---:|---:|
| Carrier | 1.0 | 1.5 | 1.0 | 0.75 | 1.0 |
| Battleship | 1.0 | 1.0 | 1.5 | 1.0 | 0.75 |
| Cruiser | 1.25 | 1.0 | 1.0 | 1.5 | 1.0 |
| Destroyer | 1.25 | 1.0 | 1.0 | 1.0 | 2.0 |
| Submarine | 3.0 | 4.0 | 2.0 | 0.5 | 1.0 |

Worked examples at \(k = 2\), using the table and the strength column (these are computed from the formula, not separate data):

| Attacker vs defender | \(x\) | \(P(\text{kill})\) |
|---|---|---:|
| Any even same-type fight (μ = 1, equal S) | 1 | 0.50 |
| Destroyer vs Submarine | \(50 \cdot 2 / 50 = 2\) | 0.80 |
| Submarine vs Destroyer | \(50 \cdot 0.5 / 50 = 0.5\) | 0.20 |
| Submarine vs Carrier | \(50 \cdot 3 / 100 = 1.5\) | 0.692 |
| Submarine vs Battleship | \(50 \cdot 4 / 125 = 1.6\) | 0.719 |
| Carrier vs Battleship | \(100 \cdot 1.5 / 125 = 1.2\) | 0.590 |
| Battleship vs Carrier | \(125 \cdot 1 / 100 = 1.25\) | 0.610 |

**Range vs spotting.** Being within `attack_range` is not enough. The attacker itself must also be within its spotting distance of that target type. A Carrier's attack range is 12, but it only spots a Destroyer at 3 and a Submarine at 2, so those are the effective engagement ranges. Another friendly scout seeing the target does not relax that check. If the attacker can spot the target, that target is also a fresh sighting for the attacking player.

---

## 7. Fog of war and spotting

Spotting is **asymmetric** and uses Manhattan distance. Observer \(o\) sees enemy target \(T\) iff **any** of \(o\)'s ships or **owned ports** is within the matrix range for that (spotter, target) pair:

\[
\text{see}(o \to T) \iff \exists\, s \in \{\text{ships and ports of } o\} :\ d(s, T) \le R(s, T).
\]

\(R\) is `battleboats/core/config/visibilityMatrix.csv` (row spots column, in tiles). Ports have their own `Port` row and column.

| spotter ↓ \ target → | Carrier | Battleship | Cruiser | Destroyer | Submarine | Landing | Merchant | Builder | Port |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Carrier | 6 | 6 | 4 | 3 | 2 | 8 | 8 | 8 | 16 |
| Battleship | 4 | 4 | 3 | 3 | 1 | 4 | 4 | 4 | 8 |
| Cruiser | 4 | 4 | 3 | 3 | 1 | 6 | 6 | 6 | 12 |
| Destroyer | 4 | 4 | 3 | 3 | 3 | 8 | 8 | 8 | 16 |
| Submarine | 4 | 4 | 3 | 2 | 1 | 2 | 2 | 2 | 4 |
| Landing | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| Merchant | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| Builder | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| Port | 16 | 16 | 12 | 12 | 1 | **8** | 8 | 8 | 16 |

So a **port spots a landing ship 8 tiles away**. A Destroyer spots a Submarine at 3, but a Submarine spots a Destroyer at 2. A Port spots a Submarine at only 1.

**Sighting records** (`Sighting`, `PortSighting` in `battleboats/core/sighting.py`):

- **Fresh:** the target is currently in range of some friendly ship or port. Position (and, for ports, current owner) is live.
- **Stale:** the observer has lost the view. The last-known position, type, and (for ports) last-known owner stay frozen, with `fresh=False` and `turn_seen` left at the last fresh observation.
- A newly visible enemy **overwrites** the record (this is also how a ship moving while watched updates its tracked position, and how a witnessed port capture updates `last_known_owner`).
- **Witnessed kills:** if a player had a *fresh* sighting of a ship that is destroyed, that sighting is deleted (they saw it die). A *stale* sighting of a ship that dies out of view is kept — the observer may still think it is alive at the old tile.
- Once an enemy port has been spotted, `is_home` is revealed and stays in the record.
- Enemy **stockpile** is never revealed, even for a fresh port sighting.
- Enemy **cargo, has_attacked, and tiles already moved** are never revealed on ship sightings.
- A player's own ships, own ports, own cash, and own stockpiles are fully known.
- The fogged observation does **not** include a full terrain grid. It does include every **coastline** tile: land adjacent to water that is not currently a port. Those tiles are static map features and are legal builder targets. Inland land that is not next to water does not appear as a token. Players do **not** automatically know the enemy home location at kickoff; they learn it when some friendly ship or port first spots that port (on a small enough map, the home may already be inside a friendly port's 16-tile port-to-port range).

Sightings are refreshed after every `step()`.

---

## 8. Maps

A map is a JSON file: `width`, `height`, a list of land tiles (everything else is water), and `ports` keyed by player id. The first port listed for each player is the **home port** (`gameEngine._infer_home_ports`). Player 0's ports are generated on the left half, player 1's on the right.

Land is a per-tile Bernoulli(`landFraction`). Ports are placed on land in each half (if there is not enough land, some water tiles are converted). Optional constraint `minPortEnemyHomeWater`: every port's adjacent spawn water tile must be at least that many water moves from a tile next to the enemy home. At 7, with landing speed 3 and the "cannot act the turn it is bought" rule, a landing ship bought on turn 0 cannot capture before turn 3.

The play-UI / training **presets** (`battleboats/ui/maps.py`) — these are the maps to use, not the retired 50×25 / 100×50 / 160×80 YAML sizes still sitting on disk:

| Preset | Size | Ports per side | Land fraction | Min water moves, spawn → enemy home |
|---|---|---:|---:|---:|
| `16x8` | 16×8 | 2 | 0.02 | 7 |
| `24x12` | 24×12 | 4 | 0.01 | 7 |
| `32x16` | 32×16 | 3 | 0.02 | 7 |
| `48x24` | 48×24 | 4 | 0.02 | 7 |
| `open-8x8` | 8×8 | 2 | 0 (open water) | 0 |
| `open-16x8` | 16×8 | 2 | 0 (open water) | 0 |

Pool admission (`battleboats/envs/map_pool.py`): every port has a water neighbor; a water path joins the two homes; homes are at least half the board width apart (Manhattan); homes sit similarly far from the midline; satellite spread is not wildly asymmetric; and the min-spawn-water constraint holds when set.

This plan's default harvest size is **32×16**.

---

## 9. Win conditions and turn caps

- **Win:** the acting player captures the opponent's home port. `engine.winner` is set to that player id. The AEC env (`battleboats/envs/battleboats_aec.py`) then gives `+1` / `-1` terminal rewards.
- **Draw / truncation:** `engine.turn >= max_turns` with no winner. The default AEC cap is `DEFAULT_MAX_TURNS = 500` (`battleboats/envs/base_aec.py`). Self-play and the play UI scale a **base of 80 turns** (and a step budget) by home Manhattan distance over a reference of 14 (the old 8×8 boards):

  \[
  \text{scale} = \max\!\bigl(1,\; d_{\text{homes}} / 14\bigr), \qquad
  \text{turns} = 10 \times \bigl\lceil 80 \cdot \text{scale} / 10 \bigr\rceil
  \]

  never below 80. On 32×16, homes are at least 16 apart (half the board width), so the scaled cap is at least 100 turns and grows with the actual home distance of that generated map. Harvest should use this same scaled cap (or record whichever cap was used).

`engine.turn` counts **rounds**, not atomic actions. A 50-turn game is 50 moves by each player after player 1's first end-turn, not 50 atomic actions.

---

## 10. How board positions are represented to a learned agent

Module: `battleboats/envs/observation.py`. `build_observation(engine, player_id)` returns:

```text
{
  "entity_tokens": float32 array of shape (N, 76),
  "globals":      float32 array of shape (8,),
}
```

\(N\) is variable. Token order is deterministic within a call, which matters because policy pointer heads index into this list:

1. Friendly ships (live `engine.ships` for this owner)
2. Friendly ports (live `engine.ports` for this owner)
3. Enemy ship sightings (fresh + stale; **never** hidden live ships)
4. Enemy port sightings (fresh + stale; **never** hidden live enemy ports)
5. Coastline tiles (land adjacent to water, excluding current ports)

There is **no positional encoding on the token axis**; entities are a set. Geometry lives *inside* each token as sinusoidal features of raw cell coordinates.

### 10.1 The 76-dim token layout

| Field | Dims | Contents |
|---|---:|---|
| Type one-hot | 10 | 8 ship types (enum order: Carrier, Battleship, Cruiser, Destroyer, Submarine, Merchant, Landing, Builder), then Port, then coastline |
| Absolute Fourier position | 32 | 8 geometric wavelengths from 2 to 4096, sin and cos, for raw \(x\) and raw \(y\) |
| Home-relative Fourier position | 16 | 4 wavelengths from 2 to 4096, sin and cos, for \(x - x_{\text{home}}\) and \(y - y_{\text{home}}\) |
| Owner | 2 | `is_friendly`, `is_enemy` relative to the observer |
| Stats | 6 | speed, strength, attack range, "mean distance this hull is spotted", "mean distance this hull spots others", cost — each normalized by the max (or, for the two spotting summaries, by the max ship-vs-ship spotting entry). Public type table; not a fog leak. |
| Ship state | 3 | cargo / 100, `has_attacked`, tiles moved this turn / speed. **Zero on enemy sightings.** |
| Port state | 3 | stockpile / 1500, `is_home`, home-port cash / 1500. Enemy stockpile and cash stay 0. Own cash is written only on the friendly **home** port token. |
| Sighting state | 2 | `fresh` flag; staleness `(turn - turn_seen) / 500`. Zero on friendlies and coastline. |
| Board size | 2 | width / 256, height / 256 (every token). |

Unused fields in a token stay zero. Coastline tokens only fill type, position, and board size.

### 10.2 Globals (8 dim)

Projected to a globals token in `PolicyNetwork` (not yet in `TransformerValueModel`):

| Index | Feature | Scale |
|---:|---|---|
| 0 | Own cash | / 1500 |
| 1 | Own ship count | / 50 |
| 2 | Own port count | / 25 |
| 3 | Fresh enemy ship sightings | / 50 |
| 4 | Fresh enemy port sightings | / 25 |
| 5 | `engine.turn` | / 500 |
| 6 | Board width | / 256 |
| 7 | Board height | / 256 |

Opponent cash is not present. Legal actions are **not** inside the observation; they are provided separately (env `infos` / an action-mask adapter that maps token indices back to engine `Action` objects).

---

## 11. Scripted agent (the former full-information baseline was removed)

The full-information teacher `scripted_action` in `battleboats/agents/scripted.py` (PR #62) has been deleted. It is not an alias of the scripted agent. The live fog-limited agent is `battleboats/agents/scripted_agent.py` (`ScriptedAgent` / `scripted_agent_action`), with `fog_commander` kept only as a config and import alias. Design and deviations: `docs/scripted_agent.md`.

The **tactical** agent remains the other fog-limited opponent: a random build queue, enemy information only from sightings, landings that chase the nearest known capturable port, merchants that flee visible threats, and combat that uses a local force-balance with softmax noise.

---

## 12. Source files

| Topic | Path |
|---|---|
| Engine, cash, income, actions, win | `battleboats/core/gameEngine.py` |
| Action dataclasses | `battleboats/core/actions.py` |
| Ship types and stats loader | `battleboats/core/shipyard/ship_type.py`, `ship_stats.py`, `ship_data.py`, `ship.py` |
| Stats CSV | `battleboats/core/config/shipTypes.csv` |
| Attack-modifier CSV | `battleboats/core/config/attackModifiers.csv` |
| Spotting-distance CSV | `battleboats/core/config/visibilityMatrix.csv` |
| Sighting records | `battleboats/core/sighting.py` |
| Ports / players | `battleboats/core/port.py`, `battleboats/core/player.py` |
| Map grid | `battleboats/core/map/Map.py` |
| Map generator | `battleboats/core/config/generate_map.py` |
| Map-pool checks and turn scaling | `battleboats/envs/map_pool.py` |
| Play-UI map presets | `battleboats/ui/maps.py` |
| Fogged observations | `battleboats/envs/observation.py` |
| AEC env, default cap, rewards | `battleboats/envs/base_aec.py`, `battleboats/envs/battleboats_aec.py` |
| Scripted agent / tactical / random / god-mode MCTS | `battleboats/agents/scripted_agent.py`, `tactical.py`, `random_agent.py`, `godmode_mcts.py` |
| Value transformer | `battleboats/training/transformer.py` |
| Combat write-up (same numbers) | `docs/combat_and_scouting.md` |
