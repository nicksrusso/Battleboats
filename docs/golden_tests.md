# Battleboats Golden Parity Test Suite

This document specifies the **language-agnostic behavioral contract** and
**canonical state / replay fixture format** that both the reference Python
engine (`battleboats/core/gameEngine.py`) and any future Rust core (via PyO3)
must satisfy.

## 1. Scope and Philosophy

- **Behavior-level only:** Golden tests interact with the engine solely through
  the `EngineBackend` / `EngineInstance` adapter in `tests/golden/backend.py`.
  Tests never call private Python methods (`_spawn_ship`, `_refresh_sightings`,
  `_destroy_ship`, `_resolve_attack`) or mutate internal fields directly.
- **Backend-parametrized:** `tests/golden/conftest.py` exposes the `backend`
  pytest fixture parametrized over `available_backends()`. Currently only
  `"python"` is registered; once the Rust PyO3 module builds, add a
  `RustEngineBackend` entry there so every golden test and replay runs against
  both implementations.
- **Fast:** The entire `tests/golden/` suite is designed to run well under 60 s.

## 2. Engine Backend Contract (`tests/golden/backend.py`)

An `EngineBackend` provides:

- `name: str` (`"python"`, `"rust"`, ...)
- `create(map_path, kill_curve_k=2.0) -> EngineInstance`
- Action constructors (`move_action`, `attack_action`, `build_ship_action`,
  `build_port_action`, `capture_port_action`, `merchant_load_action`,
  `merchant_unload_action`, `end_turn_action`) and `encode_action` /
  `decode_action`.
- Static rule constants & lookup tables:
  - `starting_cash() -> 250`
  - `port_production() -> 25`
  - `cash_per_material() -> 2`
  - `merchant_capacity() -> 100`
  - `ship_types() -> ("Carrier", "Battleship", "Cruiser", "Destroyer", "Submarine", "Merchant", "Landing", "Builder")`
  - `base_stats(ship_type) -> {"speed": int, "cost": int, "strength": float, "attack_range": int}`
  - `attack_modifier(attacker_type, defender_type) -> float`
  - `spot_distance(spotter_key, target_key) -> float` (keys are the 8 ship types plus `"Port"`).

An `EngineInstance` provides:

- `reset(seed=None)`
- `reset_from_scenario(scenario)` — loads `scenario["map_path"]`, resets with
  `scenario["seed"]`, overrides `player_0.cash` / `player_1.cash`, spawns the
  listed ships (`bought_this_turn=False`, `built_turn=-1`), sets `turn=0`,
  `current_player=0`, `winner=None`, and refreshes sightings.
- `step(action)` — no-op if `winner is not None` or if action preconditions fail;
  otherwise applies the action and refreshes sightings for both players.
- `enumerate_legal(player_id) -> list[Action]` — empty if terminal or
  `player_id != current_player`; otherwise all legal actions matching `step()`
  preconditions (with canonical spawn tile for `BuildShipAction`), plus
  `EndTurnAction`.
- `is_terminal() -> bool` (`winner is not None`)
- `clone() -> EngineInstance` — deep independent snapshot.
- Fog-of-war queries (sorted deterministically):
  - `visible_enemy_ships(player_id)`
  - `known_enemy_ships(player_id)`
  - `visible_enemy_ports(player_id)`
  - `known_enemy_ports(player_id)`
- `canonical_state() -> dict` and `state_hash() -> str`.

## 3. Canonical State Serialization & Hash Specification

`canonical_state()` returns a JSON object with `schema_version = 1`. Every
collection is sorted deterministically so hash-map / set iteration differences
between Python and Rust never affect the hash:

```json
{
  "schema_version": 1,
  "width": 32,
  "height": 16,
  "turn": 0,
  "current_player": 0,
  "winner": null,
  "kill_curve_k": 2.0,
  "next_ship_id": 0,
  "terrain_land": [[x, y], ...],
  "players": [
    {
      "id": 0,
      "home_port": [x, y],
      "cash": 250,
      "ship_sightings": [
        {
          "ship_id": 1,
          "type": "Destroyer",
          "position": [x, y],
          "turn_seen": 0,
          "fresh": true
        }
      ],
      "port_sightings": [
        {
          "position": [x, y],
          "last_known_owner": 1,
          "is_home": true,
          "turn_seen": 0,
          "fresh": true
        }
      ]
    }
  ],
  "ports": [
    {
      "position": [x, y],
      "owner": 0,
      "is_home": true,
      "stockpile": 0,
      "locked_this_turn": false
    }
  ],
  "ships": [
    {
      "id": 0,
      "owner": 0,
      "type": "Destroyer",
      "position": [x, y],
      "tiles_moved_this_turn": 0,
      "has_attacked": false,
      "cargo": 0,
      "bought_this_turn": true,
      "built_turn": 0
    }
  ]
}
```

Sorting rules:
- `terrain_land`: lexicographically by `(x, y)`
- `players`: by `id` (`0`, then `1`)
- `ship_sightings`: by `ship_id` ascending
- `port_sightings`: lexicographically by `(x, y)`
- `ports`: lexicographically by `(x, y)`
- `ships`: by `id` ascending

Hash formula:
```
state_hash = sha256(utf8(json.dumps(canonical_state, sort_keys=True, separators=(",", ":")))).hexdigest()
```
Internal PRNG state is intentionally excluded from `canonical_state` because
PRNG representations differ across languages; RNG parity is checked separately.

## 4. Compact Action Encoding

Matches `scripts/harvest_value.py`:
- Move: `["m", ship_id, x, y]`
- Attack: `["a", attacker_id, target_id]`
- BuildShip: `["b", port_x, port_y, spawn_x, spawn_y, ship_type_str]`
- BuildPort: `["p", builder_ship_id, target_x, target_y]`
- CapturePort: `["c", landing_ship_id, target_x, target_y]`
- MerchantLoad: `["l", merchant_id, port_x, port_y]`
- MerchantUnload: `["u", merchant_id, port_x, port_y]`
- EndTurn: `["e"]`

## 5. Golden Replay Fixtures (`tests/golden/fixtures/replays.json.gz`)

Generated by `tests/golden/fixtures/generate_fixtures.py` across multiple map
sizes (`16x8`, `32x16`, `48x24`) and agent mixes (scripted agent doctrine
matchups, scripted agent vs Random, Random vs Random).

Each fixture entry contains:
- `id`, `matchup`, `map_preset`, `map` (full map JSON), `seed`, `kill_curve_k`
- `actions`: full encoded action list
- `step_hashes`: per-step SHA-256 hashes (1-indexed after each action; `initial_hash` before action 0)
- `combat_outcomes`: `[[step_idx, killed_bool], ...]` for every attack that resolved RNG (allows a Rust port to test either strict PCG64 RNG parity or rules-only parity with recorded combat rolls)
- `obs_checkpoints`: sampled fog-of-war observation summaries (`visible_enemy_ships`, `known_enemy_ships`, `visible_enemy_ports`, `known_enemy_ports`, plus `obs_token_count` and `obs_globals`) at key points in the game
- `final_turn`, `winner`, `final_state_hash`

### RNG Note for the Rust Port

`gameEngine` uses `numpy.random.default_rng(seed)` (PCG64) solely in
`_resolve_attack` (`float(rng.random()) < p`). To pass strict seeded replay
parity (`test_golden_replays.py`) without overriding combat rolls, the Rust core
should either match NumPy's `SeedSequence` + `PCG64` or accept an optional RNG
callback/seed stream. The Rust backend does that: `test_golden_replay_exact_parity` calls
`set_combat_outcomes` when the instance provides it, and Rust applies those
recorded kills instead of matching NumPy PCG64. Python ignores the hook and
still rolls with `default_rng`.

## 6. Running Only the Golden Suite

```bash
poetry run pytest tests/golden/ -v
```
