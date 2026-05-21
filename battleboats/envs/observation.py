"""Observation construction for the AEC env.

Builds a fog-of-war-filtered observation dict for a given player from the
engine state. The observation is consumed by the policy network's encoder
(transformer over entity tokens + small MLP over globals).

Token schema (per docs/policy_architecture.md):
    type_onehot | position | owner | stats | ship_state | port_state | sighting_state

Variable-length token list — pointer heads handle the variable cardinality
natively, so no max-N padding is needed at this layer. (Batching may pad
later in the training loop; that's not the env's concern.)

Fog-of-war discipline (load-bearing):
    Enemy ships  → from player.sightings (fresh + stale), NEVER engine.ships.
    Enemy ports  → from player.port_sightings, NEVER engine.ports for enemies.
    Friendly entities → directly from engine.ships / engine.ports.
    Coastline tiles → static map property; safe to read from engine.map.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType

if TYPE_CHECKING:
    from battleboats.core.gameEngine import gameEngine


# ----------------------------------------------------------------------------
# Token field layout
# ----------------------------------------------------------------------------

TOKEN_TYPE_ONEHOT_DIM: int = 10  # one for each ship type plus ports and land
TOKEN_POSITION_DIM: int = 2  # x/y position
TOKEN_OWNER_DIM: int = 2  # is_friendly, is_enemy (relative to observer)
TOKEN_STATS_DIM: int = 6  # one for each ship stat
TOKEN_SHIP_STATE_DIM: int = 3  # cargo carried, has_attacked, tiles moved this turn
TOKEN_PORT_STATE_DIM: int = 2  # mats in stockpile, is home
TOKEN_SIGHTING_STATE_DIM: int = 2  # fresh, staleness

TOKEN_DIM: int = (
    TOKEN_TYPE_ONEHOT_DIM
    + TOKEN_POSITION_DIM
    + TOKEN_OWNER_DIM
    + TOKEN_STATS_DIM
    + TOKEN_SHIP_STATE_DIM
    + TOKEN_PORT_STATE_DIM
    + TOKEN_SIGHTING_STATE_DIM
)

# Normalization scales — pick sensible defaults; tune later if values consistently
# saturate near 0 or 1 in early training.
CASH_SCALE: float = 1500
STOCKPILE_SCALE: float = 1500
MAX_TURNS_SCALE: int = 500

# Ship-stat scales derived from BASE_STATS so they auto-track CSV retunes.
# Each scale is the max value of that stat across all ship types, so dividing
# yields a [0, 1] normalized field.
_ALL_STATS = list(BASE_STATS.values())
SPEED_SCALE: float = max(s.speed for s in _ALL_STATS)
STRENGTH_SCALE: float = max(s.strength for s in _ALL_STATS)
ATTACK_RANGE_SCALE: float = max(s.attack_range for s in _ALL_STATS)
VISIBILITY_SCALE: float = max(s.visibility for s in _ALL_STATS)
SCOUTING_SCALE: float = max(s.scouting for s in _ALL_STATS)
COST_SCALE: float = max(s.cost for s in _ALL_STATS)

# Precompute mapping for ship type to token
SHIP_TYPE_INDEX: Dict[ShipType, int] = {t: i for i, t in enumerate(ShipType)}
PORT_TYPE_INDEX: int = len(SHIP_TYPE_INDEX)  # 8
COASTLINE_TYPE_INDEX: int = PORT_TYPE_INDEX + 1  # 9

# Field offsets — start index of each field group inside a single token row.
# Use as: row[STATS_OFFSET + i] = ... rather than counting dims yourself.
TYPE_OFFSET: int = 0
POSITION_OFFSET: int = TYPE_OFFSET + TOKEN_TYPE_ONEHOT_DIM
OWNER_OFFSET: int = POSITION_OFFSET + TOKEN_POSITION_DIM
STATS_OFFSET: int = OWNER_OFFSET + TOKEN_OWNER_DIM
SHIP_STATE_OFFSET: int = STATS_OFFSET + TOKEN_STATS_DIM
PORT_STATE_OFFSET: int = SHIP_STATE_OFFSET + TOKEN_SHIP_STATE_DIM
SIGHTING_STATE_OFFSET: int = PORT_STATE_OFFSET + TOKEN_PORT_STATE_DIM

# ----------------------------------------------------------------------------
# Globals field layout
# ----------------------------------------------------------------------------
# Fixed-length vector of board-wide summary features that don't fit as
# per-entity tokens. Concatenated with the pooled token embedding before
# feeding the policy/value heads. Cheap shortcut for facts the network would
# otherwise have to aggregate via attention.
#
# Fields (in order written by build_globals):
#   0: own cash                   (cash / CASH_SCALE)
#   1: own ship count             (n / SHIP_COUNT_SCALE)
#   2: own port count             (n / PORT_COUNT_SCALE)
#   3: sighted enemy ship count   (fresh only, n / SHIP_COUNT_SCALE)
#   4: sighted enemy port count   (fresh only, n / PORT_COUNT_SCALE)
#   5: turn                       (turn / MAX_TURNS_SCALE)

SHIP_COUNT_SCALE: float = 50  # generous upper bound for fielded ships per side
PORT_COUNT_SCALE: float = 25  # generous upper bound for owned ports per side

GLOBALS_DIM: int = 6


# ============================================================================
# Public API
# ============================================================================


def build_observation(engine: "gameEngine", player_id: int) -> Dict[str, Any]:
    """Build the full obs dict for `player_id`.

    Returns:
        {
            "entity_tokens": np.ndarray of shape (N, TOKEN_DIM), dtype float32
            "globals": np.ndarray of shape (GLOBALS_DIM,), dtype float32
        }

    N is variable per call. Legal-action info is delivered separately via
    the env's `infos` dict (not in the observation itself), per PettingZoo
    convention.
    """
    return {
        "entity_tokens": build_entity_tokens(engine, player_id),
        "globals": build_globals(engine, player_id),
    }


def build_entity_tokens(engine: "gameEngine", player_id: int) -> np.ndarray:
    """Assemble the (N, TOKEN_DIM) token tensor.

    Token sources, in deterministic order so token indices are stable
    within a single call (matters because policy heads point at indices):

        1. Friendly ships:  engine.ships, filtered by owner == player_id.
        2. Friendly ports:  engine.ports, filtered by owner == player_id.
        3. Enemy ship sightings:  fresh + stale records from player.sightings.
        4. Enemy port sightings:  fresh + stale from player.port_sightings.
        5. Coastline tiles:  static map property; buildable land adjacent to water.

    Each token's vector is mostly zeros except for the fields applicable to
    its kind. The per-kind _write_*_token helpers handle the field population.
    """
    tokens: List[np.ndarray] = []

    # Friendly ships
    for ship in engine.ships.values():
        if ship.owner != player_id:
            continue
        row = np.zeros(TOKEN_DIM, dtype=np.float32)
        _write_friendly_ship_token(row, ship, engine)
        tokens.append(row)

    # Friendly ports
    for port in engine.ports.values():
        if port.owner != player_id:
            continue
        row = np.zeros(TOKEN_DIM, dtype=np.float32)
        _write_friendly_port_token(row, port, engine)
        tokens.append(row)

    # Enemy ship sightings (fresh + stale)
    for sighting in engine.known_enemy_ships(player_id):
        row = np.zeros(TOKEN_DIM, dtype=np.float32)
        _write_enemy_ship_sighting_token(row, sighting, engine)
        tokens.append(row)

    # Enemy port sightings (fresh + stale)
    for port_sighting in engine.known_enemy_ports(player_id):
        row = np.zeros(TOKEN_DIM, dtype=np.float32)
        _write_enemy_port_sighting_token(row, port_sighting, engine)
        tokens.append(row)

    # Coastline tiles (static map property — buildable land adjacent to water)
    for pos in _coastline_tiles(engine):
        row = np.zeros(TOKEN_DIM, dtype=np.float32)
        _write_coastline_token(row, pos, engine)
        tokens.append(row)

    # Stack into (N, TOKEN_DIM); explicit empty case so callers get a valid
    # zero-row array rather than crashing on np.stack([]).
    if tokens:
        return np.stack(tokens)
    return np.zeros((0, TOKEN_DIM), dtype=np.float32)


def build_globals(engine: "gameEngine", player_id: int) -> np.ndarray:
    """Assemble the (GLOBALS_DIM,) global features vector.

    Field order matches the layout block at module top:
        0: own cash, 1: own ships, 2: own ports,
        3: sighted enemy ships (fresh), 4: sighted enemy ports (fresh),
        5: turn
    """
    g = np.zeros(GLOBALS_DIM, dtype=np.float32)
    player = engine.players[player_id]
    g[0] = player.cash / CASH_SCALE
    g[1] = len(player.owned_ship_ids) / SHIP_COUNT_SCALE
    g[2] = len(player.owned_port_positions) / PORT_COUNT_SCALE
    g[3] = sum(1 for s in player.sightings.values() if s.fresh) / SHIP_COUNT_SCALE
    g[4] = sum(1 for s in player.port_sightings.values() if s.fresh) / PORT_COUNT_SCALE
    g[5] = engine.turn / MAX_TURNS_SCALE
    return g


# ============================================================================
# Internal helpers
# ============================================================================


def _coastline_tiles(engine: "gameEngine") -> List[Tuple[int, int]]:
    """All land tiles adjacent to water and NOT currently ports.

    These are candidate `build_port` targets. Static for a given map; could
    be cached on env construction. For now recompute per observation — the
    cost is O(W * H * 4) which is negligible at 160x80.
    """
    m = engine.map
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    out: List[Tuple[int, int]] = []
    for x in range(m.width):
        for y in range(m.height):
            pos = (x, y)
            if not m.is_land(pos) or m.is_port(pos):
                continue
            for dx, dy in neighbors:
                nb = (x + dx, y + dy)
                if m.in_bounds(nb) and m.is_water(nb):
                    out.append(pos)
                    break
    return out


def _write_friendly_ship_token(row: np.ndarray, ship, engine: "gameEngine") -> None:
    """Populate `row` (shape (TOKEN_DIM,)) for a friendly Ship.

    Caller is expected to have zero-initialized the row (np.zeros). We only
    write the fields that apply — port_state and sighting_state stay zero.
    """
    # Type one-hot
    row[SHIP_TYPE_INDEX[ship.type]] = 1.0

    # Position, normalized to [0, 1]
    x, y = ship.position
    row[POSITION_OFFSET] = x / engine.map.width
    row[POSITION_OFFSET + 1] = y / engine.map.height

    # Owner — is_friendly=1, is_enemy=0
    row[OWNER_OFFSET] = 1.0

    # Stats, each normalized by its max across all ship types
    s = ship.stats
    row[STATS_OFFSET + 0] = s.speed / SPEED_SCALE
    row[STATS_OFFSET + 1] = s.strength / STRENGTH_SCALE
    row[STATS_OFFSET + 2] = s.attack_range / ATTACK_RANGE_SCALE
    row[STATS_OFFSET + 3] = s.visibility / VISIBILITY_SCALE
    row[STATS_OFFSET + 4] = s.scouting / SCOUTING_SCALE
    row[STATS_OFFSET + 5] = s.cost / COST_SCALE

    # Ship state — only the dynamic per-turn bits
    row[SHIP_STATE_OFFSET + 0] = ship.cargo / MERCHANT_CAPACITY
    row[SHIP_STATE_OFFSET + 1] = 1.0 if ship.has_attacked else 0.0
    row[SHIP_STATE_OFFSET + 2] = ship.tiles_moved_this_turn / s.speed


def _write_friendly_port_token(row: np.ndarray, port, engine: "gameEngine") -> None:
    """Populate `row` for a friendly Port.

    Stats and ship_state stay zero (ports aren't ships). Sighting_state stays
    zero (friendly entities aren't sightings).
    """
    # Type one-hot — the dedicated port slot
    row[PORT_TYPE_INDEX] = 1.0

    # Position, normalized
    x, y = port.position
    row[POSITION_OFFSET] = x / engine.map.width
    row[POSITION_OFFSET + 1] = y / engine.map.height

    # Owner — is_friendly=1
    row[OWNER_OFFSET] = 1.0

    # Port state
    row[PORT_STATE_OFFSET + 0] = port.stockpile / STOCKPILE_SCALE
    row[PORT_STATE_OFFSET + 1] = 1.0 if port.is_home else 0.0


def _write_enemy_ship_sighting_token(row: np.ndarray, sighting, engine: "gameEngine") -> None:
    """Populate `row` for an enemy Sighting (fresh or stale).

    Stats come from BASE_STATS[sighting.type] — ship types are public knowledge
    so this isn't a fog-of-war leak. Per-instance dynamic state (cargo,
    has_attacked, tiles_moved) is genuinely hidden and stays zero.
    """
    # Type one-hot
    row[SHIP_TYPE_INDEX[sighting.type]] = 1.0

    # Position — last-known location (current if fresh, frozen at last
    # observation if stale)
    x, y = sighting.position
    row[POSITION_OFFSET] = x / engine.map.width
    row[POSITION_OFFSET + 1] = y / engine.map.height

    # Owner — is_enemy=1 (is_friendly stays 0 from zero-init)
    row[OWNER_OFFSET + 1] = 1.0

    # Stats from the type — these are public, not hidden by fog of war
    s = BASE_STATS[sighting.type]
    row[STATS_OFFSET + 0] = s.speed / SPEED_SCALE
    row[STATS_OFFSET + 1] = s.strength / STRENGTH_SCALE
    row[STATS_OFFSET + 2] = s.attack_range / ATTACK_RANGE_SCALE
    row[STATS_OFFSET + 3] = s.visibility / VISIBILITY_SCALE
    row[STATS_OFFSET + 4] = s.scouting / SCOUTING_SCALE
    row[STATS_OFFSET + 5] = s.cost / COST_SCALE

    # Sighting state
    row[SIGHTING_STATE_OFFSET + 0] = 1.0 if sighting.fresh else 0.0
    row[SIGHTING_STATE_OFFSET + 1] = (engine.turn - sighting.turn_seen) / MAX_TURNS_SCALE


def _write_enemy_port_sighting_token(row: np.ndarray, port_sighting, engine: "gameEngine") -> None:
    """Populate `row` for an enemy PortSighting (fresh or stale).

    Stockpile stays zero — private operational info even when the port is
    sighted. is_home is included (revealed on first sighting and frozen in
    the record per design).
    """
    # Type one-hot — port slot
    row[PORT_TYPE_INDEX] = 1.0

    # Position — last-known location (ports are stationary so this is just
    # the port's tile; the freshness flag tells the network whether our
    # ownership record is current).
    x, y = port_sighting.position
    row[POSITION_OFFSET] = x / engine.map.width
    row[POSITION_OFFSET + 1] = y / engine.map.height

    # Owner — is_enemy=1 (is_friendly stays 0 from zero-init)
    row[OWNER_OFFSET + 1] = 1.0

    # Port state — stockpile stays zero (hidden); is_home is observable
    row[PORT_STATE_OFFSET + 1] = 1.0 if port_sighting.is_home else 0.0

    # Sighting state
    row[SIGHTING_STATE_OFFSET + 0] = 1.0 if port_sighting.fresh else 0.0
    row[SIGHTING_STATE_OFFSET + 1] = (engine.turn - port_sighting.turn_seen) / MAX_TURNS_SCALE


def _write_coastline_token(row: np.ndarray, position: Tuple[int, int], engine: "gameEngine") -> None:
    """Populate `row` for a coastline tile.

    Coastline tiles are static map features — candidates for build_port
    targeting. Only type and position carry information; everything else
    stays zero.
    """
    row[COASTLINE_TYPE_INDEX] = 1.0
    x, y = position
    row[POSITION_OFFSET] = x / engine.map.width
    row[POSITION_OFFSET + 1] = y / engine.map.height
