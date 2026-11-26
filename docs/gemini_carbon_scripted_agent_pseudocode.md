# Gemini 4 Carbon scripted-agent pseudocode

Generated from `docs/task.md` as a reusable design artifact. Not executed or wired into the engine. Same prompt as the Haiku 5.5 request. One call; no retry (finish_reason=`stop`).

| Field | Value |
|---|---|
| Model | `google/gemini-4-carbon` |
| Reasoning effort | `high` |
| Date | 2026-11-25 (America/Chicago) |
| Request started | 2026-11-25 07:49 CST (America/Chicago) |
| Latency | 987.6 s |
| Prompt tokens | 9,617 |
| Completion tokens | 58,424 (of which 30,983 reasoning) |
| Total tokens | 68,041 |
| Cost (OpenRouter `usage.cost`) | $0.603474 |
| Finish reason | `stop` |
| max_tokens | 256,000 |

---
Below is complete, self-contained pseudocode for **`scripted_agent_action(engine, player_id, rng)`**, a strictly fog-limited, rule-based Battleboats agent designed as a drop-in counterpart to `scripted_action(engine, player_id, rng)` and `tactical.py` for harvesting diverse, decisive value-network training games.

### Design Highlights (Mapped to Rules)
1. **Hard Fog-of-War Barrier (`extract_fogged_view`, §7 & §10):** Reads only `player_id`'s own ships/ports/cash, `player_id`'s fresh and stale sightings, static coastline tokens, board dimensions, `engine.turn`, and `enumerate_legal(player_id)`. Navigable water is reconstructed strictly by flood-filling from known water seeds while treating `coastline ∪ own_ports ∪ sighted_ports` as land (any cardinal land neighbor of water is by §7/§10 either a coastline tile or a port, so inland land can never be stepped onto).
2. **Phantom Sighting Pruning (§7):** Because the engine freezes stale `Sighting` records at `last_pos` until that ship is spotted again elsewhere, the agent tracks when tiles were last covered by friendly spotting radii `R(spotter, ship_type)`. If a stale sighting's `last_pos` has been inspected *since* `turn_seen` (or is currently covered while `fresh == False`), the agent knows the enemy ship vacated that tile and avoids swarming empty ghosts.
3. **Turn-Order Discipline (§3 & §4):** Newborn ships (`bought_this_turn`) cannot act, while `MerchantUnloadAction` immediately converts cargo to `2 * cargo` cash without spending movement. Executing captures, unloads, reachable merchant deliveries, and port-unclogging moves *before* `BuildShipAction` prevents newborn ships from blocking port exits and maximizes same-turn cash before buying.
4. **Combat Matchups, Spotting vs. Attack Range, Kiting, and Port Body-Blocking (§2, §3, §5, §6, §7):**
   - Enforces `eff_range = min(attack_range, VISIBILITY[attacker][defender])` (e.g., `Carrier` snipes `Landing`/`Merchant` at `8` and kites `Battleship` from `6` out to `8`; `Battleship`/`Cruiser` only spot `Submarine` at `1` while `Submarine` strikes them from `3` and avoids `Destroyer`'s `3`-tile sub radar).
   - Assigns low-strength attackers to `strength = 0` targets (`P(kill) = 1.0`) when heavy warships have armored targets.
   - Uses `Landing`'s inability to attack (§5) and 1-ship-per-tile occupancy (§2) both **defensively** (body-blocking adjacent water tiles of `own_home` against an imminent enemy `Landing`) and **offensively** (keeping friendly escorts off `enemy_home`-adjacent water tiles so they never self-block their own `Landing`!).
5. **Diversity Without Stalls (§1.1, §4, §5):** Samples one of **6 Strategic Doctrines** per seat per game via `rng`, pairs softmax hull selection with a persistent `planned_buy` target (so the agent saves across turns for `200`–`250` hulls like `Landing`, `Carrier`, `Battleship`, and `Builder` instead of collapsing onto `100`-cost hulls), and applies emergency home-defense, anti-poverty, and late-game `Landing` checkmate overrides so games stay decisive.

---

```python
# ==============================================================================
# 1. ENGINE CONSTANTS & RULE TABLES (§4, §5, §6, §7)
# ==============================================================================

STARTING_CASH      = 250    # §4: At reset(); Player 0 gets no turn-0 income; Player 1 gets +50 before first move (§3)
PORT_PRODUCTION    = 25     # §4: Materials produced at start of owner's turn per owned port
CASH_PER_MATERIAL  = 2      # §4: Home port gives 25*2 = 50 cash/turn; 100-cargo merchant unload = 200 cash
MERCHANT_CAPACITY  = 100    # §4: Max materials carried by a Merchant
KILL_CURVE_K       = 2.0    # §6: Default kill-curve exponent

ALL_SHIP_TYPES = [
    "Carrier", "Battleship", "Cruiser", "Destroyer",
    "Submarine", "Landing", "Merchant", "Builder",
]
COMBAT_TYPES     = {"Carrier", "Battleship", "Cruiser", "Destroyer", "Submarine"}
NON_COMBAT_TYPES = {"Landing", "Merchant", "Builder"}  # §5: Strength 0, cannot attack, P(kill)=1 if attacked

# §5: Ship stats from shipTypes.csv
SHIP_STATS = {
    "Carrier":    {"cost": 250, "strength": 100, "attack_range": 12, "speed": 6},
    "Battleship": {"cost": 200, "strength": 125, "attack_range":  4, "speed": 3},
    "Cruiser":    {"cost": 150, "strength":  75, "attack_range":  3, "speed": 5},
    "Destroyer":  {"cost": 100, "strength":  50, "attack_range":  3, "speed": 5},
    "Submarine":  {"cost": 125, "strength":  50, "attack_range":  3, "speed": 3},
    "Landing":    {"cost": 250, "strength":   0, "attack_range":  0, "speed": 3},
    "Merchant":   {"cost": 100, "strength":   0, "attack_range":  0, "speed": 3},
    "Builder":    {"cost": 250, "strength":   0, "attack_range":  0, "speed": 3},
}

# §6: Attack modifiers mu(attacker, defender) from attackModifiers.csv
ATTACK_MOD = {
    "Carrier":    {"Carrier": 1.00, "Battleship": 1.50, "Cruiser": 1.00, "Destroyer": 0.75, "Submarine": 1.00},
    "Battleship": {"Carrier": 1.00, "Battleship": 1.00, "Cruiser": 1.50, "Destroyer": 1.00, "Submarine": 0.75},
    "Cruiser":    {"Carrier": 1.25, "Battleship": 1.00, "Cruiser": 1.00, "Destroyer": 1.50, "Submarine": 1.00},
    "Destroyer":  {"Carrier": 1.25, "Battleship": 1.00, "Cruiser": 1.00, "Destroyer": 1.00, "Submarine": 2.00},
    "Submarine":  {"Carrier": 3.00, "Battleship": 4.00, "Cruiser": 2.00, "Destroyer": 0.50, "Submarine": 1.00},
}

# §7: Asymmetric Manhattan spotting matrix R(spotter, target) from visibilityMatrix.csv
VISIBILITY = {
    "Carrier":    {"Carrier":  6, "Battleship":  6, "Cruiser":  4, "Destroyer":  3, "Submarine": 2, "Landing": 8, "Merchant": 8, "Builder": 8, "Port": 16},
    "Battleship": {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 1, "Landing": 4, "Merchant": 4, "Builder": 4, "Port":  8},
    "Cruiser":    {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 1, "Landing": 6, "Merchant": 6, "Builder": 6, "Port": 12},
    "Destroyer":  {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 3, "Landing": 8, "Merchant": 8, "Builder": 8, "Port": 16},
    "Submarine":  {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  2, "Submarine": 1, "Landing": 2, "Merchant": 2, "Builder": 2, "Port":  4},
    "Landing":    {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 1, "Landing": 2, "Merchant": 2, "Builder": 2, "Port":  4},
    "Merchant":   {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 1, "Landing": 2, "Merchant": 2, "Builder": 2, "Port":  4},
    "Builder":    {"Carrier":  4, "Battleship":  4, "Cruiser":  3, "Destroyer":  3, "Submarine": 1, "Landing": 2, "Merchant": 2, "Builder": 2, "Port":  4},
    "Port":       {"Carrier": 16, "Battleship": 16, "Cruiser": 12, "Destroyer": 12, "Submarine": 1, "Landing": 8, "Merchant": 8, "Builder": 8, "Port": 16},
}

CARDINAL_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


function manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


function effective_attack_range(attacker_type: str, defender_type: str) -> int:
    # §6: Target must be within BOTH attack_range AND the attacker's own spotting distance vs target_type.
    if attacker_type not in COMBAT_TYPES:
        return 0
    return min(SHIP_STATS[attacker_type]["attack_range"], VISIBILITY[attacker_type][defender_type])


function kill_probability(attacker_type: str, defender_type: str) -> float:
    # §6: Single Bernoulli trial; strength-0 hulls always die if attacked.
    if attacker_type not in COMBAT_TYPES:
        return 0.0
    def_strength = SHIP_STATS[defender_type]["strength"]
    if def_strength == 0:
        return 1.0
    mu = ATTACK_MOD[attacker_type][defender_type]
    x  = (SHIP_STATS[attacker_type]["strength"] * mu) / float(def_strength)
    xk = x ** KILL_CURVE_K
    return xk / (1.0 + xk)


# ==============================================================================
# 2. STRATEGIC DOCTRINES FOR HARVEST DIVERSITY (§1.1)
# ==============================================================================
# Sampled once per (game, seat) via caller's seeded RNG so replays are deterministic
# while exposing the value net to varied openings, tempos, and fleet mixes.

DOCTRINES = {
    "RAIDER_BLITZ": {
        "weights": {"Destroyer": 1.5, "Cruiser": 1.4, "Landing": 1.3, "Merchant": 1.1,
                    "Submarine": 1.0, "Carrier": 0.8, "Battleship": 0.8, "Builder": 0.6},
        "aggression": 1.15,          # Willing to take slightly sharper combat trades
        "builder_mode": "FORWARD",   # Prefers forward island naval base if Builder is rolled
        "target_merchants": 2,
        "buy_temperature": 0.45,
    },
    "ECONOMIC_BOOM": {
        "weights": {"Merchant": 1.6, "Builder": 1.4, "Carrier": 1.3, "Battleship": 1.3,
                    "Landing": 1.1, "Destroyer": 1.0, "Cruiser": 0.9, "Submarine": 0.9},
        "aggression": 0.90,
        "builder_mode": "ECO_HOME",  # Prefers founding a safe close-to-home satellite port early
        "target_merchants": 3,
        "buy_temperature": 0.45,
    },
    "WOLFPACK_STEALTH": {
        "weights": {"Submarine": 1.6, "Destroyer": 1.4, "Merchant": 1.2, "Landing": 1.2,
                    "Cruiser": 1.0, "Carrier": 0.9, "Builder": 0.8, "Battleship": 0.7},
        "aggression": 1.05,
        "builder_mode": "FORWARD",
        "target_merchants": 2,
        "buy_temperature": 0.45,
    },
    "CAPITAL_ARMADA": {
        "weights": {"Carrier": 1.5, "Battleship": 1.5, "Merchant": 1.3, "Destroyer": 1.2,
                    "Landing": 1.1, "Cruiser": 0.9, "Submarine": 0.8, "Builder": 0.8},
        "aggression": 0.95,
        "builder_mode": "ECO_HOME",
        "target_merchants": 3,
        "buy_temperature": 0.45,
    },
    "ISLAND_HOPPER": {
        "weights": {"Builder": 1.4, "Landing": 1.4, "Cruiser": 1.3, "Destroyer": 1.3,
                    "Merchant": 1.2, "Submarine": 0.9, "Carrier": 0.9, "Battleship": 0.9},
        "aggression": 1.05,
        "builder_mode": "FORWARD",
        "target_merchants": 2,
        "buy_temperature": 0.50,
    },
    "BALANCED_ADAPTIVE": {
        "weights": {"Carrier": 1.1, "Battleship": 1.1, "Cruiser": 1.1, "Destroyer": 1.1,
                    "Submarine": 1.1, "Landing": 1.1, "Merchant": 1.1, "Builder": 1.0},
        "aggression": 1.00,
        "builder_mode": "ADAPTIVE",
        "target_merchants": 2,
        "buy_temperature": 0.50,
    },
}


# ==============================================================================
# 3. FOGGED VIEW & PER-SEAT MEMORY (§7, §10)
# ==============================================================================

struct OwnShipView:
    ship_id: any
    ship_type: str
    pos: (int, int)
    cargo: int
    has_attacked: bool
    tiles_moved: int
    bought_this_turn: bool
    remaining_moves: int

struct OwnPortView:
    pos: (int, int)
    is_home: bool
    stockpile: int
    locked_this_turn: bool

struct EnemyShipSightingView:
    ship_id: any            # Tracked sighting key; cargo/has_attacked/tiles_moved NEVER exposed (§7)
    ship_type: str
    last_pos: (int, int)
    fresh: bool
    turn_seen: int

struct EnemyPortSightingView:
    pos: (int, int)         # Stockpile NEVER exposed (§7); is_home stays known once spotted
    last_known_owner: int
    is_home: bool
    fresh: bool
    turn_seen: int

struct FoggedView:
    player_id: int
    width: int
    height: int
    turn: int
    own_cash: int
    own_ships: List[OwnShipView]
    own_ports: List[OwnPortView]
    own_home: OwnPortView
    ship_sightings: List[EnemyShipSightingView]
    port_sightings: List[EnemyPortSightingView]
    coastline_tiles: Set[(int, int)]
    legal_actions: List[Action]


function extract_fogged_view(engine, player_id: int) -> FoggedView:
    """
    HARD FOG-OF-WAR GATE (§7, §10):
    Extracts ONLY information legally observable by `player_id`.
    Never touches opponent live ships, unsighted ports, opponent cash/stockpiles,
    or inland terrain tiles.
    """
    own_ships = []
    for s in engine.get_player_ships(player_id):
        speed = SHIP_STATS[s.ship_type]["speed"]
        rem   = 0 if s.bought_this_turn else max(0, speed - s.tiles_moved_this_turn)
        own_ships.append(OwnShipView(
            ship_id=s.id, ship_type=s.ship_type, pos=s.position,
            cargo=s.cargo, has_attacked=s.has_attacked,
            tiles_moved=s.tiles_moved_this_turn,
            bought_this_turn=s.bought_this_turn, remaining_moves=rem
        ))

    own_ports = []
    own_home  = None
    for p in engine.get_player_ports(player_id):
        pv = OwnPortView(pos=p.position, is_home=p.is_home,
                         stockpile=p.stockpile, locked_this_turn=p.locked_this_turn)
        own_ports.append(pv)
        if pv.is_home:
            own_home = pv

    # Sightings strictly from `player_id`'s sighting tables (§7)
    ship_sightings = [
        EnemyShipSightingView(
            ship_id=rec.ship_id, ship_type=rec.ship_type,
            last_pos=rec.position, fresh=rec.fresh, turn_seen=rec.turn_seen
        )
        for rec in engine.get_ship_sightings(player_id)
    ]
    port_sightings = [
        EnemyPortSightingView(
            pos=rec.position, last_known_owner=rec.last_known_owner,
            is_home=rec.is_home, fresh=rec.fresh, turn_seen=rec.turn_seen
        )
        for rec in engine.get_port_sightings(player_id)
    ]

    # Static coastline tokens exactly as exposed in build_observation (§7, §10):
    # land adjacent to water, excluding current ports.
    coastline_tiles = set(engine.get_observable_coastline_tiles())
    legal_actions   = engine.enumerate_legal(player_id)

    return FoggedView(
        player_id=player_id,
        width=engine.map.width,
        height=engine.map.height,
        turn=engine.turn,
        own_cash=engine.get_player_cash(player_id),
        own_ships=own_ships,
        own_ports=own_ports,
        own_home=own_home,
        ship_sightings=ship_sightings,
        port_sightings=port_sightings,
        coastline_tiles=coastline_tiles,
        legal_actions=legal_actions,
    )


struct SeatMemory:
    doctrine_name: str
    doctrine: dict
    type_jitter: Dict[str, float]                      # Per-game mild multiplicative noise in [0.85, 1.15]
    planned_buy: Optional[str]                         # Saved target hull so 200/250-cost ships get built (§4, §5)
    built_counts: Dict[str, int]                       # Tracks own buys this game for fleet-mix diversity
    ports_built_count: int
    known_water_seeds: Set[(int, int)]                 # Confirmed water positions seen from own ships/spawns
    ever_checked_for_ports: Set[(int, int)]            # Tiles ever covered by R(spotter, "Port") (§7)
    last_inspected_stamp: Dict[str, Dict[(int,int), (int,int)]]  # ship_type -> pos -> (turn, step)
    active_turn: int                                   # Round index of current intra-turn state
    atomic_step_in_turn: int                           # Monotonic step counter within current turn
    visited_this_turn: Dict[any, Set[(int, int)]]      # Prevents intra-turn move oscillation per ship_id
    sidestepped_this_turn: Set[any]                    # At most 1 spawn-clearing sidestep per idle ship/turn
    builder_assignments: Dict[any, (int, int)]         # Persistent target coastline per Builder ship_id


function get_or_init_seat_memory(engine, view: FoggedView, rng) -> SeatMemory:
    """
    Stores strictly private per-seat memory on `engine._scripted_agent_mem[player_id]`,
    auto-resetting at kickoff (`turn == 0` with no prior turns played).
    """
    if not hasattr(engine, "_scripted_agent_mem"):
        engine._scripted_agent_mem = {}

    mem = engine._scripted_agent_mem.get(view.player_id, None)
    is_new_game = (
        mem is None
        or view.turn < mem.active_turn
        or (view.turn == 0 and mem.active_turn > 0)
    )
    if is_new_game:
        doc_names = sorted(DOCTRINES.keys())
        chosen_name = doc_names[rng.randrange(len(doc_names))]
        jitter = {t: rng.uniform(0.85, 1.15) for t in ALL_SHIP_TYPES}
        mem = SeatMemory(
            doctrine_name=chosen_name,
            doctrine=DOCTRINES[chosen_name],
            type_jitter=jitter,
            planned_buy=None,
            built_counts={t: 0 for t in ALL_SHIP_TYPES},
            ports_built_count=0,
            known_water_seeds=set(),
            ever_checked_for_ports=set(),
            last_inspected_stamp={t: {} for t in ALL_SHIP_TYPES},
            active_turn=-1,
            atomic_step_in_turn=0,
            visited_this_turn={},
            sidestepped_this_turn=set(),
            builder_assignments={},
        )
        engine._scripted_agent_mem[view.player_id] = mem

    # Reset intra-turn trackers whenever a new turn begins for this seat
    if mem.active_turn != view.turn:
        mem.active_turn = view.turn
        mem.atomic_step_in_turn = 0
        mem.visited_this_turn = {s.ship_id: {s.pos} for s in view.own_ships}
        mem.sidestepped_this_turn = set()
    else:
        mem.atomic_step_in_turn += 1
        for s in view.own_ships:
            if s.ship_id not in mem.visited_this_turn:
                mem.visited_this_turn[s.ship_id] = {s.pos}
            else:
                mem.visited_this_turn[s.ship_id].add(s.pos)

    # Prune dead/consumed Builders from assignments
    live_ids = {s.ship_id for s in view.own_ships}
    for bid in list(mem.builder_assignments.keys()):
        if bid not in live_ids:
            del mem.builder_assignments[bid]

    return mem


# ==============================================================================
# 4. FOG-LEGAL WATER GRAPH, PHANTOM PRUNING & THREAT MAPS (§6, §7, §8, §10)
# ==============================================================================

struct TacticalWorld:
    water_tiles: Set[(int, int)]
    own_port_set: Set[(int, int)]
    fresh_enemies: List[EnemyShipSightingView]
    fresh_enemy_by_pos: Dict[(int, int), EnemyShipSightingView]
    active_stale_enemies: List[EnemyShipSightingView]  # Stale sightings NOT yet cleared by Phantom Pruning
    enemy_home_port: Optional[EnemyPortSightingView]   # Known iff spotted at least once (§7)
    capturable_satellites: List[EnemyPortSightingView] # Sighted ports not owned by us & not enemy home
    unchecked_port_prior: Dict[(int, int), float]      # Prior weight over tiles never checked for ports
    threat_by_hull: Dict[str, Dict[(int, int), float]] # Hull-specific expected kill risk on opponent's next turn
    sub_danger_tiles: Set[(int, int)]                  # Tiles within R(Destroyer,Sub)=3 or R(Port,Sub)=1 of enemy


function in_bounds(pos: (int, int), width: int, height: int) -> bool:
    return 0 <= pos[0] < width and 0 <= pos[1] < height


function neighbors4(pos: (int, int), width: int, height: int) -> List[(int, int)]:
    res = []
    for dx, dy in CARDINAL_DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height:
            res.append((nx, ny))
    return res


function build_tactical_world(view: FoggedView, mem: SeatMemory) -> TacticalWorld:
    W, H = view.width, view.height
    own_port_set     = {p.pos for p in view.own_ports}
    sighted_port_set = {ps.pos for ps in view.port_sightings}
    known_land       = view.coastline_tiles | own_port_set | sighted_port_set

    # 1. Reconstruct navigable water strictly from fog-legal observations (§7, §10):
    # Every friendly ship tile, fresh enemy ship tile, and legal BuildShipAction spawn tile is water.
    # Proof of exactness: any cardinal neighbor `n` of a water tile `w` is adjacent to water (`w`);
    # hence if `n` is land, `n` MUST be either a coastline token or a port (never inland land).
    # Unsighted ports become sighted at Manhattan distance >= 4 (min of Port column in VISIBILITY),
    # long before any ship gets adjacent to them.
    for s in view.own_ships:
        mem.known_water_seeds.add(s.pos)
    for es in view.ship_sightings:
        if es.fresh:
            mem.known_water_seeds.add(es.last_pos)
    for act in view.legal_actions:
        if act is BuildShipAction:
            mem.known_water_seeds.add(act.spawn_position)
        elif act is MoveAction:
            mem.known_water_seeds.add(act.destination)

    # Remove any seed if a newly built/sighted port ever occupies it
    mem.known_water_seeds -= known_land

    water_tiles = set()
    queue = deque(mem.known_water_seeds)
    for seed in queue:
        water_tiles.add(seed)
    while queue:
        cur = queue.popleft()
        for nb in neighbors4(cur, W, H):
            if nb not in water_tiles and nb not in known_land:
                water_tiles.add(nb)
                queue.append(nb)

    # 2. Stamp current spotting coverage (§7) for Ports and all 8 ShipTypes
    stamp = (view.turn, mem.atomic_step_in_turn)
    currently_covered_for_type = {t: set() for t in ALL_SHIP_TYPES}

    spotters = [(s.pos, s.ship_type) for s in view.own_ships] + [(p.pos, "Port") for p in view.own_ports]
    for spos, skind in spotters:
        # Port coverage
        r_port = VISIBILITY[skind]["Port"]
        for x in range(max(0, spos[0] - r_port), min(W, spos[0] + r_port + 1)):
            rem_y = r_port - abs(x - spos[0])
            for y in range(max(0, spos[1] - rem_y), min(H, spos[1] + rem_y + 1)):
                mem.ever_checked_for_ports.add((x, y))

        # Ship-type coverage
        for t in ALL_SHIP_TYPES:
            r_t = VISIBILITY[skind][t]
            for x in range(max(0, spos[0] - r_t), min(W, spos[0] + r_t + 1)):
                rem_y = r_t - abs(x - spos[0])
                for y in range(max(0, spos[1] - rem_y), min(H, spos[1] + rem_y + 1)):
                    currently_covered_for_type[t].add((x, y))
                    mem.last_inspected_stamp[t][(x, y)] = stamp

    # 3. Classify enemy ports (filtering out any previously sighted port that WE now own)
    enemy_home_port = None
    capturable_satellites = []
    for ps in view.port_sightings:
        if ps.pos in own_port_set:
            continue
        if ps.is_home:
            enemy_home_port = ps
        else:
            capturable_satellites.append(ps)

    # 4. Prior map over unchecked tiles for finding unsighted Enemy Home Port (§8 map symmetries):
    # Player 0 starts on left half, Player 1 on right half; homes are >= W//2 apart and similarly
    # placed relative to midline. We NEVER read hidden ports—only score tiles not in `ever_checked_for_ports`.
    unchecked_port_prior = {}
    if enemy_home_port is None and view.own_home is not None:
        hx, hy = view.own_home.pos
        mirror_x, mirror_y = (W - 1 - hx), (H - 1 - hy)
        enemy_half_is_right = (hx < W / 2.0)
        for x in range(W):
            for y in range(H):
                if (x, y) in mem.ever_checked_for_ports or (x, y) in own_port_set:
                    continue
                if manhattan((x, y), (hx, hy)) < (W // 2):
                    continue  # §8: Map pool guarantees homes are at least W//2 Manhattan apart
                in_enemy_half = (x >= W // 2) if enemy_half_is_right else (x < W // 2)
                base_w = 3.0 if in_enemy_half else 0.4
                # Smooth preference near mirrored x-band and coastal adjacency
                sym_dist = abs(x - mirror_x) + 0.35 * min(abs(y - hy), abs(y - mirror_y))
                coastal_bonus = 1.6 if any(nb in water_tiles for nb in neighbors4((x, y), W, H)) else 1.0
                unchecked_port_prior[(x, y)] = base_w * coastal_bonus / (1.0 + 0.15 * sym_dist)

    # 5. Classify enemy ship sightings & apply Phantom Pruning (§7):
    # Stale records stay frozen in engine until re-spotted. If `last_pos` is currently inside our
    # spotting radius for `es.ship_type` (with fresh=False) OR was inspected on a strictly later
    # turn than `es.turn_seen`, that ship is guaranteed NOT to be sitting at `last_pos` anymore.
    fresh_enemies = []
    fresh_enemy_by_pos = {}
    active_stale_enemies = []
    for es in view.ship_sightings:
        if es.fresh:
            fresh_enemies.append(es)
            fresh_enemy_by_pos[es.last_pos] = es
        else:
            age = view.turn - es.turn_seen
            if es.last_pos in currently_covered_for_type[es.ship_type]:
                continue  # Looking right at `last_pos` and ship isn't there -> phantom-pruned!
            last_chk = mem.last_inspected_stamp[es.ship_type].get(es.last_pos, (-1, -1))
            if last_chk[0] > es.turn_seen:
                continue  # Swept `last_pos` on a turn after `turn_seen` -> phantom-pruned!
            if es.ship_type == "Builder" and age > 4:
                continue  # Stale Builders likely consumed themselves founding a port (§3)
            if age <= 8:
                active_stale_enemies.append(es)

    # 6. Compute hull-specific Next-Turn Threat Maps (§5, §6, §7):
    # On opponent's turn, a fresh enemy warship `e` can move up to `speed(e)` water steps and fire
    # up to `effective_attack_range(e.type, my_hull)` Manhattan tiles.
    threat_by_hull = {t: defaultdict(float) for t in ALL_SHIP_TYPES}
    # Combine probability of survival `prod(1 - p_kill)` -> convert back to total kill risk `1 - survival`
    survival_by_hull = {t: defaultdict(lambda: 1.0) for t in ALL_SHIP_TYPES}

    for es in fresh_enemies:
        if es.ship_type not in COMBAT_TYPES:
            continue
        spd = SHIP_STATS[es.ship_type]["speed"]
        # Exact depth-`spd` water BFS (spd <= 6 visits <= 85 tiles)
        reachable = water_within_steps(es.last_pos, spd, water_tiles, W, H)
        for my_hull in ALL_SHIP_TYPES:
            pk = kill_probability(es.ship_type, my_hull)
            er = effective_attack_range(es.ship_type, my_hull)
            threatened_set = expand_manhattan(reachable, er, water_tiles, W, H)
            for tile in threatened_set:
                survival_by_hull[my_hull][tile] *= (1.0 - pk)

    # Discounted caution plume around un-cleared recent stale combat sightings (age <= 3)
    for es in active_stale_enemies:
        age = max(1, view.turn - es.turn_seen)
        if es.ship_type not in COMBAT_TYPES or age > 3:
            continue
        discount = (0.35 ** age)
        spd = SHIP_STATS[es.ship_type]["speed"]
        approx_zone = expand_manhattan({es.last_pos}, spd + 2, water_tiles, W, H)
        for my_hull in ALL_SHIP_TYPES:
            pk = discount * kill_probability(es.ship_type, my_hull)
            for tile in approx_zone:
                survival_by_hull[my_hull][tile] *= (1.0 - pk)

    for my_hull in ALL_SHIP_TYPES:
        for tile, surv in survival_by_hull[my_hull].items():
            threat_by_hull[my_hull][tile] = 1.0 - surv

    # 7. Anti-Submarine detection danger zone for friendly Submarines (§7):
    # Enemy Destroyers spot Sub at 3 (and kill 80% at range 3); Carriers spot Sub at 2; Ports spot Sub at 1.
    sub_danger_tiles = set()
    for es in fresh_enemies:
        if es.ship_type == "Destroyer":
            reach = water_within_steps(es.last_pos, SHIP_STATS["Destroyer"]["speed"], water_tiles, W, H)
            sub_danger_tiles |= expand_manhattan(reach, 3, water_tiles, W, H)
        elif es.ship_type == "Carrier":
            sub_danger_tiles |= expand_manhattan({es.last_pos}, 3, water_tiles, W, H)
    for ps in view.port_sightings:
        if ps.pos not in own_port_set:
            sub_danger_tiles |= expand_manhattan({ps.pos}, 1, water_tiles, W, H)

    return TacticalWorld(
        water_tiles=water_tiles,
        own_port_set=own_port_set,
        fresh_enemies=fresh_enemies,
        fresh_enemy_by_pos=fresh_enemy_by_pos,
        active_stale_enemies=active_stale_enemies,
        enemy_home_port=enemy_home_port,
        capturable_satellites=capturable_satellites,
        unchecked_port_prior=unchecked_port_prior,
        threat_by_hull=threat_by_hull,
        sub_danger_tiles=sub_danger_tiles,
    )


# ==============================================================================
# 5. SPATIAL & WATER-PATHFINDING HELPERS
# ==============================================================================

function water_within_steps(start: (int, int), max_steps: int,
                            water_tiles: Set[(int, int)], W: int, H: int,
                            blocked: Set[(int, int)] = set()) -> Dict[(int, int), int]:
    """Exact water-step distance map up to `max_steps` from `start`."""
    dists = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        d = dists[cur]
        if d >= max_steps:
            continue
        for nb in neighbors4(cur, W, H):
            if nb in water_tiles and nb not in blocked and nb not in dists:
                dists[nb] = d + 1
                q.append(nb)
    return dists


function expand_manhattan(centers: Iterable[(int, int)], radius: int,
                          domain: Set[(int, int)], W: int, H: int) -> Set[(int, int)]:
    out = set()
    for cx, cy in centers:
        for x in range(max(0, cx - radius), min(W, cx + radius + 1)):
            rem = radius - abs(x - cx)
            for y in range(max(0, cy - rem), min(H, cy + rem + 1)):
                if (x, y) in domain:
                    out.add((x, y))
    return out


function water_neighbors_of_tile(tile: (int, int), world: TacticalWorld, W: int, H: int) -> List[(int, int)]:
    return [nb for nb in neighbors4(tile, W, H) if nb in world.water_tiles]


function multi_source_dijkstra(goals: Set[(int, int)], world: TacticalWorld, view: FoggedView,
                               hull_type: str, risk_weight: float,
                               hard_blocked: Set[(int, int)],
                               soft_blocked: Set[(int, int)],
                               extra_tile_penalty: Dict[(int, int), float] = {}) -> Dict[(int, int), float]:
    """
    Backward Dijkstra from `goals` over `world.water_tiles`:
      - Base step cost = 1.0
      - Movable friendly ship tile (`soft_blocked`) adds +1.5 congestion cost so ships route around traffic
      - Threat risk `threat_by_hull[hull_type][tile] * risk_weight` steers vulnerable hulls around danger
    """
    W, H = view.width, view.height
    threat_map = world.threat_by_hull[hull_type]
    dist = {}
    pq = []
    for g in goals:
        if g in world.water_tiles and g not in hard_blocked:
            dist[g] = 0.0
            heappush(pq, (0.0, g))

    while pq:
        d, u = heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v in neighbors4(u, W, H):
            if v not in world.water_tiles or v in hard_blocked:
                continue
            step_cost = 1.0
            if u in soft_blocked:
                step_cost += 1.5
            step_cost += risk_weight * threat_map.get(u, 0.0)
            step_cost += extra_tile_penalty.get(u, 0.0)
            nd = d + step_cost
            if nd < dist.get(v, INF):
                dist[v] = nd
                heappush(pq, (nd, v))
    return dist


function pick_best_move_toward(ship: OwnShipView, goals: Set[(int, int)],
                               view: FoggedView, world: TacticalWorld, mem: SeatMemory,
                               risk_weight: float, rng,
                               extra_blocked: Set[(int, int)] = set(),
                               extra_penalty: Dict[(int, int), float] = {},
                               require_strict_progress: bool = True) -> Optional[MoveAction]:
    """
    Selects a legal single-tile `MoveAction` for `ship` from `view.legal_actions`
    that minimizes threat/congestion-aware water distance to `goals`, while forbidding
    tiles already visited by `ship` earlier in this turn (no intra-turn ping-pong).
    """
    legal_moves = [
        act for act in view.legal_actions
        if act is MoveAction and act.ship_id == ship.ship_id
        and act.destination not in mem.visited_this_turn[ship.ship_id]
        and act.destination not in extra_blocked
    ]
    if not legal_moves or not goals:
        return None
    if ship.pos in goals and require_strict_progress:
        return None

    hard_blocked = (
        set(world.fresh_enemy_by_pos.keys())
        | {s.pos for s in view.own_ships if s.ship_id != ship.ship_id and s.remaining_moves == 0}
        | extra_blocked
    )
    soft_blocked = {
        s.pos for s in view.own_ships
        if s.ship_id != ship.ship_id and s.remaining_moves > 0
    }

    cost_map = multi_source_dijkstra(
        goals=goals, world=world, view=view, hull_type=ship.ship_type,
        risk_weight=risk_weight, hard_blocked=hard_blocked,
        soft_blocked=soft_blocked, extra_tile_penalty=extra_penalty
    )
    # Fallback ignoring immobile friendly ships if a narrow channel is temporarily corked
    if ship.pos not in cost_map and all(m.destination not in cost_map for m in legal_moves):
        cost_map = multi_source_dijkstra(
            goals=goals, world=world, view=view, hull_type=ship.ship_type,
            risk_weight=risk_weight, hard_blocked=set(world.fresh_enemy_by_pos.keys()),
            soft_blocked={s.pos for s in view.own_ships if s.ship_id != ship.ship_id},
            extra_tile_penalty=extra_penalty
        )

    cur_cost = cost_map.get(ship.pos, INF)
    scored = []
    for mv in legal_moves:
        dc = cost_map.get(mv.destination, INF)
        if dc >= INF:
            continue
        if require_strict_progress and dc >= cur_cost - 1e-6:
            continue
        # Mild stochastic tie-break among nearly identical steps (<= 0.15 cost diff) for trace diversity
        scored.append((dc, mv))

    if not scored:
        return None
    best_c = min(c for c, _ in scored)
    near_ties = [mv for c, mv in scored if c <= best_c + 0.15]
    return near_ties[rng.randrange(len(near_ties))]


# ==============================================================================
# 6. SPECIALIZED DECISION MODULES (CAPTURES, ECONOMY, COMBAT, BUILDERS, FLEET, BUYS)
# ==============================================================================

# ------------------------------------------------------------------------------
# 6.1 Immediate Captures & Free Adjacent Merchant Transfers (§2, §3, §4)
# ------------------------------------------------------------------------------

function select_immediate_capture(view: FoggedView, world: TacticalWorld) -> Optional[CapturePortAction]:
    """
    §2 & §3: Capturing the enemy home port wins immediately. Capturing any adjacent
    non-owned port does NOT consume the Landing and does NOT spend movement.
    """
    captures = [a for a in view.legal_actions if a is CapturePortAction]
    if not captures:
        return None
    # ALWAYS prioritize Enemy Home Port first (instant victory!)
    if world.enemy_home_port is not None:
        for cap in captures:
            if cap.target == world.enemy_home_port.pos:
                return cap
    return captures[0]


function select_adjacent_merchant_transfer(view: FoggedView) -> Optional[Action]:
    """
    §3 & §4: Unloading at Home Port gives `cargo * 2` cash immediately without spending speed;
    loading at an adjacent owned non-home port with stockpile > 0 is also free of move cost.
    """
    unloads = [a for a in view.legal_actions if a is MerchantUnloadAction]
    if unloads:
        return unloads[0]

    # Only load if merchant still has room (`cargo < 100`, enforced by enumerate_legal too)
    loads = [a for a in view.legal_actions if a is MerchantLoadAction]
    if loads:
        # Prefer loading from the largest adjacent stockpile first
        port_stock = {p.pos: p.stockpile for p in view.own_ports}
        loads.sort(key=lambda a: port_stock.get(a.port, 0), reverse=True)
        return loads[0]
    return None


function select_same_turn_cash_delivery_move(view: FoggedView, world: TacticalWorld,
                                             mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    Moves any loaded Merchant (`cargo > 0`) that can reach a water tile adjacent to
    `own_home` THIS TURN (`<= remaining_moves` steps) so its cash is realized BEFORE
    `BuildShipAction` runs later in the turn.
    """
    if view.own_home is None:
        return None
    home_adj = set(water_neighbors_of_tile(view.own_home.pos, world, view.width, view.height))
    occupied = {s.pos for s in view.own_ships} | set(world.fresh_enemy_by_pos.keys())

    for s in view.own_ships:
        if s.ship_type != "Merchant" or s.cargo <= 0 or s.remaining_moves <= 0:
            continue
        if s.pos in home_adj:
            continue  # Already handled by MerchantUnloadAction
        reach = water_within_steps(
            s.pos, s.remaining_moves, world.water_tiles, view.width, view.height,
            blocked=occupied - {s.pos}
        )
        if any(g in reach for g in home_adj):
            mv = pick_best_move_toward(s, home_adj, view, world, mem, risk_weight=8.0, rng=rng)
            if mv is not None:
                return mv
    return None


# ------------------------------------------------------------------------------
# 6.2 Coordinated Combat: Target Valuation, Matchup Assignment & Kiting (§5, §6, §7)
# ------------------------------------------------------------------------------

function strategic_target_value(enemy: EnemyShipSightingView, view: FoggedView,
                                world: TacticalWorld) -> float:
    """
    Scores an enemy target by economic cost + existential/win-condition impact (§2, §5, §7).
    """
    val = float(SHIP_STATS[enemy.ship_type]["cost"])

    # 1. Existential priority: Enemy Landing near our Home Port (§2: only way we can lose!)
    if enemy.ship_type == "Landing":
        val = 550.0
        if view.own_home is not None:
            d_home = manhattan(enemy.last_pos, view.own_home.pos)
            if d_home <= 9:
                val = 2200.0 + 250.0 * (10 - d_home)

    # 2. High-value soft hulls (guaranteed P(kill) = 1.0 when in range, §5 & §6)
    elif enemy.ship_type == "Builder":
        val = 420.0  # Kill before it consumes itself to found a permanent port
    elif enemy.ship_type == "Merchant":
        val = 280.0  # 100 hull + up to 200 cash of unrevealed cargo denied

    # 3. Exposed Submarine bonus: stealth unit (`spot=1` vs most hulls, §7); kill while visible!
    elif enemy.ship_type == "Submarine":
        val += 130.0

    # 4. Escort/Breakthrough bonus: enemy warship defending Enemy Home or threatening our Landing
    if world.enemy_home_port is not None and manhattan(enemy.last_pos, world.enemy_home_port.pos) <= 5:
        val *= 1.35
    for s in view.own_ships:
        if s.ship_type == "Landing" and manhattan(enemy.last_pos, s.pos) <= 6:
            val *= 1.45
            break

    return val


function select_combat_action(view: FoggedView, world: TacticalWorld,
                              mem: SeatMemory, rng) -> Optional[Action]:
    """
    Coordinates attacks across all own combat ships that have not yet attacked (`not has_attacked`):
      - Respects BOTH `attack_range` AND hull-pair `VISIBILITY` (`effective_attack_range`, §6).
      - Avoids wasting a Battleship/Submarine shot on a strength-0 Merchant/Builder when a
        Destroyer/Cruiser can guarantee the 100% kill instead.
      - Allows moving into firing range this turn when expected kill value justifies exposure.
    """
    if not world.fresh_enemies:
        return None

    legal_attacks = {(a.attacker_id, a.target_id): a for a in view.legal_actions if a is AttackAction}
    occupied_all  = {s.pos for s in view.own_ships} | set(world.fresh_enemy_by_pos.keys())
    aggression    = mem.doctrine["aggression"]

    # Enumerate candidate (ship, target, fire_pos, steps_needed, net_score)
    candidates = []
    for s in view.own_ships:
        if s.ship_type not in COMBAT_TYPES or s.has_attacked or s.bought_this_turn:
            continue
        reachable = water_within_steps(
            s.pos, s.remaining_moves, world.water_tiles, view.width, view.height,
            blocked=occupied_all - {s.pos}
        )
        my_cost = float(SHIP_STATS[s.ship_type]["cost"])
        my_str  = float(SHIP_STATS[s.ship_type]["strength"])

        for enemy in world.fresh_enemies:
            er = effective_attack_range(s.ship_type, enemy.ship_type)
            if er <= 0:
                continue
            pk   = kill_probability(s.ship_type, enemy.ship_type)
            tval = strategic_target_value(enemy, view, world)
            expected_gain = pk * tval

            # Save high-strength anti-armor hulls when shooting strength-0 targets (§5, §6)
            overkill_pen = (0.30 * my_str) if SHIP_STATS[enemy.ship_type]["strength"] == 0 else 0.0

            # Find best reachable tile within effective engagement range `er`
            best_fire_tile = None
            best_tile_score = -INF
            for tile, steps_used in reachable.items():
                if manhattan(tile, enemy.last_pos) > er:
                    continue
                # If `tile == s.pos`, ensure engine actually lists AttackAction as legal
                if steps_used == 0 and (s.ship_id, enemy.ship_id) not in legal_attacks:
                    continue

                # Estimate retaliation risk at `tile`, discounting the target itself by `pk`
                raw_risk = world.threat_by_hull[s.ship_type].get(tile, 0.0)
                adj_risk = raw_risk * (1.0 - 0.65 * pk)
                # Bonus if ship still has leftover moves AFTER firing to shoot-and-scoot!
                moves_left_after = s.remaining_moves - steps_used
                kite_discount = 0.65 if moves_left_after >= 2 else 1.0
                exposure_cost = my_cost * adj_risk * kite_discount

                # Standoff preference: Carrier/Battleship prefer firing near max effective range
                standoff_bonus = 2.5 * manhattan(tile, enemy.last_pos)
                if s.ship_type == "Submarine" and tile in world.sub_danger_tiles:
                    exposure_cost += 75.0  # Keep Submarines out of Destroyer/Port radar when possible

                net = (expected_gain * aggression) - exposure_cost - overkill_pen + standoff_bonus - 2.0 * steps_used
                if net > best_tile_score:
                    best_tile_score = net
                    best_fire_tile = (tile, steps_used)

            # Accept shot if net-positive OR defending Home Port against an enemy Landing
            is_home_emergency = (
                enemy.ship_type == "Landing" and view.own_home is not None
                and manhattan(enemy.last_pos, view.own_home.pos) <= 8
            )
            if best_fire_tile is not None and (best_tile_score > -15.0 or is_home_emergency):
                fire_pos, steps_used = best_fire_tile
                candidates.append((best_tile_score, steps_used, s, enemy, fire_pos))

    if not candidates:
        return None

    # Prefer highest net tactical score; execute immediate zero-step shots before setup moves
    # so resolved kills instantly free tiles and prevent wasted movement!
    immediate = [c for c in candidates if c[1] == 0]
    if immediate:
        immediate.sort(key=lambda c: c[0], reverse=True)
        _, _, best_s, best_e, _ = immediate[0]
        return legal_attacks[(best_s.ship_id, best_e.ship_id)]

    candidates.sort(key=lambda c: c[0], reverse=True)
    for _, _, best_s, _, fire_pos in candidates:
        mv = pick_best_move_toward(
            best_s, {fire_pos}, view, world, mem, risk_weight=1.5, rng=rng
        )
        if mv is not None:
            return mv
    return None


function get_imminent_home_landing_threat(view: FoggedView, world: TacticalWorld) -> bool:
    if view.own_home is None:
        return False
    for es in world.fresh_enemies:
        if es.ship_type == "Landing" and manhattan(es.last_pos, view.own_home.pos) <= 5:
            return True
    return False


function get_reserved_enemy_home_capture_tiles(view: FoggedView, world: TacticalWorld) -> Set[(int, int)]:
    """
    OFFENSIVE ANTI-SELF-BLOCK RULE (§2, §3):
    If a friendly Landing is within 6 Manhattan tiles of `enemy_home`, non-Landing friendly
    ships must NOT occupy all water tiles adjacent to `enemy_home`, or our own Landing
    will be blocked from stepping adjacent to execute `CapturePortAction`!
    """
    if world.enemy_home_port is None:
        return set()
    eh = world.enemy_home_port.pos
    friendly_landing_near = any(
        s.ship_type == "Landing" and manhattan(s.pos, eh) <= 6
        for s in view.own_ships
    )
    if not friendly_landing_near:
        return set()
    return set(water_neighbors_of_tile(eh, world, view.width, view.height))


function select_post_attack_or_bodyblock_move(view: FoggedView, world: TacticalWorld,
                                              mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    1. DEFENSIVE BODY-BLOCK (§2, §3, §5): If a surviving enemy Landing is within 5 tiles of
       `own_home`, move ships with spare speed onto open water tiles adjacent to `own_home`
       (since Landings cannot attack and ships occupy 1 per tile, blocking those tiles prevents capture!).
    2. OFFENSIVE LANE-CLEARING: Any non-Landing standing on a water tile adjacent to `enemy_home`
       when a friendly Landing is approaching steps back to distance 2..3.
    3. SHOOT-AND-SCOOT KITING: Combat ships that ALREADY attacked (`has_attacked == True`) and
       still have `remaining_moves > 0` step toward lower-threat water tiles (e.g., Carrier firing
       at range 6 then retreating to range 8 outside Battleship's 4+3=7 threat reach).
    """
    W, H = view.width, view.height

    # 1. Emergency own-home body-block against surviving enemy Landing
    if get_imminent_home_landing_threat(view, world):
        home_adj = set(water_neighbors_of_tile(view.own_home.pos, world, W, H))
        occupied_own = {s.pos for s in view.own_ships}
        open_home_adj = home_adj - occupied_own - set(world.fresh_enemy_by_pos.keys())
        if open_home_adj:
            # Prefer ships that already attacked, or non-loaded ships nearby
            blockers = [
                s for s in view.own_ships
                if s.remaining_moves > 0 and s.pos not in home_adj
                and (s.has_attacked or s.ship_type in COMBAT_TYPES or (s.ship_type == "Merchant" and s.cargo == 0))
            ]
            blockers.sort(key=lambda s: manhattan(s.pos, view.own_home.pos))
            for s in blockers:
                mv = pick_best_move_toward(s, open_home_adj, view, world, mem, risk_weight=0.5, rng=rng)
                if mv is not None:
                    return mv

    # 2. Clear `enemy_home`-adjacent capture tiles for approaching friendly Landing
    reserved_eh_adj = get_reserved_enemy_home_capture_tiles(view, world)
    if reserved_eh_adj:
        eh = world.enemy_home_port.pos
        ring_2_3 = {
            t for t in expand_manhattan({eh}, 3, world.water_tiles, W, H)
            if manhattan(t, eh) >= 2
        }
        for s in view.own_ships:
            if s.ship_type != "Landing" and s.pos in reserved_eh_adj and s.remaining_moves > 0:
                mv = pick_best_move_toward(s, ring_2_3, view, world, mem, risk_weight=2.0, rng=rng)
                if mv is not None:
                    return mv

    # 3. Shoot-and-scoot kiting for ships that already spent their attack this turn
    home_adj_hold = (
        set(water_neighbors_of_tile(view.own_home.pos, world, W, H))
        if get_imminent_home_landing_threat(view, world) else set()
    )
    for s in view.own_ships:
        if s.ship_type not in COMBAT_TYPES or not s.has_attacked or s.remaining_moves <= 0:
            continue
        if s.pos in home_adj_hold:
            continue  # Stay anchored on own_home body-block tile!
        cur_threat = world.threat_by_hull[s.ship_type].get(s.pos, 0.0)
        if cur_threat <= 0.05 and not (s.ship_type == "Submarine" and s.pos in world.sub_danger_tiles):
            continue

        legal_steps = [
            a for a in view.legal_actions
            if a is MoveAction and a.ship_id == s.ship_id
            and a.destination not in mem.visited_this_turn[s.ship_id]
            and a.destination not in reserved_eh_adj
        ]
        best_kite = None
        best_score = -(cur_threat + (0.4 if s.ship_type == "Submarine" and s.pos in world.sub_danger_tiles else 0.0))
        for mv in legal_steps:
            nt = world.threat_by_hull[s.ship_type].get(mv.destination, 0.0)
            if s.ship_type == "Submarine" and mv.destination in world.sub_danger_tiles:
                nt += 0.4
            score = -nt
            if score > best_score + 0.02:
                best_score = score
                best_kite = mv
        if best_kite is not None:
            return best_kite

    return None


# ------------------------------------------------------------------------------
# 6.3 Port Spawn Unclogging (Before Movement / Purchasing)
# ------------------------------------------------------------------------------

function clear_clogged_port_spawns(view: FoggedView, world: TacticalWorld,
                                   mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    If we can afford a ship (`own_cash >= 100`) and an unlocked owned port has all (or nearly all)
    adjacent water tiles occupied by movable friendly ships, nudge a blocking ship 1 step outward
    so `BuildShipAction` is not choked.
    """
    if view.own_cash < 100 or get_imminent_home_landing_threat(view, world):
        return None
    W, H = view.width, view.height
    ship_at = {s.pos: s for s in view.own_ships}

    for p in view.own_ports:
        if p.locked_this_turn:
            continue
        adj_w = water_neighbors_of_tile(p.pos, world, W, H)
        open_w = [t for t in adj_w if t not in ship_at and t not in world.fresh_enemy_by_pos]
        if len(open_w) > 0:
            continue  # Port already has a free spawn tile
        for t in adj_w:
            blocker = ship_at.get(t, None)
            if blocker is None or blocker.remaining_moves <= 0 or blocker.ship_id in mem.sidestepped_this_turn:
                continue
            step_outs = [
                a for a in view.legal_actions
                if a is MoveAction and a.ship_id == blocker.ship_id
                and a.destination not in adj_w
                and world.threat_by_hull[blocker.ship_type].get(a.destination, 0.0) <= 0.35
            ]
            if step_outs:
                mem.sidestepped_this_turn.add(blocker.ship_id)
                return step_outs[rng.randrange(len(step_outs))]
    return None


# ------------------------------------------------------------------------------
# 6.4 Builder Port-Founding (`BuildPortAction` & Builder Navigation, §3, §4, §7, §10)
# ------------------------------------------------------------------------------

function score_coastline_for_port(coast: (int, int), mode: str,
                                  view: FoggedView, world: TacticalWorld) -> float:
    """
    Scores a coastline tile `coast` (legal `BuildPortAction` target, §7 & §10):
      - Must touch >= 1 navigable water tile (prefer >= 2 so new port never clogs).
      - Avoid clustering right next to an existing port (`manhattan <= 2`) or inside enemy guns.
      - `ECO_HOME`: favors coastline 3..6 Manhattan tiles from `own_home` (fast 1-2 turn Merchant
        shuttle for +50 cash/turn value + extra backline shipyard!).
      - `FORWARD`: favors safe mid-map coastline (`x` near midline / forward third) to project
        a 16-tile Port radar bubble (§7) and forward warship/Landing spawning!
    """
    W, H = view.width, view.height
    adj_water = water_neighbors_of_tile(coast, world, W, H)
    if not adj_water:
        return -INF

    all_known_ports = [p.pos for p in view.own_ports] + [ps.pos for ps in view.port_sightings]
    min_port_dist = min((manhattan(coast, pp) for pp in all_known_ports), default=99)
    if min_port_dist <= 2:
        return -INF  # Too cramped against an existing port

    # Reject coastline under active enemy combat or Landing threat
    local_threat = max(world.threat_by_hull["Builder"].get(w, 0.0) for w in adj_water)
    if local_threat > 0.25:
        return -INF

    openness_bonus = 12.0 * min(3, len(adj_water))
    d_home = manhattan(coast, view.own_home.pos) if view.own_home is not None else 10

    if mode == "ECO_HOME" or (mode == "ADAPTIVE" and len(view.own_ports) <= 3):
        # Sweet spot is 3..7 tiles from own_home: ultra-short merchant shuttle!
        dist_score = 45.0 - 5.0 * abs(d_home - 4.5)
    else:
        # Forward operating base near midfield / toward enemy half
        mid_x = W / 2.0
        dist_score = 40.0 - 2.2 * abs(coast[0] - mid_x)
        if world.enemy_home_port is not None:
            d_eh = manhattan(coast, world.enemy_home_port.pos)
            # Outside enemy home's 8-tile Landing radar, close enough to stage assaults (9..15)
            dist_score += 20.0 - 2.0 * abs(d_eh - 11.0)

    return openness_bonus + dist_score - 40.0 * local_threat


function select_builder_action(view: FoggedView, world: TacticalWorld,
                               mem: SeatMemory, rng) -> Optional[Action]:
    """
    §3: `BuildPortAction(builder_ship_id, target)` consumes the Builder (freeing its water tile
    as an immediate spawn neighbor of the new port!) and creates an unlocked-next-turn non-home port.
    """
    builders = [s for s in view.own_ships if s.ship_type == "Builder"]
    if not builders:
        return None

    mode = mem.doctrine["builder_mode"]

    # 1. Execute `BuildPortAction` immediately if adjacent to a strong coastline site
    legal_builds = [a for a in view.legal_actions if a is BuildPortAction]
    if legal_builds:
        scored_builds = [
            (score_coastline_for_port(a.target, mode, view, world), a)
            for a in legal_builds
        ]
        scored_builds.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_act = scored_builds[0]
        # Accept good site, or any safe site if Builder is threatened or assigned here
        assigned = mem.builder_assignments.get(best_act.builder_ship_id, None)
        if best_sc > 5.0 or best_act.target == assigned:
            mem.ports_built_count += 1
            return best_act

    # 2. Otherwise route movable Builders toward their best assigned coastline's adjacent water tiles
    claimed_coasts = set(mem.builder_assignments.values())
    for b in builders:
        if b.remaining_moves <= 0:
            continue
        target_coast = mem.builder_assignments.get(b.ship_id, None)
        if (
            target_coast is None
            or target_coast not in view.coastline_tiles
            or score_coastline_for_port(target_coast, mode, view, world) <= -100.0
        ):
            best_c, best_sc = None, -INF
            for c in view.coastline_tiles:
                if c in claimed_coasts:
                    continue
                sc = score_coastline_for_port(c, mode, view, world) - 1.2 * manhattan(b.pos, c)
                if sc > best_sc:
                    best_sc, best_c = sc, c
            if best_c is not None:
                mem.builder_assignments[b.ship_id] = best_c
                claimed_coasts.add(best_c)
                target_coast = best_c

        if target_coast is not None:
            goal_water = set(water_neighbors_of_tile(target_coast, world, view.width, view.height))
            mv = pick_best_move_toward(b, goal_water, view, world, mem, risk_weight=9.0, rng=rng)
            if mv is not None:
                return mv
    return None


# ------------------------------------------------------------------------------
# 6.5 Fleet Role Navigation: Merchants, Landings, Scouts, Subs, Carriers & Escorts
# ------------------------------------------------------------------------------

function select_merchant_move(view: FoggedView, world: TacticalWorld,
                              mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    §4: Routes Merchants to maximize cash throughput `(2 * delivered_cargo) / turns_to_deliver`
    while steering around enemy combat threat zones (`strength = 0` -> instant death if attacked).
    """
    if view.own_home is None:
        return None
    W, H = view.width, view.height
    merchants = [s for s in view.own_ships if s.ship_type == "Merchant" and s.remaining_moves > 0]
    if not merchants:
        return None

    home_adj = set(water_neighbors_of_tile(view.own_home.pos, world, W, H))
    satellites = [p for p in view.own_ports if not p.is_home]
    virtual_stock = {p.pos: p.stockpile for p in satellites}

    # Precompute rough water distance to Home Port for throughput estimation
    dist_to_home = multi_source_dijkstra(
        goals=home_adj, world=world, view=view, hull_type="Merchant",
        risk_weight=6.0, hard_blocked=set(world.fresh_enemy_by_pos.keys()), soft_blocked=set()
    )

    # Sort fullest merchants first so loaded merchants get priority on home approach
    merchants.sort(key=lambda m: m.cargo, reverse=True)
    for m in merchants:
        # Flee toward Home if currently standing in immediate enemy threat
        cur_risk = world.threat_by_hull["Merchant"].get(m.pos, 0.0)
        if cur_risk > 0.35 and m.pos not in home_adj:
            mv = pick_best_move_toward(m, home_adj, view, world, mem, risk_weight=12.0, rng=rng)
            if mv is not None:
                return mv

        best_goals = None
        best_rate  = -INF
        best_sat   = None

        # Option A: Deliver current cargo to Home Port (`MerchantUnloadAction` gives `2 * cargo`)
        if m.cargo > 0:
            d_h = dist_to_home.get(m.pos, 30.0)
            turns_h = max(0.5, math.ceil(d_h / 3.0))
            # Urgency boost if unloading unlocks a 250-cost hull soon
            urgency = 1.35 if (view.own_cash < 250 and view.own_cash + 2 * m.cargo >= 250) else 1.0
            best_rate  = urgency * (2.0 * m.cargo) / turns_h
            best_goals = home_adj

        # Option B: Visit an owned non-home satellite port to load up to `100 - m.cargo`
        if m.cargo < MERCHANT_CAPACITY:
            for sat in satellites:
                sat_adj = set(water_neighbors_of_tile(sat.pos, world, W, H))
                if not sat_adj:
                    continue
                # Estimate turns to reach `sat` + turns from `sat` back to Home
                d_to_sat = min(manhattan(m.pos, w) for w in sat_adj)
                t_to_sat = math.ceil(d_to_sat / 3.0)
                d_sat_home = min((dist_to_home.get(w, 30.0) for w in sat_adj), default=30.0)
                t_sat_home = max(1.0, math.ceil(d_sat_home / 3.0))

                # Reject heavily threatened satellites
                sat_risk = min(world.threat_by_hull["Merchant"].get(w, 0.0) for w in sat_adj)
                if sat_risk > 0.45:
                    continue

                # Projected stockpile upon arrival (+25 per turn, §4)
                proj_stock = virtual_stock[sat.pos] + PORT_PRODUCTION * t_to_sat
                gain = min(MERCHANT_CAPACITY - m.cargo, proj_stock)
                if gain <= 0:
                    continue
                total_turns = max(1.0, t_to_sat + t_sat_home)
                rate = (2.0 * (m.cargo + gain)) / total_turns
                if rate > best_rate:
                    best_rate  = rate
                    best_goals = sat_adj
                    best_sat   = (sat.pos, min(MERCHANT_CAPACITY - m.cargo, virtual_stock[sat.pos]))

        if best_sat is not None:
            spos, claimed = best_sat
            virtual_stock[spos] = max(0, virtual_stock[spos] - claimed)

        if best_goals is not None:
            # If waiting adjacent to a 0-stockpile satellite for next turn's +25 production, hold position
            if m.pos in best_goals:
                continue
            mv = pick_best_move_toward(m, best_goals, view, world, mem, risk_weight=9.0, rng=rng)
            if mv is not None:
                return mv
    return None


function select_landing_move(view: FoggedView, world: TacticalWorld,
                             mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    §2, §3, §5, §7: Routes Landings (`speed=3, strength=0`) to win the game:
      - Primary Landing targets `enemy_home` once sighted (taking <= 2-tile drive-by satellite
        captures along the way since `CapturePortAction` does not consume the Landing!).
      - Secondary Landings can peel off to capture enemy satellite ports (zeroing enemy stockpiles).
      - Wave synchronization: if approaching `enemy_home`'s 8-tile Port radar unescorted while
        friendly escorts are catching up within 5 tiles, pauses 1 turn outside threat to group up
        (unless `enemy_home` can be captured this turn, is observed undefended close by, or turn >= 55).
      - If `enemy_home` is STILL unsighted, stages safely near a capturable frontier satellite or
        midfield screen behind our scouts.
    """
    W, H = view.width, view.height
    landings = [s for s in view.own_ships if s.ship_type == "Landing" and s.remaining_moves > 0]
    if not landings:
        return None

    own_combat = [s for s in view.own_ships if s.ship_type in COMBAT_TYPES]

    if world.enemy_home_port is not None:
        eh_pos = world.enemy_home_port.pos
        eh_adj = set(water_neighbors_of_tile(eh_pos, world, W, H))
        landings.sort(key=lambda l: manhattan(l.pos, eh_pos))

        for idx, lnd in enumerate(landings):
            d_eh = manhattan(lnd.pos, eh_pos)

            # Sprint without risk-aversion if `eh_adj` is reachable THIS TURN (instant win!)
            reach_now = water_within_steps(
                lnd.pos, lnd.remaining_moves, world.water_tiles, W, H,
                blocked={s.pos for s in view.own_ships if s.ship_id != lnd.ship_id}
                        | set(world.fresh_enemy_by_pos.keys())
            )
            if any(g in reach_now for g in eh_adj):
                mv = pick_best_move_toward(lnd, eh_adj, view, world, mem, risk_weight=0.0, rng=rng)
                if mv is not None:
                    return mv

            # Secondary Landing (`idx >= 1`) can strip a nearby enemy satellite port
            goals = eh_adj
            if idx >= 1 and world.capturable_satellites:
                nearest_sat = min(world.capturable_satellites, key=lambda ps: manhattan(lnd.pos, ps.pos))
                if manhattan(lnd.pos, nearest_sat.pos) < d_eh:
                    goals = set(water_neighbors_of_tile(nearest_sat.pos, world, W, H))
            elif world.capturable_satellites:
                # Primary Landing takes free drive-by capture if almost directly on route
                for sat in world.capturable_satellites:
                    if manhattan(lnd.pos, sat.pos) + manhattan(sat.pos, eh_pos) <= d_eh + 2:
                        sat_adj = set(water_neighbors_of_tile(sat.pos, world, W, H))
                        if sat_adj:
                            goals = sat_adj
                            break

            # Wave-sync check before crossing enemy Port's 8-tile Landing spotting ring (§7)
            escort_close   = any(manhattan(c.pos, lnd.pos) <= 4 for c in own_combat)
            escort_catching = any(4 < manhattan(c.pos, lnd.pos) <= 8 for c in own_combat)
            enemy_defenders_near = any(manhattan(e.last_pos, eh_pos) <= 6 for e in world.fresh_enemies)
            eh_known_clear = (world.enemy_home_port.fresh and not enemy_defenders_near and d_eh <= 6)

            if (
                7 <= d_eh <= 10 and not escort_close and escort_catching
                and not eh_known_clear and view.turn < 55
            ):
                # Hold outside enemy radar/threat for 1 turn so fast escorts (speed 5-6) catch up!
                if world.threat_by_hull["Landing"].get(lnd.pos, 0.0) <= 0.10:
                    continue

            risk_w = 2.5 if (escort_close or view.turn >= 55) else 7.0
            mv = pick_best_move_toward(lnd, goals, view, world, mem, risk_weight=risk_w, rng=rng)
            if mv is not None:
                return mv
    else:
        # `enemy_home` not yet discovered: capture a known frontier satellite if escorted, else stage midfield
        for lnd in landings:
            escort_close = any(manhattan(c.pos, lnd.pos) <= 4 for c in own_combat)
            if world.capturable_satellites and escort_close:
                sat = min(world.capturable_satellites, key=lambda ps: manhattan(lnd.pos, ps.pos))
                goals = set(water_neighbors_of_tile(sat.pos, world, W, H))
            else:
                # Safe midfield staging ring waiting for Destroyer/Carrier 16-tile Port sweep
                mid_x = W // 2
                goals = {
                    t for t in world.water_tiles
                    if abs(t[0] - mid_x) <= 3 and world.threat_by_hull["Landing"].get(t, 0.0) <= 0.10
                }
            mv = pick_best_move_toward(lnd, goals, view, world, mem, risk_weight=8.0, rng=rng)
            if mv is not None:
                return mv

    return None


function select_scout_goals(scout: OwnShipView, view: FoggedView,
                            world: TacticalWorld) -> Set[(int, int)]:
    """
    Computes high-information water goals for a scout (`Destroyer`, `Carrier`, `Cruiser`):
      - If `enemy_home` is unsighted: scores water tiles in the forward half by the total
        `unchecked_port_prior` weight swept within `VISIBILITY[scout.ship_type]["Port"]` (§7: up to 16!).
      - Once `enemy_home` is known: refreshes stale vision on `enemy_home` (staying at safe standoff
        distance 5..8 so `R(Destroyer/Carrier, Landing/Merchant) = 8` sees defenders/merchants!) or
        sweeps enemy satellite shipping lanes.
    """
    W, H = view.width, view.height
    if world.enemy_home_port is None and world.unchecked_port_prior:
        r_p = VISIBILITY[scout.ship_type]["Port"]
        best_val = -INF
        scored_tiles = []
        # Sample coarse 2x2 grid over navigable water for fast O(N) sweep evaluation
        for wx, wy in world.water_tiles:
            if (wx + wy) % 2 != 0:
                continue
            if world.threat_by_hull[scout.ship_type].get((wx, wy), 0.0) > 0.45:
                continue
            gain = 0.0
            for tx, ty in world.unchecked_port_prior:
                if abs(wx - tx) + abs(wy - ty) <= r_p:
                    gain += world.unchecked_port_prior[(tx, ty)]
            if gain <= 0.0:
                continue
            val = gain - 0.8 * manhattan(scout.pos, (wx, wy))
            scored_tiles.append((val, (wx, wy)))
            if val > best_val:
                best_val = val
        if scored_tiles:
            return {pos for val, pos in scored_tiles if val >= 0.80 * best_val}

    # Enemy home IS known: peek around stale enemy home or raid enemy satellite merchant lanes
    if world.enemy_home_port is not None:
        eh = world.enemy_home_port.pos
        if not world.enemy_home_port.fresh or (view.turn - world.enemy_home_port.turn_seen) >= 3:
            return {
                t for t in expand_manhattan({eh}, 7, world.water_tiles, W, H)
                if manhattan(t, eh) >= 4
            }
    if world.capturable_satellites:
        sat = max(world.capturable_satellites, key=lambda ps: view.turn - ps.turn_seen)
        return expand_manhattan({sat.pos}, 5, world.water_tiles, W, H)
    return set()


function select_combat_fleet_move(view: FoggedView, world: TacticalWorld,
                                  mem: SeatMemory, rng) -> Optional[MoveAction]:
    """
    Assigns remaining movable combat ships across 5 coordinated naval roles:
      1. HOME INTERCEPTOR: Counter-matched defense against raiders/Landings within 10 tiles of `own_home`.
      2. FOG SCOUT: Dedicated fast high-port-vision ship sweeping unsighted enemy home / stale lanes.
      3. STEALTH SUBMARINE: Hunts enemy Battleships/Carriers/Cruisers/Merchants while penalizing
         `sub_danger_tiles` (`Destroyer` R=3, `Port` R=1, §7).
      4. CARRIER STANDOFF: Positions behind friendly screen at distance 5..7 from surface targets.
      5. LANDING ESCORT & MAIN BATTLE GROUP: Screens advancing friendly Landings 1..3 tiles ahead
         (never clogging `enemy_home` `dist==1` tiles!) or masses local superiority to blockade `enemy_home`.
    """
    W, H = view.width, view.height
    movable_combat = [
        s for s in view.own_ships
        if s.ship_type in COMBAT_TYPES and s.remaining_moves > 0 and not s.has_attacked
    ]
    if not movable_combat:
        return None

    reserved_eh_adj = get_reserved_enemy_home_capture_tiles(view, world)
    assigned_ids = set()

    # --- Role 1: Home & Economic Defense Interceptors ---
    if view.own_home is not None:
        home_threats = [
            e for e in world.fresh_enemies
            if manhattan(e.last_pos, view.own_home.pos) <= 10
            or any(not p.is_home and manhattan(e.last_pos, p.pos) <= 5 for p in view.own_ports)
        ]
        if home_threats:
            # Prioritize enemy Landings near `own_home` above all else
            home_threats.sort(key=lambda e: (
                0 if e.ship_type == "Landing" else 1,
                manhattan(e.last_pos, view.own_home.pos)
            ))
            target_e = home_threats[0]
            # Pick best matchup defender(s) nearby
            interceptors = sorted(
                movable_combat,
                key=lambda s: (
                    -kill_probability(s.ship_type, target_e.ship_type),
                    manhattan(s.pos, target_e.last_pos)
                )
            )
            for ic in interceptors[:2]:
                er = max(1, effective_attack_range(ic.ship_type, target_e.ship_type))
                goals = expand_manhattan({target_e.last_pos}, er, world.water_tiles, W, H)
                mv = pick_best_move_toward(ic, goals, view, world, mem, risk_weight=1.0, rng=rng)
                if mv is not None:
                    return mv
                assigned_ids.add(ic.ship_id)

    # --- Role 2: Dedicated Fog Scout (highest Port spotting + speed) ---
    need_scout = (
        world.enemy_home_port is None
        or (not world.enemy_home_port.fresh and view.turn - world.enemy_home_port.turn_seen >= 4)
    )
    if need_scout:
        scout_pool = [s for s in movable_combat if s.ship_id not in assigned_ids]
        if scout_pool:
            scout_pool.sort(
                key=lambda s: (VISIBILITY[s.ship_type]["Port"], SHIP_STATS[s.ship_type]["speed"]),
                reverse=True
            )
            scout = scout_pool[0]
            s_goals = select_scout_goals(scout, view, world) - reserved_eh_adj
            if s_goals:
                mv = pick_best_move_toward(scout, s_goals, view, world, mem, risk_weight=3.0, rng=rng,
                                           extra_blocked=reserved_eh_adj)
                if mv is not None:
                    return mv
                assigned_ids.add(scout.ship_id)

    # --- Compute Local Force Balance for Offensive vs Regroup Posture ---
    own_power = sum(SHIP_STATS[s.ship_type]["strength"] for s in view.own_ships if s.ship_type in COMBAT_TYPES)
    seen_enemy_power = sum(
        SHIP_STATS[e.ship_type]["strength"] for e in world.fresh_enemies if e.ship_type in COMBAT_TYPES
    )
    have_advantage = (
        own_power >= 0.90 * seen_enemy_power * (1.0 / mem.doctrine["aggression"])
        or view.turn >= 55
    )

    own_landings = [s for s in view.own_ships if s.ship_type == "Landing"]
    escorts_assigned = 0

    for s in movable_combat:
        if s.ship_id in assigned_ids:
            continue

        # --- Role 3: Submarine Stealth Hunting (§6, §7) ---
        if s.ship_type == "Submarine":
            sub_pen = {t: 6.0 for t in world.sub_danger_tiles}
            juicy_prey = [
                e for e in (world.fresh_enemies + world.active_stale_enemies)
                if e.ship_type in {"Battleship", "Carrier", "Cruiser", "Merchant", "Landing"}
            ]
            if juicy_prey:
                prey = min(juicy_prey, key=lambda e: manhattan(s.pos, e.last_pos))
                er = effective_attack_range("Submarine", prey.ship_type)
                # Prefer exact standoff ring `2..er` where Battleship/Cruiser (`spot=1`) cannot even see us!
                min_r = 2 if er >= 2 and prey.ship_type in {"Battleship", "Cruiser", "Carrier"} else 1
                goals = {
                    t for t in expand_manhattan({prey.last_pos}, er, world.water_tiles, W, H)
                    if manhattan(t, prey.last_pos) >= min_r
                } - reserved_eh_adj
                mv = pick_best_move_toward(
                    s, goals, view, world, mem, risk_weight=2.0, rng=rng,
                    extra_blocked=reserved_eh_adj, extra_penalty=sub_pen
                )
                if mv is not None:
                    return mv

        # --- Role 4: Escort Advancing Friendly Landings (up to 2 escorts per Landing) ---
        if own_landings and escorts_assigned < 2 * len(own_landings):
            vip = min(own_landings, key=lambda l: manhattan(s.pos, l.pos))
            escorts_assigned += 1
            if world.enemy_home_port is not None:
                eh = world.enemy_home_port.pos
                # Screen 1..3 tiles ahead of Landing toward `enemy_home`, NEVER on `dist==1` of `eh`!
                ahead_ring = {
                    t for t in expand_manhattan({vip.pos}, 3, world.water_tiles, W, H)
                    if manhattan(t, eh) <= manhattan(vip.pos, eh) and manhattan(t, eh) >= 2
                } - reserved_eh_adj
            else:
                ahead_ring = expand_manhattan({vip.pos}, 2, world.water_tiles, W, H)
            if ahead_ring:
                mv = pick_best_move_toward(
                    s, ahead_ring, view, world, mem, risk_weight=1.8, rng=rng,
                    extra_blocked=reserved_eh_adj
                )
                if mv is not None:
                    return mv

        # --- Role 5: Carrier Standoff / Main Battle Fleet / Raider ---
        if have_advantage:
            # Pick best counter-matched enemy target (fresh first, then un-cleared recent stale)
            all_targets = world.fresh_enemies if world.fresh_enemies else world.active_stale_enemies
            if all_targets:
                best_t = max(
                    all_targets,
                    key=lambda e: (
                        kill_probability(s.ship_type, e.ship_type) * strategic_target_value(e, view, world)
                        - 6.0 * manhattan(s.pos, e.last_pos)
                    )
                )
                er = max(1, effective_attack_range(s.ship_type, best_t.ship_type))
                min_standoff = 5 if (s.ship_type == "Carrier" and er >= 6) else 1
                goals = {
                    t for t in expand_manhattan({best_t.last_pos}, er, world.water_tiles, W, H)
                    if manhattan(t, best_t.last_pos) >= min_standoff
                } - reserved_eh_adj
            elif world.enemy_home_port is not None:
                # Blockade ring at distance 2..4 around `enemy_home` (leaving dist==1 open for our Landing!)
                eh = world.enemy_home_port.pos
                goals = {
                    t for t in expand_manhattan({eh}, 4, world.water_tiles, W, H)
                    if manhattan(t, eh) >= 2
                } - reserved_eh_adj
            else:
                goals = select_scout_goals(s, view, world)
        else:
            # Locally outgunned: rally near our most forward owned port to mass a coherent wave
            fwd_port = select_most_forward_owned_port(view, world)
            goals = expand_manhattan({fwd_port.pos}, 3, world.water_tiles, W, H)

        if goals:
            mv = pick_best_move_toward(
                s, goals, view, world, mem, risk_weight=2.2, rng=rng,
                extra_blocked=reserved_eh_adj
            )
            if mv is not None:
                return mv

    return None


# ------------------------------------------------------------------------------
# 6.6 Diverse, Counter-Adaptive Buy Policy (`BuildShipAction`, §1.1, §3, §4, §5)
# ------------------------------------------------------------------------------

function select_most_forward_owned_port(view: FoggedView, world: TacticalWorld) -> OwnPortView:
    if world.enemy_home_port is not None:
        return min(view.own_ports, key=lambda p: manhattan(p.pos, world.enemy_home_port.pos))
    mid_target = (view.width - 1 - view.own_home.pos[0], view.own_home.pos[1])
    return min(view.own_ports, key=lambda p: manhattan(p.pos, mid_target))


function score_hull_priorities(view: FoggedView, world: TacticalWorld,
                               mem: SeatMemory) -> Dict[str, float]:
    """
    Computes log-preference scores over ALL 8 ship types combining:
      1. Seat's sampled Strategic Doctrine + per-game hull jitter (cross-game diversity)
      2. Under-built hull curiosity bonus (prevents any of the 8 types from vanishing in harvest)
      3. Fog-legal counter-building against fresh + active stale enemy sightings (§6, §7)
      4. Economic, scouting, port-building, and Landing win-condition pacing
    """
    own_counts = Counter(s.ship_type for s in view.own_ships)
    combat_count = sum(own_counts[t] for t in COMBAT_TYPES)
    satellites   = [p for p in view.own_ports if not p.is_home]

    # Base score from doctrine * mild game-specific jitter + under-sampled hull bonus
    total_built = max(1, sum(mem.built_counts.values()))
    scores = {}
    for t in ALL_SHIP_TYPES:
        base = math.log(mem.doctrine["weights"][t] * mem.type_jitter[t])
        share = mem.built_counts[t] / float(total_built)
        # Gentle bonus if hull share is well below 1/8 = 0.125 so all 8 types appear in traces
        diversity_bonus = 0.45 * max(0.0, 0.125 - share) / 0.125
        scores[t] = base + diversity_bonus

    # --- Economy (`Merchant`, cost 100) ---
    desired_merchants = min(len(satellites), mem.doctrine["target_merchants"])
    if len(satellites) == 0 or own_counts["Merchant"] >= len(satellites) + 1:
        scores["Merchant"] -= 4.0  # Never spam idle merchants past satellite capacity
    elif own_counts["Merchant"] == 0:
        scores["Merchant"] += 1.6  # Strong early priority for 1st merchant
    elif own_counts["Merchant"] < desired_merchants:
        scores["Merchant"] += 0.6

    # --- Early Scouting (when `enemy_home` is still hidden in fog) ---
    if world.enemy_home_port is None:
        scores["Destroyer"] += 1.1   # Cost 100, speed 5, Port spot 16, Sub spot 3
        scores["Cruiser"]   += 0.5   # Cost 150, speed 5, Port spot 12
        scores["Carrier"]   += 0.4   # Cost 250, speed 6, Port spot 16
        if view.turn == 0:
            scores["Landing"] -= 1.2 # Don't blind-buy unescorted turn-0 Landing before scouting

    # --- Fog-Legal Counter-Fleet Adaptation (§6 modifiers & §7 spotting) ---
    tracked_enemies = world.fresh_enemies + world.active_stale_enemies
    enemy_counts = Counter(e.ship_type for e in tracked_enemies)

    if enemy_counts["Submarine"] > 0:
        scores["Destroyer"] += 1.2 * min(2, enemy_counts["Submarine"])  # 2.0x mod, 80% kill, R=3 spot!
    if enemy_counts["Battleship"] + enemy_counts["Carrier"] > 0:
        scores["Submarine"] += 0.9 * min(2, enemy_counts["Battleship"] + enemy_counts["Carrier"])  # 4.0x / 3.0x!
        scores["Carrier"]   += 0.4
    if enemy_counts["Destroyer"] > 0:
        scores["Cruiser"]    += 0.8 * min(2, enemy_counts["Destroyer"])  # 1.5x mod -> 83.5% kill
        scores["Battleship"] += 0.6
    if enemy_counts["Cruiser"] > 0:
        scores["Battleship"] += 0.8 * min(2, enemy_counts["Cruiser"])    # 1.5x mod -> 86.2% kill
        scores["Submarine"]  += 0.4                                      # 2.0x mod -> 64.0% kill
    if enemy_counts["Landing"] + enemy_counts["Merchant"] > 0:
        scores["Carrier"]   += 0.6  # Snipes strength-0 hulls at range 8!
        scores["Destroyer"] += 0.4  # Fast speed-5 raider, spots non-combats at 8

    # --- Port Expansion (`Builder`, cost 250) ---
    if own_counts["Builder"] >= 1 or mem.ports_built_count >= 3 or len(view.coastline_tiles) == 0:
        scores["Builder"] -= 4.5
    elif view.turn < 2 and mem.doctrine_name != "ECONOMIC_BOOM":
        scores["Builder"] -= 1.0
    elif view.own_cash >= 250 and combat_count >= 1:
        scores["Builder"] += 0.3

    # --- Win Condition (`Landing`, cost 250) — Decisiveness Engine (§2) ---
    max_landings = 3 if view.turn >= 45 else 2
    if own_counts["Landing"] >= max_landings:
        scores["Landing"] -= 3.5
    else:
        if combat_count >= 2 and own_counts["Landing"] == 0:
            scores["Landing"] += 1.3
        if world.enemy_home_port is not None and own_counts["Landing"] == 0:
            scores["Landing"] += 1.2
        if view.turn >= 25 and own_counts["Landing"] == 0:
            scores["Landing"] += 1.5  # Anti-stall ramp

    return scores


function softmax_sample_hull(scores: Dict[str, float], temperature: float, rng) -> str:
    max_s = max(scores.values())
    weights = [math.exp((scores[t] - max_s) / max(0.10, temperature)) for t in ALL_SHIP_TYPES]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for t, w in zip(ALL_SHIP_TYPES, weights):
        acc += w
        if r <= acc:
            return t
    return ALL_SHIP_TYPES[-1]


function pick_best_build_action_for_type(ship_type: str, legal_builds: List[BuildShipAction],
                                         view: FoggedView, world: TacticalWorld,
                                         mem: SeatMemory) -> Optional[BuildShipAction]:
    """
    Chooses WHICH unlocked owned port should spawn `ship_type`:
      - Combat / Scout / Landing -> most forward safe unlocked port (closest to `enemy_home` / midline),
        unless defending a threatened Home Port!
      - Merchant -> unlocked non-home satellite port with highest stockpile (instant load next turn!),
        else port closest to stocked satellites.
      - Builder -> Home Port for `ECO_HOME`, forward port for `FORWARD`.
    """
    matching = [b for b in legal_builds if b.ship_type == ship_type]
    if not matching:
        return None

    port_by_pos = {p.pos: p for p in view.own_ports}

    # Home defense emergency: spawn combat defender at/closest to `own_home`!
    home_under_attack = (
        view.own_home is not None
        and any(manhattan(e.last_pos, view.own_home.pos) <= 8 for e in world.fresh_enemies)
    )
    if ship_type in COMBAT_TYPES and home_under_attack:
        return min(matching, key=lambda b: manhattan(b.port, view.own_home.pos))

    if ship_type == "Merchant":
        def merchant_port_score(b: BuildShipAction) -> float:
            p = port_by_pos[b.port]
            threat_pen = 200.0 * world.threat_by_hull["Merchant"].get(b.spawn_position, 0.0)
            stock_bonus = (2.0 * p.stockpile + 30.0) if not p.is_home else 0.0
            return stock_bonus - threat_pen
        return max(matching, key=merchant_port_score)

    if ship_type == "Builder" and mem.doctrine["builder_mode"] == "ECO_HOME" and view.own_home is not None:
        return min(matching, key=lambda b: manhattan(b.port, view.own_home.pos))

    # Default for Combat, Scout, Landing, Forward Builder: forward-most safe port
    target_ref = (
        world.enemy_home_port.pos if world.enemy_home_port is not None
        else (view.width - 1 - view.own_home.pos[0], view.own_home.pos[1])
    )
    def forward_port_score(b: BuildShipAction) -> float:
        d = manhattan(b.spawn_position, target_ref)
        # Don't spawn a strength-0 Landing/Builder straight into an enemy warship's guns!
        risk = world.threat_by_hull[ship_type].get(b.spawn_position, 0.0)
        risk_pen = (300.0 * risk) if ship_type in NON_COMBAT_TYPES else (40.0 * risk)
        return -d - risk_pen
    return max(matching, key=forward_port_score)


function select_ship_purchase(view: FoggedView, world: TacticalWorld,
                              mem: SeatMemory, rng) -> Optional[BuildShipAction]:
    """
    Selects a `BuildShipAction` (after movement & same-turn merchant unloads):
      1. Emergency Home Defense override (buy strongest affordable counter immediately).
      2. Checkmate Landing override (if `enemy_home` known & lightly defended and we have 0 Landings).
      3. Anti-poverty override (don't sit on 0 Merchants with stocked satellites while saving 3+ turns).
      4. Otherwise buy `mem.planned_buy` if affordable, else SAVE cash across turns so 200/250-cost
         hulls (`Carrier`, `Battleship`, `Landing`, `Builder`) are built at healthy rates!
    """
    legal_builds = [a for a in view.legal_actions if a is BuildShipAction]
    if not legal_builds:
        return None

    affordable_types = {b.ship_type for b in legal_builds}
    own_counts = Counter(s.ship_type for s in view.own_ships)

    # 1. Emergency Home Defense Override
    if view.own_home is not None:
        close_enemies = [e for e in world.fresh_enemies if manhattan(e.last_pos, view.own_home.pos) <= 8]
        local_defenders = [
            s for s in view.own_ships
            if s.ship_type in COMBAT_TYPES and manhattan(s.pos, view.own_home.pos) <= 7
        ]
        if close_enemies and len(local_defenders) < len(close_enemies) + 1:
            affordable_combat = [t for t in COMBAT_TYPES if t in affordable_types]
            if affordable_combat:
                target_e = min(close_enemies, key=lambda e: manhattan(e.last_pos, view.own_home.pos))
                best_def = max(
                    affordable_combat,
                    key=lambda t: (kill_probability(t, target_e.ship_type), SHIP_STATS[t]["speed"])
                )
                act = pick_best_build_action_for_type(best_def, legal_builds, view, world, mem)
                if act is not None:
                    mem.built_counts[best_def] += 1
                    return act

    # 2. Checkmate Override: `enemy_home` known, strong fleet / clear coast, 0 live Landings, 250+ cash
    if (
        "Landing" in affordable_types and own_counts["Landing"] == 0
        and world.enemy_home_port is not None
        and (sum(own_counts[t] for t in COMBAT_TYPES) >= 2 or view.turn >= 20)
    ):
        act = pick_best_build_action_for_type("Landing", legal_builds, view, world, mem)
        if act is not None:
            mem.built_counts["Landing"] += 1
            if mem.planned_buy == "Landing":
                mem.planned_buy = None
            return act

    # 3. Sample `planned_buy` if none is currently queued
    if mem.planned_buy is None:
        scores = score_hull_priorities(view, world, mem)
        mem.planned_buy = softmax_sample_hull(scores, mem.doctrine["buy_temperature"], rng)

    # 4. Anti-Starvation Guard: if saving for an expensive hull with 0 Merchants and a stocked satellite,
    # buy 1 Merchant (100) first so income doesn't crawl at +50/turn!
    has_stocked_sat = any(not p.is_home and p.stockpile >= 25 for p in view.own_ports)
    if (
        mem.planned_buy not in affordable_types
        and own_counts["Merchant"] == 0 and has_stocked_sat
        and "Merchant" in affordable_types
    ):
        act = pick_best_build_action_for_type("Merchant", legal_builds, view, world, mem)
        if act is not None:
            mem.built_counts["Merchant"] += 1
            return act

    # 5. Execute `planned_buy` if affordable at an unlocked open port; otherwise SAVE cash for next turn!
    if mem.planned_buy in affordable_types:
        chosen = mem.planned_buy
        act = pick_best_build_action_for_type(chosen, legal_builds, view, world, mem)
        if act is not None:
            mem.built_counts[chosen] += 1
            mem.planned_buy = None  # Clear so extra cash + extra ports can roll a 2nd ship this turn!
            return act

    return None


# ==============================================================================
# 7. TOP-LEVEL ATOMIC STEP ENTRY POINT
# ==============================================================================

function scripted_agent_action(engine, player_id: int, rng) -> Action:
    """
    Drop-in atomic policy for Battleboats under strict fog of war (§3, §7, §10).
    Returns exactly one legal `Action` from `engine.enumerate_legal(player_id)` per call.
    """
    view = extract_fogged_view(engine, player_id)

    # Safety cap against unforeseen intra-turn loops; `EndTurnAction` is always legal (§3)
    end_turn = next(a for a in view.legal_actions if a is EndTurnAction)
    if len(view.legal_actions) == 1:
        return end_turn

    mem = get_or_init_seat_memory(engine, view, rng)
    if mem.atomic_step_in_turn > 250:
        return end_turn

    world = build_tactical_world(view, mem)

    # Phase 1: Instant Win (`enemy_home`) or free adjacent satellite port capture (§2, §3)
    act = select_immediate_capture(view, world)
    if act is not None:
        return act

    # Phase 2: Free adjacent Merchant Unload (+2*cargo cash) / Merchant Load (§3, §4)
    act = select_adjacent_merchant_transfer(view)
    if act is not None:
        return act

    # Phase 3: Move loaded Merchants within 1-turn reach of Home so cash arrives BEFORE buying
    act = select_same_turn_cash_delivery_move(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 4: Coordinated Combat (matchup-weighted attacks & step-into-range setups, §5, §6, §7)
    act = select_combat_action(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 5: Emergency `own_home` body-block, `enemy_home` lane-clearing & post-shot kiting (§2, §3)
    act = select_post_attack_or_bodyblock_move(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 6: Unclog owned unlocked port spawn tiles before remaining navigation & buying
    act = clear_clogged_port_spawns(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 7: Builder Port-Founding (`BuildPortAction`) and Builder navigation (§3, §4, §10)
    act = select_builder_action(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 8: Economic Merchant routing, escorted Landing pushes, Scouts & Fleet positioning
    act = select_merchant_move(view, world, mem, rng)
    if act is not None:
        return act

    act = select_landing_move(view, world, mem, rng)
    if act is not None:
        return act

    act = select_combat_fleet_move(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 9: Ship Purchasing (`BuildShipAction`) — runs AFTER port tiles clear & cash unloads!
    # Because the engine steps atomically, if a newly spawned Destroyer/Carrier reveals an enemy
    # Submarine within range of an un-attacked friendly ship, Phase 4 still fires on the next call!
    act = select_ship_purchase(view, world, mem, rng)
    if act is not None:
        return act

    # Phase 10: End Turn (§3: resets ship flags/port locks and pays incoming player's start-of-turn income)
    return end_turn
```
