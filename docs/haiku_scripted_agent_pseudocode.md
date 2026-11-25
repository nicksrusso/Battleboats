# Haiku 5.5 scripted-agent pseudocode

Generated from `docs/task.md` as a reusable design artifact. Not executed or wired into the engine.

The first OpenRouter call (`max_tokens=65536`) hit the output cap: all 65,536 completion tokens were spent on reasoning and no answer was returned (`finish_reason=length`, 321.7 s, $0.0341278). The table below is the successful retry with `max_tokens=128000`. Combined OpenRouter cost for both calls: **$0.0888804**.

| Field | Value |
|---|---|
| Model | `anthropic/claude-haiku-5.5` |
| Reasoning effort | `high` |
| Date | 2026-11-25 (America/Chicago) |
| Request started | 2026-11-25 07:34 CST (America/Chicago) |
| Latency | 469.5 s |
| Prompt tokens | 13,641 |
| Completion tokens | 106,777 (of which 87,319 reasoning) |
| Total tokens | 120,418 |
| Cost (OpenRouter `usage.cost`) | $0.0547526 (this call) |
| Finish reason | `stop` |
| max_tokens | 128,000 |

---
# Fog-Safe Scripted Agent for Battleboats: Pseudocode

The agent below plays only from what a legal player knows: its own fleet and ports, fresh and stale enemy sightings, enemy ports (whose positions never change), and terrain learned from its own units' legal moves and the coastline list. Each call to `next_action` returns one atomic action from `enumerate_legal`. `play_turn` loops over these calls until the agent ends its turn.

Section 0 lists the adapter assumptions I had to make. Where the brief does not pin down a field name or a rule detail, the pseudocode says so.

---

## 0. Interface assumptions

The brief does not give exact attribute names for the engine adapter. The agent depends only on the items below. Map them to the real API.

| Name used here | Meaning | Source in brief |
|---|---|---|
| `FogView(engine, me)` | Wrapper that exposes only fogged information | `observation.py`, `sighting.py` |
| `view.turn`, `view.width`, `view.height` | Round counter and board size | `engine.turn`, globals 5–7 (§10.2) |
| `view.cash()` | Own cash | §10.1 (own cash on home port) |
| `view.friendly_ships()` → `(id, type, pos, cargo, has_attacked)` | Own ships | §10 token 1 |
| `view.friendly_ports()` → `(id, pos, stockpile, is_home)` | Own ports | §10 token 2 |
| `view.enemy_ship_sightings()` → `(id, type, pos, fresh, turn_seen)` | Fresh and stale enemy hulls | §7 `Sighting` |
| `view.enemy_port_sightings()` → `(id, pos, fresh, turn_seen, is_home, last_known_owner)` | Enemy port records | §7 `PortSighting` |
| `view.coastline()` → set of `(x, y)` | Coastline land tiles, excluding ports | §7, §10 token 5 |
| `engine.enumerate_legal(me)` → list of actions with `.kind` and fields | Legal atomic actions | §3 |
| `engine.step(action)` | Applies one action | §3 |
| `max_turns` (the scaled cap) | Supplied by the harness from env config | §9 |

Action field names: `Move(ship_id, destination)`, `Attack(attacker_id, target_id)`, `BuildShip(port_id, spawn, ship_type)`, `BuildPort(builder_id, target)`, `Capture(landing_id, target)`, `MerchantLoad(merchant_id, port_id)`, `MerchantUnload(merchant_id, port_id)`, `EndTurn()`.

Assumptions the engine is authoritative on. If a check disagrees, the agent still plays only legal actions.

- Attack and move on the same ship are independent. A ship may move, attack, and move again, within its speed.
- Capture needs only adjacency, not remaining speed.
- Sighting records carry the engine ship id, so an attack target can be matched to a fresh track.

---

## 1. Rule constants and combat math

```python
# ============================================================
# A. Rule constants (task.md §3–§7). Copied from the brief.
#    Tests in §11 assert these against the CSVs.
# ============================================================
SHIPS   = ["Carrier","Battleship","Cruiser","Destroyer","Submarine","Landing","Merchant","Builder"]
COMBAT  = ["Carrier","Battleship","Cruiser","Destroyer","Submarine"]    # §5: only these may attack
TARGETS = SHIPS + ["Port"]                                              # §7 column order

COST = {"Carrier":250,"Battleship":200,"Cruiser":150,"Destroyer":100,"Submarine":125,
        "Landing":250,"Merchant":100,"Builder":250}                     # §5
STRENGTH = {"Carrier":100,"Battleship":125,"Cruiser":75,"Destroyer":50,"Submarine":50,
            "Landing":0,"Merchant":0,"Builder":0}                       # §5; 0 => always dies (§6)
SPEED = {"Carrier":6,"Battleship":3,"Cruiser":5,"Destroyer":5,"Submarine":3,
         "Landing":3,"Merchant":3,"Builder":3}                          # §5
ATTACK_RANGE = {"Carrier":12,"Battleship":4,"Cruiser":3,"Destroyer":3,"Submarine":3,
                "Landing":0,"Merchant":0,"Builder":0}                   # §5

# SPOT[spotter][target]: visibilityMatrix.csv (§7). Rows = spotter, columns in TARGETS order.
SPOT_ROWS = {
  "Carrier":    [6,6,4,3,2,8,8,8,16],
  "Battleship": [4,4,3,3,1,4,4,4,8],
  "Cruiser":    [4,4,3,3,1,6,6,6,12],
  "Destroyer":  [4,4,3,3,3,8,8,8,16],
  "Submarine":  [4,4,3,2,1,2,2,2,4],
  "Landing":    [4,4,3,3,1,2,2,2,4],
  "Merchant":   [4,4,3,3,1,2,2,2,4],
  "Builder":    [4,4,3,3,1,2,2,2,4],
  "Port":       [16,16,12,12,1,8,8,8,16],
}
SPOT = {s: dict(zip(TARGETS, row)) for s, row in SPOT_ROWS.items()}

# MU[attacker][defender] for combat pairs (attackModifiers.csv, §6). Missing pairs default to 1.0.
MU = {
  "Carrier":    {"Carrier":1.0,"Battleship":1.5,"Cruiser":1.0, "Destroyer":0.75,"Submarine":1.0},
  "Battleship": {"Carrier":1.0,"Battleship":1.0,"Cruiser":1.5, "Destroyer":1.0, "Submarine":0.75},
  "Cruiser":    {"Carrier":1.25,"Battleship":1.0,"Cruiser":1.0,"Destroyer":1.5, "Submarine":1.0},
  "Destroyer":  {"Carrier":1.25,"Battleship":1.0,"Cruiser":1.0,"Destroyer":1.0, "Submarine":2.0},
  "Submarine":  {"Carrier":3.0,"Battleship":4.0,"Cruiser":2.0, "Destroyer":0.5, "Submarine":1.0},
}
K_KILL = 2.0                                       # §6 kill-curve exponent

CASH_PER_MATERIAL   = 2                            # §4
PORT_PRODUCTION     = 25                           # §4
MERCHANT_CAPACITY   = 100                          # §4
HOME_CASH_PER_TURN  = PORT_PRODUCTION * CASH_PER_MATERIAL   # 50: home converts 25 mat -> 50 cash (§4)

# Agent tuning constants (not rules; the rules above are authoritative).
STALE_HORIZON        = 6     # rounds; stale hulls older than this are ignored for planning
SEARCH_REFRESH       = 8     # rounds before a searched cell becomes worth re-searching
PORT_SITE_IDEAL_DIST = 6     # preferred satellite distance from home (short merchant trips)
BUILD_PORT_CUTOFF    = 0.5   # no new ports after this fraction of the turn cap (payback)
ALLIN_FRAC           = 0.70  # after this fraction of the cap, accept risk to force a decision
SAVE_HORIZON_TURNS   = 3     # save for an unaffordable head only if it is <= this many turns of income away
SAVE_MAX_ROUNDS      = 4     # consecutive rounds we may save for one type
W_PROGRESS           = 1.0   # score per tile of objective progress
W_EXPOSE             = 3.0   # score cost per unit exposure, scaled by unit cost/100
W_ATTACK             = 1/50  # attack-enabling value: p_kill * value * W_ATTACK
CAPTURE_BONUS        = 25.0
WIN_BONUS            = 1000.0


def in_bounds(c, W, H):
    return 0 <= c[0] < W and 0 <= c[1] < H

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors4(c):
    x, y = c
    return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

def cells_within(c, r, W, H):
    """All in-bounds cells at Manhattan distance <= r from c."""
    for dx in range(-r, r+1):
        rem = r - abs(dx)
        for dy in range(-rem, rem+1):
            x, y = c[0]+dx, c[1]+dy
            if 0 <= x < W and 0 <= y < H:
                yield (x, y)
```

Derived combat tables:

```python
def eng(att, tgt):
    """Engagement distance: an attack needs Manhattan d <= ATTACK_RANGE[att] AND d <= SPOT[att][tgt] (§6).
    Example: Carrier vs Destroyer = min(12, 3) = 3, not 12. Submarine vs Destroyer = min(3, 2) = 2.
    Non-combat hulls return 0 (they cannot attack, §5)."""
    if att not in COMBAT:
        return 0
    return min(ATTACK_RANGE[att], SPOT[att][tgt])

def reach(att, tgt):
    """How far a hull can hit a target of type tgt on its next turn: move SPEED tiles, then engage (§3, §6).
    Example: Carrier vs Landing = 6 + min(12, 8) = 14. Submarine vs Landing = 3 + 2 = 5."""
    return SPEED[att] + eng(att, tgt)

def p_kill(att, dfn):
    """Single Bernoulli trial (§6). Strength-0 defenders (Landing, Merchant, Builder) always die.
    Non-combat attackers never attack."""
    if STRENGTH[dfn] == 0:
        return 1.0
    if att not in COMBAT:
        return 0.0
    x = STRENGTH[att] * MU[att][dfn] / STRENGTH[dfn]
    return x**K_KILL / (1.0 + x**K_KILL)

def turn_cap(home_manhattan):
    """Reference only. The harness supplies the cap from env config. Never compute it from
    an enemy home position the agent does not know (§9)."""
    scale = max(1.0, home_manhattan / 14.0)
    return max(80, 10 * math.ceil(80 * scale / 10))
```

---

## 2. Agent state and doctrines

```python
@dataclass
class Doctrine:
    name: str
    w: dict              # relative buy weight per type (soft preference; diminishing returns below)
    landing_deadline: int  # round by which we force one Landing if none is alive (§5 Landing role)
    port_goal: int       # satellite ports to aim for (§4 economy)
    scouts: int          # scout-capable hulls wanted while the enemy home is unknown (§7 Port spotting)
    aggression: float    # 0..1; higher accepts more exposure (§6)
    forward_base: bool   # buy combat hulls at the port nearest the front rather than home

# Fixed doctrine templates. Each seat samples one; per-game jitter keeps fleets diverse.
DOCTRINE_TEMPLATES = {
  "RAIDER": Doctrine("RAIDER",
      {"Carrier":.5,"Battleship":.5,"Cruiser":1.0,"Destroyer":1.2,"Submarine":.8,
       "Landing":2.5,"Merchant":1.0,"Builder":.5},
      landing_deadline=12, port_goal=2, scouts=1, aggression=0.75, forward_base=True),
  "ECONOMIST": Doctrine("ECONOMIST",
      {"Carrier":.6,"Battleship":.5,"Cruiser":.8,"Destroyer":.8,"Submarine":.5,
       "Landing":1.0,"Merchant":2.0,"Builder":1.5},
      landing_deadline=25, port_goal=4, scouts=1, aggression=0.4, forward_base=False),
  "FLEET": Doctrine("FLEET",
      {"Carrier":1.5,"Battleship":1.5,"Cruiser":1.2,"Destroyer":1.0,"Submarine":.5,
       "Landing":.8,"Merchant":.8,"Builder":.5},
      landing_deadline=30, port_goal=2, scouts=1, aggression=0.5, forward_base=False),
  "WOLFPACK": Doctrine("WOLFPACK",
      {"Carrier":.3,"Battleship":.3,"Cruiser":.8,"Destroyer":2.0,"Submarine":2.0,
       "Landing":1.0,"Merchant":1.0,"Builder":.7},
      landing_deadline=22, port_goal=2, scouts=1, aggression=0.6, forward_base=True),
}

def sample_doctrine(rng):
    """Each seat samples independently, so mismatched seats produce varied matchups (§1.1 diversity).
    BALANCED draws every weight uniformly, so no fleet recipe dominates."""
    name = rng.choice(["RAIDER","ECONOMIST","FLEET","WOLFPACK","BALANCED"])
    if name == "BALANCED":
        d = Doctrine("BALANCED",
            {t: rng.uniform(0.5, 1.5) for t in SHIPS},
            landing_deadline=rng.randint(15, 30), port_goal=rng.choice([2,3]),
            scouts=1, aggression=rng.uniform(0.3, 0.8), forward_base=(rng.random() < 0.5))
    else:
        t = DOCTRINE_TEMPLATES[name]
        d = Doctrine(name, dict(t.w), t.landing_deadline, t.port_goal, t.scouts,
                     t.aggression, t.forward_base)
    for k in d.w:                                    # log-normal jitter, sigma = 0.25
        d.w[k] *= math.exp(rng.gauss(0.0, 0.25))
    d.aggression = min(1.0, max(0.0, d.aggression + rng.uniform(-0.1, 0.1)))
    return d

@dataclass
class AgentState:
    doctrine: Doctrine
    cap: int                               # turn cap from env config (§9)
    use_map_prior: bool = True             # §8 generator property; set False for custom maps
    known_land: set = field(default_factory=set)
    known_water: set = field(default_factory=set)
    terrain_version: int = 0
    searched_turn: dict = field(default_factory=dict)   # cell -> last round our port-spot covered it
    role: dict = field(default_factory=dict)            # ship_id -> role string
    role_prev: dict = field(default_factory=dict)
    explore_target: dict = field(default_factory=dict)  # ship_id -> cell
    hunt_target: dict = field(default_factory=dict)     # ship_id -> enemy Track
    merchant_claim: dict = field(default_factory=dict)  # merchant ship_id -> own Port
    builder_site: dict = field(default_factory=dict)    # builder ship_id -> coastline tile
    dead_sites: set = field(default_factory=set)
    landing_target: object = None
    moves_used: dict = field(default_factory=dict)      # ship_id -> MoveActions issued this turn
    saving: Counter = field(default_factory=Counter)    # type -> consecutive rounds we saved for it
    saved_this_turn: set = field(default_factory=set)
    allin: bool = False
    in_turn: bool = False
    field_cache: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
```

---

## 3. Fog-safe belief

The belief is rebuilt on every atomic step, because sightings refresh after every `step()`. Nothing in it comes from enemy internals.

```python
@dataclass
class Ship:
    id: int; type: str; pos: tuple; cargo: int; has_attacked: bool

@dataclass
class OwnPort:
    id: int; pos: tuple; stockpile: int; is_home: bool

@dataclass
class Track:          # enemy hull sighting (fresh or stale)
    id: int; type: str; pos: tuple; fresh: bool; age: int   # age = rounds since turn_seen

@dataclass
class PortTrack:      # enemy port sighting; port positions never change (§4)
    pos: tuple; is_home: bool; fresh: bool

@dataclass
class Belief:
    turn: int; W: int; H: int; cash: int; cap: int
    mine: list; my_ports: list; home: OwnPort
    enemy: list; enemy_ports: list; enemy_home: PortTrack
    coast: set


def build_belief(view, st):
    """Only fogged information (§7, §10). Stale hulls are kept for STALE_HORIZON rounds and given drift later.
    Enemy ports are reliable: their positions never change, and ownership only changes by our captures,
    which we see (own ports are fully known)."""
    B = Belief(turn=view.turn, W=view.width, H=view.height, cash=view.cash(), cap=st.cap,
               mine=[], my_ports=[], home=None, enemy=[], enemy_ports=[],
               enemy_home=None, coast=set(view.coastline()))
    B.mine = [Ship(s.id, s.type, s.pos, s.cargo, s.has_attacked) for s in view.friendly_ships()]
    B.my_ports = [OwnPort(p.id, p.pos, p.stockpile, p.is_home) for p in view.friendly_ports()]
    B.home = next(p for p in B.my_ports if p.is_home)      # own home is always known (§7)
    own_port_pos = {p.pos for p in B.my_ports}

    for r in view.enemy_ship_sightings():                  # §7 Sighting records
        age = max(0, B.turn - r.turn_seen)
        if not r.fresh and age > STALE_HORIZON:
            continue                                       # too uncertain to plan around
        B.enemy.append(Track(r.id, r.type, r.pos, r.fresh, age))

    for r in view.enemy_port_sightings():                  # §7 PortSighting
        if r.last_known_owner != OPP_OWNER_TAG:            # OPP_OWNER_TAG: opponent's id in view
            continue
        if r.pos in own_port_pos:                          # we captured it; it is ours now
            continue
        B.enemy_ports.append(PortTrack(r.pos, bool(r.is_home), r.fresh))
    B.enemy_home = next((p for p in B.enemy_ports if p.is_home), None)   # revealed once spotted (§7)

    note_terrain(B, st)
    mark_searched(B, st)
    return B


def note_terrain(B, st):
    """Terrain we may know: coastline tiles (§7), port tiles (land), and water under our/visible hulls."""
    before = len(st.known_land)
    st.known_land |= B.coast | {p.pos for p in B.my_ports} | {p.pos for p in B.enemy_ports}
    st.known_water |= {s.pos for s in B.mine} | {t.pos for t in B.enemy if t.fresh}
    if len(st.known_land) != before:
        st.terrain_version += 1


def mark_searched(B, st):
    """A cell is 'searched' when one of our hulls or ports could spot an enemy PORT there (§7 Port column).
    This is what drives scouting."""
    for u in B.mine:
        for c in cells_within(u.pos, SPOT[u.type]["Port"], B.W, B.H):
            st.searched_turn[c] = B.turn
    for p in B.my_ports:
        for c in cells_within(p.pos, SPOT["Port"]["Port"], B.W, B.H):
            st.searched_turn[c] = B.turn


def learn_terrain_from_legal(view, legal, st):
    """At turn start every own hull has full speed (reset at EndTurn, §3), so its legal first moves are exactly
    its open-water neighbours. A neighbour that is neither a legal move nor occupied is land. Adjacent hulls are
    always spotted (the smallest spotting radius is 1, §7), so 'occupied' is known."""
    legal_dest = defaultdict(set)
    for a in legal:
        if a.kind == "Move":
            legal_dest[a.ship_id].add(a.destination)
    occupied = ({s.pos for s in view.friendly_ships()} |
                {r.pos for r in view.enemy_ship_sightings() if r.fresh} |
                {p.pos for p in view.friendly_ports()} |
                {p.pos for p in view.enemy_port_sightings()})
    for s in view.friendly_ships():
        for n in neighbors4(s.pos):
            if not in_bounds(n, view.width, view.height):
                continue
            if n in legal_dest[s.id]:
                st.known_water.add(n)
            elif n not in occupied:
                st.known_land.add(n)
    st.terrain_version += 1
```

---

## 4. Fields, threat, value

```python
def approach_tiles(p, st, W, H):
    """Tiles from which a ship can act on a port at p: its 4-neighbours that are not known land.
    Landing and Builder need an adjacent water tile (§3, §4)."""
    return [n for n in neighbors4(p) if in_bounds(n, W, H) and n not in st.known_land]

def bfs_field(tiles, st, W, H):
    """Multi-source BFS distance over tiles that are not known land. Unknown tiles are optimistically passable.
    Failed moves are self-correcting: a blocked tile shows up as a missing legal move and is then learned as land."""
    dist, q = {}, deque()
    for t in tiles:
        if in_bounds(t, W, H) and t not in st.known_land and t not in dist:
            dist[t] = 0
            q.append(t)
    while q:
        c = q.popleft()
        for n in neighbors4(c):
            if n in dist or not in_bounds(n, W, H) or n in st.known_land:
                continue
            dist[n] = dist[c] + 1
            q.append(n)
    return dist

def exposure(B, tile, utype):
    """Probability that a KNOWN enemy combat hull kills a unit of type utype on `tile` next turn (§3, §6).
    Reach = SPEED + eng; stale tracks drift SPEED per round since last seen (capped). Unseen hulls are not
    modelled, so this is a lower bound on risk. Strength-0 units die with P=1 (§6)."""
    surv = 1.0
    for e in B.enemy:
        if e.type not in COMBAT:
            continue
        drift = 0 if e.fresh else min(SPEED[e.type] * (e.age + 1), 12)
        if manhattan(tile, e.pos) <= reach(e.type, utype) + drift:
            surv *= 1.0 - p_kill(e.type, utype)
    return 1.0 - surv

def value(t, B):
    """Value of killing or capturing an enemy hull. Cost-based (§5), with multipliers:
    Landing x2 (it is the only capture threat, §3); near our home x1.5; stale x0.5 (uncertain)."""
    v = COST[t.type]
    if t.type == "Landing":
        v *= 2.0
    if manhattan(t.pos, B.home.pos) <= 8:
        v *= 1.5
    if not t.fresh:
        v *= 0.5
    return v

def strategic_anchor(B, st):
    """Where the enemy home is, or a prior point. The prior is the mirror of our home across the midline
    (generator puts homes on opposite halves, §8). It is a search prior, not a fact."""
    if B.enemy_home is not None:
        return B.enemy_home.pos
    hx, hy = B.home.pos
    return (B.W - 1 - hx, hy)

def enemy_home_region(B, st):
    """§8: homes are >= W/2 apart (Manhattan) and on opposite halves. Public map-preset property, so it is a
    legitimate search prior. Disable with st.use_map_prior = False for custom maps."""
    cells = [(x, y) for x in range(B.W) for y in range(B.H)]
    if not st.use_map_prior:
        return cells
    mid = (B.W - 1) / 2.0
    side = 1 if B.home.pos[0] < mid else -1
    return [c for c in cells
            if (c[0] - mid) * side > 0 and manhattan(c, B.home.pos) >= B.W // 2]

def mine_by_id(B, sid):
    return next((s for s in B.mine if s.id == sid), None)

def enemy_by_id(B, tid):
    return next((t for t in B.enemy if t.id == tid), None)

def enemy_port_at(B, pos):
    return next((p for p in B.enemy_ports if p.pos == pos), None)

def satellites(B):
    return [p for p in B.my_ports if not p.is_home]

def income_estimate(B):
    """Heuristic: home cash 50/turn, plus 25 cash/turn per satellite assuming half is delivered (§4)."""
    return HOME_CASH_PER_TURN + PORT_PRODUCTION * len(satellites(B))
```

---

## 5. Turn driver and immediate actions

The engine enumerates legal actions every step, so attacks, captures, and transfers are re-checked after each change. The order is: free sure-thing actions first, then purchases, then movement, then end turn.

```python
def play_turn(engine, me, st, rng):
    """Harness loop. Calls next_action until an EndTurn is issued or the step budget runs out."""
    steps = 0
    while True:
        a = next_action(engine, me, st, rng)
        engine.step(a)
        steps += 1
        if a.kind == "EndTurn" or engine.winner is not None:
            break
        if steps >= MAX_STEPS_PER_TURN:
            engine.step(EndTurnAction())
            break

MAX_STEPS_PER_TURN = 300   # safety; every atomic action spends a finite resource, so the loop terminates


def next_action(engine, me, st, rng):
    view = FogView(engine, me)
    legal = engine.enumerate_legal(me)
    if not st.in_turn:
        on_turn_start(view, legal, st, rng)
        st.in_turn = True
    B = build_belief(view, st)
    for step in (choose_capture, choose_attack, choose_merchant_transfer,
                 choose_buy, choose_build_port, choose_move):
        a = step(B, legal, st, rng)
        if a is not None:
            if a.kind == "Move":
                st.moves_used[a.ship_id] = st.moves_used.get(a.ship_id, 0) + 1
            return a
    return end_turn(st)


def on_turn_start(view, legal, st, rng):
    st.moves_used = {}
    st.saved_this_turn = set()
    st.field_cache = {}
    learn_terrain_from_legal(view, legal, st)
    B = build_belief(view, st)
    st.allin = B.turn >= ALLIN_FRAC * st.cap
    st.landing_target = choose_landing_target(B, st)
    assign_roles(B, st)
    assign_merchant_claims(B, st)
    assign_builder_sites(B, st, legal)
    assign_explore_targets(B, st)
    for s in B.mine:
        if st.role.get(s.id) == "HUNT":
            st.hunt_target[s.id] = pick_hunt_target(s, B)
    st.log.append(("turn", B.turn, st.doctrine.name, len(B.mine), B.cash))


def end_turn(st):
    for t in SHIPS:
        st.saving[t] = st.saving[t] + 1 if t in st.saved_this_turn else 0
    st.in_turn = False
    return EndTurnAction()


def choose_capture(B, legal, st, rng):
    """§3 CapturePortAction: free, and the landing is NOT consumed. Capturing the enemy home ends the game (§9).
    Capture adjacent ports whenever legal: the port's stockpile is lost to the enemy (§4), and we gain a port."""
    caps = [a for a in legal if a.kind == "Capture"]
    if not caps:
        return None
    def key(a):
        p = enemy_port_at(B, a.target)     # adjacent => visible, so is_home is known
        return 0 if (p is not None and p.is_home) else 1
    return min(caps, key=key)


def choose_attack(B, legal, st, rng):
    """§6: an attack is a single Bernoulli trial. The defender never fires back on our turn, so an attack costs
    nothing except the action itself. Take the highest expected value p_kill * value.
    The engine's legal list already enforces range and spotting (§6). We re-check it, so an ID mismatch is skipped."""
    best, best_v = None, -1.0
    for a in legal:
        if a.kind != "Attack":
            continue
        att = mine_by_id(B, a.attacker_id)
        tgt = enemy_by_id(B, a.target_id)
        if att is None or tgt is None or not tgt.fresh:
            st.log.append(("warn", "attack target is not a fresh sighting: adapter mismatch"))
            continue
        if manhattan(att.pos, tgt.pos) > eng(att.type, tgt.type):   # §6 range vs spotting
            continue
        v = p_kill(att.type, tgt.type) * value(tgt, B) + 1e-6 * rng.random()
        if v > best_v:
            best, best_v = a, v
    if best is not None:
        st.log.append(("attack", best.attacker_id, best.target_id))
    return best


def choose_merchant_transfer(B, legal, st, rng):
    """§3/§4: unload only at the home port (cash = cargo*2). Load from an adjacent owned non-home port.
    Both are always beneficial, so do them whenever legal."""
    for a in legal:
        if a.kind == "MerchantUnload":
            return a
    loads = [a for a in legal if a.kind == "MerchantLoad"]
    if not loads:
        return None
    return max(loads, key=lambda a: next(p.stockpile for p in B.my_ports if p.id == a.port_id))
```

---

## 6. Buy policy

Two layers. Hard needs are guaranteed fleet shares (scout, merchant floor, landing floor, builder). The soft layer buys the type with the highest `w / (1 + count)`, so each type has diminishing returns. A save rule prevents the "stall" failure in §1.1: save only when the top choice is close, and never for more than a few rounds.

```python
def choose_buy(B, legal, st, rng):
    opts = defaultdict(list)
    for a in legal:
        if a.kind == "BuildShip":
            opts[a.ship_type].append(a)
    if not opts:
        return None
    counts = Counter(s.type for s in B.mine)

    for t in hard_needs(B, st, counts):
        if t not in opts:
            continue
        if COST[t] <= B.cash:
            return pick_spawn(t, opts[t], B, st)
        if save_ok(B, st, t):
            st.saved_this_turn.add(t)
            return None
        break

    ranked = sorted(range(len(SHIPS)),
                    key=lambda i: (-st.doctrine.w[SHIPS[i]] / (1 + counts[SHIPS[i]]), i))
    legal_ranked = [SHIPS[i] for i in ranked if SHIPS[i] in opts]
    if not legal_ranked:
        return None
    top = legal_ranked[0]
    if COST[top] <= B.cash:
        return pick_spawn(top, opts[top], B, st)
    if save_ok(B, st, top):
        st.saved_this_turn.add(top)
        return None
    for t in legal_ranked[1:]:
        if COST[t] <= B.cash:
            return pick_spawn(t, opts[t], B, st)
    return None


def hard_needs(B, st, counts):
    """Ordered guarantees. Each reason is cited to the rule it protects."""
    D = st.doctrine
    n_scout_capable = sum(1 for s in B.mine if s.type in COMBAT and SPOT[s.type]["Port"] >= 12)
    # Scouting (§7): Port spotting radius >= 12 needed to find an enemy home. Destroyer is the cheapest (§5).
    if n_scout_capable == 0 and B.enemy_home is None and B.turn < 0.5 * st.cap:
        yield "Destroyer"
    # Economy (§4): merchants haul satellite stockpiles home.
    if satellites(B) and counts["Merchant"] < merchant_target(B, st):
        yield "Merchant"
    # Win condition (§9): a landing must exist. Force one by the doctrine deadline, or once the home is known.
    if counts["Landing"] == 0 and (B.turn >= D.landing_deadline or B.enemy_home is not None):
        yield "Landing"
    # Endgame (§9): if the home is known and the turn cap nears, keep two landings in the field.
    if st.allin and B.enemy_home is not None and counts["Landing"] < 2:
        yield "Landing"
    # Ports (§4): a satellite is worth 25 materials/turn, so build early.
    if counts["Builder"] == 0 and want_new_port(B, st):
        yield "Builder"


def save_ok(B, st, t):
    """Save for t only if it is close (<= SAVE_HORIZON_TURNS of income) and we have not saved for it
    SAVE_MAX_ROUNDS rounds in a row. Prevents stalls with no guaranteed early merchant (§1.1)."""
    eta = (COST[t] - B.cash) / max(1, income_estimate(B))
    return eta <= SAVE_HORIZON_TURNS and st.saving[t] < SAVE_MAX_ROUNDS


def merchant_target(B, st):
    """Merchants so that satellite output (25 materials/turn each, §4) can be hauled home (100 per trip, §4).
    Round trip uses Manhattan distance (a lower bound on water path, so slightly optimistic) at SPEED 3."""
    sats = satellites(B)
    if not sats:
        return 0
    d = min(manhattan(B.home.pos, p.pos) for p in sats)
    rt = 2.0 * d / SPEED["Merchant"] + 1.0
    need = math.floor(PORT_PRODUCTION * len(sats) * rt / MERCHANT_CAPACITY + 0.5)
    return max(1, min(3, need))


def want_new_port(B, st):
    return (len(satellites(B)) < st.doctrine.port_goal
            and B.turn < BUILD_PORT_CUTOFF * st.cap
            and len(site_candidates(B, st)) > 0)


def pick_spawn(t, acts, B, st):
    """Choose which legal (port, spawn) pair to use. enumerate_legal gives one canonical spawn per (port, type) (§3).
    Landing: nearest the strategic anchor. Merchant: richest satellite. Builder: nearest its site.
    Combat: forward base or home, per doctrine."""
    def key(a):
        p = next(q for q in B.my_ports if q.id == a.port_id)
        if t == "Landing":
            return manhattan(p.pos, strategic_anchor(B, st))
        if t == "Merchant":
            return 0 if p.is_home else -p.stockpile
        if t == "Builder":
            site = next(iter(st.builder_site.values()), None)
            return manhattan(p.pos, site) if site else 0
        if st.doctrine.forward_base:
            return manhattan(p.pos, strategic_anchor(B, st))
        return 0 if p.is_home else 1 + manhattan(p.pos, B.home.pos)
    return min(acts, key=key)
```

---

## 7. Economy: merchants and ports

```python
def assign_merchant_claims(B, st):
    """Merchants with no cargo claim a satellite with stockpile (§4). Each stockpile is shared among claimants."""
    st.merchant_claim = {}
    claimed = Counter()
    sats = [p for p in satellites(B) if p.stockpile > 0]
    empties = sorted((m for m in B.mine if m.type == "Merchant" and m.cargo == 0),
                     key=lambda m: min((manhattan(m.pos, p.pos) for p in sats), default=999))
    for m in empties:
        best, best_v = None, -1.0
        for p in sats:
            avail = p.stockpile - MERCHANT_CAPACITY * claimed[p.id]
            if avail <= 0:
                continue
            v = min(avail, MERCHANT_CAPACITY) / (1.0 + 0.2 * manhattan(m.pos, p.pos))
            if v > best_v:
                best, best_v = p, v
        if best is not None:
            st.merchant_claim[m.id] = best
            claimed[best.id] += 1


def site_candidates(B, st):
    """Builder targets: coastline tiles (§7, §10 token 5) that are not ports and not known dead.
    Never infer hidden enemy ports from holes in the coastline list (see §11 known issues)."""
    occupied = {p.pos for p in B.my_ports} | {p.pos for p in B.enemy_ports}
    return [c for c in B.coast
            if c not in occupied and c not in st.dead_sites
            and approach_tiles(c, st, B.W, B.H)]


def site_score(c, B, st):
    """§4: a satellite makes 25 materials/turn, which must be hauled home. Prefer sites about 6 tiles from home.
    Penalise fresh enemy hulls nearby, because landings can capture any port (§3)."""
    dh = manhattan(c, B.home.pos)
    enemy_near = sum(1 for t in B.enemy if t.fresh and manhattan(c, t.pos) <= 8)
    return -abs(dh - PORT_SITE_IDEAL_DIST) - 2.0 * enemy_near


def assign_builder_sites(B, st, legal):
    """A builder is consumed when it founds a port (§3). If it stands next to its site with no legal BuildPort there,
    the site is dead and we re-plan."""
    for b in [s for s in B.mine if s.type == "Builder"]:
        site = st.builder_site.get(b.id)
        if site is not None and manhattan(b.pos, site) == 1 and not any(
                a.kind == "BuildPort" and a.builder_id == b.id and a.target == site for a in legal):
            st.dead_sites.add(site)
            site = None
        if site is None or site in st.dead_sites:
            cands = site_candidates(B, st)
            site = max(cands, key=lambda c: site_score(c, B, st) - 0.3 * manhattan(b.pos, c)) if cands else None
        st.builder_site[b.id] = site


def choose_build_port(B, legal, st, rng):
    """§3 BuildPortAction: the builder is consumed. The new port is not home, has stockpile 0, and is locked this turn.
    Prefer the assigned site if it is legal now, otherwise the best legal adjacent site."""
    acts = [a for a in legal if a.kind == "BuildPort"]
    if not acts:
        return None
    assigned = {b.id: st.builder_site.get(b.id) for b in B.mine if b.type == "Builder"}
    for a in acts:
        if assigned.get(a.builder_id) == a.target:
            return a
    return max(acts, key=lambda a: site_score(a.target, B, st))
```

---

## 8. Roles, scouting, landing, and combat planning

Roles are assigned once per turn. Each role has a movement objective, and the move chooser (§9) follows it.

```python
def choose_landing_target(B, st):
    """§9: capturing the enemy home wins. Target it once it is revealed (§7 is_home).
    Otherwise take the nearest known enemy satellite. Any capture reduces enemy income (§4) and gives us a forward port."""
    if B.enemy_home is not None:
        return B.enemy_home
    sats = [p for p in B.enemy_ports if not p.is_home]
    if not sats:
        return None
    lands = [s for s in B.mine if s.type == "Landing"]
    anchor = lands[0].pos if lands else B.home.pos
    return min(sats, key=lambda p: manhattan(p.pos, anchor))


def assign_roles(B, st):
    """Roles: LANDING, MERCHANT, BUILDER, SCOUT, ESCORT, PUSH, HUNT, GUARD."""
    D = st.doctrine
    st.role_prev, st.role = st.role, {}
    for s in B.mine:
        if s.type == "Landing":
            st.role[s.id] = "LANDING"
        elif s.type == "Merchant":
            st.role[s.id] = "MERCHANT"
        elif s.type == "Builder":
            st.role[s.id] = "BUILDER"

    combat = [s for s in B.mine if s.type in COMBAT]
    # Scouts (§7): keep prior scouts alive, then fill with the cheapest scout-capable hulls (Port spot >= 12).
    if B.enemy_home is None:
        scout_capable = [s for s in combat if SPOT[s.type]["Port"] >= 12]
        keep_ids = [s.id for s in scout_capable if st.role_prev.get(s.id) == "SCOUT"]
        fill = sorted([s for s in scout_capable if s.id not in keep_ids], key=lambda s: COST[s.type])
        chosen = [s for s in scout_capable if s.id in keep_ids] + fill
        for s in chosen[:D.scouts]:
            st.role[s.id] = "SCOUT"

    landing_alive = any(s.type == "Landing" for s in B.mine)
    for s in combat:
        if s.id in st.role:
            continue
        if B.enemy_home is not None and landing_alive and not st.allin:
            st.role[s.id] = "ESCORT"          # §9: a landing dies to any hit (§6), so keep hulls near it
        elif B.enemy_home is not None or st.allin:
            st.role[s.id] = "PUSH"            # fleet pushes toward the home
        elif pick_hunt_target(s, B) is not None:
            st.role[s.id] = "HUNT"
        else:
            st.role[s.id] = "GUARD"


def pick_hunt_target(s, B):
    """Hunt score = p_kill * value, discounted by distance. Stale tracks get value x0.5 (in value())."""
    best, best_v = None, 0.0
    for t in B.enemy:
        pk = p_kill(s.type, t.type)
        if pk <= 0:
            continue
        v = pk * value(t, B) / (1.0 + 0.15 * manhattan(s.pos, t.pos))
        if v > best_v:
            best, best_v = t, v
    return best


def explore_pool(B, st):
    """Far-side cells not searched recently (§8 prior; §7 coverage)."""
    region = enemy_home_region(B, st)
    fresh = [c for c in region if c not in st.known_land
             and B.turn - st.searched_turn.get(c, -10**9) >= SEARCH_REFRESH]
    return fresh or [c for c in region if c not in st.known_land]


def assign_explore_targets(B, st):
    """Scouts, and any landing with no target, pick a unique nearby cell from the explore pool."""
    explorers = [s for s in B.mine if st.role.get(s.id) == "SCOUT"
                 or (st.role.get(s.id) == "LANDING" and st.landing_target is None)]
    pool = explore_pool(B, st)
    taken = set()
    for u in sorted(explorers, key=lambda s: s.id):
        cands = [c for c in pool if c not in taken] or pool
        if not cands:
            continue
        c = min(cands, key=lambda c: manhattan(c, u.pos))
        st.explore_target[u.id] = c
        taken.add(c)


def objective_tiles(B, st, s):
    """Returns (tiles, cache_key) for the role's movement objective, or (None, None)."""
    role = st.role.get(s.id, "GUARD")
    if role == "LANDING":
        if st.landing_target is not None:
            p = st.landing_target
            return approach_tiles(p.pos, st, B.W, B.H), ("L", p.pos)
        c = st.explore_target.get(s.id)
        return ([c], ("X", c)) if c else (None, None)
    if role == "SCOUT":
        c = st.explore_target.get(s.id)
        return ([c], ("X", c)) if c else (None, None)
    if role == "MERCHANT":
        if s.cargo > 0:
            return approach_tiles(B.home.pos, st, B.W, B.H), ("H",)
        claim = st.merchant_claim.get(s.id)
        if claim is None:
            return None, None
        return approach_tiles(claim.pos, st, B.W, B.H), ("S", claim.pos)
    if role == "BUILDER":
        site = st.builder_site.get(s.id)
        if site is None:
            return None, None
        return approach_tiles(site, st, B.W, B.H), ("B", site)
    if role == "HUNT":
        t = st.hunt_target.get(s.id)
        if t is not None:
            return list(cells_within(t.pos, eng(s.type, t.type), B.W, B.H)), ("T", t.id, t.pos)
    if role == "ESCORT":
        lands = [x for x in B.mine if x.type == "Landing"]
        if lands:
            L = min(lands, key=lambda x: manhattan(x.pos, s.pos))
            return list(cells_within(L.pos, 2, B.W, B.H)), ("E", L.id, L.pos)
    if role == "PUSH":
        if B.enemy_home is not None:
            return approach_tiles(B.enemy_home.pos, st, B.W, B.H), ("P", B.enemy_home.pos)
        a = strategic_anchor(B, st)
        return [a], ("P", a)
    # GUARD (default, and fallback for HUNT with no target)
    return list(cells_within(B.home.pos, 2, B.W, B.H)), ("G",)


def objective_field(B, st, s):
    tiles, key = objective_tiles(B, st, s)
    if tiles is None:
        return None
    k = (s.id, key, B.turn, st.terrain_version)
    if k not in st.field_cache:
        st.field_cache[k] = bfs_field(tiles, st, B.W, B.H)
    return st.field_cache[k]
```

---

## 9. Movement

Each step picks the single legal MoveAction with the best score. The score combines objective progress, attack-enabling and capture-enabling bonuses, and a penalty for exposure. A move stops when no move scores above zero. Each move spends speed, so the loop terminates.

```python
def dash_ok(B, st, s):
    """A landing can reach a capture-adjacent tile this round (distance <= remaining speed, §3).
    Then it may commit through exposure: capture needs no more movement, and capturing the home ends the game (§9).
    Assumes capture is legal from any adjacent tile; enumerate_legal is authoritative if it differs."""
    field = objective_field(B, st, s)
    if field is None or s.pos not in field:
        return False
    remaining = SPEED[s.type] - st.moves_used.get(s.id, 0)
    return field[s.pos] <= remaining


def gate_open(B, st, s):
    """Gating applies only to Landing (never move from a safe tile into a tile a known hull can kill,
    since strength-0 units always die, §6). Gate opens when all-in or when a dash commits."""
    if s.type != "Landing":
        return False
    return st.allin or dash_ok(B, st, s)


def best_attack_value_from(B, s, dest):
    """Value of a fresh target this ship could hit from `dest` this turn (§6 range vs spotting)."""
    best = 0.0
    for t in B.enemy:
        if not t.fresh:
            continue
        if manhattan(dest, t.pos) <= eng(s.type, t.type):
            best = max(best, p_kill(s.type, t.type) * value(t, B))
    return best


def capture_bonus_from(B, dest):
    """Adjacent enemy port from `dest` (§3 Capture). Home capture is a win (§9)."""
    for p in B.enemy_ports:
        if manhattan(dest, p.pos) == 1:
            return WIN_BONUS if p.is_home else CAPTURE_BONUS
    return 0.0


def choose_move(B, legal, st, rng):
    best, best_score = None, 0.0
    for a in legal:
        if a.kind != "Move":
            continue
        s = mine_by_id(B, a.ship_id)
        if s is None:
            continue
        field = objective_field(B, st, s)
        if field is None or s.pos not in field or a.destination not in field:
            continue
        progress = field[s.pos] - field[a.destination]
        expo_now = exposure(B, s.pos, s.type)
        expo_dst = exposure(B, a.destination, s.type)
        if s.type in ("Landing", "Merchant", "Builder") and expo_dst > 0 and expo_now == 0 \
                and not gate_open(B, st, s):
            continue
        score = W_PROGRESS * progress
        if s.type in COMBAT and not s.has_attacked:
            score += W_ATTACK * best_attack_value_from(B, s, a.destination)
        if s.type == "Landing":
            score += capture_bonus_from(B, a.destination)
        score -= W_EXPOSE * expo_dst * COST[s.type] / 100.0 * (1.0 - 0.5 * st.doctrine.aggression)
        score += 1e-6 * rng.random()
        if score > best_score:
            best, best_score = a, score
    return best
```

Role summary:

- **LANDING.** Approach the target's capture-adjacent tiles (home if revealed, else the nearest known satellite, else explore). The gate keeps it out of threat until a dash can finish the capture this round. Any adjacent enemy port is captured at once by `choose_capture`.
- **SCOUT.** Go to a far-side cell not searched recently. Search means in range of an enemy port by the Port column (§7). Destroyers, cheapest and speed 5, are preferred.
- **MERCHANT.** Load at the claimed satellite, then haul home and unload. Flee threats by exposure penalty (strength-0, so exposure is lethal).
- **BUILDER.** Move to water next to the chosen site, then `choose_build_port` consumes the builder to found a port.
- **HUNT.** Move to a tile where the target is within `eng` (§6), then attack. Attacks are free, so the attack bonus pulls the hull into range.
- **ESCORT.** Stay within 2 tiles of a landing. Attack anything that comes into range.
- **PUSH.** Move to the enemy home's approach tiles, or to the anchor if the home is unknown.
- **GUARD.** Stay within 2 tiles of home. Attack intruders in range.

---

## 10. Bookkeeping and harvest hooks

```python
def record_for_harvest(view, legal, action, st, outcome=None):
    """Log per step: fogged observation (build_observation), legal list, chosen action, and per-turn doctrine and
    buy counts for the decisive-rate and buy-diversity reports (§1.1, §1.4). The observation must be recorded
    before the action is applied, for each seat that acts."""
    st.log.append(("step", st.doctrine.name, action.kind))
    # In the harness: keep build_observation(engine, me), legal, action, true-state seed/map/action list,
    # and the terminal outcome (+1 / -1 for the observing seat, or 0 on a draw, §1.2).
```

ε-random wrapper for diversity (§1.1): with probability ε, replace the agent's action with a uniform random legal action, excluding `EndTurn` until the agent has made at least one action this turn.

---

## 11. Tests, invariants, and known issues

**Constant and combat checks (run at import):**

- `p_kill("Destroyer","Submarine") == 0.80`
- `p_kill("Submarine","Destroyer") == 0.20`
- `p_kill("Submarine","Carrier") ≈ 0.692`
- `p_kill("Battleship","Carrier") ≈ 0.610`
- `eng("Carrier","Destroyer") == 3` (not 12)
- `eng("Submarine","Destroyer") == 2`
- `reach("Carrier","Landing") == 14`
- `turn_cap(16) == 100`

**Invariants:**

1. Every returned action is an element of `engine.enumerate_legal(me)`. Test by identity or equality.
2. `choose_attack` returns only targets with a fresh track and Manhattan `<= eng`.
3. No code reads the opponent's fleet, cash, stockpile, cargo, or hidden port flags. Grep for `engine.ships`, `engine.ports`, and `engine.players` in the agent module and forbid them.
4. Termination: the total step count per turn is bounded by `MAX_STEPS_PER_TURN`.

**Fog-invariance test (the key leak test):**

Take a game state. Make a copy in which only hidden information changes: enemy hulls outside every friendly spotting radius, enemy cash and stockpiles, enemy cargo, unsighted enemy ports. Keep the same RNG state. Run `next_action` on both copies. The two actions must match. Any mismatch is a leak.

**Harvest report (§1.4):**

- Decisive rate with a confidence interval, against the 76% check
- Seat split and matchup split
- Buy share per type. Target: roughly 8–15% each.
- Game-length distribution against the turn cap

**Known issues and open items:**

1. **Coastline leak in `observation.py`.** The coastline list excludes all ports, including unsighted enemy ports. So a hole in the list next to water can reveal a hidden port. This agent never infers from holes, and it builds sites only from present coastline tiles. Recommend that `observation.py` exclude only ports the observer owns or has sighted, or add a documented rule that the coastline list is static.
2. **Map prior.** `enemy_home_region` and `strategic_anchor` use the public generator property (§8: homes on opposite halves, Manhattan >= W/2). Disable with `use_map_prior=False` for custom maps.
3. **Stale-threat drift.** Drift is `SPEED * (age + 1)`, capped at 12. The cap and `STALE_HORIZON` are tunable. Larger values are safer but make landings more cautious.
4. **Unseen hulls.** Exposure only models known hulls. Doctrine aggression and the all-in phase (after `ALLIN_FRAC * cap`) are the decisiveness mechanisms. Without them, many games would hit the cap.
5. **One-step greedy movement.** There is no multi-turn lookahead and no model of the enemy's response beyond the exposure penalty. This is intentional. It is a teacher, and MCTS comes later (§1.3).
6. **Role plans are per turn.** A role may be stale mid-turn if a home is revealed. This is intentional, to avoid oscillation. Re-plan at the next turn.
7. **Attack/move independence and capture-without-speed** are assumptions (§0). Verify both against `enumerate_legal` before harvesting at scale.
