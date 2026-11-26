//! scripted agent. Decision order matches `battleboats.agents.scripted_agent`.
//!
//! Random draws go through [`CommanderRng`] so a caller can inject CPython's
//! `random.Random` and keep the same stream. Legal actions are normalized
//! (sorted) the same way `LegalIndex.from_actions` does, so enumerate order
//! does not change the choice.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::action::Action;
use crate::ship::ShipType;
use crate::state::{GameState, MERCHANT_CAPACITY, PORT_PRODUCTION};
use crate::tables::tables;

const BIG: i32 = 1_000_000_000;

pub trait CommanderRng {
    fn random(&mut self) -> f64;
    fn randrange(&mut self, n: usize) -> usize;
    fn uniform(&mut self, lo: f64, hi: f64) -> f64;
}

#[derive(Clone, Copy)]
struct Doctrine {
    name: &'static str,
    weights: [f64; 8],
    aggression: f64,
    builder_mode: &'static str,
    target_merchants: i32,
    buy_temperature: f64,
}

fn weight(d: &Doctrine, t: ShipType) -> f64 {
    d.weights[t.index()]
}

fn doctrines() -> &'static [Doctrine] {
    use std::sync::OnceLock;
    fn w(c: f64, b: f64, r: f64, d: f64, s: f64, m: f64, l: f64, u: f64) -> [f64; 8] {
        // Carrier, Battleship, Cruiser, Destroyer, Submarine, Merchant, Landing, Builder
        [c, b, r, d, s, m, l, u]
    }
    static D: OnceLock<Vec<Doctrine>> = OnceLock::new();
    D.get_or_init(|| {
        vec![
            Doctrine {
                name: "RAIDER_BLITZ",
                weights: w(0.8, 0.8, 1.4, 1.5, 1.0, 1.1, 1.3, 0.6),
                aggression: 1.15,
                builder_mode: "FORWARD",
                target_merchants: 2,
                buy_temperature: 0.45,
            },
            Doctrine {
                name: "ECONOMIC_BOOM",
                weights: w(1.3, 1.3, 0.9, 1.0, 0.9, 1.6, 1.1, 1.4),
                aggression: 0.90,
                builder_mode: "ECO_HOME",
                target_merchants: 3,
                buy_temperature: 0.45,
            },
            Doctrine {
                name: "WOLFPACK_STEALTH",
                weights: w(0.9, 0.7, 1.0, 1.4, 1.6, 1.2, 1.2, 0.8),
                aggression: 1.05,
                builder_mode: "FORWARD",
                target_merchants: 2,
                buy_temperature: 0.45,
            },
            Doctrine {
                name: "CAPITAL_ARMADA",
                weights: w(1.5, 1.5, 0.9, 1.2, 0.8, 1.3, 1.1, 0.8),
                aggression: 0.95,
                builder_mode: "ECO_HOME",
                target_merchants: 3,
                buy_temperature: 0.45,
            },
            Doctrine {
                name: "ISLAND_HOPPER",
                weights: w(0.9, 0.9, 1.3, 1.3, 0.9, 1.2, 1.4, 1.4),
                aggression: 1.05,
                builder_mode: "FORWARD",
                target_merchants: 2,
                buy_temperature: 0.50,
            },
            Doctrine {
                name: "BALANCED_ADAPTIVE",
                weights: w(1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0),
                aggression: 1.00,
                builder_mode: "ADAPTIVE",
                target_merchants: 2,
                buy_temperature: 0.50,
            },
        ]
    })
    .as_slice()
}

fn doctrine_by_name(name: &str) -> Doctrine {
    *doctrines()
        .iter()
        .find(|d| d.name == name)
        .unwrap_or(&doctrines()[5])
}

#[derive(Clone, Copy)]
struct OwnShip {
    id: i64,
    ship_type: ShipType,
    pos: (i32, i32),
    cargo: i64,
    has_attacked: bool,
    bought: bool,
    remaining: i32,
}

#[derive(Clone, Copy)]
struct OwnPort {
    pos: (i32, i32),
    is_home: bool,
    stockpile: i64,
    locked: bool,
}

#[derive(Clone, Copy)]
struct EnemyShip {
    id: i64,
    ship_type: ShipType,
    pos: (i32, i32),
    fresh: bool,
    turn_seen: i64,
}

#[derive(Clone, Copy)]
struct EnemyPort {
    pos: (i32, i32),
    is_home: bool,
    fresh: bool,
    turn_seen: i64,
}

struct View {
    width: i32,
    height: i32,
    turn: i64,
    cash: i64,
    kill_k: f64,
    ships: Vec<OwnShip>,
    ports: Vec<OwnPort>,
    home: OwnPort,
    enemy_ships: Vec<EnemyShip>,
    enemy_ports: Vec<EnemyPort>,
    moves: HashMap<i64, Vec<(i32, i32)>>,
    attacks: HashSet<(i64, i64)>,
    captures: Vec<(i64, (i32, i32))>,
    unloads: Vec<(i64, (i32, i32))>,
    loads: Vec<(i64, (i32, i32))>,
    port_builds: Vec<(i64, (i32, i32))>,
    builds: HashMap<(i32, i32), HashSet<ShipType>>,
}

struct Memory {
    water: HashSet<(i32, i32)>,
    coast: Vec<(i32, i32)>,
    doctrine: Doctrine,
    jitter: [f64; 8],
    planned_buy: Option<ShipType>,
    built_counts: [i32; 8],
    ports_built: i32,
    builder_assignments: Vec<(i64, (i32, i32))>,
    pruned: HashMap<i64, i64>,
    checked_coast: HashSet<(i32, i32)>,
    turn_key: Option<i64>,
    steps_in_turn: i32,
    visited: HashMap<i64, HashSet<(i32, i32)>>,
    sidestepped: HashSet<i64>,
}

pub struct ScriptedAgentConfig {
    pub max_steps_per_turn: i32,
    pub stale_horizon: i64,
    pub late_game_turn: i64,
    pub landing_ramp_turn: i64,
    pub checkmate_turn: i64,
    pub third_landing_turn: i64,
    pub use_map_prior: bool,
    pub doctrine: Option<String>,
}

impl Default for ScriptedAgentConfig {
    fn default() -> Self {
        Self {
            max_steps_per_turn: 250,
            stale_horizon: 8,
            late_game_turn: 55,
            landing_ramp_turn: 25,
            checkmate_turn: 20,
            third_landing_turn: 45,
            use_map_prior: true,
            doctrine: None,
        }
    }
}

pub struct ScriptedAgent {
    config: ScriptedAgentConfig,
    memory: HashMap<i64, Memory>,
}

impl ScriptedAgent {
    pub fn new(config: ScriptedAgentConfig) -> Self {
        Self {
            config,
            memory: HashMap::new(),
        }
    }

    pub fn doctrine_name(&self, pid: i64) -> Option<&str> {
        self.memory.get(&pid).map(|m| m.doctrine.name)
    }

    pub fn act(&mut self, state: &GameState, pid: i64, rng: &mut dyn CommanderRng) -> Action {
        let view = build_view(state, pid);
        if view.moves.is_empty()
            && view.attacks.is_empty()
            && view.captures.is_empty()
            && view.builds.is_empty()
            && view.port_builds.is_empty()
            && view.loads.is_empty()
            && view.unloads.is_empty()
        {
            // End turn is always legal when the seat may act; if the seat may
            // not, the caller should not have asked. Still return end-turn.
            let _ = &view;
        }
        if !self.memory.contains_key(&pid) {
            let mem = init_memory(state, &self.config, rng);
            self.memory.insert(pid, mem);
        }
        let mem = self.memory.get_mut(&pid).unwrap();
        decide(&view, mem, rng, &self.config)
    }
}

fn combat(t: ShipType) -> bool {
    let s = tables().stats(t);
    s.attack_range > 0 && s.strength > 0.0
}

fn manh(a: (i32, i32), b: (i32, i32)) -> i32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn neighbors(p: (i32, i32), w: i32, h: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity(4);
    if p.0 > 0 {
        out.push((p.0 - 1, p.1));
    }
    if p.0 + 1 < w {
        out.push((p.0 + 1, p.1));
    }
    if p.1 > 0 {
        out.push((p.0, p.1 - 1));
    }
    if p.1 + 1 < h {
        out.push((p.0, p.1 + 1));
    }
    out
}

fn eff_range(att: ShipType, dfn: ShipType) -> i32 {
    if !combat(att) {
        return 0;
    }
    let book = tables();
    let ar = book.stats(att).attack_range as i32;
    let spot = book.spot_distance(
        crate::ship::SpotKey::Ship(att),
        crate::ship::SpotKey::Ship(dfn),
    );
    ar.min(spot as i32)
}

fn kill_p(att: ShipType, dfn: ShipType, k: f64) -> f64 {
    if !combat(att) {
        return 0.0;
    }
    let book = tables();
    let sd = book.stats(dfn).strength;
    if sd <= 0.0 {
        return 1.0;
    }
    let x = book.stats(att).strength * book.attack_modifier(att, dfn) / sd;
    let xk = x.powf(k);
    xk / (1.0 + xk)
}

fn build_view(state: &GameState, pid: i64) -> View {
    let player = &state.players[pid as usize];
    let mut ships = Vec::new();
    let mut ids: Vec<i64> = player.owned_ship_ids.iter().copied().collect();
    ids.sort_unstable();
    for sid in ids {
        let s = &state.ships[&sid];
        let rem = if s.bought_this_turn {
            0
        } else {
            (s.stats.speed as i32 - s.tiles_moved_this_turn as i32).max(0)
        };
        ships.push(OwnShip {
            id: sid,
            ship_type: s.ship_type,
            pos: s.position,
            cargo: s.cargo,
            has_attacked: s.has_attacked,
            bought: s.bought_this_turn,
            remaining: rem,
        });
    }
    let mut port_pos: Vec<(i32, i32)> = player.owned_port_positions.iter().copied().collect();
    port_pos.sort_unstable();
    let mut ports = Vec::new();
    for pos in port_pos {
        let p = &state.ports[&pos];
        ports.push(OwnPort {
            pos,
            is_home: p.is_home,
            stockpile: p.stockpile,
            locked: p.locked_this_turn,
        });
    }
    let home = ports
        .iter()
        .copied()
        .find(|p| p.is_home)
        .unwrap_or(OwnPort {
            pos: player.home_port,
            is_home: true,
            stockpile: 0,
            locked: true,
        });
    let mut enemy_ships: Vec<EnemyShip> = player
        .sightings
        .values()
        .map(|s| EnemyShip {
            id: s.ship_id,
            ship_type: s.ship_type,
            pos: s.position,
            fresh: s.fresh,
            turn_seen: s.turn_seen,
        })
        .collect();
    enemy_ships.sort_by_key(|s| s.id);
    let own_ports = &player.owned_port_positions;
    let mut enemy_ports: Vec<EnemyPort> = player
        .port_sightings
        .iter()
        .filter(|(pos, _)| !own_ports.contains(*pos))
        .map(|(pos, s)| EnemyPort {
            pos: *pos,
            is_home: s.is_home,
            fresh: s.fresh,
            turn_seen: s.turn_seen,
        })
        .collect();
    enemy_ports.sort_by_key(|p| p.pos);

    let mut moves: HashMap<i64, Vec<(i32, i32)>> = HashMap::new();
    let mut attacks = HashSet::new();
    let mut captures = Vec::new();
    let mut unloads = Vec::new();
    let mut loads = Vec::new();
    let mut port_builds = Vec::new();
    let mut builds: HashMap<(i32, i32), HashSet<ShipType>> = HashMap::new();
    for action in state.enumerate_legal(pid) {
        match action {
            Action::Move { ship_id, x, y } => moves.entry(ship_id).or_default().push((x, y)),
            Action::Attack {
                attacker_id,
                target_id,
            } => {
                attacks.insert((attacker_id, target_id));
            }
            Action::CapturePort {
                landing_ship_id,
                x,
                y,
            } => captures.push((landing_ship_id, (x, y))),
            Action::MerchantUnload { merchant_id, x, y } => unloads.push((merchant_id, (x, y))),
            Action::MerchantLoad { merchant_id, x, y } => loads.push((merchant_id, (x, y))),
            Action::BuildPort {
                builder_ship_id,
                x,
                y,
            } => port_builds.push((builder_ship_id, (x, y))),
            Action::BuildShip {
                port_x,
                port_y,
                ship_type,
                ..
            } => {
                builds
                    .entry((port_x, port_y))
                    .or_default()
                    .insert(ship_type);
            }
            Action::EndTurn => {}
        }
    }
    for v in moves.values_mut() {
        v.sort_unstable();
    }
    captures.sort_unstable();
    unloads.sort_unstable();
    loads.sort_unstable();
    port_builds.sort_unstable();
    View {
        width: state.map.width,
        height: state.map.height,
        turn: state.turn,
        cash: player.cash,
        kill_k: state.kill_curve_k,
        ships,
        ports,
        home,
        enemy_ships,
        enemy_ports,
        moves,
        attacks,
        captures,
        unloads,
        loads,
        port_builds,
        builds,
    }
}

fn init_memory(state: &GameState, config: &ScriptedAgentConfig, rng: &mut dyn CommanderRng) -> Memory {
    let w = state.map.width;
    let h = state.map.height;
    let mut water = HashSet::new();
    let mut coast = Vec::new();
    for x in 0..w {
        for y in 0..h {
            let pos = (x, y);
            if state.map.is_land(x, y).unwrap_or(false) {
                if neighbors(pos, w, h)
                    .into_iter()
                    .any(|n| state.map.is_water(n.0, n.1).unwrap_or(false))
                {
                    coast.push(pos);
                }
            } else {
                water.insert(pos);
            }
        }
    }
    let doctrine = if let Some(name) = &config.doctrine {
        doctrine_by_name(name)
    } else {
        doctrines()[rng.randrange(doctrines().len())]
    };
    let mut jitter = [1.0; 8];
    for t in ShipType::ALL {
        jitter[t.index()] = rng.uniform(0.85, 1.15);
    }
    Memory {
        water,
        coast,
        doctrine,
        jitter,
        planned_buy: None,
        built_counts: [0; 8],
        ports_built: 0,
        builder_assignments: Vec::new(),
        pruned: HashMap::new(),
        checked_coast: HashSet::new(),
        turn_key: None,
        steps_in_turn: 0,
        visited: HashMap::new(),
        sidestepped: HashSet::new(),
    }
}

fn decide(view: &View, mem: &mut Memory, rng: &mut dyn CommanderRng, config: &ScriptedAgentConfig) -> Action {
    if mem.turn_key != Some(view.turn) {
        mem.turn_key = Some(view.turn);
        mem.steps_in_turn = 0;
        mem.visited.clear();
        mem.sidestepped.clear();
    }
    mem.steps_in_turn += 1;
    for s in &view.ships {
        mem.visited.entry(s.id).or_default().insert(s.pos);
    }
    if mem.steps_in_turn > config.max_steps_per_turn {
        return Action::EndTurn;
    }
    let mut d = Decision::new(view, mem, config);
    d.run(rng).unwrap_or(Action::EndTurn)
}

struct Decision<'a> {
    v: &'a View,
    m: &'a mut Memory,
    cfg: &'a ScriptedAgentConfig,
    k: f64,
    threat: HashMap<((i32, i32), u8), f64>,
    field_cache: HashMap<Vec<(i32, i32)>, Vec<i32>>,
    own_pos: HashSet<(i32, i32)>,
    own_port_set: HashSet<(i32, i32)>,
    fresh: Vec<EnemyShip>,
    stale: Vec<EnemyShip>,
    fresh_by_pos: HashMap<(i32, i32), EnemyShip>,
    occupied: HashSet<(i32, i32)>,
    enemy_home: Option<EnemyPort>,
    satellites: Vec<EnemyPort>,
    known_ports: HashSet<(i32, i32)>,
    prior: Vec<((i32, i32), f64)>,
    home_threat: bool,
    reserved_eh_adj: HashSet<(i32, i32)>,
}

impl<'a> Decision<'a> {
    fn new(v: &'a View, m: &'a mut Memory, cfg: &'a ScriptedAgentConfig) -> Self {
        let mut d = Self {
            k: v.kill_k,
            threat: HashMap::new(),
            field_cache: HashMap::new(),
            own_pos: v.ships.iter().map(|s| s.pos).collect(),
            own_port_set: v.ports.iter().map(|p| p.pos).collect(),
            fresh: Vec::new(),
            stale: Vec::new(),
            fresh_by_pos: HashMap::new(),
            occupied: HashSet::new(),
            enemy_home: None,
            satellites: Vec::new(),
            known_ports: HashSet::new(),
            prior: Vec::new(),
            home_threat: false,
            reserved_eh_adj: HashSet::new(),
            v,
            m,
            cfg,
        };
        d.build_world();
        d
    }

    fn build_world(&mut self) {
        let mut fresh = Vec::new();
        let mut stale = Vec::new();
        for e in &self.v.enemy_ships {
            if e.fresh {
                fresh.push(*e);
                continue;
            }
            if self.m.pruned.get(&e.id) == Some(&e.turn_seen) {
                continue;
            }
            if self.covered(e.pos, Some(e.ship_type)) {
                self.m.pruned.insert(e.id, e.turn_seen);
                continue;
            }
            if self.v.turn - e.turn_seen <= self.cfg.stale_horizon {
                stale.push(*e);
            }
        }
        self.fresh_by_pos = fresh.iter().map(|e| (e.pos, *e)).collect();
        self.occupied = self
            .own_pos
            .union(&self.fresh_by_pos.keys().copied().collect())
            .copied()
            .collect();
        // union of HashSet doesn't work that way. rebuild:
        self.occupied = self.own_pos.clone();
        for p in self.fresh_by_pos.keys() {
            self.occupied.insert(*p);
        }
        self.enemy_home = self.v.enemy_ports.iter().copied().find(|p| p.is_home);
        self.satellites = self
            .v
            .enemy_ports
            .iter()
            .copied()
            .filter(|p| !p.is_home)
            .collect();
        self.known_ports = self.own_port_set.clone();
        for p in &self.v.enemy_ports {
            self.known_ports.insert(p.pos);
        }
        self.prior.clear();
        if self.enemy_home.is_none() {
            let hx = self.v.home.pos.0;
            let coast = self.m.coast.clone();
            for c in coast {
                if self.m.checked_coast.contains(&c) || self.known_ports.contains(&c) {
                    continue;
                }
                if self.covered(c, None) {
                    self.m.checked_coast.insert(c);
                    continue;
                }
                let mut wgt = 1.0;
                if self.cfg.use_map_prior {
                    let same = (c.0 < self.v.width / 2) == (hx < self.v.width / 2);
                    wgt *= if same { 0.3 } else { 3.0 };
                    if manh(c, self.v.home.pos) < self.v.width / 2 {
                        wgt *= 0.5;
                    }
                    wgt *= 1.0 + 1.0 / (1.0 + (c.0 - (self.v.width - 1 - hx)).abs() as f64);
                }
                self.prior.push((c, wgt));
            }
        }
        self.home_threat = fresh
            .iter()
            .any(|e| e.ship_type == ShipType::Landing && manh(e.pos, self.v.home.pos) <= 5);
        self.reserved_eh_adj.clear();
        if let Some(eh) = self.enemy_home {
            if self
                .v
                .ships
                .iter()
                .any(|s| s.ship_type == ShipType::Landing && manh(s.pos, eh.pos) <= 6)
            {
                self.reserved_eh_adj = self.water_adj(eh.pos).into_iter().collect();
            }
        }
        self.fresh = fresh;
        self.stale = stale;
    }

    fn covered(&self, target: (i32, i32), kind: Option<ShipType>) -> bool {
        let book = tables();
        for s in &self.v.ships {
            let reach = match kind {
                Some(t) => book.spot_distance(
                    crate::ship::SpotKey::Ship(s.ship_type),
                    crate::ship::SpotKey::Ship(t),
                ),
                None => book.spot_distance(
                    crate::ship::SpotKey::Ship(s.ship_type),
                    crate::ship::SpotKey::Port,
                ),
            };
            if f64::from(manh(s.pos, target)) <= reach {
                return true;
            }
        }
        for p in &self.v.ports {
            let reach = match kind {
                Some(t) => {
                    book.spot_distance(crate::ship::SpotKey::Port, crate::ship::SpotKey::Ship(t))
                }
                None => book.spot_distance(crate::ship::SpotKey::Port, crate::ship::SpotKey::Port),
            };
            if f64::from(manh(p.pos, target)) <= reach {
                return true;
            }
        }
        false
    }

    fn water_adj(&self, tile: (i32, i32)) -> Vec<(i32, i32)> {
        neighbors(tile, self.v.width, self.v.height)
            .into_iter()
            .filter(|n| self.m.water.contains(n))
            .collect()
    }

    fn ring(&self, center: (i32, i32), rmax: i32, rmin: i32) -> HashSet<(i32, i32)> {
        let mut out = HashSet::new();
        let (cx, cy) = center;
        for x in (cx - rmax).max(0)..(cx + rmax + 1).min(self.v.width) {
            let rem = rmax - (x - cx).abs();
            for y in (cy - rem).max(0)..(cy + rem + 1).min(self.v.height) {
                let p = (x, y);
                if self.m.water.contains(&p) && manh(p, center) >= rmin {
                    out.insert(p);
                }
            }
        }
        out
    }

    fn threat_of(&mut self, tile: (i32, i32), hull: ShipType) -> f64 {
        let key = (tile, hull.index() as u8);
        if let Some(t) = self.threat.get(&key) {
            return *t;
        }
        let mut surv = 1.0;
        for e in &self.fresh {
            if !combat(e.ship_type) {
                continue;
            }
            let er = eff_range(e.ship_type, hull);
            if er <= 0 {
                continue;
            }
            let speed = tables().stats(e.ship_type).speed as i32;
            if manh(e.pos, tile) <= speed + er {
                surv *= 1.0 - kill_p(e.ship_type, hull, self.k);
            }
        }
        for e in &self.stale {
            if !combat(e.ship_type) {
                continue;
            }
            let age = (self.v.turn - e.turn_seen).max(1);
            if age > 3 || eff_range(e.ship_type, hull) <= 0 {
                continue;
            }
            let speed = tables().stats(e.ship_type).speed as i32;
            if manh(e.pos, tile) <= speed + 2 {
                surv *= 1.0 - 0.35f64.powi(age as i32) * kill_p(e.ship_type, hull, self.k);
            }
        }
        let t = 1.0 - surv;
        self.threat.insert(key, t);
        t
    }

    fn sub_danger(&self, tile: (i32, i32)) -> bool {
        for e in &self.fresh {
            if e.ship_type == ShipType::Destroyer {
                let speed = tables().stats(ShipType::Destroyer).speed as i32;
                if manh(e.pos, tile) <= speed + 3 {
                    return true;
                }
            }
            if e.ship_type == ShipType::Carrier && manh(e.pos, tile) <= 3 {
                return true;
            }
        }
        self.v.enemy_ports.iter().any(|p| manh(p.pos, tile) <= 1)
    }

    fn dist_field(&mut self, goals: &HashSet<(i32, i32)>) -> Vec<i32> {
        let mut key: Vec<(i32, i32)> = goals.iter().copied().collect();
        key.sort_unstable();
        if let Some(f) = self.field_cache.get(&key) {
            return f.clone();
        }
        let w = self.v.width;
        let h = self.v.height;
        let mut f = vec![BIG; (w * h) as usize];
        let mut q = VecDeque::new();
        for g in &key {
            let i = (g.0 * h + g.1) as usize;
            if self.m.water.contains(g) && f[i] == BIG {
                f[i] = 0;
                q.push_back(*g);
            }
        }
        while let Some(cur) = q.pop_front() {
            let d = f[(cur.0 * h + cur.1) as usize] + 1;
            for n in neighbors(cur, w, h) {
                let i = (n.0 * h + n.1) as usize;
                if f[i] == BIG && self.m.water.contains(&n) {
                    f[i] = d;
                    q.push_back(n);
                }
            }
        }
        self.field_cache.insert(key, f.clone());
        f
    }

    fn reach(&self, ship: &OwnShip) -> HashMap<(i32, i32), i32> {
        let mut out = HashMap::new();
        out.insert(ship.pos, 0);
        let mut q = VecDeque::new();
        q.push_back(ship.pos);
        while let Some(cur) = q.pop_front() {
            let d = out[&cur];
            if d >= ship.remaining {
                continue;
            }
            for n in neighbors(cur, self.v.width, self.v.height) {
                if out.contains_key(&n) || !self.m.water.contains(&n) || self.occupied.contains(&n)
                {
                    continue;
                }
                out.insert(n, d + 1);
                q.push_back(n);
            }
        }
        out
    }

    fn move_toward(
        &mut self,
        ship: &OwnShip,
        goals: &HashSet<(i32, i32)>,
        risk: f64,
        extra_blocked: &HashSet<(i32, i32)>,
        penalty: &HashMap<(i32, i32), f64>,
        strict: bool,
        rng: &mut dyn CommanderRng,
    ) -> Option<Action> {
        if goals.is_empty() || ship.remaining <= 0 {
            return None;
        }
        if strict && goals.contains(&ship.pos) {
            return None;
        }
        let visited = self.m.visited.get(&ship.id).cloned().unwrap_or_default();
        let dests: Vec<(i32, i32)> = self
            .v
            .moves
            .get(&ship.id)
            .map(|v| {
                v.iter()
                    .copied()
                    .filter(|d| !visited.contains(d) && !extra_blocked.contains(d))
                    .collect()
            })
            .unwrap_or_default();
        if dests.is_empty() {
            return None;
        }
        let field = self.dist_field(goals);
        let h = self.v.height;
        let at = |p: (i32, i32)| field[(p.0 * h + p.1) as usize];
        let cur_d = at(ship.pos);
        let cur_cost = cur_d as f64
            + risk * self.threat_of(ship.pos, ship.ship_type)
            + penalty.get(&ship.pos).copied().unwrap_or(0.0);
        let mut scored = Vec::new();
        for d in dests {
            let fd = at(d);
            if fd >= BIG {
                continue;
            }
            if strict && fd >= cur_d {
                continue;
            }
            let cost = fd as f64
                + risk * self.threat_of(d, ship.ship_type)
                + penalty.get(&d).copied().unwrap_or(0.0);
            if strict && cur_d < BIG && cost >= cur_cost {
                continue;
            }
            scored.push((cost, d));
        }
        if scored.is_empty() {
            return None;
        }
        let best = scored.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
        let mut ties: Vec<(i32, i32)> = scored
            .into_iter()
            .filter(|(c, _)| *c <= best + 0.15)
            .map(|(_, d)| d)
            .collect();
        ties.sort_unstable();
        let dest = if ties.len() > 1 {
            ties[rng.randrange(ties.len())]
        } else {
            ties[0]
        };
        Some(Action::Move {
            ship_id: ship.id,
            x: dest.0,
            y: dest.1,
        })
    }

    fn run(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        if let Some(a) = self.capture() {
            return Some(a);
        }
        if let Some(a) = self.merchant_transfer() {
            return Some(a);
        }
        if let Some(a) = self.cash_delivery(rng) {
            return Some(a);
        }
        if let Some(a) = self.combat(rng) {
            return Some(a);
        }
        if let Some(a) = self.post_attack(rng) {
            return Some(a);
        }
        if let Some(a) = self.unclog(rng) {
            return Some(a);
        }
        if let Some(a) = self.builder(rng) {
            return Some(a);
        }
        if let Some(a) = self.merchant_move(rng) {
            return Some(a);
        }
        if let Some(a) = self.landing_move(rng) {
            return Some(a);
        }
        if let Some(a) = self.fleet_move(rng) {
            return Some(a);
        }
        self.purchase(rng)
    }

    fn capture(&self) -> Option<Action> {
        if self.v.captures.is_empty() {
            return None;
        }
        if let Some(eh) = self.enemy_home {
            if let Some((sid, tgt)) = self.v.captures.iter().find(|(_, t)| *t == eh.pos) {
                return Some(Action::CapturePort {
                    landing_ship_id: *sid,
                    x: tgt.0,
                    y: tgt.1,
                });
            }
        }
        let (sid, tgt) = self.v.captures[0];
        Some(Action::CapturePort {
            landing_ship_id: sid,
            x: tgt.0,
            y: tgt.1,
        })
    }

    fn merchant_transfer(&self) -> Option<Action> {
        if let Some((mid, port)) = self.v.unloads.first() {
            return Some(Action::MerchantUnload {
                merchant_id: *mid,
                x: port.0,
                y: port.1,
            });
        }
        if !self.v.loads.is_empty() {
            let stock: HashMap<(i32, i32), i64> =
                self.v.ports.iter().map(|p| (p.pos, p.stockpile)).collect();
            let mut loads = self.v.loads.clone();
            loads.sort_by(|a, b| {
                let sa = stock.get(&a.1).copied().unwrap_or(0);
                let sb = stock.get(&b.1).copied().unwrap_or(0);
                sb.cmp(&sa).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1))
            });
            let (mid, port) = loads[0];
            return Some(Action::MerchantLoad {
                merchant_id: mid,
                x: port.0,
                y: port.1,
            });
        }
        None
    }

    fn cash_delivery(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let home_adj: HashSet<(i32, i32)> = self.water_adj(self.v.home.pos).into_iter().collect();
        for s in &self.v.ships.clone() {
            if s.ship_type != ShipType::Merchant
                || s.cargo <= 0
                || s.remaining <= 0
                || home_adj.contains(&s.pos)
            {
                continue;
            }
            let r = self.reach(s);
            if home_adj.iter().any(|g| r.contains_key(g)) {
                if let Some(mv) = self.move_toward(
                    s,
                    &home_adj,
                    8.0,
                    &HashSet::new(),
                    &HashMap::new(),
                    true,
                    rng,
                ) {
                    return Some(mv);
                }
            }
        }
        None
    }

    fn target_value(&self, e: &EnemyShip) -> f64 {
        let mut val = tables().stats(e.ship_type).cost as f64;
        if e.ship_type == ShipType::Landing {
            val = 550.0;
            let dh = manh(e.pos, self.v.home.pos);
            if dh <= 9 {
                val = 2200.0 + 250.0 * f64::from(10 - dh);
            }
        } else if e.ship_type == ShipType::Builder {
            val = 420.0;
        } else if e.ship_type == ShipType::Merchant {
            val = 280.0;
        } else if e.ship_type == ShipType::Submarine {
            val += 130.0;
        }
        if let Some(eh) = self.enemy_home {
            if manh(e.pos, eh.pos) <= 5 {
                val *= 1.35;
            }
        }
        if self
            .v
            .ships
            .iter()
            .any(|s| s.ship_type == ShipType::Landing && manh(e.pos, s.pos) <= 6)
        {
            val *= 1.45;
        }
        val
    }

    fn combat(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        if self.fresh.is_empty() {
            return None;
        }
        let aggr = self.m.doctrine.aggression;
        let mut cands = Vec::new();
        let ships = self.v.ships.clone();
        let fresh = self.fresh.clone();
        for s in &ships {
            if !combat(s.ship_type) || s.has_attacked || s.bought {
                continue;
            }
            let r = self.reach(s);
            let my_cost = tables().stats(s.ship_type).cost as f64;
            let my_str = tables().stats(s.ship_type).strength;
            for e in &fresh {
                let er = eff_range(s.ship_type, e.ship_type);
                if er <= 0 {
                    continue;
                }
                let pk = kill_p(s.ship_type, e.ship_type, self.k);
                let gain = pk * self.target_value(e);
                let overkill = if tables().stats(e.ship_type).strength <= 0.0 {
                    0.30 * my_str
                } else {
                    0.0
                };
                let mut best = f64::NEG_INFINITY;
                let mut best_tile: Option<((i32, i32), i32)> = None;
                let mut tiles: Vec<(i32, i32)> = r.keys().copied().collect();
                tiles.sort_unstable();
                for tile in tiles {
                    let steps = r[&tile];
                    let dist = manh(tile, e.pos);
                    if dist > er {
                        continue;
                    }
                    if steps == 0 && !self.v.attacks.contains(&(s.id, e.id)) {
                        continue;
                    }
                    let adj_risk = self.threat_of(tile, s.ship_type) * (1.0 - 0.65 * pk);
                    let kite = if s.remaining - steps >= 2 { 0.65 } else { 1.0 };
                    let mut exposure = my_cost * adj_risk * kite;
                    if s.ship_type == ShipType::Submarine && self.sub_danger(tile) {
                        exposure += 75.0;
                    }
                    let net = gain * aggr - exposure - overkill + 2.5 * f64::from(dist)
                        - 2.0 * f64::from(steps);
                    if net > best {
                        best = net;
                        best_tile = Some((tile, steps));
                    }
                }
                let emergency =
                    e.ship_type == ShipType::Landing && manh(e.pos, self.v.home.pos) <= 8;
                if let Some((tile, steps)) = best_tile {
                    if best > -15.0 || emergency {
                        cands.push((best, steps, s.id, e.id, tile, *s));
                    }
                }
            }
        }
        if cands.is_empty() {
            return None;
        }
        let mut immediate: Vec<_> = cands.iter().copied().filter(|c| c.1 == 0).collect();
        if !immediate.is_empty() {
            immediate.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.2.cmp(&b.2))
                    .then(a.4.cmp(&b.4))
            });
            let c = immediate[0];
            return Some(Action::Attack {
                attacker_id: c.2,
                target_id: c.3,
            });
        }
        cands.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(a.2.cmp(&b.2))
                .then(a.4.cmp(&b.4))
        });
        for c in &cands {
            let mut goals = HashSet::new();
            goals.insert(c.4);
            if let Some(mv) = self.move_toward(
                &c.5,
                &goals,
                1.5,
                &HashSet::new(),
                &HashMap::new(),
                true,
                rng,
            ) {
                return Some(mv);
            }
        }
        None
    }

    fn post_attack(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let home_adj: HashSet<(i32, i32)> = self.water_adj(self.v.home.pos).into_iter().collect();
        if self.home_threat {
            let open_adj: HashSet<(i32, i32)> =
                home_adj.difference(&self.occupied).copied().collect();
            if !open_adj.is_empty() {
                let mut blockers: Vec<OwnShip> = self
                    .v
                    .ships
                    .iter()
                    .copied()
                    .filter(|s| {
                        s.remaining > 0
                            && !home_adj.contains(&s.pos)
                            && (s.has_attacked
                                || combat(s.ship_type)
                                || (s.ship_type == ShipType::Merchant && s.cargo == 0))
                    })
                    .collect();
                blockers.sort_by_key(|s| (manh(s.pos, self.v.home.pos), s.id));
                for s in &blockers {
                    if let Some(mv) = self.move_toward(
                        s,
                        &open_adj,
                        0.5,
                        &HashSet::new(),
                        &HashMap::new(),
                        true,
                        rng,
                    ) {
                        return Some(mv);
                    }
                }
            }
        }
        if !self.reserved_eh_adj.is_empty() {
            let eh = self.enemy_home.unwrap().pos;
            let r23 = self.ring(eh, 3, 2);
            for s in &self.v.ships.clone() {
                if s.ship_type != ShipType::Landing
                    && self.reserved_eh_adj.contains(&s.pos)
                    && s.remaining > 0
                {
                    if let Some(mv) =
                        self.move_toward(s, &r23, 2.0, &HashSet::new(), &HashMap::new(), true, rng)
                    {
                        return Some(mv);
                    }
                }
            }
        }
        let hold: HashSet<(i32, i32)> = if self.home_threat {
            home_adj.clone()
        } else {
            HashSet::new()
        };
        for s in &self.v.ships.clone() {
            if !combat(s.ship_type) || !s.has_attacked || s.remaining <= 0 || hold.contains(&s.pos)
            {
                continue;
            }
            let sub = s.ship_type == ShipType::Submarine;
            let cur = self.threat_of(s.pos, s.ship_type)
                + if sub && self.sub_danger(s.pos) {
                    0.4
                } else {
                    0.0
                };
            if cur <= 0.05 {
                continue;
            }
            let visited = self.m.visited.get(&s.id).cloned().unwrap_or_default();
            let mut best = -cur;
            let mut best_d = None;
            let moves = self.v.moves.get(&s.id).cloned().unwrap_or_default();
            for d in moves {
                if visited.contains(&d) || self.reserved_eh_adj.contains(&d) {
                    continue;
                }
                let nt = self.threat_of(d, s.ship_type)
                    + if sub && self.sub_danger(d) { 0.4 } else { 0.0 };
                if -nt > best + 0.02 {
                    best = -nt;
                    best_d = Some(d);
                }
            }
            if let Some(d) = best_d {
                return Some(Action::Move {
                    ship_id: s.id,
                    x: d.0,
                    y: d.1,
                });
            }
        }
        None
    }

    fn unclog(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        if self.v.cash < 100 || self.home_threat {
            return None;
        }
        let ship_at: HashMap<(i32, i32), OwnShip> =
            self.v.ships.iter().map(|s| (s.pos, *s)).collect();
        for p in &self.v.ports.clone() {
            if p.locked {
                continue;
            }
            let adj = self.water_adj(p.pos);
            if adj.iter().any(|t| !self.occupied.contains(t)) {
                continue;
            }
            for t in adj {
                let Some(b) = ship_at.get(&t).copied() else {
                    continue;
                };
                if b.remaining <= 0 || self.m.sidestepped.contains(&b.id) {
                    continue;
                }
                let adj_set: HashSet<(i32, i32)> = self.water_adj(p.pos).into_iter().collect();
                let outs: Vec<(i32, i32)> = self
                    .v
                    .moves
                    .get(&b.id)
                    .map(|ms| {
                        ms.iter()
                            .copied()
                            .filter(|d| {
                                !adj_set.contains(d) && self.threat_of(*d, b.ship_type) <= 0.35
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if !outs.is_empty() {
                    self.m.sidestepped.insert(b.id);
                    let d = outs[rng.randrange(outs.len())];
                    return Some(Action::Move {
                        ship_id: b.id,
                        x: d.0,
                        y: d.1,
                    });
                }
            }
        }
        None
    }

    fn coast_score(&mut self, c: (i32, i32), mode: &str) -> f64 {
        let adj = self.water_adj(c);
        if adj.is_empty() || self.known_ports.contains(&c) {
            return f64::NEG_INFINITY;
        }
        let nearest = self
            .known_ports
            .iter()
            .map(|pp| manh(c, *pp))
            .min()
            .unwrap_or(99);
        if nearest <= 2 {
            return f64::NEG_INFINITY;
        }
        let local = adj
            .iter()
            .map(|w| self.threat_of(*w, ShipType::Builder))
            .fold(f64::NEG_INFINITY, f64::max);
        if local > 0.25 {
            return f64::NEG_INFINITY;
        }
        let openness = 12.0 * (adj.len().min(3) as f64);
        let dh = manh(c, self.v.home.pos);
        let ds = if mode == "ECO_HOME" || (mode == "ADAPTIVE" && self.v.ports.len() <= 3) {
            45.0 - 5.0 * (f64::from(dh) - 4.5).abs()
        } else {
            let mut ds = 40.0 - 2.2 * (f64::from(c.0) - self.v.width as f64 / 2.0).abs();
            if let Some(eh) = self.enemy_home {
                ds += 20.0 - 2.0 * (f64::from(manh(c, eh.pos)) - 11.0).abs();
            }
            ds
        };
        openness + ds - 40.0 * local
    }

    fn builder(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let builders: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| s.ship_type == ShipType::Builder)
            .collect();
        if builders.is_empty() {
            return None;
        }
        let mode = self.m.doctrine.builder_mode;
        if !self.v.port_builds.is_empty() {
            let mut scored: Vec<(f64, i64, (i32, i32))> = self
                .v
                .port_builds
                .iter()
                .map(|(bid, t)| (self.coast_score(*t, mode), *bid, *t))
                .collect();
            scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.cmp(&b.1))
                    .then(a.2.cmp(&b.2))
            });
            let (sc, bid, tgt) = scored[0];
            let assigned = self
                .m
                .builder_assignments
                .iter()
                .find(|(id, _)| *id == bid)
                .map(|(_, p)| *p);
            if sc > 5.0 || assigned == Some(tgt) {
                self.m.ports_built += 1;
                self.m.builder_assignments.retain(|(id, _)| *id != bid);
                return Some(Action::BuildPort {
                    builder_ship_id: bid,
                    x: tgt.0,
                    y: tgt.1,
                });
            }
        }
        let live: HashSet<i64> = builders.iter().map(|b| b.id).collect();
        self.m
            .builder_assignments
            .retain(|(id, _)| live.contains(id));
        let mut claimed: HashSet<(i32, i32)> =
            self.m.builder_assignments.iter().map(|(_, p)| *p).collect();
        for b in &builders {
            if b.remaining <= 0 {
                continue;
            }
            let mut tgt = self
                .m
                .builder_assignments
                .iter()
                .find(|(id, _)| *id == b.id)
                .map(|(_, p)| *p);
            if tgt.is_none() || self.coast_score(tgt.unwrap(), mode) <= -100.0 {
                if let Some(old) = tgt {
                    claimed.remove(&old);
                }
                let mut best = f64::NEG_INFINITY;
                let mut best_c = None;
                for c in self.m.coast.clone() {
                    if claimed.contains(&c) {
                        continue;
                    }
                    let mut sc = self.coast_score(c, mode);
                    if sc == f64::NEG_INFINITY {
                        continue;
                    }
                    sc -= 1.2 * f64::from(manh(b.pos, c));
                    if sc > best {
                        best = sc;
                        best_c = Some(c);
                    }
                }
                if best_c.is_none() {
                    self.m.builder_assignments.retain(|(id, _)| *id != b.id);
                    continue;
                }
                let best_c = best_c.unwrap();
                if let Some(slot) = self
                    .m
                    .builder_assignments
                    .iter_mut()
                    .find(|(id, _)| *id == b.id)
                {
                    slot.1 = best_c;
                } else {
                    self.m.builder_assignments.push((b.id, best_c));
                }
                claimed.insert(best_c);
                tgt = Some(best_c);
            }
            let goals: HashSet<(i32, i32)> = self.water_adj(tgt.unwrap()).into_iter().collect();
            if let Some(mv) =
                self.move_toward(b, &goals, 9.0, &HashSet::new(), &HashMap::new(), true, rng)
            {
                return Some(mv);
            }
        }
        None
    }

    fn any_viable_coast(&mut self) -> bool {
        let mode = self.m.doctrine.builder_mode;
        let coast = self.m.coast.clone();
        coast
            .into_iter()
            .any(|c| self.coast_score(c, mode) > f64::NEG_INFINITY)
    }

    fn merchant_move(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let mut merchants: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| s.ship_type == ShipType::Merchant && s.remaining > 0)
            .collect();
        if merchants.is_empty() {
            return None;
        }
        let home_adj: HashSet<(i32, i32)> = self.water_adj(self.v.home.pos).into_iter().collect();
        let f_home = self.dist_field(&home_adj);
        let h = self.v.height;
        let sats: Vec<OwnPort> = self
            .v
            .ports
            .iter()
            .copied()
            .filter(|p| !p.is_home)
            .collect();
        let mut stock: HashMap<(i32, i32), i64> =
            sats.iter().map(|p| (p.pos, p.stockpile)).collect();
        merchants.sort_by(|a, b| b.cargo.cmp(&a.cargo).then(a.id.cmp(&b.id)));
        for mch in &merchants {
            if self.threat_of(mch.pos, ShipType::Merchant) > 0.35 && !home_adj.contains(&mch.pos) {
                if let Some(mv) = self.move_toward(
                    mch,
                    &home_adj,
                    12.0,
                    &HashSet::new(),
                    &HashMap::new(),
                    true,
                    rng,
                ) {
                    return Some(mv);
                }
            }
            let mut best_rate = f64::NEG_INFINITY;
            let mut best_goals: Option<HashSet<(i32, i32)>> = None;
            let mut best_sat: Option<((i32, i32), i64)> = None;
            if mch.cargo > 0 {
                let mut dh = f_home[(mch.pos.0 * h + mch.pos.1) as usize];
                if dh >= BIG {
                    dh = 30;
                }
                let turns = (dh as f64 / 3.0).ceil().max(0.5);
                let urgency = if self.v.cash < 250 && self.v.cash + 2 * mch.cargo >= 250 {
                    1.35
                } else {
                    1.0
                };
                best_rate = urgency * 2.0 * mch.cargo as f64 / turns;
                best_goals = Some(home_adj.clone());
            }
            if mch.cargo < MERCHANT_CAPACITY {
                for sat in &sats {
                    let adj = self.water_adj(sat.pos);
                    if adj.is_empty() {
                        continue;
                    }
                    if adj
                        .iter()
                        .map(|w| self.threat_of(*w, ShipType::Merchant))
                        .fold(f64::INFINITY, f64::min)
                        > 0.45
                    {
                        continue;
                    }
                    let t_to =
                        (adj.iter().map(|w| manh(mch.pos, *w)).min().unwrap() as f64 / 3.0).ceil();
                    let mut dsh = adj
                        .iter()
                        .map(|w| f_home[(w.0 * h + w.1) as usize])
                        .min()
                        .unwrap();
                    if dsh >= BIG {
                        dsh = 30;
                    }
                    let t_back = (dsh as f64 / 3.0).ceil().max(1.0);
                    let proj = stock[&sat.pos] + PORT_PRODUCTION * t_to as i64;
                    let gain = (MERCHANT_CAPACITY - mch.cargo).min(proj);
                    if gain <= 0 {
                        continue;
                    }
                    let rate = 2.0 * (mch.cargo + gain) as f64 / (t_to + t_back).max(1.0);
                    if rate > best_rate {
                        best_rate = rate;
                        best_goals = Some(adj.into_iter().collect());
                        best_sat = Some((
                            sat.pos,
                            (MERCHANT_CAPACITY - mch.cargo).min(stock[&sat.pos]),
                        ));
                    }
                }
            }
            if let Some((pos, take)) = best_sat {
                let left = stock.get(&pos).copied().unwrap_or(0) - take;
                stock.insert(pos, left.max(0));
            }
            if let Some(goals) = &best_goals {
                if !goals.contains(&mch.pos) {
                    if let Some(mv) = self.move_toward(
                        mch,
                        goals,
                        9.0,
                        &HashSet::new(),
                        &HashMap::new(),
                        true,
                        rng,
                    ) {
                        return Some(mv);
                    }
                }
            }
        }
        None
    }

    fn landing_move(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let mut landings: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| s.ship_type == ShipType::Landing && s.remaining > 0)
            .collect();
        if landings.is_empty() {
            return None;
        }
        let combat_ships: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| combat(s.ship_type))
            .collect();
        let late = self.v.turn >= self.cfg.late_game_turn;
        if let Some(ehp) = self.enemy_home {
            let eh = ehp.pos;
            let eh_adj: HashSet<(i32, i32)> = self.water_adj(eh).into_iter().collect();
            landings.sort_by_key(|s| (manh(s.pos, eh), s.id));
            for (idx, lnd) in landings.iter().enumerate() {
                let d_eh = manh(lnd.pos, eh);
                let r = self.reach(lnd);
                if eh_adj.iter().any(|g| r.contains_key(g)) {
                    if let Some(mv) = self.move_toward(
                        lnd,
                        &eh_adj,
                        0.0,
                        &HashSet::new(),
                        &HashMap::new(),
                        true,
                        rng,
                    ) {
                        return Some(mv);
                    }
                }
                let mut goals = eh_adj.clone();
                if !self.satellites.is_empty() {
                    if idx >= 1 {
                        let near = *self
                            .satellites
                            .iter()
                            .min_by_key(|p| (manh(lnd.pos, p.pos), p.pos))
                            .unwrap();
                        if manh(lnd.pos, near.pos) < d_eh && !self.water_adj(near.pos).is_empty() {
                            goals = self.water_adj(near.pos).into_iter().collect();
                        }
                    } else {
                        let mut sats = self.satellites.clone();
                        sats.sort_by_key(|p| p.pos);
                        for sat in sats {
                            if manh(lnd.pos, sat.pos) + manh(sat.pos, eh) <= d_eh + 2 {
                                let adj = self.water_adj(sat.pos);
                                if !adj.is_empty() {
                                    goals = adj.into_iter().collect();
                                    break;
                                }
                            }
                        }
                    }
                }
                let escort_close = combat_ships.iter().any(|c| manh(c.pos, lnd.pos) <= 4);
                let escort_catching = combat_ships.iter().any(|c| {
                    let d = manh(c.pos, lnd.pos);
                    d > 4 && d <= 8
                });
                let defenders = self.fresh.iter().any(|e| manh(e.pos, eh) <= 6);
                let known_clear = self.enemy_home.unwrap().fresh && !defenders && d_eh <= 6;
                if (7..=10).contains(&d_eh)
                    && !escort_close
                    && escort_catching
                    && !known_clear
                    && !late
                    && self.threat_of(lnd.pos, ShipType::Landing) <= 0.10
                {
                    continue;
                }
                let risk = if escort_close || late { 2.5 } else { 7.0 };
                if let Some(mv) = self.move_toward(
                    lnd,
                    &goals,
                    risk,
                    &HashSet::new(),
                    &HashMap::new(),
                    true,
                    rng,
                ) {
                    return Some(mv);
                }
            }
        } else {
            for lnd in &landings {
                let escort_close = combat_ships.iter().any(|c| manh(c.pos, lnd.pos) <= 4);
                let goals = if !self.satellites.is_empty() && escort_close {
                    let sat = *self
                        .satellites
                        .iter()
                        .min_by_key(|p| (manh(lnd.pos, p.pos), p.pos))
                        .unwrap();
                    self.water_adj(sat.pos).into_iter().collect()
                } else {
                    let mid = self.v.width / 2;
                    let goals: HashSet<(i32, i32)> = self
                        .m
                        .water
                        .iter()
                        .copied()
                        .filter(|t| (t.0 - mid).abs() <= 3)
                        .collect();
                    let safe: HashSet<(i32, i32)> = goals
                        .iter()
                        .copied()
                        .filter(|t| self.threat_of(*t, ShipType::Landing) <= 0.10)
                        .collect();
                    if safe.is_empty() {
                        goals
                    } else {
                        safe
                    }
                };
                if let Some(mv) = self.move_toward(
                    lnd,
                    &goals,
                    8.0,
                    &HashSet::new(),
                    &HashMap::new(),
                    true,
                    rng,
                ) {
                    return Some(mv);
                }
            }
        }
        None
    }

    fn scout_goals(&mut self, s: &OwnShip) -> HashSet<(i32, i32)> {
        if self.enemy_home.is_none() && !self.prior.is_empty() {
            let rp = tables().spot_distance(
                crate::ship::SpotKey::Ship(s.ship_type),
                crate::ship::SpotKey::Port,
            ) as i32;
            let mut prior = self.prior.clone();
            prior.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
            let mut scored = Vec::new();
            let water: Vec<(i32, i32)> = self.m.water.iter().copied().collect();
            for wt in water {
                if (wt.0 + wt.1) % 2 != 0 {
                    continue;
                }
                let mut gain = 0.0;
                for (c, pw) in &prior {
                    if manh(wt, *c) <= rp {
                        gain += pw;
                    }
                }
                if gain <= 0.0 || self.threat_of(wt, s.ship_type) > 0.45 {
                    continue;
                }
                scored.push((gain - 0.8 * f64::from(manh(s.pos, wt)), wt));
            }
            if !scored.is_empty() {
                let best = scored.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
                return scored
                    .into_iter()
                    .filter(|(val, _)| *val >= best - 0.2 * best.abs())
                    .map(|(_, p)| p)
                    .collect();
            }
        }
        if self.enemy_home.is_none() {
            let (hx, hy) = self.v.home.pos;
            return self.ring((self.v.width - 1 - hx, hy), 6, 0);
        }
        let eh = self.enemy_home.unwrap();
        if !eh.fresh || self.v.turn - eh.turn_seen >= 3 {
            return self.ring(eh.pos, 7, 4);
        }
        if !self.satellites.is_empty() {
            let sat = *self
                .satellites
                .iter()
                .max_by_key(|p| (self.v.turn - p.turn_seen, p.pos))
                .unwrap();
            return self.ring(sat.pos, 5, 0);
        }
        HashSet::new()
    }

    fn forward_port(&self) -> OwnPort {
        let reference = if let Some(eh) = self.enemy_home {
            eh.pos
        } else {
            (self.v.width - 1 - self.v.home.pos.0, self.v.home.pos.1)
        };
        *self
            .v
            .ports
            .iter()
            .min_by_key(|p| (manh(p.pos, reference), p.pos))
            .unwrap()
    }

    fn fleet_move(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        let movable: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| combat(s.ship_type) && s.remaining > 0 && !s.has_attacked)
            .collect();
        if movable.is_empty() {
            return None;
        }
        let reserved = self.reserved_eh_adj.clone();
        let mut assigned = HashSet::new();
        let mut threats: Vec<EnemyShip> = self
            .fresh
            .iter()
            .copied()
            .filter(|e| {
                manh(e.pos, self.v.home.pos) <= 10
                    || self
                        .v
                        .ports
                        .iter()
                        .any(|p| !p.is_home && manh(e.pos, p.pos) <= 5)
            })
            .collect();
        if !threats.is_empty() {
            threats.sort_by_key(|e| {
                (
                    if e.ship_type == ShipType::Landing {
                        0
                    } else {
                        1
                    },
                    manh(e.pos, self.v.home.pos),
                    e.pos,
                )
            });
            let te = threats[0];
            let mut ics = movable.clone();
            ics.sort_by(|a, b| {
                let ka = kill_p(a.ship_type, te.ship_type, self.k);
                let kb = kill_p(b.ship_type, te.ship_type, self.k);
                kb.partial_cmp(&ka)
                    .unwrap()
                    .then(manh(a.pos, te.pos).cmp(&manh(b.pos, te.pos)))
                    .then(a.id.cmp(&b.id))
            });
            for ic in ics.into_iter().take(2) {
                let er = eff_range(ic.ship_type, te.ship_type).max(1);
                let goals = self.ring(te.pos, er, 0);
                if let Some(mv) = self.move_toward(
                    &ic,
                    &goals,
                    1.0,
                    &HashSet::new(),
                    &HashMap::new(),
                    true,
                    rng,
                ) {
                    return Some(mv);
                }
                assigned.insert(ic.id);
            }
        }
        let need_scout = self.enemy_home.is_none()
            || self
                .enemy_home
                .map(|e| !e.fresh && self.v.turn - e.turn_seen >= 4)
                .unwrap_or(false);
        if need_scout {
            let mut pool: Vec<OwnShip> = movable
                .iter()
                .copied()
                .filter(|s| !assigned.contains(&s.id))
                .collect();
            if !pool.is_empty() {
                pool.sort_by(|a, b| {
                    let sa = tables().spot_distance(
                        crate::ship::SpotKey::Ship(a.ship_type),
                        crate::ship::SpotKey::Port,
                    );
                    let sb = tables().spot_distance(
                        crate::ship::SpotKey::Ship(b.ship_type),
                        crate::ship::SpotKey::Port,
                    );
                    sb.partial_cmp(&sa)
                        .unwrap()
                        .then(
                            tables()
                                .stats(b.ship_type)
                                .speed
                                .cmp(&tables().stats(a.ship_type).speed),
                        )
                        .then(a.id.cmp(&b.id))
                });
                let sc = pool[0];
                let mut goals = self.scout_goals(&sc);
                for r in &reserved {
                    goals.remove(r);
                }
                if !goals.is_empty() {
                    if let Some(mv) =
                        self.move_toward(&sc, &goals, 3.0, &reserved, &HashMap::new(), true, rng)
                    {
                        return Some(mv);
                    }
                }
                assigned.insert(sc.id);
            }
        }
        let own_power: f64 = self
            .v
            .ships
            .iter()
            .filter(|s| combat(s.ship_type))
            .map(|s| tables().stats(s.ship_type).strength)
            .sum();
        let seen_power: f64 = self
            .fresh
            .iter()
            .filter(|e| combat(e.ship_type))
            .map(|e| tables().stats(e.ship_type).strength)
            .sum();
        let advantage = own_power >= 0.9 * seen_power / self.m.doctrine.aggression
            || self.v.turn >= self.cfg.late_game_turn;
        let landings: Vec<OwnShip> = self
            .v
            .ships
            .iter()
            .copied()
            .filter(|s| s.ship_type == ShipType::Landing)
            .collect();
        let mut escorts = 0i32;
        for s in &movable {
            if assigned.contains(&s.id) {
                continue;
            }
            if s.ship_type == ShipType::Submarine {
                let mut prey: Vec<EnemyShip> = self
                    .fresh
                    .iter()
                    .chain(self.stale.iter())
                    .copied()
                    .filter(|e| {
                        matches!(
                            e.ship_type,
                            ShipType::Battleship
                                | ShipType::Carrier
                                | ShipType::Cruiser
                                | ShipType::Merchant
                                | ShipType::Landing
                        )
                    })
                    .collect();
                if !prey.is_empty() {
                    prey.sort_by_key(|e| (manh(s.pos, e.pos), e.pos));
                    let pr = prey[0];
                    let er = eff_range(ShipType::Submarine, pr.ship_type).max(1);
                    let rmin = if er >= 2
                        && matches!(
                            pr.ship_type,
                            ShipType::Battleship | ShipType::Cruiser | ShipType::Carrier
                        ) {
                        2
                    } else {
                        1
                    };
                    let mut goals = self.ring(pr.pos, er, rmin);
                    for r in &reserved {
                        goals.remove(r);
                    }
                    let mut pen = HashMap::new();
                    let moves = self.v.moves.get(&s.id).cloned().unwrap_or_default();
                    for t in goals.iter().copied().chain(moves) {
                        if self.sub_danger(t) {
                            pen.insert(t, 6.0);
                        }
                    }
                    if let Some(mv) = self.move_toward(s, &goals, 2.0, &reserved, &pen, true, rng) {
                        return Some(mv);
                    }
                }
            }
            if !landings.is_empty() && escorts < 2 * landings.len() as i32 {
                let vip = *landings
                    .iter()
                    .min_by_key(|l| (manh(s.pos, l.pos), l.id))
                    .unwrap();
                escorts += 1;
                let ahead = if let Some(eh) = self.enemy_home {
                    let mut ahead = self.ring(vip.pos, 3, 0);
                    ahead.retain(|t| {
                        manh(*t, eh.pos) <= manh(vip.pos, eh.pos) && manh(*t, eh.pos) >= 2
                    });
                    for r in &reserved {
                        ahead.remove(r);
                    }
                    ahead
                } else {
                    self.ring(vip.pos, 2, 0)
                };
                if !ahead.is_empty() {
                    if let Some(mv) =
                        self.move_toward(s, &ahead, 1.8, &reserved, &HashMap::new(), true, rng)
                    {
                        return Some(mv);
                    }
                }
            }
            let goals = if advantage {
                let targets: Vec<EnemyShip> = if !self.fresh.is_empty() {
                    self.fresh.clone()
                } else {
                    self.stale.clone()
                };
                if !targets.is_empty() {
                    let bt = *targets
                        .iter()
                        .max_by(|a, b| {
                            let ka = kill_p(s.ship_type, a.ship_type, self.k)
                                * self.target_value(a)
                                - 6.0 * f64::from(manh(s.pos, a.pos));
                            let kb = kill_p(s.ship_type, b.ship_type, self.k)
                                * self.target_value(b)
                                - 6.0 * f64::from(manh(s.pos, b.pos));
                            ka.partial_cmp(&kb).unwrap().then(b.pos.cmp(&a.pos))
                        })
                        .unwrap();
                    let er = eff_range(s.ship_type, bt.ship_type).max(1);
                    let rmin = if s.ship_type == ShipType::Carrier && er >= 6 {
                        5
                    } else {
                        1
                    };
                    let mut goals = self.ring(bt.pos, er, rmin);
                    for r in &reserved {
                        goals.remove(r);
                    }
                    goals
                } else if let Some(eh) = self.enemy_home {
                    let mut goals = self.ring(eh.pos, 4, 2);
                    for r in &reserved {
                        goals.remove(r);
                    }
                    goals
                } else {
                    self.scout_goals(s)
                }
            } else {
                self.ring(self.forward_port().pos, 3, 0)
            };
            if !goals.is_empty() {
                if let Some(mv) =
                    self.move_toward(s, &goals, 2.2, &reserved, &HashMap::new(), true, rng)
                {
                    return Some(mv);
                }
            }
        }
        None
    }

    fn hull_scores(&mut self) -> [f64; 8] {
        let mut own = [0i32; 8];
        for s in &self.v.ships {
            own[s.ship_type.index()] += 1;
        }
        let n_combat: i32 = ShipType::ALL
            .iter()
            .filter(|t| combat(**t))
            .map(|t| own[t.index()])
            .sum();
        let n_sats = self.v.ports.iter().filter(|p| !p.is_home).count() as i32;
        let total = self.m.built_counts.iter().sum::<i32>().max(1) as f64;
        let mut sc = [0.0; 8];
        for t in ShipType::ALL {
            let share = self.m.built_counts[t.index()] as f64 / total;
            sc[t.index()] = (weight(&self.m.doctrine, t) * self.m.jitter[t.index()]).ln()
                + 0.45 * (0.125 - share).max(0.0) / 0.125;
        }
        let desired = n_sats.min(self.m.doctrine.target_merchants);
        let mi = ShipType::Merchant.index();
        if n_sats == 0 || own[mi] >= n_sats + 1 {
            sc[mi] -= 4.0;
        } else if own[mi] == 0 {
            sc[mi] += 1.6;
        } else if own[mi] < desired {
            sc[mi] += 0.6;
        }
        if self.enemy_home.is_none() {
            sc[ShipType::Destroyer.index()] += 1.1;
            sc[ShipType::Cruiser.index()] += 0.5;
            sc[ShipType::Carrier.index()] += 0.4;
            if self.v.turn == 0 {
                sc[ShipType::Landing.index()] -= 1.2;
            }
        }
        let mut ec = [0i32; 8];
        for e in self.fresh.iter().chain(self.stale.iter()) {
            ec[e.ship_type.index()] += 1;
        }
        if ec[ShipType::Submarine.index()] > 0 {
            sc[ShipType::Destroyer.index()] +=
                1.2 * f64::from(ec[ShipType::Submarine.index()].min(2));
        }
        if ec[ShipType::Battleship.index()] + ec[ShipType::Carrier.index()] > 0 {
            let n = (ec[ShipType::Battleship.index()] + ec[ShipType::Carrier.index()]).min(2);
            sc[ShipType::Submarine.index()] += 0.9 * f64::from(n);
            sc[ShipType::Carrier.index()] += 0.4;
        }
        if ec[ShipType::Destroyer.index()] > 0 {
            let n = ec[ShipType::Destroyer.index()].min(2);
            sc[ShipType::Cruiser.index()] += 0.8 * f64::from(n);
            sc[ShipType::Battleship.index()] += 0.6;
        }
        if ec[ShipType::Cruiser.index()] > 0 {
            let n = ec[ShipType::Cruiser.index()].min(2);
            sc[ShipType::Battleship.index()] += 0.8 * f64::from(n);
            sc[ShipType::Submarine.index()] += 0.4;
        }
        if ec[ShipType::Landing.index()] + ec[ShipType::Merchant.index()] > 0 {
            sc[ShipType::Carrier.index()] += 0.6;
            sc[ShipType::Destroyer.index()] += 0.4;
        }
        if own[ShipType::Builder.index()] >= 1
            || self.m.ports_built >= 3
            || !self.any_viable_coast()
        {
            sc[ShipType::Builder.index()] -= 4.5;
        } else if self.v.turn < 2 && self.m.doctrine.name != "ECONOMIC_BOOM" {
            sc[ShipType::Builder.index()] -= 1.0;
        } else if self.v.cash >= 250 && n_combat >= 1 {
            sc[ShipType::Builder.index()] += 0.3;
        }
        let max_land = if self.v.turn >= self.cfg.third_landing_turn {
            3
        } else {
            2
        };
        if own[ShipType::Landing.index()] >= max_land {
            sc[ShipType::Landing.index()] -= 3.5;
        } else {
            if n_combat >= 2 && own[ShipType::Landing.index()] == 0 {
                sc[ShipType::Landing.index()] += 1.3;
            }
            if self.enemy_home.is_some() && own[ShipType::Landing.index()] == 0 {
                sc[ShipType::Landing.index()] += 1.2;
            }
            if self.v.turn >= self.cfg.landing_ramp_turn && own[ShipType::Landing.index()] == 0 {
                sc[ShipType::Landing.index()] += 1.5;
            }
        }
        sc
    }

    fn sample_hull(&self, scores: &[f64; 8], rng: &mut dyn CommanderRng) -> ShipType {
        let temp = self.m.doctrine.buy_temperature.max(0.10);
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = ShipType::ALL
            .iter()
            .map(|t| ((scores[t.index()] - mx) / temp).exp())
            .collect();
        let mut r = rng.random() * ws.iter().sum::<f64>();
        for (t, w) in ShipType::ALL.iter().zip(ws) {
            r -= w;
            if r <= 0.0 {
                return *t;
            }
        }
        ShipType::Builder
    }

    fn build_for(&mut self, t: ShipType) -> Option<Action> {
        let mut options = Vec::new();
        let mut ports: Vec<(i32, i32)> = self.v.builds.keys().copied().collect();
        ports.sort_unstable();
        for port in ports {
            if !self
                .v
                .builds
                .get(&port)
                .map(|s| s.contains(&t))
                .unwrap_or(false)
            {
                continue;
            }
            for sp in self.water_adj(port) {
                if !self.occupied.contains(&sp) {
                    options.push((port, sp));
                }
            }
        }
        if options.is_empty() {
            return None;
        }
        let port_by: HashMap<(i32, i32), OwnPort> =
            self.v.ports.iter().map(|p| (p.pos, *p)).collect();
        let home = self.v.home.pos;
        let fresh = self.fresh.clone();
        let choice = if combat(t) && fresh.iter().any(|e| manh(e.pos, home) <= 8) {
            options.into_iter().min_by(|a, b| {
                let ka = (
                    manh(a.0, home),
                    -fresh.iter().map(|e| manh(a.1, e.pos)).min().unwrap_or(0),
                    a.0,
                    a.1,
                );
                let kb = (
                    manh(b.0, home),
                    -fresh.iter().map(|e| manh(b.1, e.pos)).min().unwrap_or(0),
                    b.0,
                    b.1,
                );
                ka.cmp(&kb)
            })
        } else if t == ShipType::Merchant {
            let scored: Vec<_> = options
                .into_iter()
                .map(|o| {
                    let p = port_by[&o.0];
                    let bonus = if p.is_home {
                        0.0
                    } else {
                        2.0 * p.stockpile as f64 + 30.0
                    };
                    let score = -(bonus - 200.0 * self.threat_of(o.1, ShipType::Merchant));
                    (score, o)
                })
                .collect();
            scored
                .into_iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)))
                .map(|(_, o)| o)
        } else if t == ShipType::Builder && self.m.doctrine.builder_mode == "ECO_HOME" {
            options.into_iter().min_by(|a, b| {
                let sa = (
                    manh(a.0, home) as f64,
                    300.0 * self.threat_of(a.1, ShipType::Builder),
                );
                let sb = (
                    manh(b.0, home) as f64,
                    300.0 * self.threat_of(b.1, ShipType::Builder),
                );
                sa.0.partial_cmp(&sb.0)
                    .unwrap()
                    .then(sa.1.partial_cmp(&sb.1).unwrap())
                    .then(a.cmp(b))
            })
        } else {
            let reference = self
                .enemy_home
                .map(|e| e.pos)
                .unwrap_or((self.v.width - 1 - home.0, home.1));
            let pen = if !combat(t) { 300.0 } else { 40.0 };
            options.into_iter().min_by(|a, b| {
                let sa = f64::from(manh(a.1, reference)) + pen * self.threat_of(a.1, t);
                let sb = f64::from(manh(b.1, reference)) + pen * self.threat_of(b.1, t);
                sa.partial_cmp(&sb).unwrap().then(a.cmp(b))
            })
        };
        choice.map(|(port, spawn)| Action::BuildShip {
            port_x: port.0,
            port_y: port.1,
            spawn_x: spawn.0,
            spawn_y: spawn.1,
            ship_type: t,
        })
    }

    fn purchase(&mut self, rng: &mut dyn CommanderRng) -> Option<Action> {
        if self.v.builds.is_empty() {
            return None;
        }
        let mut affordable = HashSet::new();
        for set in self.v.builds.values() {
            affordable.extend(set.iter().copied());
        }
        let mut own = [0i32; 8];
        for s in &self.v.ships {
            own[s.ship_type.index()] += 1;
        }
        let close: Vec<EnemyShip> = self
            .fresh
            .iter()
            .copied()
            .filter(|e| manh(e.pos, self.v.home.pos) <= 8)
            .collect();
        if !close.is_empty() {
            let defenders = self
                .v
                .ships
                .iter()
                .filter(|s| combat(s.ship_type) && manh(s.pos, self.v.home.pos) <= 7)
                .count();
            if defenders < close.len() + 1 {
                let aff: Vec<ShipType> = ShipType::ALL
                    .into_iter()
                    .filter(|t| combat(*t) && affordable.contains(t))
                    .collect();
                if !aff.is_empty() {
                    let te = *close
                        .iter()
                        .min_by_key(|e| (manh(e.pos, self.v.home.pos), e.pos))
                        .unwrap();
                    let best = *aff
                        .iter()
                        .max_by(|a, b| {
                            let ka = kill_p(**a, te.ship_type, self.k);
                            let kb = kill_p(**b, te.ship_type, self.k);
                            ka.partial_cmp(&kb)
                                .unwrap()
                                .then(tables().stats(**a).speed.cmp(&tables().stats(**b).speed))
                                .then(b.index().cmp(&a.index()))
                        })
                        .unwrap();
                    if let Some(act) = self.build_for(best) {
                        if let Action::BuildShip { ship_type, .. } = act {
                            self.m.built_counts[ship_type.index()] += 1;
                        }
                        return Some(act);
                    }
                }
            }
        }
        let n_combat: i32 = ShipType::ALL
            .iter()
            .filter(|t| combat(**t))
            .map(|t| own[t.index()])
            .sum();
        if affordable.contains(&ShipType::Landing)
            && own[ShipType::Landing.index()] == 0
            && self.enemy_home.is_some()
            && (n_combat >= 2 || self.v.turn >= self.cfg.checkmate_turn)
        {
            if let Some(act) = self.build_for(ShipType::Landing) {
                self.m.built_counts[ShipType::Landing.index()] += 1;
                if self.m.planned_buy == Some(ShipType::Landing) {
                    self.m.planned_buy = None;
                }
                return Some(act);
            }
        }
        if self.m.planned_buy.is_none() {
            let scores = self.hull_scores();
            self.m.planned_buy = Some(self.sample_hull(&scores, rng));
        }
        let stocked = self.v.ports.iter().any(|p| !p.is_home && p.stockpile >= 25);
        if self
            .m
            .planned_buy
            .map(|t| !affordable.contains(&t))
            .unwrap_or(false)
            && own[ShipType::Merchant.index()] == 0
            && stocked
            && affordable.contains(&ShipType::Merchant)
        {
            if let Some(act) = self.build_for(ShipType::Merchant) {
                self.m.built_counts[ShipType::Merchant.index()] += 1;
                return Some(act);
            }
        }
        if let Some(chosen) = self.m.planned_buy {
            if affordable.contains(&chosen) {
                if let Some(act) = self.build_for(chosen) {
                    self.m.built_counts[chosen.index()] += 1;
                    self.m.planned_buy = None;
                    return Some(act);
                }
            }
        }
        None
    }
}
