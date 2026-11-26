//! Value-net observation tokens. Layout matches `battleboats.envs.observation`.
//!
//! Fourier features use f64 `sin`/`cos` then a cast to f32, the same rounding
//! as assigning a Python float into a float32 row.

use crate::ship::{ShipType, SpotKey, SHIP_TYPE_COUNT};
use crate::state::{GameState, MERCHANT_CAPACITY};
use crate::tables::tables;

pub const TOKEN_TYPE_ONEHOT_DIM: usize = 10;
pub const POSITION_FREQS: usize = 8;
pub const HOME_REL_FREQS: usize = 4;
pub const TOKEN_POSITION_DIM: usize = POSITION_FREQS * 2 * 2;
pub const TOKEN_HOME_REL_DIM: usize = HOME_REL_FREQS * 2 * 2;
pub const TOKEN_OWNER_DIM: usize = 2;
pub const TOKEN_STATS_DIM: usize = 6;
pub const TOKEN_SHIP_STATE_DIM: usize = 3;
pub const TOKEN_PORT_STATE_DIM: usize = 3;
pub const TOKEN_SIGHTING_STATE_DIM: usize = 2;
pub const TOKEN_BOARD_SIZE_DIM: usize = 2;

pub const TOKEN_DIM: usize = TOKEN_TYPE_ONEHOT_DIM
    + TOKEN_POSITION_DIM
    + TOKEN_HOME_REL_DIM
    + TOKEN_OWNER_DIM
    + TOKEN_STATS_DIM
    + TOKEN_SHIP_STATE_DIM
    + TOKEN_PORT_STATE_DIM
    + TOKEN_SIGHTING_STATE_DIM
    + TOKEN_BOARD_SIZE_DIM;

pub const GLOBALS_DIM: usize = 8;

const CASH_SCALE: f64 = 1500.0;
const STOCKPILE_SCALE: f64 = 1500.0;
const MAX_TURNS_SCALE: f64 = 500.0;
const POSITION_SCALE: f64 = 256.0;
const SHIP_COUNT_SCALE: f64 = 50.0;
const PORT_COUNT_SCALE: f64 = 25.0;
const POSITION_MIN_WAVELENGTH: f64 = 2.0;
const POSITION_MAX_WAVELENGTH: f64 = 4096.0;

const TYPE_OFFSET: usize = 0;
const POSITION_OFFSET: usize = TYPE_OFFSET + TOKEN_TYPE_ONEHOT_DIM;
const HOME_REL_OFFSET: usize = POSITION_OFFSET + TOKEN_POSITION_DIM;
const OWNER_OFFSET: usize = HOME_REL_OFFSET + TOKEN_HOME_REL_DIM;
const STATS_OFFSET: usize = OWNER_OFFSET + TOKEN_OWNER_DIM;
const SHIP_STATE_OFFSET: usize = STATS_OFFSET + TOKEN_STATS_DIM;
const PORT_STATE_OFFSET: usize = SHIP_STATE_OFFSET + TOKEN_SHIP_STATE_DIM;
const SIGHTING_STATE_OFFSET: usize = PORT_STATE_OFFSET + TOKEN_PORT_STATE_DIM;
const BOARD_SIZE_OFFSET: usize = SIGHTING_STATE_OFFSET + TOKEN_SIGHTING_STATE_DIM;

const PORT_TYPE_INDEX: usize = SHIP_TYPE_COUNT;
const COASTLINE_TYPE_INDEX: usize = PORT_TYPE_INDEX + 1;

pub struct Observation {
    /// Row-major `(n, TOKEN_DIM)` float32 values.
    pub tokens: Vec<f32>,
    pub n_tokens: usize,
    pub globals: [f32; GLOBALS_DIM],
}

struct Scales {
    speed: f64,
    strength: f64,
    attack_range: f64,
    cost: f64,
    spotted_at: [f64; SHIP_TYPE_COUNT],
    spots_at: [f64; SHIP_TYPE_COUNT],
    position_waves: Vec<f64>,
    home_waves: Vec<f64>,
}

fn geometric(count: usize, shortest: f64, longest: f64) -> Vec<f64> {
    let log_lo = shortest.ln();
    let log_hi = longest.ln();
    let step = (log_hi - log_lo) / (count - 1) as f64;
    (0..count)
        .map(|i| (log_lo + i as f64 * step).exp())
        .collect()
}

fn scales() -> &'static Scales {
    use std::sync::OnceLock;
    static SCALES: OnceLock<Scales> = OnceLock::new();
    SCALES.get_or_init(|| {
        let book = tables();
        let mut speed = 0.0f64;
        let mut strength = 0.0f64;
        let mut attack_range = 0.0f64;
        let mut cost = 0.0f64;
        let mut spot_scale = 0.0f64;
        for t in ShipType::ALL {
            let s = book.stats(t);
            speed = speed.max(s.speed as f64);
            strength = strength.max(s.strength);
            attack_range = attack_range.max(s.attack_range as f64);
            cost = cost.max(s.cost as f64);
            for o in ShipType::ALL {
                spot_scale = spot_scale.max(book.spot_distance(SpotKey::Ship(o), SpotKey::Ship(t)));
            }
        }
        let n = SHIP_TYPE_COUNT as f64;
        let mut spotted_at = [0.0; SHIP_TYPE_COUNT];
        let mut spots_at = [0.0; SHIP_TYPE_COUNT];
        for t in ShipType::ALL {
            let mut seen = 0.0;
            let mut sees = 0.0;
            for o in ShipType::ALL {
                seen += book.spot_distance(SpotKey::Ship(o), SpotKey::Ship(t));
                sees += book.spot_distance(SpotKey::Ship(t), SpotKey::Ship(o));
            }
            spotted_at[t.index()] = seen / n / spot_scale;
            spots_at[t.index()] = sees / n / spot_scale;
        }
        Scales {
            speed,
            strength,
            attack_range,
            cost,
            spotted_at,
            spots_at,
            position_waves: geometric(
                POSITION_FREQS,
                POSITION_MIN_WAVELENGTH,
                POSITION_MAX_WAVELENGTH,
            ),
            home_waves: geometric(
                HOME_REL_FREQS,
                POSITION_MIN_WAVELENGTH,
                POSITION_MAX_WAVELENGTH,
            ),
        }
    })
}

fn f32_div(numer: f64, denom: f64) -> f32 {
    (numer / denom) as f32
}

fn write_fourier(row: &mut [f32], offset: usize, coord: f64, waves: &[f64]) {
    for (i, wave) in waves.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * (coord / wave);
        row[offset + 2 * i] = angle.sin() as f32;
        row[offset + 2 * i + 1] = angle.cos() as f32;
    }
}

fn write_position(row: &mut [f32], x: f64, y: f64, home: (i32, i32), scales: &Scales) {
    let hx = f64::from(home.0);
    let hy = f64::from(home.1);
    write_fourier(row, POSITION_OFFSET, x, &scales.position_waves);
    write_fourier(
        row,
        POSITION_OFFSET + 2 * scales.position_waves.len(),
        y,
        &scales.position_waves,
    );
    write_fourier(row, HOME_REL_OFFSET, x - hx, &scales.home_waves);
    write_fourier(
        row,
        HOME_REL_OFFSET + 2 * scales.home_waves.len(),
        y - hy,
        &scales.home_waves,
    );
}

fn write_stats(row: &mut [f32], ship_type: ShipType, scales: &Scales) {
    let s = tables().stats(ship_type);
    row[STATS_OFFSET] = f32_div(s.speed as f64, scales.speed);
    row[STATS_OFFSET + 1] = f32_div(s.strength, scales.strength);
    row[STATS_OFFSET + 2] = f32_div(s.attack_range as f64, scales.attack_range);
    row[STATS_OFFSET + 3] = scales.spotted_at[ship_type.index()] as f32;
    row[STATS_OFFSET + 4] = scales.spots_at[ship_type.index()] as f32;
    row[STATS_OFFSET + 5] = f32_div(s.cost as f64, scales.cost);
}

fn write_board(row: &mut [f32], width: i32, height: i32) {
    row[BOARD_SIZE_OFFSET] = f32_div(f64::from(width), POSITION_SCALE);
    row[BOARD_SIZE_OFFSET + 1] = f32_div(f64::from(height), POSITION_SCALE);
}

impl GameState {
    pub fn observation(&self, player_id: i64) -> Observation {
        let scales = scales();
        let player = &self.players[player_id as usize];
        let home = player.home_port;
        let mut tokens = Vec::new();
        let width = self.map.width;
        let height = self.map.height;

        for ship in self.ships.values().filter(|s| s.owner == player_id) {
            let mut row = vec![0.0f32; TOKEN_DIM];
            row[ship.ship_type.index()] = 1.0;
            write_position(
                &mut row,
                f64::from(ship.position.0),
                f64::from(ship.position.1),
                home,
                scales,
            );
            row[OWNER_OFFSET] = 1.0;
            write_stats(&mut row, ship.ship_type, scales);
            let speed = tables().stats(ship.ship_type).speed as f64;
            row[SHIP_STATE_OFFSET] = f32_div(ship.cargo as f64, MERCHANT_CAPACITY as f64);
            row[SHIP_STATE_OFFSET + 1] = if ship.has_attacked { 1.0 } else { 0.0 };
            row[SHIP_STATE_OFFSET + 2] = f32_div(ship.tiles_moved_this_turn as f64, speed);
            write_board(&mut row, width, height);
            tokens.extend(row);
        }
        for port in self.ports.values().filter(|p| p.owner == player_id) {
            let mut row = vec![0.0f32; TOKEN_DIM];
            row[PORT_TYPE_INDEX] = 1.0;
            write_position(
                &mut row,
                f64::from(port.position.0),
                f64::from(port.position.1),
                home,
                scales,
            );
            row[OWNER_OFFSET] = 1.0;
            row[PORT_STATE_OFFSET] = f32_div(port.stockpile as f64, STOCKPILE_SCALE);
            row[PORT_STATE_OFFSET + 1] = if port.is_home { 1.0 } else { 0.0 };
            if port.is_home {
                let cash = self.players[port.owner as usize].cash as f64;
                row[PORT_STATE_OFFSET + 2] = f32_div(cash, CASH_SCALE);
            }
            write_board(&mut row, width, height);
            tokens.extend(row);
        }
        for sighting in player.sightings.values() {
            let mut row = vec![0.0f32; TOKEN_DIM];
            row[sighting.ship_type.index()] = 1.0;
            write_position(
                &mut row,
                f64::from(sighting.position.0),
                f64::from(sighting.position.1),
                home,
                scales,
            );
            row[OWNER_OFFSET + 1] = 1.0;
            write_stats(&mut row, sighting.ship_type, scales);
            row[SIGHTING_STATE_OFFSET] = if sighting.fresh { 1.0 } else { 0.0 };
            row[SIGHTING_STATE_OFFSET + 1] =
                f32_div((self.turn - sighting.turn_seen) as f64, MAX_TURNS_SCALE);
            write_board(&mut row, width, height);
            tokens.extend(row);
        }
        for sighting in player.port_sightings.values() {
            let mut row = vec![0.0f32; TOKEN_DIM];
            row[PORT_TYPE_INDEX] = 1.0;
            write_position(
                &mut row,
                f64::from(sighting.position.0),
                f64::from(sighting.position.1),
                home,
                scales,
            );
            row[OWNER_OFFSET + 1] = 1.0;
            row[PORT_STATE_OFFSET + 1] = if sighting.is_home { 1.0 } else { 0.0 };
            row[SIGHTING_STATE_OFFSET] = if sighting.fresh { 1.0 } else { 0.0 };
            row[SIGHTING_STATE_OFFSET + 1] =
                f32_div((self.turn - sighting.turn_seen) as f64, MAX_TURNS_SCALE);
            write_board(&mut row, width, height);
            tokens.extend(row);
        }
        let coast = self.coastline_tiles(Some(player_id)).unwrap_or_default();
        for pos in coast {
            let mut row = vec![0.0f32; TOKEN_DIM];
            row[COASTLINE_TYPE_INDEX] = 1.0;
            write_position(&mut row, f64::from(pos.0), f64::from(pos.1), home, scales);
            write_board(&mut row, width, height);
            tokens.extend(row);
        }

        let n_tokens = if tokens.is_empty() {
            0
        } else {
            tokens.len() / TOKEN_DIM
        };
        let fresh_ships = player.sightings.values().filter(|s| s.fresh).count();
        let fresh_ports = player.port_sightings.values().filter(|s| s.fresh).count();
        let globals = [
            f32_div(player.cash as f64, CASH_SCALE),
            f32_div(player.owned_ship_ids.len() as f64, SHIP_COUNT_SCALE),
            f32_div(player.owned_port_positions.len() as f64, PORT_COUNT_SCALE),
            f32_div(fresh_ships as f64, SHIP_COUNT_SCALE),
            f32_div(fresh_ports as f64, PORT_COUNT_SCALE),
            f32_div(self.turn as f64, MAX_TURNS_SCALE),
            f32_div(f64::from(width), POSITION_SCALE),
            f32_div(f64::from(height), POSITION_SCALE),
        ];
        Observation {
            tokens,
            n_tokens,
            globals,
        }
    }
}
