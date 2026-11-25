//! Live game state. Field meanings match `gameEngine` / the schema_version 1
//! canonical document. Internal PRNG state is intentionally absent: it is not
//! part of the hash, and strict PCG64 parity is a later issue.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{CoreError, CoreResult};
use crate::map::Map;
use crate::ship::{ShipStats, ShipType, SpotKey};
use crate::tables::tables;

pub const STARTING_CASH: i64 = 250;
pub const PORT_PRODUCTION: i64 = 25;
pub const CASH_PER_MATERIAL: i64 = 2;
pub const MERCHANT_CAPACITY: i64 = 100;
pub const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct Ship {
    pub id: i64,
    pub ship_type: ShipType,
    pub stats: ShipStats,
    pub owner: i64,
    pub position: (i32, i32),
    pub tiles_moved_this_turn: i64,
    pub has_attacked: bool,
    pub cargo: i64,
    pub bought_this_turn: bool,
    pub built_turn: i64,
}

#[derive(Debug, Clone)]
pub struct Port {
    pub position: (i32, i32),
    pub owner: i64,
    pub is_home: bool,
    pub stockpile: i64,
    pub locked_this_turn: bool,
}

#[derive(Debug, Clone)]
pub struct ShipSighting {
    pub ship_id: i64,
    pub ship_type: ShipType,
    pub position: (i32, i32),
    pub turn_seen: i64,
    pub fresh: bool,
}

#[derive(Debug, Clone)]
pub struct PortSighting {
    pub position: (i32, i32),
    pub last_known_owner: i64,
    pub is_home: bool,
    pub turn_seen: i64,
    pub fresh: bool,
}

#[derive(Debug, Clone)]
pub struct Player {
    pub id: i64,
    pub home_port: (i32, i32),
    pub cash: i64,
    pub owned_ship_ids: BTreeSet<i64>,
    pub owned_port_positions: BTreeSet<(i32, i32)>,
    pub sightings: BTreeMap<i64, ShipSighting>,
    pub port_sightings: BTreeMap<(i32, i32), PortSighting>,
}

impl Player {
    fn new(id: i64, home_port: (i32, i32)) -> Self {
        let mut owned_port_positions = BTreeSet::new();
        owned_port_positions.insert(home_port);
        Self {
            id,
            home_port,
            cash: 0,
            owned_ship_ids: BTreeSet::new(),
            owned_port_positions,
            sightings: BTreeMap::new(),
            port_sightings: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub map: Map,
    pub players: Vec<Player>,
    pub ships: BTreeMap<i64, Ship>,
    pub ports: BTreeMap<(i32, i32), Port>,
    pub current_player: i64,
    pub turn: i64,
    pub winner: Option<i64>,
    pub kill_curve_k: f64,
    pub next_ship_id: i64,
}

#[derive(Debug, Clone)]
enum MapSource {
    Path(String),
    Json(String),
    /// Built from a canonical document. `reset` needs the original map file.
    Detached,
}

/// One engine, mirroring `gameEngine.__init__` + `reset` / `reset_from_scenario`.
#[derive(Debug, Clone)]
pub struct Engine {
    source: MapSource,
    pub kill_curve_k: f64,
    /// Accepted for API compatibility. Not part of canonical state.
    pub rng_seed: Option<i64>,
    pub state: Option<GameState>,
}

impl Engine {
    pub fn from_map_path(path: impl Into<String>, kill_curve_k: f64) -> Self {
        Self {
            source: MapSource::Path(path.into()),
            kill_curve_k,
            rng_seed: None,
            state: None,
        }
    }

    pub fn from_state(state: GameState, kill_curve_k: f64) -> Self {
        Self {
            source: MapSource::Detached,
            kill_curve_k,
            rng_seed: None,
            state: Some(state),
        }
    }

    pub fn from_map_json(text: impl Into<String>, kill_curve_k: f64) -> Self {
        Self {
            source: MapSource::Json(text.into()),
            kill_curve_k,
            rng_seed: None,
            state: None,
        }
    }

    pub fn reset(&mut self, seed: Option<i64>) -> CoreResult<&GameState> {
        self.rng_seed = seed;
        let map = self.load_map()?;
        self.state = Some(GameState::reset_from_map(map, self.kill_curve_k)?);
        Ok(self.state.as_ref().unwrap())
    }

    pub fn reset_from_scenario_json(&mut self, text: &str) -> CoreResult<&GameState> {
        let scenario: Scenario = serde_json::from_str(text)
            .map_err(|e| CoreError::new(format!("scenario json: {e}")))?;
        self.reset_from_scenario(&scenario)
    }

    fn reset_from_scenario(&mut self, scenario: &Scenario) -> CoreResult<&GameState> {
        if scenario.map_path.is_empty() {
            return Err(CoreError::new("scenario is missing map_path"));
        }
        self.source = MapSource::Path(scenario.map_path.clone());
        self.reset(scenario.seed)?;
        let state = self.state.as_mut().unwrap();
        state.players[0].cash = scenario.player_0.cash;
        state.players[1].cash = scenario.player_1.cash;
        for ship in &scenario.player_0.ships {
            state.spawn_ship(0, ShipType::from_enum_name(&ship.ship_type)?, ship.position)?;
        }
        for ship in &scenario.player_1.ships {
            state.spawn_ship(1, ShipType::from_enum_name(&ship.ship_type)?, ship.position)?;
        }
        state.current_player = 0;
        state.turn = 0;
        state.winner = None;
        state.refresh_sightings()?;
        Ok(self.state.as_ref().unwrap())
    }

    pub fn require(&self) -> CoreResult<&GameState> {
        self.state
            .as_ref()
            .ok_or_else(|| CoreError::new("engine has not been reset"))
    }

    fn load_map(&self) -> CoreResult<Map> {
        match &self.source {
            MapSource::Json(text) => Map::load_json(text),
            MapSource::Path(path) => {
                let text = fs::read_to_string(Path::new(path)).map_err(|e| {
                    CoreError::new(format!("failed to read map {path}: {e}"))
                })?;
                Map::load_json(&text)
            }
            MapSource::Detached => Err(CoreError::new(
                "reset requires a map path or map json; this engine was loaded from canonical state",
            )),
        }
    }
}

impl GameState {
    pub fn reset_from_map(map: Map, kill_curve_k: f64) -> CoreResult<Self> {
        let homes = map.infer_home_ports()?;
        let mut players = vec![Player::new(0, homes[0]), Player::new(1, homes[1])];
        for player in &mut players {
            player.cash = STARTING_CASH;
        }
        let home_set = [homes[0], homes[1]];
        let mut ports = BTreeMap::new();
        for &pos in &map.port_positions {
            let owner = i64::from(map.port_owner_at(pos.0, pos.1)?);
            if owner == 0 || owner == 1 {
                players[owner as usize].owned_port_positions.insert(pos);
            }
            ports.insert(
                pos,
                Port {
                    position: pos,
                    owner,
                    is_home: home_set.contains(&pos),
                    stockpile: 0,
                    locked_this_turn: false,
                },
            );
        }
        let mut state = Self {
            map,
            players,
            ships: BTreeMap::new(),
            ports,
            current_player: 0,
            turn: 0,
            winner: None,
            kill_curve_k,
            next_ship_id: 0,
        };
        state.refresh_sightings()?;
        Ok(state)
    }

    pub fn spawn_ship(
        &mut self,
        owner: i64,
        ship_type: ShipType,
        position: (i32, i32),
    ) -> CoreResult<i64> {
        let id = self.next_ship_id;
        self.next_ship_id += 1;
        let ship = Ship {
            id,
            ship_type,
            stats: tables().stats(ship_type),
            owner,
            position,
            tiles_moved_this_turn: 0,
            has_attacked: false,
            cargo: 0,
            bought_this_turn: false,
            built_turn: -1,
        };
        self.map.place_ship(id, position.0, position.1)?;
        self.players[owner as usize].owned_ship_ids.insert(id);
        self.ships.insert(id, ship);
        Ok(id)
    }

    /// `gameEngine._refresh_sightings`. Order of iteration does not change the
    /// result: each target is written independently.
    pub fn refresh_sightings(&mut self) -> CoreResult<()> {
        let ship_targets: Vec<(i64, ShipType, (i32, i32), i64)> = self
            .ships
            .values()
            .map(|s| (s.id, s.ship_type, s.position, s.owner))
            .collect();
        let port_targets: Vec<((i32, i32), i64, bool)> = self
            .ports
            .values()
            .map(|p| (p.position, p.owner, p.is_home))
            .collect();
        let observer_count = self.players.len();
        for observer_id in 0..observer_count {
            for sighting in self.players[observer_id].sightings.values_mut() {
                sighting.fresh = false;
            }
            for (id, ship_type, position, owner) in &ship_targets {
                if *owner == self.players[observer_id].id {
                    continue;
                }
                if !self.can_observer_see(observer_id, *position, SpotKey::Ship(*ship_type))? {
                    continue;
                }
                self.players[observer_id].sightings.insert(
                    *id,
                    ShipSighting {
                        ship_id: *id,
                        ship_type: *ship_type,
                        position: *position,
                        turn_seen: self.turn,
                        fresh: true,
                    },
                );
            }

            let owned = self.players[observer_id].owned_port_positions.clone();
            self.players[observer_id]
                .port_sightings
                .retain(|pos, _| !owned.contains(pos));
            for sighting in self.players[observer_id].port_sightings.values_mut() {
                sighting.fresh = false;
            }
            for (position, owner, is_home) in &port_targets {
                if *owner == self.players[observer_id].id {
                    continue;
                }
                if !self.can_observer_see(observer_id, *position, SpotKey::Port)? {
                    continue;
                }
                self.players[observer_id].port_sightings.insert(
                    *position,
                    PortSighting {
                        position: *position,
                        last_known_owner: *owner,
                        is_home: *is_home,
                        turn_seen: self.turn,
                        fresh: true,
                    },
                );
            }
        }
        Ok(())
    }

    fn can_observer_see(
        &self,
        observer_idx: usize,
        target: (i32, i32),
        target_key: SpotKey,
    ) -> CoreResult<bool> {
        let player = &self.players[observer_idx];
        let book = tables();
        for sid in &player.owned_ship_ids {
            let scout = self
                .ships
                .get(sid)
                .ok_or_else(|| CoreError::new(format!("missing ship {sid}")))?;
            let distance = Map::manhattan(scout.position, target);
            let reach = book.spot_distance(SpotKey::Ship(scout.ship_type), target_key);
            if f64::from(distance) <= reach {
                return Ok(true);
            }
        }
        for pos in &player.owned_port_positions {
            let distance = Map::manhattan(*pos, target);
            let reach = book.spot_distance(SpotKey::Port, target_key);
            if f64::from(distance) <= reach {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `observation._coastline_tiles`. `None` drops every real port. A player
    /// id drops ports that player owns or has sighted, so unsighted enemy
    /// ports stay coastline.
    pub fn coastline_tiles(&self, player_id: Option<i64>) -> CoreResult<Vec<(i32, i32)>> {
        let base = self.map.coastline_owned();
        let filtered = match player_id {
            None => base
                .into_iter()
                .filter(|pos| self.map.is_port(pos.0, pos.1).ok() == Some(false))
                .collect(),
            Some(pid) => {
                let player = self
                    .players
                    .iter()
                    .find(|p| p.id == pid)
                    .ok_or_else(|| CoreError::new(format!("no player {pid}")))?;
                let known = &player.owned_port_positions;
                let sighted = &player.port_sightings;
                base.into_iter()
                    .filter(|pos| !known.contains(pos) && !sighted.contains_key(pos))
                    .collect()
            }
        };
        Ok(filtered)
    }
}

#[derive(Debug, Deserialize)]
struct Scenario {
    seed: Option<i64>,
    map_path: String,
    player_0: ScenarioPlayer,
    player_1: ScenarioPlayer,
}

#[derive(Debug, Deserialize)]
struct ScenarioPlayer {
    cash: i64,
    ships: Vec<ScenarioShip>,
}

#[derive(Debug, Deserialize)]
struct ScenarioShip {
    #[serde(rename = "type")]
    ship_type: String,
    position: (i32, i32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_spots_enemy_ports_inside_sixteen_tiles() {
        let map = Map::load_json(
            r#"{"width":4,"height":3,"land":[[0,0],[1,1],[3,2]],"ports":{"0":[[0,0]],"1":[[3,2]]}}"#,
        )
        .unwrap();
        let state = GameState::reset_from_map(map, 2.0).unwrap();
        assert_eq!(state.players[0].cash, 250);
        assert_eq!(state.players[0].home_port, (0, 0));
        assert_eq!(state.players[1].home_port, (3, 2));
        assert_eq!(state.next_ship_id, 0);
        assert!(state.ships.is_empty());
        let seen = &state.players[0].port_sightings[&(3, 2)];
        assert!(seen.fresh);
        assert!(seen.is_home);
        assert_eq!(seen.last_known_owner, 1);
        assert_eq!(seen.turn_seen, 0);
        assert!(state.players[1].port_sightings.contains_key(&(0, 0)));
        let coast = state.coastline_tiles(Some(0)).unwrap();
        assert_eq!(coast, vec![(1, 1)]);
        let all_ports_hidden = state.coastline_tiles(None).unwrap();
        assert_eq!(all_ports_hidden, vec![(1, 1)]);
    }
}
