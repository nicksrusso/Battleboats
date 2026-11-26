//! schema_version 1 canonical state and its SHA-256.
//!
//! `state_hash = sha256(utf8(json.dumps(state, sort_keys=True, separators=(",", ":"))))`.
//! Collections are sorted here so map/set iteration order cannot change the digest.

use std::collections::BTreeSet;

use indexmap::IndexMap;

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{CoreError, CoreResult};
use crate::jsonfmt::{
    format_python_json_float, json_array, json_bool, json_i64, json_object, json_pos, json_string,
};
use crate::map::Map;
use crate::ship::ShipType;
use crate::state::{GameState, Player, Port, PortSighting, Ship, ShipSighting, SCHEMA_VERSION};
use crate::tables::tables;

pub fn canonical_json(state: &GameState) -> String {
    let players = json_array(state.players.iter().map(player_json).collect::<Vec<_>>());
    let mut ports: Vec<_> = state.ports.values().collect();
    ports.sort_by_key(|port| port.position);
    let ports = json_array(ports.into_iter().map(port_json));
    let mut ships: Vec<_> = state.ships.values().collect();
    ships.sort_by_key(|ship| ship.id);
    let ships = json_array(ships.into_iter().map(ship_json));
    let terrain = json_array(
        state
            .map
            .terrain_land()
            .into_iter()
            .map(|(x, y)| json_pos(x, y)),
    );
    let winner = match state.winner {
        Some(id) => json_i64(id),
        None => "null".to_string(),
    };
    json_object(&[
        ("current_player", json_i64(state.current_player)),
        ("height", json_i64(i64::from(state.map.height))),
        ("kill_curve_k", format_python_json_float(state.kill_curve_k)),
        ("next_ship_id", json_i64(state.next_ship_id)),
        ("players", players),
        ("ports", ports),
        ("schema_version", json_i64(SCHEMA_VERSION)),
        ("ships", ships),
        ("terrain_land", terrain),
        ("turn", json_i64(state.turn)),
        ("width", json_i64(i64::from(state.map.width))),
        ("winner", winner),
    ])
}

pub fn state_hash(state: &GameState) -> String {
    hash_canonical_json(&canonical_json(state))
}

pub fn hash_canonical_json(json: &str) -> String {
    let digest = Sha256::digest(json.as_bytes());
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn player_json(player: &Player) -> String {
    let mut ship_rows: Vec<_> = player.sightings.values().collect();
    ship_rows.sort_by_key(|s| s.ship_id);
    let ship_sightings = json_array(ship_rows.into_iter().map(ship_sighting_json));
    let mut port_rows: Vec<_> = player.port_sightings.values().collect();
    port_rows.sort_by_key(|s| s.position);
    let port_sightings = json_array(port_rows.into_iter().map(port_sighting_json));
    json_object(&[
        ("cash", json_i64(player.cash)),
        (
            "home_port",
            json_pos(player.home_port.0, player.home_port.1),
        ),
        ("id", json_i64(player.id)),
        ("port_sightings", port_sightings),
        ("ship_sightings", ship_sightings),
    ])
}

fn ship_sighting_json(s: &ShipSighting) -> String {
    json_object(&[
        ("fresh", json_bool(s.fresh)),
        ("position", json_pos(s.position.0, s.position.1)),
        ("ship_id", json_i64(s.ship_id)),
        ("turn_seen", json_i64(s.turn_seen)),
        ("type", json_string(s.ship_type.value())),
    ])
}

fn port_sighting_json(s: &PortSighting) -> String {
    json_object(&[
        ("fresh", json_bool(s.fresh)),
        ("is_home", json_bool(s.is_home)),
        ("last_known_owner", json_i64(s.last_known_owner)),
        ("position", json_pos(s.position.0, s.position.1)),
        ("turn_seen", json_i64(s.turn_seen)),
    ])
}

fn port_json(port: &Port) -> String {
    json_object(&[
        ("is_home", json_bool(port.is_home)),
        ("locked_this_turn", json_bool(port.locked_this_turn)),
        ("owner", json_i64(port.owner)),
        ("position", json_pos(port.position.0, port.position.1)),
        ("stockpile", json_i64(port.stockpile)),
    ])
}

fn ship_json(ship: &Ship) -> String {
    json_object(&[
        ("bought_this_turn", json_bool(ship.bought_this_turn)),
        ("built_turn", json_i64(ship.built_turn)),
        ("cargo", json_i64(ship.cargo)),
        ("has_attacked", json_bool(ship.has_attacked)),
        ("id", json_i64(ship.id)),
        ("owner", json_i64(ship.owner)),
        ("position", json_pos(ship.position.0, ship.position.1)),
        (
            "tiles_moved_this_turn",
            json_i64(ship.tiles_moved_this_turn),
        ),
        ("type", json_string(ship.ship_type.value())),
    ])
}

/// Parse a canonical document (key order does not matter) back into state.
///
/// Arrays are re-sorted on the way back out, so a shuffled Python dict still
/// hashes like `canonical_state()`.
pub fn state_from_canonical_json(text: &str) -> CoreResult<GameState> {
    let doc: CanonicalDoc =
        serde_json::from_str(text).map_err(|e| CoreError::new(format!("canonical json: {e}")))?;
    state_from_doc(doc)
}

pub fn state_from_canonical_value(value: Value) -> CoreResult<GameState> {
    let doc: CanonicalDoc = serde_json::from_value(value)
        .map_err(|e| CoreError::new(format!("canonical state: {e}")))?;
    state_from_doc(doc)
}

fn state_from_doc(doc: CanonicalDoc) -> CoreResult<GameState> {
    if doc.schema_version != SCHEMA_VERSION {
        return Err(CoreError::new(format!(
            "unsupported schema_version {}",
            doc.schema_version
        )));
    }
    let width =
        i32::try_from(doc.width).map_err(|_| CoreError::new("width does not fit in i32"))?;
    let height =
        i32::try_from(doc.height).map_err(|_| CoreError::new("height does not fit in i32"))?;
    let mut map = Map::blank(width, height)?;
    for cell in &doc.terrain_land {
        map.set_terrain(cell[0], cell[1], crate::map::LAND)?;
    }

    let mut ports = IndexMap::new();
    for port in doc.ports {
        let pos = (port.position[0], port.position[1]);
        if ports.contains_key(&pos) {
            return Err(CoreError::new(format!("duplicate port at {pos:?}")));
        }
        let owner = i8::try_from(port.owner)
            .map_err(|_| CoreError::new(format!("port owner {} does not fit in i8", port.owner)))?;
        map.set_port_owner(pos.0, pos.1, owner)?;
        map.port_positions.push(pos);
        ports.insert(
            pos,
            Port {
                position: pos,
                owner: port.owner,
                is_home: port.is_home,
                stockpile: port.stockpile,
                locked_this_turn: port.locked_this_turn,
            },
        );
    }

    let mut players = Vec::with_capacity(doc.players.len());
    let mut seen_ids = BTreeSet::new();
    for player in doc.players {
        if !seen_ids.insert(player.id) {
            return Err(CoreError::new(format!("duplicate player {}", player.id)));
        }
        let mut sightings = IndexMap::new();
        for s in player.ship_sightings {
            let ship_type = ShipType::from_value(&s.ship_type)?;
            sightings.insert(
                s.ship_id,
                ShipSighting {
                    ship_id: s.ship_id,
                    ship_type,
                    position: (s.position[0], s.position[1]),
                    turn_seen: s.turn_seen,
                    fresh: s.fresh,
                },
            );
        }
        let mut port_sightings = IndexMap::new();
        for s in player.port_sightings {
            let pos = (s.position[0], s.position[1]);
            port_sightings.insert(
                pos,
                PortSighting {
                    position: pos,
                    last_known_owner: s.last_known_owner,
                    is_home: s.is_home,
                    turn_seen: s.turn_seen,
                    fresh: s.fresh,
                },
            );
        }
        let mut owned_ports = BTreeSet::new();
        for port in ports.values() {
            if port.owner == player.id {
                owned_ports.insert(port.position);
            }
        }
        players.push(Player {
            id: player.id,
            home_port: (player.home_port[0], player.home_port[1]),
            cash: player.cash,
            owned_ship_ids: BTreeSet::new(),
            owned_port_positions: owned_ports,
            sightings,
            port_sightings,
        });
    }
    players.sort_by_key(|p| p.id);

    let mut ships = IndexMap::new();
    for ship in doc.ships {
        if ships.contains_key(&ship.id) {
            return Err(CoreError::new(format!("duplicate ship {}", ship.id)));
        }
        let ship_type = ShipType::from_value(&ship.ship_type)?;
        let position = (ship.position[0], ship.position[1]);
        map.place_ship(ship.id, position.0, position.1)?;
        if let Some(player) = players.iter_mut().find(|p| p.id == ship.owner) {
            player.owned_ship_ids.insert(ship.id);
        }
        ships.insert(
            ship.id,
            Ship {
                id: ship.id,
                ship_type,
                stats: tables().stats(ship_type),
                owner: ship.owner,
                position,
                tiles_moved_this_turn: ship.tiles_moved_this_turn,
                has_attacked: ship.has_attacked,
                cargo: ship.cargo,
                bought_this_turn: ship.bought_this_turn,
                built_turn: ship.built_turn,
            },
        );
    }

    Ok(GameState {
        map,
        players,
        ships,
        ports,
        current_player: doc.current_player,
        turn: doc.turn,
        winner: doc.winner,
        kill_curve_k: doc.kill_curve_k,
        next_ship_id: doc.next_ship_id,
    })
}

#[derive(Debug, Deserialize)]
struct CanonicalDoc {
    schema_version: i64,
    width: i64,
    height: i64,
    turn: i64,
    current_player: i64,
    winner: Option<i64>,
    kill_curve_k: f64,
    next_ship_id: i64,
    terrain_land: Vec<[i32; 2]>,
    players: Vec<CanonicalPlayer>,
    ports: Vec<CanonicalPort>,
    ships: Vec<CanonicalShip>,
}

#[derive(Debug, Deserialize)]
struct CanonicalPlayer {
    id: i64,
    home_port: [i32; 2],
    cash: i64,
    ship_sightings: Vec<CanonicalShipSighting>,
    port_sightings: Vec<CanonicalPortSighting>,
}

#[derive(Debug, Deserialize)]
struct CanonicalShipSighting {
    ship_id: i64,
    #[serde(rename = "type")]
    ship_type: String,
    position: [i32; 2],
    turn_seen: i64,
    fresh: bool,
}

#[derive(Debug, Deserialize)]
struct CanonicalPortSighting {
    position: [i32; 2],
    last_known_owner: i64,
    is_home: bool,
    turn_seen: i64,
    fresh: bool,
}

#[derive(Debug, Deserialize)]
struct CanonicalPort {
    position: [i32; 2],
    owner: i64,
    is_home: bool,
    stockpile: i64,
    locked_this_turn: bool,
}

#[derive(Debug, Deserialize)]
struct CanonicalShip {
    id: i64,
    owner: i64,
    #[serde(rename = "type")]
    ship_type: String,
    position: [i32; 2],
    tiles_moved_this_turn: i64,
    has_attacked: bool,
    cargo: i64,
    bought_this_turn: bool,
    built_turn: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;

    const TINY_MAP: &str =
        r#"{"width":4,"height":3,"land":[[0,0],[1,1],[3,2]],"ports":{"0":[[0,0]],"1":[[3,2]]}}"#;

    fn tiny_json() -> String {
        let map = Map::load_json(TINY_MAP).unwrap();
        let state = GameState::reset_from_map(map, 2.0).unwrap();
        canonical_json(&state)
    }

    #[test]
    fn reset_json_matches_hand_built_python_document() {
        // Hand-built to the schema in docs/golden_tests.md. Python parity is
        // checked again in tests/test_rust_parity.py.
        let expected = concat!(
            r#"{"current_player":0,"height":3,"kill_curve_k":2.0,"next_ship_id":0,"players":["#,
            r#"{"cash":250,"home_port":[0,0],"id":0,"port_sightings":[{"fresh":true,"is_home":true,"last_known_owner":1,"position":[3,2],"turn_seen":0}],"ship_sightings":[]},"#,
            r#"{"cash":250,"home_port":[3,2],"id":1,"port_sightings":[{"fresh":true,"is_home":true,"last_known_owner":0,"position":[0,0],"turn_seen":0}],"ship_sightings":[]}"#,
            r#"],"ports":["#,
            r#"{"is_home":true,"locked_this_turn":false,"owner":0,"position":[0,0],"stockpile":0},"#,
            r#"{"is_home":true,"locked_this_turn":false,"owner":1,"position":[3,2],"stockpile":0}"#,
            r#"],"schema_version":1,"ships":[],"terrain_land":[[0,0],[1,1],[3,2]],"turn":0,"width":4,"winner":null}"#,
        );
        let json = tiny_json();
        assert_eq!(json, expected);
        assert!(!json.contains(' '));
        assert!(!json.contains('\n'));
    }

    #[test]
    fn round_trip_resorts_shuffled_arrays() {
        let original = tiny_json();
        let mut value: Value = serde_json::from_str(&original).unwrap();
        // Reverse terrain and ports; emit must sort them again.
        value["terrain_land"].as_array_mut().unwrap().reverse();
        value["ports"].as_array_mut().unwrap().reverse();
        let state = state_from_canonical_value(value).unwrap();
        assert_eq!(canonical_json(&state), original);
        assert_eq!(state_hash(&state), hash_canonical_json(&original));
    }

    #[test]
    fn kill_curve_four_uses_python_float_spelling() {
        let map = Map::load_json(TINY_MAP).unwrap();
        let state = GameState::reset_from_map(map, 4.0).unwrap();
        let json = canonical_json(&state);
        assert!(json.contains("\"kill_curve_k\":4.0"), "{json}");
        let parsed = state_from_canonical_json(&json).unwrap();
        assert_eq!(canonical_json(&parsed), json);
    }
}
