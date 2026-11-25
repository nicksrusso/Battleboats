//! Battleboats engine core.
//!
//! Python (`battleboats.core.gameEngine`) is the reference implementation.
//! This crate owns the schema_version 1 state, its canonical JSON / SHA-256,
//! compact actions, and map / coastline queries. Rule steps (movement, combat,
//! economy) land in later issues and must keep this hash stable.

pub mod action;
pub mod canonical;
pub mod error;
pub mod jsonfmt;
pub mod map;
pub mod ship;
pub mod state;
pub mod tables;

pub use action::{decode_json, Action};
pub use canonical::{canonical_json, hash_canonical_json, state_from_canonical_json, state_hash};
pub use error::{CoreError, CoreResult};
pub use jsonfmt::format_python_json_float;
pub use map::Map;
pub use ship::{ShipStats, ShipType, SpotKey};
pub use state::{
    Engine, GameState, CASH_PER_MATERIAL, MERCHANT_CAPACITY, PORT_PRODUCTION, SCHEMA_VERSION,
    STARTING_CASH,
};
pub use tables::tables;
