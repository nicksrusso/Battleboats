//! Ship types in the same order as `battleboats.core.shipyard.ship_type.ShipType`.

use crate::error::{CoreError, CoreResult};

pub const SHIP_TYPE_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShipType {
    Carrier = 0,
    Battleship = 1,
    Cruiser = 2,
    Destroyer = 3,
    Submarine = 4,
    Merchant = 5,
    Landing = 6,
    Builder = 7,
}

impl ShipType {
    pub const ALL: [ShipType; SHIP_TYPE_COUNT] = [
        ShipType::Carrier,
        ShipType::Battleship,
        ShipType::Cruiser,
        ShipType::Destroyer,
        ShipType::Submarine,
        ShipType::Merchant,
        ShipType::Landing,
        ShipType::Builder,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// Canonical JSON / action value (`"Destroyer"`).
    pub fn value(self) -> &'static str {
        match self {
            ShipType::Carrier => "Carrier",
            ShipType::Battleship => "Battleship",
            ShipType::Cruiser => "Cruiser",
            ShipType::Destroyer => "Destroyer",
            ShipType::Submarine => "Submarine",
            ShipType::Merchant => "Merchant",
            ShipType::Landing => "Landing",
            ShipType::Builder => "Builder",
        }
    }

    /// Python enum member name (`"DESTROYER"`), used by scenario ship records.
    pub fn enum_name(self) -> &'static str {
        match self {
            ShipType::Carrier => "CARRIER",
            ShipType::Battleship => "BATTLESHIP",
            ShipType::Cruiser => "CRUISER",
            ShipType::Destroyer => "DESTROYER",
            ShipType::Submarine => "SUBMARINE",
            ShipType::Merchant => "MERCHANT",
            ShipType::Landing => "LANDING",
            ShipType::Builder => "BUILDER",
        }
    }

    pub fn from_value(value: &str) -> CoreResult<Self> {
        Self::ALL
            .into_iter()
            .find(|t| t.value() == value)
            .ok_or_else(|| CoreError::new(format!("unknown ship type value {value:?}")))
    }

    pub fn from_enum_name(name: &str) -> CoreResult<Self> {
        Self::ALL
            .into_iter()
            .find(|t| t.enum_name() == name)
            .ok_or_else(|| CoreError::new(format!("unknown ship type name {name:?}")))
    }
}

/// Spotter or target in `visibilityMatrix.csv`. Ports are not ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpotKey {
    Ship(ShipType),
    Port,
}

impl SpotKey {
    pub const COUNT: usize = SHIP_TYPE_COUNT + 1;

    pub fn index(self) -> usize {
        match self {
            SpotKey::Ship(t) => t.index(),
            SpotKey::Port => SHIP_TYPE_COUNT,
        }
    }

    pub fn parse(name: &str) -> CoreResult<Self> {
        let trimmed = name.trim();
        if trimmed.eq_ignore_ascii_case("port") {
            return Ok(SpotKey::Port);
        }
        Ok(SpotKey::Ship(ShipType::from_value(&title_case(trimmed))?))
    }
}

/// Python `str.title()` for the ASCII labels in the balance CSVs.
pub fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut new_word = true;
    for c in s.chars() {
        if c.is_ascii_alphabetic() {
            if new_word {
                out.push(c.to_ascii_uppercase());
                new_word = false;
            } else {
                out.push(c.to_ascii_lowercase());
            }
        } else {
            out.push(c);
            new_word = true;
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShipStats {
    pub speed: i64,
    pub cost: i64,
    pub strength: f64,
    pub attack_range: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_follow_python_enum_order() {
        let values: Vec<_> = ShipType::ALL.iter().map(|t| t.value()).collect();
        assert_eq!(
            values,
            vec![
                "Carrier",
                "Battleship",
                "Cruiser",
                "Destroyer",
                "Submarine",
                "Merchant",
                "Landing",
                "Builder",
            ]
        );
    }
}
