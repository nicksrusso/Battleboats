//! Balance tables compiled from the same CSVs the Python engine loads at import.

use std::sync::OnceLock;

use crate::error::{CoreError, CoreResult};
use crate::ship::{title_case, ShipStats, ShipType, SpotKey, SHIP_TYPE_COUNT};

const SHIP_TYPES_CSV: &str = include_str!("../../../battleboats/core/config/shipTypes.csv");
const ATTACK_MODIFIERS_CSV: &str =
    include_str!("../../../battleboats/core/config/attackModifiers.csv");
const VISIBILITY_CSV: &str = include_str!("../../../battleboats/core/config/visibilityMatrix.csv");

#[derive(Debug, Clone)]
pub struct Tables {
    pub base_stats: [ShipStats; SHIP_TYPE_COUNT],
    /// Missing attacker/defender pairs are 1.0, matching `attack_modifier`.
    pub attack: [[f64; SHIP_TYPE_COUNT]; SHIP_TYPE_COUNT],
    pub spot: [[f64; SpotKey::COUNT]; SpotKey::COUNT],
}

pub fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        Tables::load().unwrap_or_else(|err| panic!("failed to load ship tables: {err}"))
    })
}

impl Tables {
    fn load() -> CoreResult<Self> {
        let mut base_stats = [ShipStats {
            speed: 0,
            cost: 0,
            strength: 0.0,
            attack_range: 0,
        }; SHIP_TYPE_COUNT];
        let mut seen = [false; SHIP_TYPE_COUNT];
        let mut reader = csv::Reader::from_reader(SHIP_TYPES_CSV.as_bytes());
        for row in reader.records() {
            let row = row.map_err(|e| CoreError::new(format!("shipTypes.csv: {e}")))?;
            let name = row
                .get(0)
                .ok_or_else(|| CoreError::new("shipTypes.csv: short row"))?;
            let ship_type = ShipType::from_value(&title_case(name.trim()))?;
            let stats = ShipStats {
                cost: parse_i64(row.get(1), "Cost")?,
                strength: parse_f64(row.get(2), "Strength")?,
                attack_range: parse_i64(row.get(3), "attack range")?,
                speed: parse_i64(row.get(4), "speed")?,
            };
            base_stats[ship_type.index()] = stats;
            seen[ship_type.index()] = true;
        }
        if let Some(missing) = seen.iter().position(|ok| !ok) {
            return Err(CoreError::new(format!(
                "shipTypes.csv is missing {}",
                ShipType::ALL[missing].value()
            )));
        }

        let mut attack = [[1.0; SHIP_TYPE_COUNT]; SHIP_TYPE_COUNT];
        let mut attack_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(ATTACK_MODIFIERS_CSV.as_bytes());
        let mut attack_rows = attack_reader.records();
        let header = attack_rows
            .next()
            .ok_or_else(|| CoreError::new("attackModifiers.csv: empty"))?
            .map_err(|e| CoreError::new(format!("attackModifiers.csv: {e}")))?;
        let defenders: Vec<ShipType> = header
            .iter()
            .skip(1)
            .filter(|c| !c.trim().is_empty())
            .map(|c| ShipType::from_value(&title_case(c.trim())))
            .collect::<CoreResult<_>>()?;
        for row in attack_rows {
            let row = row.map_err(|e| CoreError::new(format!("attackModifiers.csv: {e}")))?;
            let label = row.get(0).unwrap_or("").trim();
            if label.is_empty() {
                continue;
            }
            let attacker = ShipType::from_value(&title_case(label))?;
            for (defender, cell) in defenders.iter().zip(row.iter().skip(1)) {
                if cell.trim().is_empty() {
                    continue;
                }
                attack[attacker.index()][defender.index()] = parse_f64(Some(cell), "modifier")?;
            }
        }

        let mut spot = [[0.0; SpotKey::COUNT]; SpotKey::COUNT];
        let mut spot_seen = [[false; SpotKey::COUNT]; SpotKey::COUNT];
        let mut spot_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(VISIBILITY_CSV.as_bytes());
        let mut spot_rows = spot_reader.records();
        let header = spot_rows
            .next()
            .ok_or_else(|| CoreError::new("visibilityMatrix.csv: empty"))?
            .map_err(|e| CoreError::new(format!("visibilityMatrix.csv: {e}")))?;
        let targets: Vec<SpotKey> = header
            .iter()
            .skip(1)
            .filter(|c| !c.trim().is_empty())
            .map(SpotKey::parse)
            .collect::<CoreResult<_>>()?;
        for row in spot_rows {
            let row = row.map_err(|e| CoreError::new(format!("visibilityMatrix.csv: {e}")))?;
            let label = row.get(0).unwrap_or("").trim();
            if label.is_empty() {
                continue;
            }
            let spotter = SpotKey::parse(label)?;
            for (target, cell) in targets.iter().zip(row.iter().skip(1)) {
                if cell.trim().is_empty() {
                    continue;
                }
                let distance = parse_f64(Some(cell), "spot distance")?;
                if distance < 0.0 {
                    return Err(CoreError::new(format!(
                        "visibilityMatrix.csv: negative distance {label}->{cell}"
                    )));
                }
                spot[spotter.index()][target.index()] = distance;
                spot_seen[spotter.index()][target.index()] = true;
            }
        }
        for (a, row) in spot_seen.iter().enumerate() {
            for (b, seen) in row.iter().enumerate() {
                if !seen {
                    return Err(CoreError::new(format!(
                        "visibilityMatrix.csv is missing spotting pair {a}->{b}"
                    )));
                }
            }
        }

        Ok(Tables {
            base_stats,
            attack,
            spot,
        })
    }

    pub fn stats(&self, ship_type: ShipType) -> ShipStats {
        self.base_stats[ship_type.index()]
    }

    pub fn attack_modifier(&self, attacker: ShipType, defender: ShipType) -> f64 {
        self.attack[attacker.index()][defender.index()]
    }

    pub fn spot_distance(&self, spotter: SpotKey, target: SpotKey) -> f64 {
        self.spot[spotter.index()][target.index()]
    }
}

fn parse_i64(cell: Option<&str>, field: &str) -> CoreResult<i64> {
    let text = cell
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::new(format!("missing {field}")))?;
    text.parse::<i64>()
        .map_err(|_| CoreError::new(format!("bad integer {field}: {text}")))
}

fn parse_f64(cell: Option<&str>, field: &str) -> CoreResult<f64> {
    let text = cell
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CoreError::new(format!("missing {field}")))?;
    text.parse::<f64>()
        .map_err(|_| CoreError::new(format!("bad float {field}: {text}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_stats_match_the_csv() {
        let t = tables();
        let dd = t.stats(ShipType::Destroyer);
        assert_eq!(dd.cost, 100);
        assert_eq!(dd.strength, 50.0);
        assert_eq!(dd.attack_range, 3);
        assert_eq!(dd.speed, 5);
        let sub = t.stats(ShipType::Submarine);
        assert_eq!(sub.cost, 125);
        assert_eq!(sub.strength, 50.0);
        let landing = t.stats(ShipType::Landing);
        assert_eq!(landing.cost, 250);
        assert_eq!(landing.strength, 0.0);
        assert_eq!(landing.attack_range, 0);
        assert_eq!(t.stats(ShipType::Carrier).speed, 6);
        assert_eq!(t.stats(ShipType::Merchant).cost, 100);
    }

    #[test]
    fn modifiers_and_spotting_match_the_csvs() {
        let t = tables();
        assert_eq!(
            t.attack_modifier(ShipType::Destroyer, ShipType::Submarine),
            2.0
        );
        assert_eq!(
            t.attack_modifier(ShipType::Submarine, ShipType::Battleship),
            4.0
        );
        assert_eq!(
            t.attack_modifier(ShipType::Landing, ShipType::Destroyer),
            1.0
        );
        assert_eq!(
            t.spot_distance(
                SpotKey::Ship(ShipType::Destroyer),
                SpotKey::Ship(ShipType::Submarine)
            ),
            3.0
        );
        assert_eq!(
            t.spot_distance(SpotKey::Port, SpotKey::Ship(ShipType::Submarine)),
            1.0
        );
        assert_eq!(t.spot_distance(SpotKey::Port, SpotKey::Port), 16.0);
    }
}
