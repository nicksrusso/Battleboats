//! Map JSON loading and the spatial queries `battleboats.core.map.Map` exposes.
//!
//! Terrain is stored in the same C-order as the Python numpy grid of shape
//! `(width, height)`: index `x * height + y`.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::error::{CoreError, CoreResult};

pub const WATER: u8 = 0;
pub const LAND: u8 = 1;
pub const NO_SHIP: i64 = -1;
pub const NO_OWNER: i8 = -1;

#[derive(Debug, Clone)]
pub struct Map {
    pub width: i32,
    pub height: i32,
    pub seed: Option<i64>,
    terrain: Vec<u8>,
    port_owner: Vec<i8>,
    ship_at: Vec<i64>,
    /// Port coordinates in JSON object order, then list order. The first port
    /// of each owner is that player's home (`gameEngine._infer_home_ports`).
    pub port_positions: Vec<(i32, i32)>,
    coastline_cache: Option<Vec<(i32, i32)>>,
}

impl Map {
    pub fn load_json(text: &str) -> CoreResult<Self> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| CoreError::new(format!("map json: {e}")))?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &Value) -> CoreResult<Self> {
        let width = required_i32(value, "width")?;
        let height = required_i32(value, "height")?;
        if width <= 0 || height <= 0 {
            return Err(CoreError::new(format!(
                "map must be at least 1x1, got {width}x{height}"
            )));
        }
        let cells = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| CoreError::new("map dimensions overflow"))?;
        let mut map = Self {
            width,
            height,
            seed: value.get("seed").and_then(Value::as_i64),
            terrain: vec![WATER; cells],
            port_owner: vec![NO_OWNER; cells],
            ship_at: vec![NO_SHIP; cells],
            port_positions: Vec::new(),
            coastline_cache: None,
        };
        let land = value
            .get("land")
            .and_then(Value::as_array)
            .ok_or_else(|| CoreError::new("map json missing land array"))?;
        for cell in land {
            let (x, y) = pair(cell, "land")?;
            map.set_terrain(x, y, LAND)?;
        }
        let ports = value
            .get("ports")
            .and_then(Value::as_object)
            .ok_or_else(|| CoreError::new("map json missing ports object"))?;
        // `preserve_order` keeps the JSON object order Python's json.load uses.
        for (owner_key, positions) in ports {
            let owner: i8 = owner_key.parse().map_err(|_| {
                CoreError::new(format!("map port owner key is not an int: {owner_key:?}"))
            })?;
            let positions = positions
                .as_array()
                .ok_or_else(|| CoreError::new(format!("ports[{owner_key}] is not a list")))?;
            for cell in positions {
                let (x, y) = pair(cell, "port")?;
                map.set_port_owner(x, y, owner)?;
                map.port_positions.push((x, y));
            }
        }
        Ok(map)
    }

    /// Empty water map used when rebuilding a grid from canonical state.
    pub fn blank(width: i32, height: i32) -> CoreResult<Self> {
        if width <= 0 || height <= 0 {
            return Err(CoreError::new(format!(
                "map must be at least 1x1, got {width}x{height}"
            )));
        }
        let cells = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| CoreError::new("map dimensions overflow"))?;
        Ok(Self {
            width,
            height,
            seed: None,
            terrain: vec![WATER; cells],
            port_owner: vec![NO_OWNER; cells],
            ship_at: vec![NO_SHIP; cells],
            port_positions: Vec::new(),
            coastline_cache: None,
        })
    }

    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.width && y < self.height
    }

    pub fn is_water(&self, x: i32, y: i32) -> CoreResult<bool> {
        Ok(self.terrain(x, y)? == WATER)
    }

    pub fn is_land(&self, x: i32, y: i32) -> CoreResult<bool> {
        Ok(self.terrain(x, y)? == LAND)
    }

    pub fn is_port(&self, x: i32, y: i32) -> CoreResult<bool> {
        Ok(self.port_owner_at(x, y)? != NO_OWNER)
    }

    pub fn port_owner_at(&self, x: i32, y: i32) -> CoreResult<i8> {
        let idx = self.index(x, y)?;
        Ok(self.port_owner[idx])
    }

    pub fn ship_id_at(&self, x: i32, y: i32) -> CoreResult<i64> {
        let idx = self.index(x, y)?;
        Ok(self.ship_at[idx])
    }

    pub fn is_occupied(&self, x: i32, y: i32) -> CoreResult<bool> {
        Ok(self.ship_id_at(x, y)? != NO_SHIP)
    }

    pub fn terrain(&self, x: i32, y: i32) -> CoreResult<u8> {
        let idx = self.index(x, y)?;
        Ok(self.terrain[idx])
    }

    pub fn set_terrain(&mut self, x: i32, y: i32, value: u8) -> CoreResult<()> {
        let idx = self.index(x, y)?;
        self.terrain[idx] = value;
        self.coastline_cache = None;
        Ok(())
    }

    pub fn set_port_owner(&mut self, x: i32, y: i32, owner: i8) -> CoreResult<()> {
        let idx = self.index(x, y)?;
        self.port_owner[idx] = owner;
        Ok(())
    }

    pub fn place_ship(&mut self, ship_id: i64, x: i32, y: i32) -> CoreResult<()> {
        let idx = self.index(x, y)?;
        self.ship_at[idx] = ship_id;
        Ok(())
    }

    pub fn manhattan(a: (i32, i32), b: (i32, i32)) -> i32 {
        (a.0 - b.0).abs() + (a.1 - b.1).abs()
    }

    /// In-bounds tiles within Manhattan distance `r`, in the same order as
    /// `Map.tiles_within` (increasing x, then increasing y).
    pub fn tiles_within(&self, cx: i32, cy: i32, r: i32) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        if r < 0 {
            return out;
        }
        let xmin = 0.max(cx - r);
        let xmax = (self.width - 1).min(cx + r);
        for x in xmin..=xmax {
            let dy = r - (x - cx).abs();
            let ymin = 0.max(cy - dy);
            let ymax = (self.height - 1).min(cy + dy);
            for y in ymin..=ymax {
                out.push((x, y));
            }
        }
        out
    }

    /// Land tiles orthogonally adjacent to water, ports included.
    ///
    /// Same scan as `battleboats.envs.observation._coastline_scan`: x major,
    /// then y. The cache is dropped if terrain changes.
    pub fn coastline(&mut self) -> &[(i32, i32)] {
        if self.coastline_cache.is_none() {
            self.coastline_cache = Some(self.scan_coastline());
        }
        self.coastline_cache.as_ref().unwrap()
    }

    pub fn coastline_owned(&self) -> Vec<(i32, i32)> {
        self.scan_coastline()
    }

    fn scan_coastline(&self) -> Vec<(i32, i32)> {
        const NEIGHBORS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let mut out = Vec::new();
        for x in 0..self.width {
            for y in 0..self.height {
                if self.terrain[self.index(x, y).unwrap()] != LAND {
                    continue;
                }
                for (dx, dy) in NEIGHBORS {
                    let nx = x + dx;
                    let ny = y + dy;
                    if self.in_bounds(nx, ny) && self.terrain[self.index(nx, ny).unwrap()] == WATER
                    {
                        out.push((x, y));
                        break;
                    }
                }
            }
        }
        out
    }

    /// First port listed for owners 0 and 1. Matches `_infer_home_ports`.
    pub fn infer_home_ports(&self) -> CoreResult<[(i32, i32); 2]> {
        let mut homes = [None, None];
        for &pos in &self.port_positions {
            let owner = self.port_owner_at(pos.0, pos.1)?;
            if (owner == 0 || owner == 1) && homes[owner as usize].is_none() {
                homes[owner as usize] = Some(pos);
            }
        }
        match (homes[0], homes[1]) {
            (Some(p0), Some(p1)) => Ok([p0, p1]),
            _ => Err(CoreError::new(
                "map is missing a home port for player 0 or player 1",
            )),
        }
    }

    /// Land cells in lexicographic `(x, y)` order. This is `terrain_land`.
    pub fn terrain_land(&self) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        for x in 0..self.width {
            for y in 0..self.height {
                if self.terrain[self.index(x, y).unwrap()] == LAND {
                    out.push((x, y));
                }
            }
        }
        out
    }

    pub fn clear_ships(&mut self) {
        self.ship_at.fill(NO_SHIP);
    }

    fn index(&self, x: i32, y: i32) -> CoreResult<usize> {
        if !self.in_bounds(x, y) {
            return Err(CoreError::new(format!(
                "coordinate ({x}, {y}) is outside {0}x{1}",
                self.width, self.height
            )));
        }
        Ok((x as usize) * (self.height as usize) + (y as usize))
    }
}

/// Land coordinates as a set, used when tests compare grids.
pub fn land_set(cells: &[(i32, i32)]) -> BTreeSet<(i32, i32)> {
    cells.iter().copied().collect()
}

fn required_i32(value: &Value, key: &str) -> CoreResult<i32> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| CoreError::new(format!("map json missing integer {key}")))
}

fn pair(value: &Value, what: &str) -> CoreResult<(i32, i32)> {
    let arr = value
        .as_array()
        .ok_or_else(|| CoreError::new(format!("{what} cell is not a pair")))?;
    if arr.len() < 2 {
        return Err(CoreError::new(format!("{what} cell is not a pair")));
    }
    let x = arr[0]
        .as_i64()
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| CoreError::new(format!("{what} x is not an int")))?;
    let y = arr[1]
        .as_i64()
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| CoreError::new(format!("{what} y is not an int")))?;
    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Map {
        Map::load_json(
            r#"{"width":4,"height":3,"seed":7,"land":[[3,2],[0,0],[1,1]],"ports":{"1":[[3,2]],"0":[[0,0]]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn load_preserves_port_list_order_not_key_sort() {
        // Key "1" is written first. preserve_order must not sort it ahead of
        // the array order inside each owner, and home is the first cell of
        // that owner, not the lexicographic port.
        let map = sample();
        assert_eq!(map.port_positions, vec![(3, 2), (0, 0)]);
        assert_eq!(map.infer_home_ports().unwrap(), [(0, 0), (3, 2)]);
        assert_eq!(map.seed, Some(7));
        assert_eq!(map.terrain_land(), vec![(0, 0), (1, 1), (3, 2)]);
        assert!(map.is_land(1, 1).unwrap());
        assert!(map.is_water(1, 0).unwrap());
        assert!(map.is_port(0, 0).unwrap());
        assert!(!map.is_port(1, 1).unwrap());
        assert!(!map.in_bounds(-1, 0));
        assert!(map.is_water(-1, 0).is_err());
        assert!(map.is_water(0, 3).is_err());
    }

    #[test]
    fn coastline_and_tiles_within_match_python_scan_order() {
        let mut map = sample();
        assert_eq!(map.coastline(), &[(0, 0), (1, 1), (3, 2)]);
        assert_eq!(map.tiles_within(0, 0, 1), vec![(0, 0), (0, 1), (1, 0)]);
        assert_eq!(Map::manhattan((0, 0), (3, 2)), 5);
    }
}
