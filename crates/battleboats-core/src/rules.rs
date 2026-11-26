//! Rule application. Mirrors `gameEngine.step` and `enumerate_legal`.
//!
//! Illegal actions are silent no-ops, then sightings refresh. A terminal
//! engine ignores the action and does not refresh. Combat uses `scripted_kill`
//! when the golden replay recorded an outcome for this step; otherwise a
//! uniform draw from the engine RNG.

use rand::Rng;

use crate::action::Action;
use crate::error::CoreResult;
use crate::map::Map;
use crate::ship::{ShipType, SpotKey};
use crate::state::{GameState, Port, CASH_PER_MATERIAL, MERCHANT_CAPACITY, PORT_PRODUCTION};
use crate::tables::tables;

const CARDINALS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// `BASE_STATS` insertion order (shipTypes.csv), which is the buy order.
const BUILD_ORDER: [ShipType; 8] = [
    ShipType::Carrier,
    ShipType::Battleship,
    ShipType::Cruiser,
    ShipType::Destroyer,
    ShipType::Submarine,
    ShipType::Landing,
    ShipType::Merchant,
    ShipType::Builder,
];

impl GameState {
    pub fn apply(
        &mut self,
        action: &Action,
        rng: &mut impl Rng,
        scripted_kill: Option<bool>,
    ) -> CoreResult<()> {
        if self.winner.is_some() {
            return Ok(());
        }
        match *action {
            Action::Move { ship_id, x, y } => self.do_move(ship_id, (x, y))?,
            Action::Attack {
                attacker_id,
                target_id,
            } => self.do_attack(attacker_id, target_id, rng, scripted_kill)?,
            Action::BuildShip {
                port_x,
                port_y,
                spawn_x,
                spawn_y,
                ship_type,
            } => self.do_build_ship((port_x, port_y), (spawn_x, spawn_y), ship_type)?,
            Action::BuildPort {
                builder_ship_id,
                x,
                y,
            } => self.do_build_port(builder_ship_id, (x, y))?,
            Action::CapturePort {
                landing_ship_id,
                x,
                y,
            } => self.do_capture(landing_ship_id, (x, y))?,
            Action::MerchantLoad { merchant_id, x, y } => {
                self.do_merchant_load(merchant_id, (x, y))?
            }
            Action::MerchantUnload { merchant_id, x, y } => {
                self.do_merchant_unload(merchant_id, (x, y))?
            }
            Action::EndTurn => self.do_end_turn(),
        }
        self.refresh_sightings()?;
        Ok(())
    }

    pub fn enumerate_legal(&self, player_id: i64) -> Vec<Action> {
        let mut actions = Vec::new();
        if self.winner.is_some() || player_id != self.current_player {
            return actions;
        }
        let player = &self.players[player_id as usize];
        let opp = &self.players[(1 - player_id) as usize];
        let opp_ids: Vec<i64> = opp.owned_ship_ids.iter().copied().collect();
        let ship_ids: Vec<i64> = player.owned_ship_ids.iter().copied().collect();
        let book = tables();

        for sid in ship_ids {
            let ship = &self.ships[&sid];
            if ship.bought_this_turn {
                continue;
            }
            let (sx, sy) = ship.position;
            let can_move = ship.tiles_moved_this_turn < ship.stats.speed;
            let is_builder = ship.ship_type == ShipType::Builder;
            let is_landing = ship.ship_type == ShipType::Landing;
            let is_merchant = ship.ship_type == ShipType::Merchant;
            for (dx, dy) in CARDINALS {
                let neighbor = (sx + dx, sy + dy);
                if !self.map.in_bounds(neighbor.0, neighbor.1) {
                    continue;
                }
                let water = self.map.is_water(neighbor.0, neighbor.1).unwrap_or(false);
                let occupied = self.map.is_occupied(neighbor.0, neighbor.1).unwrap_or(true);
                if can_move && water && !occupied {
                    actions.push(Action::Move {
                        ship_id: sid,
                        x: neighbor.0,
                        y: neighbor.1,
                    });
                }
                if !self.map.is_land(neighbor.0, neighbor.1).unwrap_or(false) {
                    continue;
                }
                if self.map.is_port(neighbor.0, neighbor.1).unwrap_or(false) {
                    let neighbor_owner =
                        i64::from(self.map.port_owner_at(neighbor.0, neighbor.1).unwrap_or(-1));
                    if is_landing && neighbor_owner != player_id {
                        actions.push(Action::CapturePort {
                            landing_ship_id: sid,
                            x: neighbor.0,
                            y: neighbor.1,
                        });
                    }
                    if is_merchant && neighbor_owner == player_id {
                        if ship.cargo > 0 && neighbor == player.home_port {
                            actions.push(Action::MerchantUnload {
                                merchant_id: sid,
                                x: neighbor.0,
                                y: neighbor.1,
                            });
                        }
                        let stock = self.ports.get(&neighbor).map(|p| p.stockpile).unwrap_or(0);
                        if ship.cargo < MERCHANT_CAPACITY
                            && neighbor != player.home_port
                            && stock > 0
                        {
                            actions.push(Action::MerchantLoad {
                                merchant_id: sid,
                                x: neighbor.0,
                                y: neighbor.1,
                            });
                        }
                    }
                } else if is_builder {
                    actions.push(Action::BuildPort {
                        builder_ship_id: sid,
                        x: neighbor.0,
                        y: neighbor.1,
                    });
                }
            }

            if !ship.has_attacked && ship.stats.attack_range > 0 {
                for opp_sid in &opp_ids {
                    let target = &self.ships[opp_sid];
                    let distance = Map::manhattan(ship.position, target.position);
                    if i64::from(distance) > ship.stats.attack_range {
                        continue;
                    }
                    let reach = book.spot_distance(
                        SpotKey::Ship(ship.ship_type),
                        SpotKey::Ship(target.ship_type),
                    );
                    if f64::from(distance) > reach {
                        continue;
                    }
                    actions.push(Action::Attack {
                        attacker_id: sid,
                        target_id: *opp_sid,
                    });
                }
            }
        }

        let affordable: Vec<ShipType> = BUILD_ORDER
            .into_iter()
            .filter(|t| player.cash >= book.stats(*t).cost)
            .collect();
        if !affordable.is_empty() {
            let opp_home = opp.home_port;
            let ports: Vec<(i32, i32)> = player.owned_port_positions.iter().copied().collect();
            for port in ports {
                if self
                    .ports
                    .get(&port)
                    .map(|p| p.locked_this_turn)
                    .unwrap_or(true)
                {
                    continue;
                }
                let mut spawns = Vec::new();
                for (dx, dy) in CARDINALS {
                    let spawn = (port.0 + dx, port.1 + dy);
                    if !self.map.in_bounds(spawn.0, spawn.1) {
                        continue;
                    }
                    if !self.map.is_water(spawn.0, spawn.1).unwrap_or(false) {
                        continue;
                    }
                    if self.map.is_occupied(spawn.0, spawn.1).unwrap_or(true) {
                        continue;
                    }
                    spawns.push(spawn);
                }
                let Some(canonical) = spawns
                    .into_iter()
                    .min_by_key(|s| (Map::manhattan(*s, opp_home), s.0, s.1))
                else {
                    continue;
                };
                for ship_type in &affordable {
                    actions.push(Action::BuildShip {
                        port_x: port.0,
                        port_y: port.1,
                        spawn_x: canonical.0,
                        spawn_y: canonical.1,
                        ship_type: *ship_type,
                    });
                }
            }
        }

        actions.push(Action::EndTurn);
        actions
    }

    fn do_move(&mut self, ship_id: i64, dest: (i32, i32)) -> CoreResult<()> {
        let Some(ship) = self.ships.get(&ship_id) else {
            return Ok(());
        };
        if ship.owner != self.current_player {
            return Ok(());
        }
        if ship.bought_this_turn || ship.tiles_moved_this_turn >= ship.stats.speed {
            return Ok(());
        }
        let src = ship.position;
        if !self.map.in_bounds(dest.0, dest.1) {
            return Ok(());
        }
        if Map::manhattan(src, dest) != 1 {
            return Ok(());
        }
        if !self.map.is_water(dest.0, dest.1)? {
            return Ok(());
        }
        if self.map.is_occupied(dest.0, dest.1)? {
            return Ok(());
        }
        self.map.relocate_ship(ship_id, src, dest)?;
        let ship = self.ships.get_mut(&ship_id).unwrap();
        ship.position = dest;
        ship.tiles_moved_this_turn += 1;
        Ok(())
    }

    fn do_attack(
        &mut self,
        attacker_id: i64,
        target_id: i64,
        rng: &mut impl Rng,
        scripted_kill: Option<bool>,
    ) -> CoreResult<()> {
        let Some(attacker) = self.ships.get(&attacker_id) else {
            return Ok(());
        };
        if attacker.owner != self.current_player
            || attacker.has_attacked
            || attacker.bought_this_turn
            || attacker.stats.attack_range <= 0
        {
            return Ok(());
        }
        let attacker_pos = attacker.position;
        let attacker_type = attacker.ship_type;
        let attacker_range = attacker.stats.attack_range;
        let Some(defender) = self.ships.get(&target_id) else {
            return Ok(());
        };
        if defender.owner == self.current_player {
            return Ok(());
        }
        let defender_pos = defender.position;
        let distance = Map::manhattan(attacker_pos, defender_pos);
        if i64::from(distance) > attacker_range {
            return Ok(());
        }
        let reach = tables().spot_distance(
            SpotKey::Ship(attacker_type),
            SpotKey::Ship(defender.ship_type),
        );
        if f64::from(distance) > reach {
            return Ok(());
        }
        self.ships.get_mut(&attacker_id).unwrap().has_attacked = true;
        if self.resolve_attack(attacker_id, target_id, rng, scripted_kill) {
            self.destroy_ship(target_id)?;
        }
        Ok(())
    }

    fn resolve_attack(
        &mut self,
        attacker_id: i64,
        defender_id: i64,
        rng: &mut impl Rng,
        scripted_kill: Option<bool>,
    ) -> bool {
        if let Some(killed) = scripted_kill {
            return killed;
        }
        let attacker = &self.ships[&attacker_id];
        let defender = &self.ships[&defender_id];
        if defender.stats.strength <= 0.0 {
            return true;
        }
        let modifier = tables().attack_modifier(attacker.ship_type, defender.ship_type);
        let x = (attacker.stats.strength * modifier) / defender.stats.strength;
        let xk = x.powf(self.kill_curve_k);
        let p = xk / (1.0 + xk);
        rng.gen::<f64>() < p
    }

    fn do_build_ship(
        &mut self,
        port: (i32, i32),
        spawn: (i32, i32),
        ship_type: ShipType,
    ) -> CoreResult<()> {
        let player_id = self.current_player;
        let player = &self.players[player_id as usize];
        if !player.owned_port_positions.contains(&port) {
            return Ok(());
        }
        if self
            .ports
            .get(&port)
            .map(|p| p.locked_this_turn)
            .unwrap_or(true)
        {
            return Ok(());
        }
        if !self.map.in_bounds(spawn.0, spawn.1) {
            return Ok(());
        }
        if Map::manhattan(spawn, port) != 1 {
            return Ok(());
        }
        if !self.map.is_water(spawn.0, spawn.1)? || self.map.is_occupied(spawn.0, spawn.1)? {
            return Ok(());
        }
        let cost = tables().stats(ship_type).cost;
        if self.players[player_id as usize].cash < cost {
            return Ok(());
        }
        self.players[player_id as usize].cash -= cost;
        let id = self.spawn_ship(player_id, ship_type, spawn)?;
        let ship = self.ships.get_mut(&id).unwrap();
        ship.bought_this_turn = true;
        ship.built_turn = self.turn;
        ship.tiles_moved_this_turn = ship.stats.speed;
        ship.has_attacked = true;
        Ok(())
    }

    fn do_build_port(&mut self, builder_id: i64, target: (i32, i32)) -> CoreResult<()> {
        let Some(ship) = self.ships.get(&builder_id) else {
            return Ok(());
        };
        if ship.owner != self.current_player
            || ship.ship_type != ShipType::Builder
            || ship.bought_this_turn
        {
            return Ok(());
        }
        let origin = ship.position;
        if !self.map.in_bounds(target.0, target.1) {
            return Ok(());
        }
        if !self.map.is_land(target.0, target.1)? || self.map.is_port(target.0, target.1)? {
            return Ok(());
        }
        if Map::manhattan(origin, target) != 1 {
            return Ok(());
        }
        let owner = self.current_player;
        self.map.add_port(target.0, target.1, owner as i8)?;
        self.players[owner as usize]
            .owned_port_positions
            .insert(target);
        self.ports.insert(
            target,
            Port {
                position: target,
                owner,
                is_home: false,
                stockpile: 0,
                locked_this_turn: true,
            },
        );
        self.destroy_ship(builder_id)?;
        Ok(())
    }

    fn do_capture(&mut self, landing_id: i64, target: (i32, i32)) -> CoreResult<()> {
        let Some(ship) = self.ships.get(&landing_id) else {
            return Ok(());
        };
        if ship.owner != self.current_player
            || ship.ship_type != ShipType::Landing
            || ship.bought_this_turn
        {
            return Ok(());
        }
        let origin = ship.position;
        if !self.map.in_bounds(target.0, target.1) || !self.map.is_port(target.0, target.1)? {
            return Ok(());
        }
        let prev_owner = i64::from(self.map.port_owner_at(target.0, target.1)?);
        if prev_owner == self.current_player {
            return Ok(());
        }
        if Map::manhattan(origin, target) != 1 {
            return Ok(());
        }
        let captor = self.current_player;
        self.map.set_port_owner(target.0, target.1, captor as i8)?;
        self.players[prev_owner as usize]
            .owned_port_positions
            .remove(&target);
        self.players[captor as usize]
            .owned_port_positions
            .insert(target);
        let prev_home = self.players[prev_owner as usize].home_port;
        if let Some(port) = self.ports.get_mut(&target) {
            port.owner = captor;
            port.stockpile = 0;
            port.locked_this_turn = true;
        }
        if target == prev_home {
            self.winner = Some(captor);
        }
        Ok(())
    }

    fn do_merchant_load(&mut self, merchant_id: i64, port: (i32, i32)) -> CoreResult<()> {
        let Some(merchant) = self.ships.get(&merchant_id) else {
            return Ok(());
        };
        if merchant.owner != self.current_player
            || merchant.ship_type != ShipType::Merchant
            || merchant.bought_this_turn
        {
            return Ok(());
        }
        let origin = merchant.position;
        let cargo = merchant.cargo;
        if !self.merchant_at_own_port(origin, port)? {
            return Ok(());
        }
        let stock = self.ports.get(&port).map(|p| p.stockpile).unwrap_or(0);
        let transfer = stock.min(MERCHANT_CAPACITY - cargo);
        if transfer <= 0 {
            return Ok(());
        }
        self.ports.get_mut(&port).unwrap().stockpile -= transfer;
        self.ships.get_mut(&merchant_id).unwrap().cargo += transfer;
        Ok(())
    }

    fn do_merchant_unload(&mut self, merchant_id: i64, port: (i32, i32)) -> CoreResult<()> {
        let Some(merchant) = self.ships.get(&merchant_id) else {
            return Ok(());
        };
        if merchant.owner != self.current_player
            || merchant.ship_type != ShipType::Merchant
            || merchant.bought_this_turn
            || merchant.cargo <= 0
        {
            return Ok(());
        }
        let origin = merchant.position;
        let cargo = merchant.cargo;
        if !self.merchant_at_own_port(origin, port)? {
            return Ok(());
        }
        if port != self.players[self.current_player as usize].home_port {
            return Ok(());
        }
        self.players[self.current_player as usize].cash += cargo * CASH_PER_MATERIAL;
        self.ships.get_mut(&merchant_id).unwrap().cargo = 0;
        Ok(())
    }

    fn merchant_at_own_port(&self, origin: (i32, i32), port: (i32, i32)) -> CoreResult<bool> {
        if !self.map.in_bounds(port.0, port.1) || !self.map.is_port(port.0, port.1)? {
            return Ok(false);
        }
        if i64::from(self.map.port_owner_at(port.0, port.1)?) != self.current_player {
            return Ok(false);
        }
        Ok(Map::manhattan(origin, port) == 1)
    }

    fn do_end_turn(&mut self) {
        let outgoing = self.current_player as usize;
        let ship_ids: Vec<i64> = self.players[outgoing]
            .owned_ship_ids
            .iter()
            .copied()
            .collect();
        for sid in ship_ids {
            if let Some(ship) = self.ships.get_mut(&sid) {
                ship.tiles_moved_this_turn = 0;
                ship.has_attacked = false;
                ship.bought_this_turn = false;
            }
        }
        let ports: Vec<(i32, i32)> = self.players[outgoing]
            .owned_port_positions
            .iter()
            .copied()
            .collect();
        for pos in ports {
            if let Some(port) = self.ports.get_mut(&pos) {
                port.locked_this_turn = false;
            }
        }
        if self.current_player == 1 {
            self.turn += 1;
        }
        self.current_player = 1 - self.current_player;
        self.tick_income(self.current_player);
    }

    fn tick_income(&mut self, player_id: i64) {
        let idx = player_id as usize;
        let home = self.players[idx].home_port;
        let positions: Vec<(i32, i32)> = self.players[idx]
            .owned_port_positions
            .iter()
            .copied()
            .collect();
        for pos in positions {
            if pos == home {
                self.players[idx].cash += PORT_PRODUCTION * CASH_PER_MATERIAL;
            } else if let Some(port) = self.ports.get_mut(&pos) {
                port.stockpile += PORT_PRODUCTION;
            }
        }
    }

    fn destroy_ship(&mut self, ship_id: i64) -> CoreResult<()> {
        let Some(ship) = self.ships.remove(&ship_id) else {
            return Ok(());
        };
        if let Some(player) = self.players.iter_mut().find(|p| p.id == ship.owner) {
            player.owned_ship_ids.remove(&ship_id);
        }
        self.map.remove_ship(ship.position.0, ship.position.1)?;
        for observer in &mut self.players {
            if observer.id == ship.owner {
                continue;
            }
            if observer
                .sightings
                .get(&ship_id)
                .map(|s| s.fresh)
                .unwrap_or(false)
            {
                observer.sightings.remove(&ship_id);
            }
        }
        Ok(())
    }
}
