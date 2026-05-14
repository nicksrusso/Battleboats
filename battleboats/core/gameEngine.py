from typing import Dict, List, Optional, Tuple
import numpy as np
from copy import deepcopy
from .player import Player
from .map.Map import Map
from .sighting import Sighting
from .shipyard.ship import Ship
from .shipyard.ship_type import ShipType
from .shipyard.ship_data import BASE_STATS, attack_modifier
from .actions import (
    Action,
    MoveAction,
    AttackAction,
    BuildShipAction,
    BuildPortAction,
    CapturePortAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    EndTurnAction,
)

STARTING_CASH: int = 250
PORT_PRODUCTION: int = 25  # materials per owned port per player turn
CASH_PER_MATERIAL: int = 2  # conversion at home port (production + deliveries)
MERCHANT_CAPACITY: int = 100

_CARDINAL_OFFSETS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


class gameEngine:
    """Sole owner of game state and all rules.

    The Map handles spatial truth, Players hold per-side bookkeeping, Ships
    are passive records — but every mutation flows through this class so
    invariants stay coordinated.
    """

    def __init__(self, map_json_path: str, kill_curve_k: float = 2.0) -> None:
        self.map_json_path = map_json_path
        self.kill_curve_k = kill_curve_k
        self.map: Map = Map()
        self.players: List[Player] = []
        self.ships: Dict[int, Ship] = {}
        self.current_player: int = 0
        self.turn: int = 0
        self.winner: Optional[int] = None
        self.rng: np.random.Generator = np.random.default_rng()
        self._next_ship_id: int = 0
        self._dispatch = {
            MoveAction: self._do_move,
            AttackAction: self._do_attack,
            BuildShipAction: self._do_build_ship,
            BuildPortAction: self._do_build_port,
            CapturePortAction: self._do_capture_port,
            MerchantLoadAction: self._do_merchant_load,
            MerchantUnloadAction: self._do_merchant_unload,
            EndTurnAction: self._do_end_turn,
        }

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.map = Map()
        self.map.load(self.map_json_path)
        home_ports = self._infer_home_ports()
        self.players = [Player(0, home_ports[0]), Player(1, home_ports[1])]
        for p in self.players:
            p.cash = STARTING_CASH
        for pos in self.map.port_positions:
            owner = int(self.map.port_owner[pos])
            self.players[owner].owned_port_positions.add(pos)
            self.players[owner].port_materials.setdefault(pos, 0)
        self.ships = {}
        self._next_ship_id = 0
        self.current_player = 0
        self.turn = 0
        self.winner = None
        self._refresh_sightings()

    def _infer_home_ports(self) -> List[Tuple[int, int]]:
        homes: List[Optional[Tuple[int, int]]] = [None, None]
        for pos in self.map.port_positions:
            owner = int(self.map.port_owner[pos])
            if owner in (0, 1) and homes[owner] is None:
                homes[owner] = pos
        return [h for h in homes if h is not None]

    # ------------------------------------------------------------------ step
    def step(self, action: Action) -> None:
        if self.winner is not None:
            return
        self._dispatch[type(action)](action)
        self._refresh_sightings()

    # ------------------------------------------------------------------ action handlers
    def _do_move(self, a: MoveAction) -> None:
        ship = self.ships.get(a.ship_id)
        if ship is None or ship.owner != self.current_player:
            return
        if ship.tiles_moved_this_turn >= ship.stats.speed:
            return
        if not self.map.in_bounds(a.destination):
            return
        if self.map.manhattan(ship.position, a.destination) != 1:
            return
        if not self.map.is_water(a.destination):
            return
        if self.map.is_occupied(a.destination):
            return
        self.map.relocate_ship(ship.id, ship.position, a.destination)
        ship.position = a.destination
        ship.tiles_moved_this_turn += 1

    def _do_attack(self, a: AttackAction) -> None:
        attacker = self.ships.get(a.attacker_id)
        if attacker is None or attacker.owner != self.current_player:
            return
        if attacker.has_attacked:
            return
        if attacker.stats.attack_range <= 0:
            return
        defender = self.ships.get(a.target_id)
        if defender is None or defender.owner == self.current_player:
            return
        distance = self.map.manhattan(attacker.position, defender.position)
        if distance > attacker.stats.attack_range:
            return
        if distance > self._detection_distance(attacker, defender):
            return
        attacker.has_attacked = True
        if self._resolve_attack(attacker, defender):
            self._destroy_ship(defender.id)

    def _do_build_ship(self, a: BuildShipAction) -> None:
        # Silent no-op on illegal actions; legal_actions() will mask these out
        # for trained agents. Engine stays robust to bad input during dev / random play.
        player = self.players[self.current_player]
        if a.port not in player.owned_port_positions:
            return
        if not self.map.in_bounds(a.spawn_position):
            return
        if self.map.manhattan(a.spawn_position, a.port) != 1:
            return
        if not self.map.is_water(a.spawn_position):
            return
        if self.map.is_occupied(a.spawn_position):
            return
        cost = BASE_STATS[a.ship_type].cost
        if player.cash < cost:
            return
        player.cash -= cost
        self._spawn_ship(self.current_player, a.ship_type, a.spawn_position)

    def _do_build_port(self, a: BuildPortAction) -> None:
        ship = self.ships.get(a.builder_ship_id)
        if ship is None or ship.owner != self.current_player:
            return
        if ship.type != ShipType.BUILDER:
            return
        if not self.map.in_bounds(a.target):
            return
        if not self.map.is_land(a.target):
            return
        if self.map.is_port(a.target):
            return
        if self.map.manhattan(ship.position, a.target) != 1:
            return
        player = self.players[self.current_player]
        self.map.add_port(a.target, self.current_player)
        player.owned_port_positions.add(a.target)
        player.port_materials[a.target] = 0
        self._destroy_ship(a.builder_ship_id)

    def _do_capture_port(self, a: CapturePortAction) -> None:
        ship = self.ships.get(a.landing_ship_id)
        if ship is None or ship.owner != self.current_player:
            return
        if ship.type != ShipType.LANDING:
            return
        if not self.map.in_bounds(a.target):
            return
        if not self.map.is_port(a.target):
            return
        prev_owner_id = int(self.map.port_owner[a.target])
        if prev_owner_id == self.current_player:
            return
        if self.map.manhattan(ship.position, a.target) != 1:
            return
        self.map.set_port_owner(a.target, self.current_player)
        prev_owner = self.players[prev_owner_id]
        prev_owner.owned_port_positions.discard(a.target)
        prev_owner.port_materials.pop(a.target, None)
        captor = self.players[self.current_player]
        captor.owned_port_positions.add(a.target)
        captor.port_materials[a.target] = 0
        if a.target == prev_owner.home_port:
            self.winner = self.current_player

    def _do_merchant_load(self, a: MerchantLoadAction) -> None:
        merchant = self.ships.get(a.merchant_id)
        if merchant is None or merchant.owner != self.current_player:
            return
        if merchant.type != ShipType.MERCHANT:
            return
        if not self.map.in_bounds(a.port):
            return
        if not self.map.is_port(a.port):
            return
        if int(self.map.port_owner[a.port]) != self.current_player:
            return
        if self.map.manhattan(merchant.position, a.port) != 1:
            return
        player = self.players[self.current_player]
        available = player.port_materials.get(a.port, 0)
        capacity_left = MERCHANT_CAPACITY - merchant.cargo
        transfer = min(available, capacity_left)
        if transfer <= 0:
            return
        player.port_materials[a.port] = available - transfer
        merchant.cargo += transfer

    def _do_merchant_unload(self, a: MerchantUnloadAction) -> None:
        merchant = self.ships.get(a.merchant_id)
        if merchant is None or merchant.owner != self.current_player:
            return
        if merchant.type != ShipType.MERCHANT:
            return
        if merchant.cargo <= 0:
            return
        if not self.map.in_bounds(a.port):
            return
        if not self.map.is_port(a.port):
            return
        if int(self.map.port_owner[a.port]) != self.current_player:
            return
        if self.map.manhattan(merchant.position, a.port) != 1:
            return
        player = self.players[self.current_player]
        if a.port == player.home_port:
            player.cash += merchant.cargo * CASH_PER_MATERIAL
        else:
            player.port_materials[a.port] = player.port_materials.get(a.port, 0) + merchant.cargo
        merchant.cargo = 0

    def _tick_port_income(self, player_id: int) -> None:
        """Apply per-turn production for one player. Home port converts directly
        to cash; other owned ports accumulate raw materials needing a merchant."""
        player = self.players[player_id]
        for port in player.owned_port_positions:
            if port == player.home_port:
                player.cash += PORT_PRODUCTION * CASH_PER_MATERIAL
            else:
                player.port_materials[port] = player.port_materials.get(port, 0) + PORT_PRODUCTION

    def _do_end_turn(self, a: EndTurnAction) -> None:
        for sid in self.players[self.current_player].owned_ship_ids:
            self.ships[sid].reset_turn_flags()
        self._tick_port_income(self.current_player)
        # self.turn counts rounds; increments after player 1 finishes their turn.
        if self.current_player == 1:
            self.turn += 1
        self.current_player = 1 - self.current_player

    # ------------------------------------------------------------------ internal helpers
    def _new_ship_id(self) -> int:
        sid = self._next_ship_id
        self._next_ship_id += 1
        return sid

    def _spawn_ship(
        self,
        owner: int,
        ship_type: ShipType,
        position: Tuple[int, int],
    ) -> Ship:
        sid = self._new_ship_id()
        stats = BASE_STATS[ship_type]
        ship = Ship(id=sid, type=ship_type, stats=stats, owner=owner, position=position)
        self.ships[sid] = ship
        self.players[owner].owned_ship_ids.add(sid)
        self.map.place_ship(sid, position)
        return ship

    def _destroy_ship(self, ship_id: int) -> None:
        ship = self.ships.pop(ship_id)
        self.players[ship.owner].owned_ship_ids.discard(ship_id)
        self.map.remove_ship(ship.position)
        # Witness-kill: any other player who had a fresh view of this ship
        # saw it die and forgets it. Stale sightings are preserved (the
        # observer didn't witness the death and may still think it's alive).
        for observer in self.players:
            if observer.id == ship.owner:
                continue
            sighting = observer.sightings.get(ship_id)
            if sighting is not None and sighting.fresh:
                del observer.sightings[ship_id]

    def _detection_distance(self, scout: Ship, target: Ship) -> float:
        """Range at which `scout` can spot `target`. Asymmetric by design."""
        return scout.stats.scouting * target.stats.visibility

    def _resolve_attack(self, attacker: Ship, defender: Ship) -> bool:
        """Return True if defender is destroyed. Probability is a function of
        relative strength, modified by the attacker-vs-defender type matchup.
        Engine owns the RNG so rollouts are deterministic under a fixed seed.

        P(kill) = x^k / (1 + x^k),
            x = (attacker.strength * modifier) / defender.strength
            k = self.kill_curve_k (configurable; higher = more decisive).
        Defenders with strength 0 always die (and would otherwise divide by zero).
        """
        if defender.stats.strength <= 0:
            return True
        x = (attacker.stats.strength * attack_modifier(attacker.type, defender.type)) / defender.stats.strength
        xk = x**self.kill_curve_k
        p = xk / (1.0 + xk)
        return float(self.rng.random()) < p

    def _can_observer_see(self, observer_id: int, target: Ship) -> bool:
        for sid in self.players[observer_id].owned_ship_ids:
            scout = self.ships[sid]
            if self.map.manhattan(scout.position, target.position) <= self._detection_distance(scout, target):
                return True
        return False

    def _refresh_sightings(self) -> None:
        """Recompute every player's sightings against current world state.

        Stale sightings keep their position/turn_seen; only the `fresh` flag
        flips. A newly-visible enemy overwrites with a fresh record (this is
        also how moving-while-visible updates the recorded position).
        O(my_ships * enemy_ships) per call; cache later if profiling demands.
        """
        for observer in self.players:
            for s in observer.sightings.values():
                s.fresh = False
            for target in self.ships.values():
                if target.owner == observer.id:
                    continue
                if not self._can_observer_see(observer.id, target):
                    continue
                observer.sightings[target.id] = Sighting(
                    ship_id=target.id,
                    type=target.type,
                    position=target.position,
                    turn_seen=self.turn,
                    fresh=True,
                )

    # ------------------------------------------------------------------ RL hooks
    def visible_enemy_ships(self, player_id: int) -> List[Ship]:
        """Enemy ships currently in sight of one of player_id's owned ships."""
        return [self.ships[sid] for sid, s in self.players[player_id].sightings.items() if s.fresh and sid in self.ships]

    def known_enemy_ships(self, player_id: int) -> List[Sighting]:
        """All sightings (fresh + stale) — player's last-known view of enemies."""
        return list(self.players[player_id].sightings.values())

    def enumerate_legal(self, player_id: int) -> List[Action]:
        """Concrete legal actions for `player_id` right now.

        Mirrors the validation in each `_do_*` handler so step() will accept
        every returned action. AttackAction is omitted until visibility lands
        in Phase 2. EndTurnAction is always included for the current player.
        """
        actions: List[Action] = []
        if self.winner is not None or player_id != self.current_player:
            return actions
        player = self.players[player_id]

        for sid in player.owned_ship_ids:
            ship = self.ships[sid]
            sx, sy = ship.position
            can_move = ship.tiles_moved_this_turn < ship.stats.speed
            is_builder = ship.type == ShipType.BUILDER
            is_landing = ship.type == ShipType.LANDING
            is_merchant = ship.type == ShipType.MERCHANT
            for dx, dy in _CARDINAL_OFFSETS:
                neighbor = (sx + dx, sy + dy)
                if not self.map.in_bounds(neighbor):
                    continue
                if can_move and self.map.is_water(neighbor) and not self.map.is_occupied(neighbor):
                    actions.append(MoveAction(sid, neighbor))
                if not self.map.is_land(neighbor):
                    continue
                if self.map.is_port(neighbor):
                    neighbor_owner = int(self.map.port_owner[neighbor])
                    if is_landing and neighbor_owner != player_id:
                        actions.append(CapturePortAction(sid, neighbor))
                    if is_merchant and neighbor_owner == player_id:
                        if ship.cargo > 0:
                            actions.append(MerchantUnloadAction(sid, neighbor))
                        if (
                            ship.cargo < MERCHANT_CAPACITY
                            and neighbor != player.home_port
                            and player.port_materials.get(neighbor, 0) > 0
                        ):
                            actions.append(MerchantLoadAction(sid, neighbor))
                elif is_builder:
                    actions.append(BuildPortAction(sid, neighbor))

        affordable_types = [t for t, s in BASE_STATS.items() if player.cash >= s.cost]
        if affordable_types:
            for port in player.owned_port_positions:
                px, py = port
                for dx, dy in _CARDINAL_OFFSETS:
                    spawn = (px + dx, py + dy)
                    if not self.map.in_bounds(spawn):
                        continue
                    if not self.map.is_water(spawn):
                        continue
                    if self.map.is_occupied(spawn):
                        continue
                    for t in affordable_types:
                        actions.append(BuildShipAction(port, spawn, t))

        actions.append(EndTurnAction())
        return actions

    def legal_actions(self, player_id: int):
        """RL-friendly per-action-type mask. Deferred until env layer designs its action encoding."""
        raise NotImplementedError

    def get_state(self) -> dict:
        """Full ground-truth state. The env layer applies fog-of-war masking."""
        return {
            "terrain": self.map.terrain.copy(),
            "port_owner": self.map.port_owner.copy(),
            "ship_at": self.map.ship_at.copy(),
            "ships": {sid: deepcopy(ship) for sid, ship in self.ships.items()},
            "players": [deepcopy(self.players[0]), deepcopy(self.players[1])],
            "current_player": self.current_player,
            "turn": self.turn,
            "winner": self.winner,
        }

    def is_terminal(self) -> bool:
        return self.winner is not None

    def clone(self) -> "gameEngine":
        """Fast snapshot for RL rollouts / search. Copies numpy grids and the
        ship registry without going through reset()/JSON load.
        """
        new = gameEngine.__new__(gameEngine)

        new.map_json_path = self.map_json_path
        new.kill_curve_k = self.kill_curve_k
        new.current_player = self.current_player
        new.turn = self.turn
        new.winner = self.winner
        new._next_ship_id = self._next_ship_id

        new.map = deepcopy(self.map)
        new.players = [deepcopy(p) for p in self.players]
        new.ships = {sid: deepcopy(s) for sid, s in self.ships.items()}
        new.rng = deepcopy(self.rng)

        # Bind handlers to `new` — copying the dict would keep them bound to `self`.
        new._dispatch = {
            MoveAction: new._do_move,
            AttackAction: new._do_attack,
            BuildShipAction: new._do_build_ship,
            BuildPortAction: new._do_build_port,
            CapturePortAction: new._do_capture_port,
            MerchantLoadAction: new._do_merchant_load,
            MerchantUnloadAction: new._do_merchant_unload,
            EndTurnAction: new._do_end_turn,
        }

        return new
