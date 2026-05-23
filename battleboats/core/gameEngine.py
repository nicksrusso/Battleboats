from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from copy import deepcopy

from battleboats.core.player import Player
from battleboats.core.port import Port
from battleboats.core.map.Map import Map
from battleboats.core.sighting import Sighting, PortSighting
from battleboats.core.shipyard.ship import Ship
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.core.shipyard.ship_data import BASE_STATS, attack_modifier
from battleboats.core.actions import (
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
PORT_VISIBILITY: float = 4.0  # how easy a port is to spot
PORT_SCOUTING: float = 4.0  # a port's own detection coefficient

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
        self.ports: Dict[Tuple[int, int], Port] = {}
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
        home_port_set = set(home_ports)
        self.ports = {}
        for pos in self.map.port_positions:
            owner = int(self.map.port_owner[pos])
            self.players[owner].owned_port_positions.add(pos)
            self.ports[pos] = Port(
                position=pos,
                owner=owner,
                is_home=pos in home_port_set,
                stockpile=0,
            )
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
        self.ports[a.target] = Port(
            position=a.target,
            owner=self.current_player,
            is_home=False,
            stockpile=0,
        )
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
        captor = self.players[self.current_player]
        captor.owned_port_positions.add(a.target)
        port = self.ports[a.target]
        port.owner = self.current_player
        port.stockpile = 0  # captured port's stockpile is lost
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
        port = self.ports[a.port]
        capacity_left = MERCHANT_CAPACITY - merchant.cargo
        transfer = min(port.stockpile, capacity_left)
        if transfer <= 0:
            return
        port.stockpile -= transfer
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
        # Unload restricted to home port — non-home unload would just push
        # cargo back to stockpile, which creates sterile load↔unload loops.
        if a.port != player.home_port:
            return
        player.cash += merchant.cargo * CASH_PER_MATERIAL
        merchant.cargo = 0

    def _tick_port_income(self, player_id: int) -> None:
        """Apply production for one player at the start of their turn. Home port
        converts directly to cash; other owned ports accumulate raw materials
        needing a merchant to monetize."""
        player = self.players[player_id]
        for pos in player.owned_port_positions:
            if pos == player.home_port:
                player.cash += PORT_PRODUCTION * CASH_PER_MATERIAL
            else:
                self.ports[pos].stockpile += PORT_PRODUCTION

    def _do_end_turn(self, a: EndTurnAction) -> None:
        # Outgoing player's per-turn ship bookkeeping.
        for sid in self.players[self.current_player].owned_ship_ids:
            self.ships[sid].reset_turn_flags()
        # self.turn counts rounds; increments after player 1 finishes their turn.
        if self.current_player == 1:
            self.turn += 1
        # Hand control over, then apply the new player's start-of-turn income.
        # Income lives at start-of-turn (not end-of-turn) so EndTurn carries
        # no economic delta from the actor's perspective — income arrives
        # "automatically" and is not a reward MCTS should chase by ending early.
        self.current_player = 1 - self.current_player
        self._tick_port_income(self.current_player)

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

    def _detection_distance(self, scout: Union[Ship, Port], target: Union[Ship, Port]) -> float:
        """Range at which `scout` can spot `target`. Asymmetric by design.

        Ports use module-level `PORT_SCOUTING` and `PORT_VISIBILITY` constants
        in place of per-instance stats — the formula is identical otherwise.
        """
        scouting = scout.stats.scouting if isinstance(scout, Ship) else PORT_SCOUTING
        visibility = target.stats.visibility if isinstance(target, Ship) else PORT_VISIBILITY
        return scouting * visibility

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

    def _can_observer_see(self, observer_id: int, target: Union[Ship, Port]) -> bool:
        """Does the observer have a friendly ship OR port in detection range?"""
        player = self.players[observer_id]
        for sid in player.owned_ship_ids:
            scout = self.ships[sid]
            if self.map.manhattan(scout.position, target.position) <= self._detection_distance(scout, target):
                return True
        for pos in player.owned_port_positions:
            scout = self.ports[pos]
            if self.map.manhattan(pos, target.position) <= self._detection_distance(scout, target):
                return True
        return False

    def _refresh_sightings(self) -> None:
        """Recompute every player's sightings (ships + ports) against world state.

        Stale sightings keep their last-known data; only the `fresh` flag
        flips. A newly-visible enemy overwrites with a fresh record (this is
        also how moving-while-visible updates the recorded position, and how
        a witnessed port capture updates `last_known_owner`).

        Port sightings additionally clean up entries the observer now owns
        (a captured port is removed from the captor's port_sightings on the
        next refresh tick).

        O(my_ships * enemy_ships + my_ports * enemy_ports) per call. Cache if
        profiling demands.
        """
        for observer in self.players:
            # Ship sightings
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
            # Port sightings
            own_positions = observer.owned_port_positions
            for pos in list(observer.port_sightings):
                if pos in own_positions:
                    del observer.port_sightings[pos]
            for s in observer.port_sightings.values():
                s.fresh = False
            for target_port in self.ports.values():
                if target_port.owner == observer.id:
                    continue
                if not self._can_observer_see(observer.id, target_port):
                    continue
                observer.port_sightings[target_port.position] = PortSighting(
                    position=target_port.position,
                    last_known_owner=target_port.owner,
                    is_home=target_port.is_home,
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

    def visible_enemy_ports(self, player_id: int) -> List[PortSighting]:
        """Enemy ports currently in sight of one of player_id's owned scouts."""
        return [s for s in self.players[player_id].port_sightings.values() if s.fresh]

    def known_enemy_ports(self, player_id: int) -> List[PortSighting]:
        """All port sightings (fresh + stale) — player's last-known view of enemy ports."""
        return list(self.players[player_id].port_sightings.values())

    def enumerate_legal(self, player_id: int) -> List[Action]:
        """Concrete legal actions for `player_id` right now.

        Mirrors the validation in each `_do_*` handler so step() will accept
        every returned action. EndTurnAction is always included for the
        current player.
        """
        actions: List[Action] = []
        if self.winner is not None or player_id != self.current_player:
            return actions
        player = self.players[player_id]
        opp_ship_ids = self.players[1 - player_id].owned_ship_ids

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
                        # Unload is restricted to home port — non-home unload
                        # drops cargo back into stockpile, creating sterile
                        # load↔unload loops that the logistics shaping bonus
                        # in T_ECON would otherwise reward.
                        if ship.cargo > 0 and neighbor == player.home_port:
                            actions.append(MerchantUnloadAction(sid, neighbor))
                        if (
                            ship.cargo < MERCHANT_CAPACITY
                            and neighbor != player.home_port
                            and self.ports[neighbor].stockpile > 0
                        ):
                            actions.append(MerchantLoadAction(sid, neighbor))
                elif is_builder:
                    actions.append(BuildPortAction(sid, neighbor))

            # AttackAction enumeration — mirrors _do_attack's preconditions.
            if not ship.has_attacked and ship.stats.attack_range > 0:
                for opp_sid in opp_ship_ids:
                    target = self.ships[opp_sid]
                    distance = self.map.manhattan(ship.position, target.position)
                    if distance > ship.stats.attack_range:
                        continue
                    if distance > self._detection_distance(ship, target):
                        continue
                    actions.append(AttackAction(sid, opp_sid))

        affordable_types = [t for t, s in BASE_STATS.items() if player.cash >= s.cost]
        if affordable_types:
            # Canonicalize the spawn position so MCTS sees one BuildShipAction
            # per (port, ship_type) instead of one per spawn-tile variant. Each
            # port may have up to 4 water-adjacent unoccupied spawn tiles; all
            # of them produce a ship 1 Manhattan step from the port, so the
            # choice has near-zero strategic impact relative to the 100+ turn
            # game horizon. Picking deterministically here drops root branching
            # factor by ~4× and lets MCTS spend its budget on real decisions.
            #
            # Rule: spawn closest to the opp home port (ties broken by lex
            # order on position). For combat ships and landings this points
            # them the right way out of the gate; for merchants/builders
            # they'll move anyway and the one-tile delta is irrelevant.
            opp_home = self.players[1 - player_id].home_port
            for port in player.owned_port_positions:
                px, py = port
                candidate_spawns = []
                for dx, dy in _CARDINAL_OFFSETS:
                    spawn = (px + dx, py + dy)
                    if not self.map.in_bounds(spawn):
                        continue
                    if not self.map.is_water(spawn):
                        continue
                    if self.map.is_occupied(spawn):
                        continue
                    candidate_spawns.append(spawn)
                if not candidate_spawns:
                    continue
                canonical_spawn = min(
                    candidate_spawns,
                    key=lambda s: (self.map.manhattan(s, opp_home), s),
                )
                for t in affordable_types:
                    actions.append(BuildShipAction(port, canonical_spawn, t))

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
            "ports": {pos: deepcopy(port) for pos, port in self.ports.items()},
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
        new.ports = {pos: deepcopy(p) for pos, p in self.ports.items()}
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
