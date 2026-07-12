"""Merchant delivery skill sandbox (PettingZoo AEC).

Single learning agent (player_0 by default). Opponent seat is auto-EndTurn'd
so the learner keeps the cycle without a second policy.

Reward (default, throughput): each home unload pays reward proportional to the
materials delivered that step (a full merchant load = deliver_reward), summed
over a fixed horizon (run to max_turns, no early termination). Rewards this
throughput / logistics-efficiency skill. Set terminate_on_delivery=True for the
old sparse mode (+deliver_reward once, then terminate) used as an early warmup.

Deliveries are measured by CARGO decrease (income-immune — per-turn port income
raises cash every turn, so a cash-delta reward would false-positive; cargo only
drops on a paid home unload).

Map JSON is supplied by the caller (width/height/ports of your choosing).
This env places stockpile + merchant after engine.reset().
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from battleboats.core.actions import (
    Action,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import MERCHANT_CAPACITY
from battleboats.core.map.Map import WATER
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.base_aec import BattleboatsBaseAEC

_CARDINALS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Verbs the merchant skill may use. Crucially excludes BuildShipAction: without
# this, the policy can build a LANDING ship from the home port, capture the
# enemy home, and set engine.winner — which the throughput env doesn't treat as
# terminal, so enumerate_legal then returns empty (winner set) and the next mask
# build hits a no-legal-option NaN. EndTurnAction is always enumerated, so the
# filtered list is never empty.
_MERCHANT_VERBS: Tuple[type, ...] = (
    MoveAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    EndTurnAction,
)


class MerchantSkillAEC(BattleboatsBaseAEC):
    """1p merchant skill: load (optional) and deliver materials to home."""

    metadata = {"name": "battleboats_merchant_skill_v0", "is_parallelizable": False}

    def __init__(
        self,
        map_json_path: str,
        *,
        max_turns: int = 50,
        learner_id: int = 0,
        stockpile: int = 10_000,
        starting_cargo: int = 0,
        deliver_reward: float = 1.0,
        terminate_on_delivery: bool = False,
        restrict_verbs: bool = True,
        randomize_ports: bool = False,
        merchant_position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            map_json_path: map with at least learner home port; non-home port
                recommended if stockpile > 0 (materials to pick up).
            max_turns: truncation budget (engine turn counter) = the horizon over
                which throughput is scored.
            learner_id: engine player id that trains (usually 0).
            stockpile: raw materials placed on each non-home port of learner.
                Defaults large so throughput is limited by round-trip time, not by
                exhausting the satellite (the skill is cycle efficiency).
            starting_cargo: cargo already on the merchant at reset (skips load).
            deliver_reward: reward for delivering a FULL merchant load. Per-material
                reward = deliver_reward / MERCHANT_CAPACITY, so a full load ~= +1
                (comparable to a landing capture, for later multi-task mixing).
            terminate_on_delivery: if True, revert to sparse mode — flat
                deliver_reward on the first home unload, then terminate. Default
                False = throughput (accumulate per-delivery reward to max_turns).
            restrict_verbs: if True (default), filter the learner's legal actions
                to move/load/unload/endturn (see _MERCHANT_VERBS). Hides BuildShip
                (the home port can build), which otherwise lets the policy build a
                landing ship and win by capture — breaking the throughput episode.
            randomize_ports: if True, place the satellite (resource) port on a
                random tile every reset (home + spawn fixed). Forces relational
                logistics — navigate to wherever the satellite is — instead of
                memorizing a fixed route. The satellite is a FRIENDLY port so it is
                always observed (no fog), unlike the landing target.
            merchant_position: optional spawn tile; else auto-pick water next
                to a relevant port.
        """
        self.learner_id = learner_id
        self.stockpile = stockpile
        self.starting_cargo = starting_cargo
        self.deliver_reward = deliver_reward
        self.terminate_on_delivery = terminate_on_delivery
        self.restrict_verbs = restrict_verbs
        self.randomize_ports = randomize_ports
        self._reward_per_material = deliver_reward / MERCHANT_CAPACITY
        self.merchant_position = merchant_position
        self._delivered = False
        self._tmp_map_path: Optional[str] = None
        super().__init__(map_json_path=map_json_path, max_turns=max_turns)
        if self.randomize_ports:
            self._init_random_ports()

    def _possible_agents(self) -> List[str]:
        return [self._agent_name(self.learner_id)]

    # ------------------------------------------------------------- port randomize
    def _init_random_ports(self) -> None:
        """Cache board size + fixed home/opponent-home; open a reusable temp map."""
        with open(self.map_json_path) as f:
            base = json.load(f)
        self._board = (int(base["width"]), int(base["height"]))
        self._home = tuple(base["ports"][str(self.learner_id)][0])
        self._opp_home = tuple(base["ports"][str(1 - self.learner_id)][0])
        fd, self._tmp_map_path = tempfile.mkstemp(suffix=".json", prefix="merchant_rand_")
        os.close(fd)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Regenerate a random-satellite layout (if enabled) before the engine loads."""
        if self.randomize_ports and not (options or {}).get("scenario"):
            self._write_random_layout(seed)
        super().reset(seed=seed, options=options)

    def _write_random_layout(self, seed: Optional[int]) -> None:
        """Write a fresh layout: fixed home + random satellite + fixed opp home.

        Home is listed FIRST in player 0's ports so _infer_home_ports keeps it as
        home regardless of the satellite tile. Home and its neighbors are excluded
        from candidates so the spawn (auto-picked water next to home) stays stable.
        """
        w, h = self._board
        home, opp = self._home, self._opp_home
        rng = random.Random(seed)
        exclude = {home, opp}
        for dx, dy in _CARDINALS:
            exclude.add((home[0] + dx, home[1] + dy))
        candidates = [
            (x, y) for x in range(w) for y in range(h) if (x, y) not in exclude
        ]
        if not candidates:
            raise RuntimeError("MerchantSkillAEC: board too small to place a satellite.")
        satellite = rng.choice(candidates)
        layout = {
            "width": w,
            "height": h,
            "seed": 0,
            "land": [list(home), list(satellite), list(opp)],
            "ports": {
                str(self.learner_id): [list(home), list(satellite)],  # home first
                str(1 - self.learner_id): [list(opp)],
            },
        }
        with open(self._tmp_map_path, "w") as f:
            json.dump(layout, f)
        self.map_json_path = self._tmp_map_path

    def close(self) -> None:
        if self._tmp_map_path and os.path.exists(self._tmp_map_path):
            try:
                os.remove(self._tmp_map_path)
            except OSError:
                pass

    def _configure_scenario(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> None:
        assert self.engine is not None
        opts = options or {}
        stockpile = int(opts.get("stockpile", self.stockpile))
        starting_cargo = int(opts.get("starting_cargo", self.starting_cargo))
        merchant_position = opts.get("merchant_position", self.merchant_position)
        if merchant_position is not None:
            merchant_position = tuple(merchant_position)

        learner = self.engine.players[self.learner_id]
        home = learner.home_port
        non_home = [p for p in learner.owned_port_positions if p != home]

        for pos in non_home:
            self.engine.ports[pos].stockpile = stockpile

        if merchant_position is None:
            if self.randomize_ports:
                anchor = home  # stable spawn near fixed home; the satellite moves
            elif non_home and stockpile > 0:
                anchor = non_home[0]
            else:
                anchor = home
            if starting_cargo > 0 and stockpile <= 0:
                anchor = home
            merchant_position = self._find_adjacent_water(anchor)
            if merchant_position is None:
                raise RuntimeError(
                    f"MerchantSkillAEC: no free water adjacent to {anchor}; "
                    "pass merchant_position= or fix the map JSON."
                )

        ship = self.engine._spawn_ship(
            self.learner_id, ShipType.MERCHANT, merchant_position
        )
        if starting_cargo > 0:
            ship.cargo = min(int(starting_cargo), MERCHANT_CAPACITY)

        self._delivered = False
        # Ensure learner to move first for a clean skill episode.
        self.engine.current_player = self.learner_id
        self.engine.turn = 0
        self.engine.winner = None
        self.engine._refresh_sightings()

        if self.restrict_verbs:
            self._install_verb_filter()

    def _install_verb_filter(self) -> None:
        """Wrap this episode's engine so enumerate_legal only yields _MERCHANT_VERBS.

        ActionMasks reads enumerate_legal directly, so filtering here is what
        reaches the policy. Applied to the per-episode engine instance (the engine
        never calls enumerate_legal internally, and reset() builds a fresh engine,
        so this neither perturbs engine mechanics nor compounds across resets).
        """
        assert self.engine is not None
        raw = self.engine.enumerate_legal

        def enumerate_legal(player_id: int, _raw=raw) -> List[Action]:
            return [a for a in _raw(player_id) if isinstance(a, _MERCHANT_VERBS)]

        self.engine.enumerate_legal = enumerate_legal

    def _capture_reward_state(self) -> Dict[str, Any]:
        assert self.engine is not None
        pid = self.learner_id
        cargo = 0
        for sid in self.engine.players[pid].owned_ship_ids:
            ship = self.engine.ships[sid]
            if ship.type is ShipType.MERCHANT:
                cargo += ship.cargo
        return {
            "cash": int(self.engine.players[pid].cash),
            "cargo": int(cargo),
        }

    def _reward_after_step(
        self,
        agent: str,
        action: Optional[Action],
        pre: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = {a: 0.0 for a in self.agents}
        delivered = self._materials_delivered(pre)
        if delivered > 0:
            learner = self._agent_name(self.learner_id)
            if self.terminate_on_delivery:
                self._delivered = True
                rewards[learner] = float(self.deliver_reward)  # sparse warmup mode
            else:
                rewards[learner] = delivered * self._reward_per_material
        return rewards

    def _is_terminated(self) -> bool:
        return self.terminate_on_delivery and self._delivered

    def _is_truncated(self) -> bool:
        return self.engine is not None and self.engine.turn >= self.max_turns

    def _after_engine_step(self, agent: str, action: Optional[Action]) -> None:
        """Auto-EndTurn the non-learner seat so AEC only exposes the skill agent."""
        assert self.engine is not None
        guard = 0
        while (
            self.engine.current_player != self.learner_id
            and self.engine.winner is None
            and self.engine.turn < self.max_turns
            and guard < 8
        ):
            self.engine.step(EndTurnAction())
            guard += 1

    def _materials_delivered(self, pre: Dict[str, Any]) -> int:
        """Materials unloaded at home this step, measured by cargo decrease.

        Cargo only drops on a paid home unload (non-home unload is illegal;
        loading raises cargo), so cargo-delta is an income-immune count of
        delivered materials — cash is not, since per-turn port income inflates
        it every turn regardless of delivery.
        """
        assert self.engine is not None
        post = self._capture_reward_state()
        return max(0, pre.get("cargo", 0) - post["cargo"])

    def _find_adjacent_water(
        self, port: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        assert self.engine is not None
        m = self.engine.map
        for dx, dy in _CARDINALS:
            pos = (port[0] + dx, port[1] + dy)
            if not m.in_bounds(pos):
                continue
            if m.terrain[pos] != WATER:
                continue
            if m.is_occupied(pos):
                continue
            return pos
        return None
