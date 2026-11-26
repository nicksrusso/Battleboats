"""One local game: human vs tactical, scripted_agent, scripted, random, or a policy checkpoint.

The human sees fog of war strictly per the spotting rules:
- Own ships and own ports in full.
- Enemy ships only via ``player.sightings`` (fresh = currently spotted,
  stale = last-known position and turn seen).
- Enemy ports only via ``player.port_sightings``; unsighted enemy/neutral
  ports render as plain land.
- Terrain coastline is visible.
- The turn log and opponent move markers never disclose unseen enemy moves,
  unseen builds, or unseen attacker coordinates.
"""

from __future__ import annotations

import atexit
import html
import json
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from battleboats.agents.heuristics import pk
from battleboats.agents.random_agent import random_action
from battleboats.agents.registry import normalize_agent_kind
from battleboats.agents.scripted_agent import scripted_agent_action
from battleboats.agents.scripted import scripted_action
from battleboats.agents.tactical import tactical_scripted_action
from battleboats.core.actions import (
    Action,
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.gameEngine import CASH_PER_MATERIAL, gameEngine
from battleboats.core.shipyard.ship_data import BASE_STATS, PORT, spot_distance
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.llm.openrouter import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OpenRouterClient,
    decide_llm_action,
    format_usage_line,
    resolve_model_id,
    resolve_reasoning,
    safe_error,
)
from battleboats.training.acting import policy_action
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.device import resolve_device
from battleboats.ui.game_log import (
    GameLog,
    combat_result,
    decision_record,
    default_log_dir,
    end_record,
    llm_fields,
    log_filename,
    serialize_engine,
    start_record,
)
from battleboats.llm.prompt import build_view
from battleboats.ui.maps import ResolvedMap, resolve_map

Pos = Tuple[int, int]
_CARDINALS: Tuple[Pos, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

_SHIP_CODE = {
    ShipType.CARRIER: "CV",
    ShipType.BATTLESHIP: "BB",
    ShipType.CRUISER: "CA",
    ShipType.DESTROYER: "DD",
    ShipType.SUBMARINE: "SS",
    ShipType.MERCHANT: "AK",
    ShipType.LANDING: "LC",
    ShipType.BUILDER: "CB",
}

OPPONENT_LABELS = {
    "tactical": "Tactical (fog of war)",
    "scripted_agent": "Scripted agent (fog of war)",
    "scripted": "Scripted baseline (full-information opponent)",
    "random": "Random",
    "checkpoint": "Checkpoint",
    "llm": "LLM (OpenRouter)",
    "inkling": "Inkling (high thinking)",
}

LLM_OPPONENTS = ("llm", "inkling")


class _TypeStandIn:
    __slots__ = ("type", "stats")

    def __init__(self, ship_type: ShipType) -> None:
        self.type = ship_type
        self.stats = BASE_STATS[ship_type]


def describe_action(action: Action, engine: gameEngine, viewer_id: Optional[int] = None) -> str:
    """Short fog-safe label for an action."""
    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        kind = _SHIP_CODE.get(ship.type, "?") if ship else "?"
        return f"Move {kind} #{action.ship_id} -> {action.destination}"
    if isinstance(action, AttackAction):
        a = engine.ships.get(action.attacker_id)
        ak = _SHIP_CODE.get(a.type, "?") if a else "?"
        if viewer_id is not None and a is not None and a.owner == viewer_id:
            sight = engine.players[viewer_id].sightings.get(action.target_id)
            tk = _SHIP_CODE.get(sight.type, "?") if sight else "?"
        else:
            t = engine.ships.get(action.target_id)
            tk = _SHIP_CODE.get(t.type, "?") if t else "?"
        return f"Attack {ak} #{action.attacker_id} -> {tk} #{action.target_id}"
    if isinstance(action, BuildShipAction):
        return f"Build {action.ship_type.value} at port {action.port}"
    if isinstance(action, BuildPortAction):
        return f"Build port at {action.target}"
    if isinstance(action, CapturePortAction):
        return f"Capture port {action.target}"
    if isinstance(action, MerchantLoadAction):
        return f"Load merchant #{action.merchant_id} at {action.port}"
    if isinstance(action, MerchantUnloadAction):
        return f"Unload merchant #{action.merchant_id} at {action.port}"
    if isinstance(action, EndTurnAction):
        return "End turn"
    return type(action).__name__


class PlaySession:
    """Human seat vs one opponent.

    Engine mutations go through ``self.lock``. LLM turns run in a background
    thread so the HTTP server can keep serving ``/api/state`` polls.
    """

    def __init__(
        self,
        *,
        opponent: str = "tactical",
        human_side: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        map_spec: Optional[str] = None,
        seed: int = 0,
        max_turns: Optional[int] = None,
        map_dir: Path = Path("runs/play_maps"),
        checkpoint: Optional[Path] = None,
        device: str = "auto",
        step_budget: int = 1000,
        llm_model: Optional[str] = None,
        llm_reasoning: Optional[str] = None,
        llm_client: Optional[Callable[[str, float], str]] = None,
        llm_timeout: float = DEFAULT_TIMEOUT,
        llm_max_tokens: int = DEFAULT_MAX_TOKENS,
        llm_temperature: float = DEFAULT_TEMPERATURE,
        log_dir: Optional[Path] = None,
    ) -> None:
        opponent = normalize_agent_kind(opponent)
        if opponent not in ("tactical", "scripted_agent", "random", "scripted", "checkpoint", "llm", "inkling"):
            raise ValueError(f"unknown opponent {opponent!r}")
        if human_side not in (0, 1):
            raise ValueError("human_side must be 0 or 1")
        self.opponent = opponent
        self.human_side = human_side
        self.seed = int(seed)
        self.step_budget = step_budget
        self.rng = random.Random(self.seed)
        self.entries: List[dict] = []
        self.recent_opponent_moves: List[dict] = []
        self.recent_combat: List[dict] = []
        self.error = ""
        self.lock = threading.RLock()
        self.thinking = False
        self.thinking_message = ""
        self.thinking_started: Optional[float] = None
        self.last_llm: dict = {}
        self.llm_totals: dict = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        self._worker: Optional[threading.Thread] = None
        self.llm_model = resolve_model_id(llm_model, opponent=opponent)
        self.llm_reasoning = resolve_reasoning(llm_reasoning or DEFAULT_REASONING)
        self.llm_temperature = float(llm_temperature)
        self.llm_client = llm_client
        if self._uses_llm() and self.llm_client is None:
            self.llm_client = OpenRouterClient(
                model=self.llm_model,
                reasoning_effort=self.llm_reasoning,
                timeout=llm_timeout,
                max_tokens=llm_max_tokens,
            )

        resolved: ResolvedMap = resolve_map(
            map_spec,
            seed=self.seed,
            map_dir=Path(map_dir),
            width=width,
            height=height,
        )
        self.map_name = resolved.name
        self.map_path = resolved.path
        self.max_turns = (
            int(max_turns) if max_turns is not None else resolved.recommended_max_turns
        )

        self.engine = gameEngine(str(self.map_path))
        self.engine.reset(seed=self.seed)
        self.policy = None
        self.checkpoint = None
        if opponent == "checkpoint":
            if checkpoint is None or not Path(checkpoint).exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            self.checkpoint = Path(checkpoint)
            dev = resolve_device(device)
            self.policy, _ = load_checkpoint(self.checkpoint, dev)
            self.policy.eval()

        self._log_steps = 0
        self._pending_llm_meta: Optional[dict] = None
        self._game_log_closed = False
        log_root = Path(log_dir) if log_dir is not None else default_log_dir(map_dir)
        self.log_path = log_root / log_filename(self.map_name, self.opponent, self.seed)
        self.game_log = GameLog(self.log_path)
        self.game_log.write(start_record(self))
        atexit.register(self._finalize_log_atexit)
        self._add_log(
            f"Game started on {self.map_name} ({self.engine.map.width}x{self.engine.map.height}), "
            f"seed {self.seed}. You are Player {self.human_side}. "
            f"Log: {self.log_path}",
            kind="info",
        )
        self._play_opponent_until_human()
        if self.over:
            self._finalize_log(complete=True)

    @property
    def log(self) -> List[str]:
        return [e["text"] for e in self.entries]

    @property
    def over(self) -> bool:
        return self.engine.winner is not None or self.engine.turn >= self.max_turns

    @property
    def result_text(self) -> str:
        if self.engine.winner is None and self.engine.turn >= self.max_turns:
            return "Draw — turn limit reached."
        if self.engine.winner is None:
            return ""
        if self.engine.winner == self.human_side:
            return "Victory! You captured the enemy home port."
        return "Defeat — your home port was captured."

    @property
    def opponent_label(self) -> str:
        if self.opponent == "checkpoint" and self.checkpoint is not None:
            return f"Checkpoint {self.checkpoint}"
        if self._uses_llm():
            name = "Inkling" if "inkling" in self.llm_model.lower() or self.opponent == "inkling" else "LLM"
            return f"{name} ({self.llm_model}, reasoning={self.llm_reasoning})"
        return OPPONENT_LABELS[self.opponent]

    def _uses_llm(self) -> bool:
        return self.opponent in LLM_OPPONENTS

    def _add_log(self, text: str, kind: str = "info") -> None:
        self.entries.append({"turn": self.engine.turn, "kind": kind, "text": text})

    def legal(self) -> List[Action]:
        if self.over or self.engine.current_player != self.human_side:
            return []
        return self.engine.enumerate_legal(self.human_side)

    def apply(self, index: int) -> None:
        self.error = ""
        if self.thinking:
            self.error = "Wait — the opponent is still thinking."
            return
        legal = self.legal()
        if not legal:
            self.error = "No legal actions."
            return
        if index < 0 or index >= len(legal):
            self.error = f"Action {index} is out of range."
            return
        action = legal[index]
        if isinstance(action, EndTurnAction):
            self.recent_opponent_moves = []
            self.recent_combat = []
        self._apply(action, self.human_side)
        if not self.over and self.engine.current_player != self.human_side:
            self._play_opponent_until_human()

    def apply_spec(self, spec: dict) -> None:
        """Apply a structured UI action payload."""
        self.error = ""
        if self.over:
            self.error = "Game is already over."
            return
        if self.thinking:
            self.error = "Wait — the opponent is still thinking."
            return
        if self.engine.current_player != self.human_side:
            self.error = "Not your turn."
            return

        if "index" in spec and spec["index"] is not None:
            self.apply(int(spec["index"]))
            return

        kind = spec.get("type")
        if kind == "end_turn":
            legal = self.legal()
            for i, a in enumerate(legal):
                if isinstance(a, EndTurnAction):
                    self.apply(i)
                    return
            self.error = "End turn is not legal right now."
            return

        if kind == "move_path":
            ship_id = int(spec["ship_id"])
            to = spec["to"]
            self.move_ship_to(ship_id, (int(to[0]), int(to[1])))
            return

        if kind == "move":
            ship_id = int(spec["ship_id"])
            dst = (int(spec["to"][0]), int(spec["to"][1]))
            for i, a in enumerate(self.legal()):
                if isinstance(a, MoveAction) and a.ship_id == ship_id and a.destination == dst:
                    self.apply(i)
                    return
            self.error = f"Cannot move ship #{ship_id} to {dst}."
            return

        if kind == "attack":
            att_id = int(spec["attacker_id"])
            tgt_id = int(spec["target_id"])
            for i, a in enumerate(self.legal()):
                if isinstance(a, AttackAction) and a.attacker_id == att_id and a.target_id == tgt_id:
                    self.apply(i)
                    return
            self.error = f"Cannot attack target #{tgt_id} with ship #{att_id}."
            return

        if kind == "build_ship":
            port = (int(spec["port"][0]), int(spec["port"][1]))
            ship_name = str(spec["ship_type"]).strip().lower()
            for i, a in enumerate(self.legal()):
                if (
                    isinstance(a, BuildShipAction)
                    and a.port == port
                    and a.ship_type.value.lower() == ship_name
                ):
                    self.apply(i)
                    return
            self.error = f"Cannot build {spec.get('ship_type')} at port {port}."
            return

        if kind == "capture":
            ship_id = int(spec["ship_id"])
            target = (int(spec["target"][0]), int(spec["target"][1]))
            for i, a in enumerate(self.legal()):
                if (
                    isinstance(a, CapturePortAction)
                    and a.landing_ship_id == ship_id
                    and a.target == target
                ):
                    self.apply(i)
                    return
            self.error = f"Cannot capture port {target} with ship #{ship_id}."
            return

        if kind == "build_port":
            ship_id = int(spec["ship_id"])
            target = (int(spec["target"][0]), int(spec["target"][1]))
            for i, a in enumerate(self.legal()):
                if (
                    isinstance(a, BuildPortAction)
                    and a.builder_ship_id == ship_id
                    and a.target == target
                ):
                    self.apply(i)
                    return
            self.error = f"Cannot build port at {target} with ship #{ship_id}."
            return

        if kind == "load":
            ship_id = int(spec["ship_id"])
            port = (int(spec["port"][0]), int(spec["port"][1]))
            for i, a in enumerate(self.legal()):
                if (
                    isinstance(a, MerchantLoadAction)
                    and a.merchant_id == ship_id
                    and a.port == port
                ):
                    self.apply(i)
                    return
            self.error = f"Cannot load merchant #{ship_id} at {port}."
            return

        if kind == "unload":
            ship_id = int(spec["ship_id"])
            port = (int(spec["port"][0]), int(spec["port"][1]))
            for i, a in enumerate(self.legal()):
                if (
                    isinstance(a, MerchantUnloadAction)
                    and a.merchant_id == ship_id
                    and a.port == port
                ):
                    self.apply(i)
                    return
            self.error = f"Cannot unload merchant #{ship_id} at {port}."
            return

        self.error = f"Unknown action type: {kind!r}"

    def _human_known_blocked_water(self) -> set:
        """Water tiles the human KNOWS are occupied (own ships + fresh enemy sightings)."""
        human = self.engine.players[self.human_side]
        blocked = {
            self.engine.ships[sid].position
            for sid in human.owned_ship_ids
            if sid in self.engine.ships
        }
        for s in human.sightings.values():
            if s.fresh:
                blocked.add(s.position)
        return blocked

    def reachable_tiles(self, ship_id: int) -> List[List[int]]:
        """Water tiles reachable by ``ship_id`` within remaining speed on known water."""
        ship = self.engine.ships.get(ship_id)
        if (
            ship is None
            or ship.owner != self.human_side
            or ship.bought_this_turn
            or self.over
            or self.engine.current_player != self.human_side
        ):
            return []
        budget = ship.stats.speed - ship.tiles_moved_this_turn
        if budget <= 0:
            return []
        blocked = self._human_known_blocked_water()
        blocked.discard(ship.position)
        m = self.engine.map
        dist: Dict[Pos, int] = {ship.position: 0}
        q = deque([ship.position])
        out: List[List[int]] = []
        while q:
            cur = q.popleft()
            d = dist[cur]
            if d >= budget:
                continue
            for dx, dy in _CARDINALS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in dist or not m.in_bounds(nxt):
                    continue
                if not m.is_water(nxt) or nxt in blocked:
                    continue
                dist[nxt] = d + 1
                q.append(nxt)
                out.append([nxt[0], nxt[1]])
        return out

    def move_ship_to(self, ship_id: int, destination: Pos) -> None:
        """Move ``ship_id`` step-by-step along shortest known water path to ``destination``."""
        self.error = ""
        ship = self.engine.ships.get(ship_id)
        if ship is None or ship.owner != self.human_side:
            self.error = f"Ship #{ship_id} is not yours."
            return
        if ship.bought_this_turn:
            self.error = "Ships bought this turn cannot act until next turn."
            return
        budget = ship.stats.speed - ship.tiles_moved_this_turn
        if budget <= 0:
            self.error = "Ship has no moves left this turn."
            return
        if destination == ship.position:
            return

        blocked = self._human_known_blocked_water()
        blocked.discard(ship.position)
        m = self.engine.map
        parent: Dict[Pos, Optional[Pos]] = {ship.position: None}
        depth: Dict[Pos, int] = {ship.position: 0}
        q = deque([ship.position])
        found = False
        while q:
            cur = q.popleft()
            if cur == destination:
                found = True
                break
            if depth[cur] >= budget:
                continue
            for dx, dy in _CARDINALS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in parent or not m.in_bounds(nxt):
                    continue
                if not m.is_water(nxt) or nxt in blocked:
                    continue
                parent[nxt] = cur
                depth[nxt] = depth[cur] + 1
                q.append(nxt)

        if not found:
            self.error = f"Destination {destination} is not reachable this turn."
            return

        path: List[Pos] = []
        cur: Optional[Pos] = destination
        while cur is not None and cur != ship.position:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        for nxt in path:
            if self.over:
                break
            if m.is_occupied(nxt):
                # Refresh sightings ensures the blocking enemy is now spotted; stop cleanly.
                self.engine._refresh_sightings()
                self.error = f"Movement stopped at {ship.position}: {nxt} is occupied."
                self._add_log(f"Movement stopped: enemy spotted blocking {nxt}!", kind="spot")
                break
            self._apply(MoveAction(ship_id, nxt), self.human_side)

    def board_cells(self) -> List[List[dict]]:
        """Rows from high y to low y, so (0, 0) sits at the bottom-left."""
        engine = self.engine
        me = self.human_side
        player = engine.players[me]
        w, h = engine.map.width, engine.map.height
        rows = []
        for y in range(h - 1, -1, -1):
            row = []
            for x in range(w):
                pos = (x, y)
                cell = {
                    "x": x,
                    "y": y,
                    "terrain": "land" if engine.map.is_land(pos) else "water",
                    "port": None,
                    "ship": None,
                    "home": False,
                    "stale": False,
                }
                if pos in player.owned_port_positions:
                    port = engine.ports[pos]
                    cell["port"] = "own"
                    cell["home"] = bool(port.is_home)
                else:
                    sight = player.port_sightings.get(pos)
                    if sight is not None:
                        cell["port"] = "enemy"
                        cell["home"] = bool(sight.is_home)
                        cell["stale"] = not sight.fresh
                    elif engine.map.is_port(pos):
                        # Unsighted port: show as land so the enemy base stays hidden.
                        cell["terrain"] = "land"
                        cell["port"] = None
                own_sid = engine.map.ship_id_at(pos)
                if own_sid >= 0 and own_sid in player.owned_ship_ids:
                    ship = engine.ships[own_sid]
                    cell["ship"] = _SHIP_CODE.get(ship.type, "?")
                    cell["ship_owner"] = "own"
                else:
                    for sight in player.sightings.values():
                        if sight.position == pos:
                            cell["ship"] = _SHIP_CODE.get(sight.type, "?")
                            cell["ship_owner"] = "enemy"
                            cell["stale"] = not sight.fresh
                            break
                row.append(cell)
            rows.append(row)
        return rows

    def status(self) -> dict:
        me = self.engine.players[self.human_side]
        sighted = sum(1 for s in me.sightings.values() if s.fresh)
        return {
            "turn": self.engine.turn,
            "max_turns": self.max_turns,
            "cash": me.cash,
            "ships": len(me.owned_ship_ids),
            "ports": len(me.owned_port_positions),
            "sighted_ships": sighted,
            "to_move": None if self.over else self.engine.current_player,
            "over": self.over,
            "winner": self.engine.winner if self.over else None,
            "result": self.result_text,
            "opponent": self.opponent,
            "opponent_label": self.opponent_label,
            "human_side": self.human_side,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "map_name": self.map_name,
            "width": self.engine.map.width,
            "height": self.engine.map.height,
            "seed": self.seed,
            "thinking": bool(self.thinking),
            "thinking_message": self.thinking_message,
            "thinking_for_s": (
                round(time.monotonic() - self.thinking_started, 1)
                if self.thinking and self.thinking_started is not None
                else None
            ),
            "llm_model": self.llm_model if self._uses_llm() else "",
            "llm_reasoning": self.llm_reasoning if self._uses_llm() else "",
            "last_llm": self._public_last_llm(),
            "llm_totals": dict(self.llm_totals),
            "log_path": str(self.log_path) if getattr(self, "log_path", None) else "",
        }

    def to_state_dict(self) -> dict:
        """Complete fog-of-war-safe state snapshot for the Canvas UI."""
        engine = self.engine
        me_id = self.human_side
        me = engine.players[me_id]
        w, h = engine.map.width, engine.map.height

        # Terrain grid: rows y=0..h-1, x=0..w-1 (0=water, 1=land).
        # Unsighted ports stay masked as land (1).
        terrain = [
            [1 if engine.map.is_land((x, y)) else 0 for x in range(w)]
            for y in range(h)
        ]

        legal = self.legal()

        # Index legal actions by ship/port for fast UI lookups.
        buildable_ports: Dict[Pos, List[str]] = {}
        attack_targets_by_ship: Dict[int, List[dict]] = {}
        special_actions_by_ship: Dict[int, List[dict]] = {}
        legal_records: List[dict] = []

        for idx, action in enumerate(legal):
            rec = {"index": idx, "label": describe_action(action, engine, viewer_id=me_id)}
            if isinstance(action, MoveAction):
                rec.update({"type": "move", "ship_id": action.ship_id, "to": list(action.destination)})
            elif isinstance(action, AttackAction):
                att = engine.ships.get(action.attacker_id)
                sight = me.sightings.get(action.target_id)
                prob = None
                if att is not None and sight is not None:
                    prob = round(
                        pk(att, _TypeStandIn(sight.type), engine.kill_curve_k), 3
                    )
                rec.update(
                    {
                        "type": "attack",
                        "attacker_id": action.attacker_id,
                        "target_id": action.target_id,
                        "pk": prob,
                    }
                )
                attack_targets_by_ship.setdefault(action.attacker_id, []).append(
                    {
                        "index": idx,
                        "target_id": action.target_id,
                        "pos": list(sight.position) if sight else None,
                        "code": _SHIP_CODE.get(sight.type, "?") if sight else "?",
                        "ship_type": sight.type.value if sight else "Unknown",
                        "pk": prob,
                    }
                )
            elif isinstance(action, BuildShipAction):
                rec.update(
                    {
                        "type": "build_ship",
                        "port": list(action.port),
                        "ship_type": action.ship_type.value,
                        "cost": BASE_STATS[action.ship_type].cost,
                    }
                )
                buildable_ports.setdefault(action.port, []).append(action.ship_type.value)
            elif isinstance(action, BuildPortAction):
                rec.update(
                    {
                        "type": "build_port",
                        "ship_id": action.builder_ship_id,
                        "target": list(action.target),
                    }
                )
                special_actions_by_ship.setdefault(action.builder_ship_id, []).append(rec)
            elif isinstance(action, CapturePortAction):
                rec.update(
                    {
                        "type": "capture",
                        "ship_id": action.landing_ship_id,
                        "target": list(action.target),
                    }
                )
                special_actions_by_ship.setdefault(action.landing_ship_id, []).append(rec)
            elif isinstance(action, MerchantLoadAction):
                rec.update(
                    {
                        "type": "load",
                        "ship_id": action.merchant_id,
                        "port": list(action.port),
                    }
                )
                special_actions_by_ship.setdefault(action.merchant_id, []).append(rec)
            elif isinstance(action, MerchantUnloadAction):
                rec.update(
                    {
                        "type": "unload",
                        "ship_id": action.merchant_id,
                        "port": list(action.port),
                    }
                )
                special_actions_by_ship.setdefault(action.merchant_id, []).append(rec)
            elif isinstance(action, EndTurnAction):
                rec.update({"type": "end_turn"})
            legal_records.append(rec)

        own_ports = []
        for pos in sorted(me.owned_port_positions):
            p = engine.ports[pos]
            own_ports.append(
                {
                    "pos": [pos[0], pos[1]],
                    "is_home": bool(p.is_home),
                    "stockpile": int(p.stockpile),
                    "locked": bool(p.locked_this_turn),
                    "buildable_types": buildable_ports.get(pos, []),
                }
            )

        enemy_ports = []
        for pos, ps in sorted(me.port_sightings.items()):
            enemy_ports.append(
                {
                    "pos": [pos[0], pos[1]],
                    "is_home": bool(ps.is_home),
                    "owner": int(ps.last_known_owner),
                    "fresh": bool(ps.fresh),
                    "turn_seen": int(ps.turn_seen),
                    "stale_turns": int(engine.turn - ps.turn_seen),
                }
            )

        own_ships = []
        for sid in sorted(me.owned_ship_ids):
            ship = engine.ships[sid]
            moves_left = 0 if ship.bought_this_turn else max(0, ship.stats.speed - ship.tiles_moved_this_turn)
            can_attack = bool(
                not ship.bought_this_turn
                and not ship.has_attacked
                and ship.stats.attack_range > 0
            )
            own_ships.append(
                {
                    "id": ship.id,
                    "type": ship.type.value,
                    "code": _SHIP_CODE[ship.type],
                    "pos": [ship.position[0], ship.position[1]],
                    "speed": int(ship.stats.speed),
                    "tiles_moved": int(ship.tiles_moved_this_turn),
                    "moves_left": int(moves_left),
                    "strength": float(ship.stats.strength),
                    "attack_range": int(ship.stats.attack_range),
                    "cost": int(ship.stats.cost),
                    "has_attacked": bool(ship.has_attacked),
                    "can_attack": can_attack,
                    "bought_this_turn": bool(ship.bought_this_turn),
                    "cargo": int(ship.cargo),
                    "reachable": self.reachable_tiles(ship.id),
                    "attack_targets": attack_targets_by_ship.get(ship.id, []),
                    "special_actions": special_actions_by_ship.get(ship.id, []),
                    "spot_surface": int(spot_distance(ship.type, ShipType.CARRIER)),
                    "spot_sub": int(spot_distance(ship.type, ShipType.SUBMARINE)),
                    "spot_port": int(spot_distance(ship.type, PORT)),
                }
            )

        enemy_ships = []
        for sid, sight in sorted(me.sightings.items()):
            stats = BASE_STATS[sight.type]
            enemy_ships.append(
                {
                    "id": sight.ship_id,
                    "type": sight.type.value,
                    "code": _SHIP_CODE[sight.type],
                    "pos": [sight.position[0], sight.position[1]],
                    "fresh": bool(sight.fresh),
                    "turn_seen": int(sight.turn_seen),
                    "stale_turns": int(engine.turn - sight.turn_seen),
                    "strength": float(stats.strength),
                    "attack_range": int(stats.attack_range),
                    "speed": int(stats.speed),
                }
            )

        ship_catalog = [
            {
                "type": t.value,
                "code": _SHIP_CODE[t],
                "cost": int(BASE_STATS[t].cost),
                "strength": float(BASE_STATS[t].strength),
                "attack_range": int(BASE_STATS[t].attack_range),
                "speed": int(BASE_STATS[t].speed),
            }
            for t in ShipType
        ]

        return {
            "status": self.status(),
            "terrain": terrain,
            "own_ports": own_ports,
            "enemy_ports": enemy_ports,
            "own_ships": own_ships,
            "enemy_ships": enemy_ships,
            "ship_catalog": ship_catalog,
            "legal_actions": legal_records,
            "log": self.entries[-60:],
            "recent_opponent_moves": self.recent_opponent_moves,
            "recent_combat": self.recent_combat,
            "error": self.error,
        }

    def render_html(self) -> str:
        """Render the interactive Canvas Play UI page with embedded initial state."""
        st = self.status()
        state_json = json.dumps(self.to_state_dict()).replace("</", "<\\/")
        opp_text = html.escape(self.opponent_label)
        log_items = "".join(
            f'<li class="log-{html.escape(e["kind"])}">{html.escape(e["text"])}</li>'
            for e in self.entries[-18:]
        )
        err_html = f'<p id="errorBanner" class="err">{html.escape(self.error)}</p>' if self.error else '<p id="errorBanner" class="err" hidden></p>'
        result_html = f'<div id="resultBanner" class="result">{html.escape(st["result"])}</div>' if st["result"] else '<div id="resultBanner" class="result" hidden></div>'

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Battleboats — Play vs {opp_text}</title>
<style>
  :root {{
    --bg: #0b1118;
    --panel: #131c26;
    --border: #233243;
    --text: #e7ecf1;
    --muted: #93a4b8;
    --accent: #38bdf8;
    --gold: #ffd400;
    --danger: #ef4444;
    --ok: #22c55e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0;
    padding: 0.9rem 1.2rem;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .topbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.65rem 0.9rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.75rem;
  }}
  .stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
  }}
  .stat-val {{ font-weight: 700; color: #fff; }}
  .cash-val {{ font-weight: 700; color: var(--gold); }}
  .controls {{ display: flex; gap: 0.5rem; align-items: center; }}
  button {{
    background: #1d3557;
    color: #fff;
    border: 1px solid #2e4c77;
    border-radius: 4px;
    padding: 0.4rem 0.75rem;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }}
  button:hover:not(:disabled) {{ background: #264875; }}
  button:disabled {{ opacity: 0.42; cursor: not-allowed; }}
  button.primary {{ background: #0f766e; border-color: #14b8a6; }}
  button.primary:hover:not(:disabled) {{ background: #115e59; }}
  button.danger {{ background: #991b1b; border-color: #ef4444; }}
  button.danger:hover:not(:disabled) {{ background: #b91c1c; }}
  .err {{
    background: #451a1a;
    border: 1px solid #b91c1c;
    color: #fecaca;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    margin: 0 0 0.75rem 0;
  }}
  .result {{
    background: #172536;
    border: 2px solid var(--gold);
    color: var(--gold);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }}
  .workspace {{
    display: grid;
    grid-template-columns: minmax(320px, 1fr) 360px;
    gap: 0.9rem;
    align-items: start;
  }}
  @media (max-width: 960px) {{
    .workspace {{ grid-template-columns: 1fr; }}
  }}
  .board-wrap {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.65rem;
    overflow: auto;
  }}
  canvas#board {{
    display: block;
    cursor: pointer;
    border-radius: 3px;
  }}
  .legend {{
    margin-top: 0.5rem;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.45;
  }}
  .sidebar {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.7rem 0.85rem;
  }}
  .card h2 {{
    margin: 0 0 0.5rem 0;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
  }}
  .kv {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 0.75rem;
    font-size: 12px;
  }}
  .kv .k {{ color: var(--muted); }}
  .btn-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
    margin-top: 0.45rem;
  }}
  .ship-btn {{
    text-align: left;
    padding: 0.38rem 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.3rem;
  }}
  .ship-btn small {{ color: var(--muted); font-weight: 400; font-size: 10px; }}
  .act-list {{ display: flex; flex-direction: column; gap: 0.35rem; margin-top: 0.45rem; }}
  .log-list {{
    margin: 0;
    padding-left: 1.2rem;
    max-height: 260px;
    overflow-y: auto;
    font-size: 12px;
    line-height: 1.4;
  }}
  .log-list li {{ margin-bottom: 0.22rem; }}
  .log-combat_win {{ color: #4ade80; font-weight: 600; }}
  .log-combat_loss {{ color: #f87171; font-weight: 600; }}
  .log-combat_miss {{ color: var(--muted); }}
  .log-spot {{ color: #fbbf24; }}
  .log-enemy_move {{ color: #fda4af; }}
  .log-turn {{ color: var(--accent); font-weight: 600; list-style: none; margin-left: -1rem; margin-top: 0.35rem; }}
  .muted {{ color: var(--muted); }}
  .port-tabs {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.45rem; }}
  .port-tab {{ padding: 0.25rem 0.5rem; font-size: 11px; }}
  .port-tab.active {{ background: #1e40af; border-color: var(--gold); }}
  .thinking {{
    background: #1e3a5f;
    border: 1px solid #38bdf8;
    color: #bae6fd;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.75rem;
    font-weight: 700;
  }}
  .log-llm {{ color: #7dd3fc; }}
  .log-error {{ color: #f87171; font-weight: 600; }}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <strong>Battleboats</strong> ·
    You: <span class="stat-val">Player {st['human_side']}</span> ·
    Opponent: <span class="stat-val">{opp_text}</span> ·
    Map: <span class="stat-val">{html.escape(str(st['map_name']))}</span>
    (<span id="boardDims">{st['width']}x{st['height']}</span>, seed {st['seed']})
  </div>
  <div class="stats">
    <span>Turn <span id="turnStat" class="stat-val">{st['turn']}/{st['max_turns']}</span></span>
    <span>Cash <span id="cashStat" class="cash-val">${st['cash']}</span></span>
    <span>Ships <span id="shipsStat" class="stat-val">{st['ships']}</span></span>
    <span>Ports <span id="portsStat" class="stat-val">{st['ports']}</span></span>
    <span>Sighted <span id="sightedStat" class="stat-val">{st['sighted_ships']}</span></span>
    <span id="llmStat" class="muted"></span>
  </div>
  <div class="controls">
    <button id="nextShipBtn" title="Cycle ready ships (Tab)">Next Ship</button>
    <button id="endTurnBtn" class="primary" title="End Turn (E)">End Turn</button>
    <a href="/?new=1"><button type="button">New Game</button></a>
  </div>
</div>

{result_html}
{err_html}
<div id="thinkingBanner" class="thinking" hidden>Inkling thinking…</div>

<div class="workspace">
  <div class="board-wrap">
    <canvas id="board"></canvas>
    <div id="hoverInfo" class="legend">Click a friendly ship to move/attack, or a friendly port to buy ships. Keys: Arrows/WASD move, Tab next ship, E end turn.</div>
    <div class="legend">
      <strong>Legend:</strong>
      Blue = your ships · Gold outline = your ports (H home, P satellite) ·
      Bright Red = enemy in sight · Faded Red = stale last-known sighting ·
      Unsighted enemy ports appear as land.
      Codes: CV Carrier, BB Battleship, CA Cruiser, DD Destroyer, SS Submarine, AK Merchant, LC Landing, CB Builder.
    </div>
  </div>

  <div class="sidebar">
    <div class="card" id="inspectorCard">
      <h2>Selection</h2>
      <div id="inspectorBody"><p class="muted">Select one of your ships or ports on the grid.</p></div>
    </div>

    <div class="card" id="shipyardCard">
      <h2>Shipyard</h2>
      <div id="portTabs" class="port-tabs"></div>
      <div id="buyButtons" class="btn-grid"></div>
    </div>

    <div class="card">
      <h2>Turn Log</h2>
      <ol id="logList" class="log-list">{log_items}</ol>
    </div>
  </div>
</div>

<script>
const INITIAL_STATE = {state_json};
(function() {{
  let state = INITIAL_STATE;
  let selectedShipId = null;
  let selectedPortKey = null;
  let hoverTile = null;
  let busy = false;
  let pollTimer = null;
  let cellSize = 36;
  const PAD_L = 26;
  const PAD_B = 22;
  const PAD_T = 10;
  const PAD_R = 10;

  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  const hoverInfo = document.getElementById("hoverInfo");
  const errorBanner = document.getElementById("errorBanner");
  const resultBanner = document.getElementById("resultBanner");
  const thinkingBanner = document.getElementById("thinkingBanner");

  function posKey(p) {{ return p[0] + "," + p[1]; }}

  function pickCellSize(w, h) {{
    const maxW = Math.max(420, window.innerWidth - 440);
    const maxH = Math.max(300, window.innerHeight - 230);
    const s = Math.floor(Math.min((maxW - PAD_L - PAD_R) / w, (maxH - PAD_T - PAD_B) / h));
    return Math.max(20, Math.min(48, s));
  }}

  function gridToCanvas(gx, gy, h) {{
    return [PAD_L + gx * cellSize, PAD_T + (h - 1 - gy) * cellSize];
  }}

  function canvasToGrid(cx, cy, w, h) {{
    const gx = Math.floor((cx - PAD_L) / cellSize);
    const gy = h - 1 - Math.floor((cy - PAD_T) / cellSize);
    if (gx < 0 || gx >= w || gy < 0 || gy >= h) return null;
    return [gx, gy];
  }}

  function selectedShip() {{
    if (selectedShipId === null) return null;
    return state.own_ships.find(s => s.id === selectedShipId) || null;
  }}

  function syncSelections() {{
    if (selectedShipId !== null && !state.own_ships.some(s => s.id === selectedShipId)) {{
      selectedShipId = null;
    }}
    if (state.own_ports.length > 0) {{
      if (!selectedPortKey || !state.own_ports.some(p => posKey(p.pos) === selectedPortKey)) {{
        const unlocked = state.own_ports.find(p => !p.locked) || state.own_ports[0];
        selectedPortKey = posKey(unlocked.pos);
      }}
    }} else {{
      selectedPortKey = null;
    }}
  }}

  function formatLlmStat(st) {{
    const last = st.last_llm || {{}};
    const parts = [];
    if (last.latency_s != null) parts.push(Number(last.latency_s).toFixed(1) + "s");
    if (last.total_tokens != null) {{
      let tok = Number(last.total_tokens).toLocaleString() + " tok";
      if (last.reasoning_tokens) tok += " (" + Number(last.reasoning_tokens).toLocaleString() + " reason)";
      parts.push(tok);
    }}
    if (last.cost_usd != null) parts.push("$" + Number(last.cost_usd).toFixed(4));
    if (!parts.length) return "";
    return "Last LLM: " + parts.join(" · ");
  }}

  function ensurePoll() {{
    if (state.status && state.status.thinking) {{
      if (!pollTimer) {{
        pollTimer = setInterval(async () => {{
          try {{
            const resp = await fetch("/api/state");
            if (!resp.ok) return;
            state = await resp.json();
            syncSelections();
            render();
          }} catch (err) {{}}
        }}, 600);
      }}
    }} else if (pollTimer) {{
      clearInterval(pollTimer);
      pollTimer = null;
    }}
  }}

  async function sendAction(payload) {{
    if (busy || (state.status && state.status.thinking)) return;
    busy = true;
    try {{
      const resp = await fetch("/api/action", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      state = await resp.json();
      syncSelections();
      render();
    }} finally {{
      busy = false;
    }}
  }}

  function drawShipBadge(cx, cy, code, fill, stroke, alpha, subLabel) {{
    ctx.save();
    ctx.globalAlpha = alpha;
    const r = Math.max(7, cellSize * 0.37);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold " + Math.max(9, Math.floor(cellSize * 0.34)) + "px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(code, cx, cy + 0.5);

    if (subLabel) {{
      ctx.font = "bold " + Math.max(8, Math.floor(cellSize * 0.23)) + "px sans-serif";
      ctx.fillStyle = "#ffd400";
      ctx.fillText(subLabel, cx + r * 0.72, cy - r * 0.72);
    }}
    ctx.restore();
  }}

  function drawBoard() {{
    const w = state.status.width;
    const h = state.status.height;
    cellSize = pickCellSize(w, h);
    canvas.width = PAD_L + w * cellSize + PAD_R;
    canvas.height = PAD_T + h * cellSize + PAD_B;

    ctx.fillStyle = "#0b1118";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const ownPortMap = new Map(state.own_ports.map(p => [posKey(p.pos), p]));
    const enemyPortMap = new Map(state.enemy_ports.map(p => [posKey(p.pos), p]));
    const ship = selectedShip();
    const reachSet = new Set((ship ? ship.reachable : []).map(posKey));
    const attackMap = new Map((ship ? ship.attack_targets : []).filter(t => t.pos).map(t => [posKey(t.pos), t]));
    const specialMap = new Map((ship ? ship.special_actions : []).map(a => [posKey(a.target || a.port), a]));

    // Terrain & ports
    for (let gy = 0; gy < h; gy++) {{
      for (let gx = 0; gx < w; gx++) {{
        const [px, py] = gridToCanvas(gx, gy, h);
        const key = gx + "," + gy;
        const isLand = state.terrain[gy][gx] === 1;
        ctx.fillStyle = isLand ? "#2d6a4f" : "#154372";
        ctx.fillRect(px, py, cellSize, cellSize);

        const op = ownPortMap.get(key);
        const ep = enemyPortMap.get(key);
        if (op) {{
          ctx.fillStyle = "#1d3557";
          ctx.fillRect(px + 1, py + 1, cellSize - 2, cellSize - 2);
          ctx.strokeStyle = selectedPortKey === key ? "#ffffff" : "#ffd400";
          ctx.lineWidth = selectedPortKey === key ? 3 : 2;
          ctx.strokeRect(px + 2, py + 2, cellSize - 4, cellSize - 4);
          ctx.fillStyle = "#ffd400";
          ctx.font = "bold " + Math.max(9, Math.floor(cellSize * 0.34)) + "px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(op.is_home ? "H" : "P", px + cellSize / 2, py + cellSize / 2);
          if (op.stockpile > 0) {{
            ctx.fillStyle = "#bae6fd";
            ctx.font = "bold " + Math.max(7, Math.floor(cellSize * 0.22)) + "px sans-serif";
            ctx.fillText(String(op.stockpile), px + cellSize - 8, py + cellSize - 6);
          }}
        }} else if (ep) {{
          ctx.save();
          ctx.globalAlpha = ep.fresh ? 1.0 : 0.52;
          ctx.fillStyle = "#7f1d1d";
          ctx.fillRect(px + 1, py + 1, cellSize - 2, cellSize - 2);
          ctx.strokeStyle = "#ef4444";
          ctx.lineWidth = 2;
          ctx.strokeRect(px + 2, py + 2, cellSize - 4, cellSize - 4);
          ctx.fillStyle = "#fecaca";
          ctx.font = "bold " + Math.max(9, Math.floor(cellSize * 0.34)) + "px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(ep.is_home ? "H" : "P", px + cellSize / 2, py + cellSize / 2);
          ctx.restore();
        }}

        // Reachable move highlight
        if (reachSet.has(key)) {{
          ctx.fillStyle = "rgba(34, 197, 94, 0.26)";
          ctx.fillRect(px + 1, py + 1, cellSize - 2, cellSize - 2);
          ctx.fillStyle = "rgba(74, 222, 128, 0.7)";
          ctx.beginPath();
          ctx.arc(px + cellSize / 2, py + cellSize / 2, Math.max(2, cellSize * 0.09), 0, Math.PI * 2);
          ctx.fill();
        }}

        // Special action target highlight (capture / build_port / load / unload)
        if (specialMap.has(key)) {{
          ctx.strokeStyle = "#fb923c";
          ctx.lineWidth = 3;
          ctx.strokeRect(px + 2, py + 2, cellSize - 4, cellSize - 4);
        }}

        // Grid line
        ctx.strokeStyle = "#1d2d3e";
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, cellSize, cellSize);
      }}
    }}

    // Axis labels
    ctx.fillStyle = "#7b8fa6";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const stepX = w > 28 ? 5 : (w > 16 ? 2 : 1);
    for (let gx = 0; gx < w; gx += stepX) {{
      const [px, _] = gridToCanvas(gx, 0, h);
      ctx.fillText(String(gx), px + cellSize / 2, PAD_T + h * cellSize + 11);
    }}
    ctx.textAlign = "right";
    const stepY = h > 16 ? 2 : 1;
    for (let gy = 0; gy < h; gy += stepY) {{
      const [_, py] = gridToCanvas(0, gy, h);
      ctx.fillText(String(gy), PAD_L - 5, py + cellSize / 2);
    }}

    // Visible opponent move trails
    for (const mv of state.recent_opponent_moves) {{
      if (mv.kind === "move" && mv.from && mv.to) {{
        const [fx, fy] = gridToCanvas(mv.from[0], mv.from[1], h);
        const [tx, ty] = gridToCanvas(mv.to[0], mv.to[1], h);
        ctx.save();
        ctx.strokeStyle = "rgba(251, 146, 60, 0.85)";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(fx + cellSize / 2, fy + cellSize / 2);
        ctx.lineTo(tx + cellSize / 2, ty + cellSize / 2);
        ctx.stroke();
        ctx.restore();
      }}
    }}

    // Combat lines/markers
    for (const cb of state.recent_combat) {{
      if (cb.to) {{
        const [tx, ty] = gridToCanvas(cb.to[0], cb.to[1], h);
        if (cb.from) {{
          const [fx, fy] = gridToCanvas(cb.from[0], cb.from[1], h);
          ctx.save();
          ctx.strokeStyle = cb.killed ? "#facc15" : "#94a3b8";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(fx + cellSize / 2, fy + cellSize / 2);
          ctx.lineTo(tx + cellSize / 2, ty + cellSize / 2);
          ctx.stroke();
          ctx.restore();
        }}
        ctx.save();
        ctx.strokeStyle = cb.killed ? "#facc15" : "#cbd5e1";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(tx + 3, ty + 3, cellSize - 6, cellSize - 6);
        ctx.restore();
      }}
    }}

    // Enemy ships (fresh and stale sightings only)
    for (const es of state.enemy_ships) {{
      const [px, py] = gridToCanvas(es.pos[0], es.pos[1], h);
      const cx = px + cellSize / 2;
      const cy = py + cellSize / 2;
      const isTarget = attackMap.has(posKey(es.pos));
      drawShipBadge(
        cx,
        cy,
        es.code,
        es.fresh ? "#dc2626" : "#7f1d1d",
        isTarget ? "#ffd400" : (es.fresh ? "#fca5a5" : "#991b1b"),
        es.fresh ? 1.0 : 0.55,
        es.fresh ? "" : "?"
      );
      if (isTarget) {{
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3;
        ctx.strokeRect(px + 2, py + 2, cellSize - 4, cellSize - 4);
      }}
    }}

    // Own ships
    for (const os of state.own_ships) {{
      const [px, py] = gridToCanvas(os.pos[0], os.pos[1], h);
      const cx = px + cellSize / 2;
      const cy = py + cellSize / 2;
      const isSel = selectedShipId === os.id;
      const ready = os.moves_left > 0 || os.can_attack || os.special_actions.length > 0;
      const fill = isSel ? "#0284c7" : (ready ? "#1d4ed8" : "#1e3a5f");
      const stroke = isSel ? "#ffd400" : (ready ? "#7dd3fc" : "#475569");
      const badge = os.bought_this_turn ? "N" : (os.cargo > 0 ? String(os.cargo) : "");
      drawShipBadge(cx, cy, os.code, fill, stroke, os.bought_this_turn ? 0.65 : 1.0, badge);
      if (isSel) {{
        ctx.strokeStyle = "#ffd400";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(px + 1.5, py + 1.5, cellSize - 3, cellSize - 3);
      }}
    }}

    // Hover outline
    if (hoverTile) {{
      const [hx, hy] = gridToCanvas(hoverTile[0], hoverTile[1], h);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hx + 1, hy + 1, cellSize - 2, cellSize - 2);
    }}
  }}

  function renderStats() {{
    const st = state.status;
    document.getElementById("turnStat").textContent = st.turn + "/" + st.max_turns;
    document.getElementById("cashStat").textContent = "$" + st.cash;
    document.getElementById("shipsStat").textContent = String(st.ships);
    document.getElementById("portsStat").textContent = String(st.ports);
    document.getElementById("sightedStat").textContent = String(st.sighted_ships);
    const blocked = st.over || st.thinking || st.to_move !== st.human_side;
    document.getElementById("endTurnBtn").disabled = blocked;
    document.getElementById("nextShipBtn").disabled = st.over || st.thinking || state.own_ships.length === 0;
    const llmStat = document.getElementById("llmStat");
    if (llmStat) llmStat.textContent = formatLlmStat(st);
    if (st.thinking) {{
      let msg = st.thinking_message || "Inkling thinking…";
      if (st.thinking_for_s != null) msg += " (" + st.thinking_for_s + "s)";
      thinkingBanner.textContent = msg;
      thinkingBanner.hidden = false;
    }} else {{
      thinkingBanner.hidden = true;
    }}

    if (state.error) {{
      errorBanner.textContent = state.error;
      errorBanner.hidden = false;
    }} else {{
      errorBanner.hidden = true;
    }}

    if (st.result) {{
      resultBanner.innerHTML = "";
      const span = document.createElement("span");
      span.textContent = st.result;
      const a = document.createElement("a");
      a.href = "/?new=1";
      const btn = document.createElement("button");
      btn.textContent = "Play Again / New Game";
      a.appendChild(btn);
      resultBanner.appendChild(span);
      resultBanner.appendChild(a);
      if (st.log_path) {{
        const logLine = document.createElement("div");
        logLine.className = "muted";
        logLine.style.fontWeight = "400";
        logLine.style.fontSize = "12px";
        logLine.style.marginTop = "0.35rem";
        logLine.textContent = "Game log: " + st.log_path;
        resultBanner.appendChild(logLine);
      }}
      resultBanner.hidden = false;
    }} else {{
      resultBanner.hidden = true;
    }}
  }}

  function renderInspector() {{
    const body = document.getElementById("inspectorBody");
    const ship = selectedShip();
    if (!ship) {{
      const port = state.own_ports.find(p => posKey(p.pos) === selectedPortKey);
      if (port) {{
        body.innerHTML =
          '<div class="kv">' +
          '<span class="k">Port</span><span>(' + port.pos[0] + ',' + port.pos[1] + ')' + (port.is_home ? ' (Home)' : ' (Satellite)') + '</span>' +
          '<span class="k">Stockpile</span><span>' + port.stockpile + '</span>' +
          '<span class="k">Status</span><span>' + (port.locked ? 'Locked this turn' : 'Ready to build') + '</span>' +
          '</div>';
      }} else {{
        body.innerHTML = '<p class="muted">Select one of your ships or ports on the grid.</p>';
      }}
      return;
    }}

    let html = '<div class="kv">' +
      '<span class="k">Unit</span><span><strong>' + ship.type + ' (' + ship.code + ' #' + ship.id + ')</strong></span>' +
      '<span class="k">Position</span><span>(' + ship.pos[0] + ',' + ship.pos[1] + ')</span>' +
      '<span class="k">Moves left</span><span>' + ship.moves_left + ' / ' + ship.speed + (ship.bought_this_turn ? ' (bought this turn)' : '') + '</span>' +
      '<span class="k">Combat</span><span>Str ' + ship.strength + ' · Range ' + ship.attack_range + ' · ' + (ship.can_attack ? 'Ready' : (ship.has_attacked ? 'Fired' : 'No gun')) + '</span>' +
      '<span class="k">Spotting</span><span>Surface ' + ship.spot_surface + ' · Sub ' + ship.spot_sub + ' · Port ' + ship.spot_port + '</span>';
    if (ship.type === "Merchant") {{
      html += '<span class="k">Cargo</span><span>' + ship.cargo + ' / 100</span>';
    }}
    html += '</div>';

    if (ship.attack_targets.length > 0 || ship.special_actions.length > 0) {{
      html += '<div class="act-list">';
      for (const tgt of ship.attack_targets) {{
        const pct = tgt.pk !== null ? Math.round(tgt.pk * 100) + '% kill' : '';
        html += '<button class="danger" data-act-index="' + tgt.index + '">Attack ' + tgt.code + ' #' + tgt.target_id + ' (' + pct + ')</button>';
      }}
      for (const sa of ship.special_actions) {{
        html += '<button class="primary" data-act-index="' + sa.index + '">' + sa.label + '</button>';
      }}
      html += '</div>';
    }}
    body.innerHTML = html;
    body.querySelectorAll("button[data-act-index]").forEach(btn => {{
      btn.addEventListener("click", () => sendAction({{ index: Number(btn.dataset.actIndex) }}));
    }});
  }}

  function renderShipyard() {{
    const tabs = document.getElementById("portTabs");
    const btns = document.getElementById("buyButtons");
    tabs.innerHTML = "";
    btns.innerHTML = "";
    if (state.own_ports.length === 0) {{
      btns.innerHTML = '<p class="muted">No owned ports.</p>';
      return;
    }}
    for (const p of state.own_ports) {{
      const k = posKey(p.pos);
      const b = document.createElement("button");
      b.className = "port-tab" + (k === selectedPortKey ? " active" : "");
      b.textContent = (p.is_home ? "Home " : "Port ") + "(" + p.pos[0] + "," + p.pos[1] + ")" + (p.locked ? " [locked]" : "");
      b.addEventListener("click", () => {{ selectedPortKey = k; selectedShipId = null; render(); }});
      tabs.appendChild(b);
    }}
    const activePort = state.own_ports.find(p => posKey(p.pos) === selectedPortKey) || state.own_ports[0];
    const buildable = new Set(activePort.buildable_types);
    for (const item of state.ship_catalog) {{
      const b = document.createElement("button");
      b.className = "ship-btn";
      b.disabled = state.status.over || !buildable.has(item.type);
      b.title = "Str " + item.strength + ", Range " + item.attack_range + ", Speed " + item.speed;
      b.innerHTML = '<span>' + item.code + ' ' + item.type + '<br><small>Str ' + item.strength + ' · R' + item.attack_range + ' · Spd' + item.speed + '</small></span><strong>$' + item.cost + '</strong>';
      b.addEventListener("click", () => sendAction({{
        type: "build_ship",
        port: activePort.pos,
        ship_type: item.type,
      }}));
      btns.appendChild(b);
    }}
  }}

  function renderLog() {{
    const list = document.getElementById("logList");
    list.innerHTML = "";
    const recent = state.log.slice(-30);
    for (const entry of recent) {{
      const li = document.createElement("li");
      li.className = "log-" + entry.kind;
      li.textContent = entry.text;
      list.appendChild(li);
    }}
    list.scrollTop = list.scrollHeight;
  }}

  function render() {{
    syncSelections();
    renderStats();
    drawBoard();
    renderInspector();
    renderShipyard();
    renderLog();
    ensurePoll();
  }}

  canvas.addEventListener("click", (ev) => {{
    if (state.status.over || state.status.thinking) return;
    const rect = canvas.getBoundingClientRect();
    const pos = canvasToGrid(ev.clientX - rect.left, ev.clientY - rect.top, state.status.width, state.status.height);
    if (!pos) return;
    const key = posKey(pos);

    const ownShip = state.own_ships.find(s => posKey(s.pos) === key);
    if (ownShip) {{
      selectedShipId = ownShip.id;
      render();
      return;
    }}

    const ship = selectedShip();
    if (ship) {{
      const tgt = ship.attack_targets.find(t => t.pos && posKey(t.pos) === key);
      if (tgt) {{
        sendAction({{ index: tgt.index }});
        return;
      }}
      const spec = ship.special_actions.find(a => posKey(a.target || a.port) === key);
      if (spec) {{
        sendAction({{ index: spec.index }});
        return;
      }}
      if (ship.reachable.some(r => posKey(r) === key)) {{
        sendAction({{ type: "move_path", ship_id: ship.id, to: pos }});
        return;
      }}
    }}

    const ownPort = state.own_ports.find(p => posKey(p.pos) === key);
    if (ownPort) {{
      selectedShipId = null;
      selectedPortKey = key;
      render();
      return;
    }}

    selectedShipId = null;
    render();
  }});

  canvas.addEventListener("mousemove", (ev) => {{
    const rect = canvas.getBoundingClientRect();
    const pos = canvasToGrid(ev.clientX - rect.left, ev.clientY - rect.top, state.status.width, state.status.height);
    hoverTile = pos;
    if (!pos) {{
      hoverInfo.textContent = "Click a friendly ship or port.";
      drawBoard();
      return;
    }}
    const key = posKey(pos);
    const parts = ["(" + pos[0] + "," + pos[1] + ")"];
    parts.push(state.terrain[pos[1]][pos[0]] === 1 ? "Land" : "Water");
    const os = state.own_ships.find(s => posKey(s.pos) === key);
    if (os) parts.push("Your " + os.type + " #" + os.id + " (moves " + os.moves_left + "/" + os.speed + ")");
    const es = state.enemy_ships.find(s => posKey(s.pos) === key);
    if (es) parts.push((es.fresh ? "Enemy " : "Last seen ") + es.type + " #" + es.id + (es.fresh ? "" : " (turn " + es.turn_seen + ")"));
    const op = state.own_ports.find(p => posKey(p.pos) === key);
    if (op) parts.push("Your " + (op.is_home ? "Home Port" : "Port") + " (stock " + op.stockpile + ")");
    const ep = state.enemy_ports.find(p => posKey(p.pos) === key);
    if (ep) parts.push((ep.fresh ? "Enemy " : "Last seen enemy ") + (ep.is_home ? "Home Port" : "Port"));
    const ship = selectedShip();
    if (ship) {{
      const tgt = ship.attack_targets.find(t => t.pos && posKey(t.pos) === key);
      if (tgt && tgt.pk !== null) parts.push("Click to Attack (" + Math.round(tgt.pk * 100) + "% kill)");
    }}
    hoverInfo.textContent = parts.join(" · ");
    drawBoard();
  }});

  document.getElementById("endTurnBtn").addEventListener("click", () => sendAction({{ type: "end_turn" }}));

  document.getElementById("nextShipBtn").addEventListener("click", () => {{
    const ready = state.own_ships.filter(s => s.moves_left > 0 || s.can_attack || s.special_actions.length > 0);
    const pool = ready.length > 0 ? ready : state.own_ships;
    if (pool.length === 0) return;
    const idx = pool.findIndex(s => s.id === selectedShipId);
    selectedShipId = pool[(idx + 1) % pool.length].id;
    render();
  }});

  document.addEventListener("keydown", (ev) => {{
    if (ev.target && ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName)) return;
    if (state.status && state.status.thinking) return;
    if (ev.key === "e" || ev.key === "E") {{
      ev.preventDefault();
      if (!state.status.over) sendAction({{ type: "end_turn" }});
      return;
    }}
    if (ev.key === "Tab") {{
      ev.preventDefault();
      document.getElementById("nextShipBtn").click();
      return;
    }}
    if (ev.key === "Escape") {{
      selectedShipId = null;
      render();
      return;
    }}
    const ship = selectedShip();
    if (!ship) return;
    const dirs = {{
      ArrowUp: [0, 1], w: [0, 1], W: [0, 1],
      ArrowDown: [0, -1], s: [0, -1], S: [0, -1],
      ArrowLeft: [-1, 0], a: [-1, 0], A: [-1, 0],
      ArrowRight: [1, 0], d: [1, 0], D: [1, 0],
    }};
    const d = dirs[ev.key];
    if (d) {{
      ev.preventDefault();
      sendAction({{ type: "move", ship_id: ship.id, to: [ship.pos[0] + d[0], ship.pos[1] + d[1]] }});
    }}
  }});

  window.addEventListener("resize", () => drawBoard());
  render();
}})();
</script>
</body>
</html>"""

    def _apply(self, action: Action, pid: int) -> None:
        human_id = self.human_side
        human = self.engine.players[human_id]
        pre_fresh_ships = {sid: s.position for sid, s in human.sightings.items() if s.fresh}
        pre_fresh_ports = {pos for pos, s in human.port_sightings.items() if s.fresh}
        pre_own_ports = set(human.owned_port_positions)
        legal_before = list(self.engine.enumerate_legal(pid))
        state_before = serialize_engine(self.engine)
        observations = {
            str(seat): build_view(self.engine, seat, list(self.engine.enumerate_legal(seat)))
            for seat in (0, 1)
        }

        move_meta = None
        attack_meta = None
        if isinstance(action, MoveAction):
            ship = self.engine.ships.get(action.ship_id)
            if ship is not None:
                move_meta = (ship.id, ship.type, ship.position, action.destination)
        elif isinstance(action, AttackAction):
            att = self.engine.ships.get(action.attacker_id)
            tgt = self.engine.ships.get(action.target_id)
            if att is not None and tgt is not None:
                attack_meta = (
                    att.id,
                    att.type,
                    att.position,
                    tgt.id,
                    tgt.type,
                    tgt.position,
                )

        self.engine.step(action)

        human = self.engine.players[human_id]
        post_fresh_ships = {sid: s.position for sid, s in human.sightings.items() if s.fresh}
        post_fresh_ports = {pos for pos, s in human.port_sightings.items() if s.fresh}

        if pid == human_id:
            self._log_human_action(
                action,
                move_meta=move_meta,
                attack_meta=attack_meta,
                pre_fresh_ships=pre_fresh_ships,
                post_fresh_ships=post_fresh_ships,
                pre_fresh_ports=pre_fresh_ports,
                post_fresh_ports=post_fresh_ports,
            )
        else:
            self._log_opponent_action(
                action,
                move_meta=move_meta,
                attack_meta=attack_meta,
                pre_fresh_ships=pre_fresh_ships,
                post_fresh_ships=post_fresh_ships,
                pre_fresh_ports=pre_fresh_ports,
                post_fresh_ports=post_fresh_ports,
                pre_own_ports=pre_own_ports,
            )

        self._write_decision_log(
            action,
            pid,
            legal=legal_before,
            state_before=state_before,
            observations=observations,
            combat=combat_result(action, attack_meta, self.engine),
        )
        if self.over:
            self._finalize_log(complete=True)

    def close(self, complete=None) -> None:
        # Flush and close the game log. Marks incomplete unless the game ended.
        if complete is None:
            complete = bool(self.over)
        self._finalize_log(complete=complete)

    def _finalize_log_atexit(self) -> None:
        try:
            self._finalize_log(complete=bool(self.over))
        except Exception:
            pass

    def _finalize_log(self, *, complete: bool) -> None:
        log = getattr(self, "game_log", None)
        if log is None or log.closed:
            return
        if complete:
            if self.engine.winner is not None:
                reason = "capture"
            elif self.engine.turn >= self.max_turns:
                reason = "turn_limit"
            else:
                reason = "complete"
        else:
            reason = "incomplete"
        log.write(end_record(self, complete=complete, reason=reason))
        log._complete = complete
        log.close()

    def _write_decision_log(
        self,
        action: Action,
        pid: int,
        *,
        legal,
        state_before: dict,
        observations: dict,
        combat,
    ) -> None:
        log = getattr(self, "game_log", None)
        if log is None or log.closed:
            return
        self._log_steps += 1
        log.write(
            decision_record(
                self,
                action,
                pid,
                legal=legal,
                state_before=state_before,
                observations=observations,
                combat=combat,
                llm=llm_fields(self._pending_llm_meta),
                step=self._log_steps,
            )
        )
        self._pending_llm_meta = None

    def _log_human_action(
        self,
        action: Action,
        *,
        move_meta,
        attack_meta,
        pre_fresh_ships: dict,
        post_fresh_ships: dict,
        pre_fresh_ports: set,
        post_fresh_ports: set,
    ) -> None:
        human = self.engine.players[self.human_side]
        if isinstance(action, AttackAction) and attack_meta is not None:
            aid, atype, apos, tid, ttype, tpos = attack_meta
            killed = tid not in self.engine.ships
            outcome = "SUNK" if killed else "MISSED"
            kind = "combat_win" if killed else "combat_miss"
            self._add_log(
                f"You: {_SHIP_CODE[atype]} #{aid} attacked {_SHIP_CODE[ttype]} #{tid} at {tpos} — {outcome}!",
                kind=kind,
            )
            self.recent_combat.append(
                {
                    "by": "human",
                    "from": list(apos),
                    "to": list(tpos),
                    "killed": killed,
                    "outcome": outcome,
                }
            )
        elif isinstance(action, MoveAction) and move_meta is not None:
            sid, stype, src, dst = move_meta
            self._add_log(f"You: Moved {_SHIP_CODE[stype]} #{sid} {src} -> {dst}", kind="info")
        elif isinstance(action, BuildShipAction):
            self._add_log(
                f"You: Built {action.ship_type.value} at port {action.port} -> {action.spawn_position}",
                kind="info",
            )
        elif isinstance(action, BuildPortAction):
            self._add_log(f"You: Built port at {action.target}", kind="info")
        elif isinstance(action, CapturePortAction):
            self._add_log(f"You: Captured port at {action.target}!", kind="combat_win")
        elif isinstance(action, MerchantLoadAction):
            ship = self.engine.ships.get(action.merchant_id)
            cargo = ship.cargo if ship else 0
            self._add_log(
                f"You: Loaded AK #{action.merchant_id} at {action.port} (cargo {cargo})",
                kind="info",
            )
        elif isinstance(action, MerchantUnloadAction):
            self._add_log(
                f"You: Unloaded AK #{action.merchant_id} at home {action.port} "
                f"(cash ${human.cash}, +{CASH_PER_MATERIAL}/mat)",
                kind="info",
            )
        elif isinstance(action, EndTurnAction):
            self._add_log("You: Ended turn.", kind="turn")

        for sid in sorted(set(post_fresh_ships) - set(pre_fresh_ships)):
            s = human.sightings[sid]
            self._add_log(
                f"Spotted enemy {_SHIP_CODE[s.type]} #{sid} at {s.position}!",
                kind="spot",
            )
        for pos in sorted(post_fresh_ports - pre_fresh_ports):
            ps = human.port_sightings[pos]
            label = "home port" if ps.is_home else "port"
            self._add_log(f"Spotted enemy {label} at {pos}!", kind="spot")

    def _log_opponent_action(
        self,
        action: Action,
        *,
        move_meta,
        attack_meta,
        pre_fresh_ships: dict,
        post_fresh_ships: dict,
        pre_fresh_ports: set,
        post_fresh_ports: set,
        pre_own_ports: set,
    ) -> None:
        if isinstance(action, MoveAction) and move_meta is not None:
            sid, stype, src, dst = move_meta
            code = _SHIP_CODE[stype]
            was_seen = sid in pre_fresh_ships
            now_seen = sid in post_fresh_ships
            if was_seen and now_seen:
                if (
                    self.recent_opponent_moves
                    and self.recent_opponent_moves[-1].get("kind") == "move"
                    and self.recent_opponent_moves[-1].get("ship_id") == sid
                    and self.recent_opponent_moves[-1].get("to") == list(src)
                ):
                    mv = self.recent_opponent_moves[-1]
                    mv["to"] = list(dst)
                    if self.entries and self.entries[-1]["kind"] == "enemy_move" and f"#{sid} moved" in self.entries[-1]["text"]:
                        self.entries[-1]["text"] = (
                            f"Opponent: {code} #{sid} moved {tuple(mv['from'])} -> {dst}"
                        )
                    else:
                        self._add_log(f"Opponent: {code} #{sid} moved {src} -> {dst}", kind="enemy_move")
                else:
                    self.recent_opponent_moves.append(
                        {"kind": "move", "ship_id": sid, "code": code, "from": list(src), "to": list(dst)}
                    )
                    self._add_log(f"Opponent: {code} #{sid} moved {src} -> {dst}", kind="enemy_move")
            elif not was_seen and now_seen:
                self.recent_opponent_moves.append(
                    {"kind": "spotted", "ship_id": sid, "code": code, "to": list(dst)}
                )
                self._add_log(f"Opponent: {code} #{sid} appeared at {dst}", kind="spot")
            elif was_seen and not now_seen:
                self.recent_opponent_moves.append(
                    {"kind": "lost", "ship_id": sid, "code": code, "from": list(src)}
                )
                self._add_log(f"Opponent: {code} #{sid} slipped out of sight from {src}", kind="enemy_move")
        elif isinstance(action, AttackAction) and attack_meta is not None:
            aid, atype, apos, tid, ttype, tpos = attack_meta
            killed = tid not in self.engine.ships
            outcome = "SUNK" if killed else "MISSED"
            att_seen = (aid in pre_fresh_ships) or (aid in post_fresh_ships)
            att_label = f"{_SHIP_CODE[atype]} #{aid} at {apos}" if att_seen else "Unseen enemy"
            self._add_log(
                f"Opponent: {att_label} attacked your {_SHIP_CODE[ttype]} #{tid} at {tpos} — {outcome}!",
                kind="combat_loss" if killed else "combat_miss",
            )
            evt = {
                "by": "opponent",
                "from": list(apos) if att_seen else None,
                "to": list(tpos),
                "killed": killed,
                "outcome": outcome,
            }
            self.recent_combat.append(evt)
            self.recent_opponent_moves.append({"kind": "attack", **evt})
        elif isinstance(action, BuildShipAction):
            sid_at = self.engine.map.ship_id_at(action.spawn_position)
            if sid_at >= 0 and sid_at in post_fresh_ships:
                self.recent_opponent_moves.append(
                    {"kind": "build", "to": list(action.spawn_position), "code": _SHIP_CODE[action.ship_type]}
                )
                self._add_log(
                    f"Opponent: Built {action.ship_type.value} at {action.spawn_position}",
                    kind="spot",
                )
        elif isinstance(action, BuildPortAction):
            if action.target in post_fresh_ports:
                self._add_log(f"Opponent: Built port at {action.target}", kind="spot")
        elif isinstance(action, CapturePortAction):
            if action.target in pre_own_ports:
                self.recent_opponent_moves.append({"kind": "capture", "to": list(action.target)})
                self._add_log(f"Opponent: Captured your port at {action.target}!", kind="combat_loss")
            elif action.target in pre_fresh_ports or action.target in post_fresh_ports:
                self.recent_opponent_moves.append({"kind": "capture", "to": list(action.target)})
                self._add_log(f"Opponent: Captured port at {action.target}", kind="enemy_move")
        elif isinstance(action, EndTurnAction):
            if not self.over:
                self._add_log(f"--- Turn {self.engine.turn}: Your turn ---", kind="turn")

    def _opponent_action(self) -> Action:
        pid = self.engine.current_player
        if self.opponent == "tactical":
            return tactical_scripted_action(self.engine, pid, self.rng)
        if self.opponent == "scripted_agent":
            return scripted_agent_action(self.engine, pid, self.rng)
        if self.opponent == "random":
            return random_action(self.engine, pid, self.rng)
        if self.opponent == "scripted":
            return scripted_action(self.engine, pid, self.rng)
        if self._uses_llm():
            action, meta = self._decide_llm(pid)
            self._record_llm_meta(meta)
            self._log_llm_meta(meta)
            return action
        assert self.policy is not None
        return policy_action(self.policy, self.engine, pid)

    def _thinking_label(self) -> str:
        if "inkling" in self.llm_model.lower() or self.opponent == "inkling":
            return "Inkling thinking…"
        return "LLM thinking…"

    def _decide_llm(self, pid: int) -> tuple:
        assert self.llm_client is not None
        return decide_llm_action(
            self.engine,
            pid,
            self.rng,
            self.llm_client,
            temperature=self.llm_temperature,
        )

    def _accumulate_usage(self, summary: dict) -> None:
        self.llm_totals["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            if key in summary:
                self.llm_totals[key] = int(self.llm_totals.get(key, 0)) + int(summary[key])
        if "cost_usd" in summary:
            self.llm_totals["cost_usd"] = float(self.llm_totals.get("cost_usd", 0.0)) + float(summary["cost_usd"])

    def _record_llm_meta(self, meta: dict) -> None:
        self._pending_llm_meta = meta
        summary = dict(meta.get("usage_summary") or {})
        latency = meta.get("latency_s")
        if latency is not None:
            summary["latency_s"] = latency
        self.last_llm = self._public_last_llm_from(summary, meta)
        self._accumulate_usage(summary)

    def _public_last_llm_from(self, summary: dict, meta: dict) -> dict:
        """Usage only — never include model reasoning, replies, or prompts."""
        public = {
            "latency_s": meta.get("latency_s"),
            "fallback": bool(meta.get("fallback")),
            "passed": bool(meta.get("passed")),
        }
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cost_usd",
        ):
            if key in summary:
                public[key] = summary[key]
        return public

    def _public_last_llm(self) -> dict:
        allowed = {
            "latency_s",
            "fallback",
            "passed",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cost_usd",
        }
        return {k: v for k, v in dict(self.last_llm).items() if k in allowed}

    def _log_llm_meta(self, meta: dict) -> None:
        summary = dict(meta.get("usage_summary") or {})
        line = format_usage_line(summary, meta.get("latency_s"))
        name = "Inkling" if "inkling" in self.llm_model.lower() or self.opponent == "inkling" else "LLM"
        bits = [name]
        if meta.get("passed"):
            bits.append("API error, passed the turn")
        elif meta.get("fallback"):
            bits.append("invalid reply, used a random legal move")
        if line:
            bits.append(line)
        kind = "error" if meta.get("passed") or meta.get("error") else "llm"
        self._add_log(": ".join([bits[0], " · ".join(bits[1:])]) if len(bits) > 1 else bits[0], kind=kind)

    def _start_llm_worker(self) -> None:
        if self.thinking or self.over or self.engine.current_player == self.human_side:
            return
        self.thinking = True
        self.thinking_started = time.monotonic()
        self.thinking_message = self._thinking_label()
        self._llm_turn_decisions = 0
        worker = threading.Thread(target=self._llm_worker, name="llm-turn", daemon=True)
        self._worker = worker
        worker.start()

    def _llm_worker(self) -> None:
        try:
            guard = 0
            while guard < self.step_budget:
                with self.lock:
                    if self.over or self.engine.current_player == self.human_side:
                        break
                    pid = self.engine.current_player
                    self.thinking_message = (
                        f"{self._thinking_label()} (decision {self._llm_turn_decisions + 1} this turn)"
                    )
                try:
                    action, meta = self._decide_llm(pid)
                except Exception as exc:
                    action = EndTurnAction()
                    meta = {
                        "fallback": True,
                        "passed": True,
                        "error": safe_error(exc),
                        "latency_s": 0,
                        "usage_summary": {},
                        "reasoning": "",
                    }
                with self.lock:
                    if self.over or self.engine.current_player != pid:
                        break
                    self._llm_turn_decisions += 1
                    self._record_llm_meta(meta)
                    self._apply(action, pid)
                    self._log_llm_meta(meta)
                    if self.over or self.engine.current_player == self.human_side:
                        break
                guard += 1
            with self.lock:
                if (
                    not self.over
                    and self.engine.current_player != self.human_side
                    and guard >= self.step_budget
                ):
                    self._apply(EndTurnAction(), self.engine.current_player)
                    # _apply already logs the action; keep a budget note if still to move.
                    if not self.over and self.engine.current_player != self.human_side:
                        self._add_log("Opponent: forced end turn (step budget)", kind="turn")
        finally:
            with self.lock:
                self.thinking = False
                self.thinking_message = ""
                self.thinking_started = None

    def _play_opponent_until_human(self) -> None:
        if self._uses_llm():
            self._start_llm_worker()
            return
        guard = 0
        while (
            not self.over
            and self.engine.current_player != self.human_side
            and guard < self.step_budget
        ):
            self._apply(self._opponent_action(), self.engine.current_player)
            guard += 1
        if (
            not self.over
            and self.engine.current_player != self.human_side
            and guard >= self.step_budget
        ):
            self._apply(EndTurnAction(), self.engine.current_player)
            if not self.over and self.engine.current_player != self.human_side:
                self._add_log("Opponent: forced end turn (step budget)", kind="turn")
