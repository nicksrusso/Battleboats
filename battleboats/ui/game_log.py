"""Append-only JSONL logs of human-vs-agent play-UI games.

One file per game under ``runs/human_games/`` by default (the ``runs/``
tree is gitignored). Lines are flushed as they are written so a crash or
abandoned game still leaves a usable prefix; a missing ``game_end`` line,
or ``complete: false``, marks the file incomplete.

The schema is meant to be reused later as behavior-cloning / DAgger data:
each ``decision`` has the acting seat's fog-of-war observation (FORMAT 4
``build_view``), the legal action list, the chosen action, and a
JSON-safe ground-truth snapshot. Replaying ``seed`` + ``map`` + the
action sequence through ``gameEngine`` reconstructs the game (combat uses
the engine RNG started from that seed).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np

from battleboats.core.actions import Action, AttackAction, EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.llm.prompt import action_record
from battleboats.core.shipyard.ship_type import ShipType

LOG_FORMAT_VERSION = 1
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "core" / "config"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILES = ("shipTypes.csv", "attackModifiers.csv", "visibilityMatrix.csv")


def default_log_dir(map_dir: Path | str) -> Path:
    """``runs/play_maps`` → ``runs/human_games``; tests using a tmp map dir stay in tmp."""
    return Path(map_dir).resolve().parent / "human_games"


def git_sha(repo_root: Path = _REPO_ROOT) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = out.strip()
        return sha or None
    except (OSError, subprocess.CalledProcessError):
        return None


def rules_fingerprint(kill_curve_k: float = 2.0) -> dict:
    digest = hashlib.sha256()
    for name in _CONFIG_FILES:
        path = _CONFIG_DIR / name
        digest.update(name.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(f"kill_curve_k={kill_curve_k}".encode())
    return {
        "config_sha256": digest.hexdigest(),
        "git_sha": git_sha(),
        "kill_curve_k": float(kill_curve_k),
        "config_files": list(_CONFIG_FILES),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return [_json_default(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_default(x) for x in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _json_default(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def dumps(record: dict) -> str:
    return json.dumps(record, default=_json_default, separators=(",", ":"))


def serialize_ship(ship) -> dict:
    return {
        "id": int(ship.id),
        "type": ship.type.value,
        "owner": int(ship.owner),
        "position": [int(ship.position[0]), int(ship.position[1])],
        "tiles_moved_this_turn": int(ship.tiles_moved_this_turn),
        "has_attacked": bool(ship.has_attacked),
        "cargo": int(ship.cargo),
        "bought_this_turn": bool(ship.bought_this_turn),
        "built_turn": int(ship.built_turn),
        "speed": int(ship.stats.speed),
        "cost": int(ship.stats.cost),
        "strength": float(ship.stats.strength),
        "attack_range": int(ship.stats.attack_range),
    }


def serialize_engine(engine: gameEngine) -> dict:
    """JSON-safe ground-truth snapshot (enough, with seed+map, to replay)."""
    land = [
        [int(x), int(y)]
        for x in range(engine.map.width)
        for y in range(engine.map.height)
        if engine.map.is_land((x, y))
    ]
    ports = []
    for pos, port in sorted(engine.ports.items()):
        ports.append(
            {
                "position": [int(pos[0]), int(pos[1])],
                "owner": int(port.owner),
                "is_home": bool(port.is_home),
                "stockpile": int(port.stockpile),
                "locked_this_turn": bool(port.locked_this_turn),
            }
        )
    players = []
    for player in engine.players:
        sightings = []
        for sid, s in sorted(player.sightings.items()):
            sightings.append(
                {
                    "ship_id": int(s.ship_id),
                    "type": s.type.value,
                    "position": [int(s.position[0]), int(s.position[1])],
                    "turn_seen": int(s.turn_seen),
                    "fresh": bool(s.fresh),
                }
            )
        port_sightings = []
        for pos, s in sorted(player.port_sightings.items()):
            port_sightings.append(
                {
                    "position": [int(pos[0]), int(pos[1])],
                    "last_known_owner": int(s.last_known_owner),
                    "is_home": bool(s.is_home),
                    "turn_seen": int(s.turn_seen),
                    "fresh": bool(s.fresh),
                }
            )
        players.append(
            {
                "id": int(player.id),
                "home_port": [int(player.home_port[0]), int(player.home_port[1])],
                "cash": int(player.cash),
                "owned_ship_ids": sorted(int(s) for s in player.owned_ship_ids),
                "owned_port_positions": [
                    [int(p[0]), int(p[1])] for p in sorted(player.owned_port_positions)
                ],
                "sightings": sightings,
                "port_sightings": port_sightings,
            }
        )
    return {
        "width": int(engine.map.width),
        "height": int(engine.map.height),
        "land": land,
        "ports": ports,
        "ships": [serialize_ship(s) for _, s in sorted(engine.ships.items())],
        "players": players,
        "current_player": int(engine.current_player),
        "turn": int(engine.turn),
        "winner": engine.winner,
        "kill_curve_k": float(engine.kill_curve_k),
        "next_ship_id": int(engine._next_ship_id),
    }


def _safe_token(value: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (cleaned or "x")[:limit]


def log_filename(map_name: str, opponent: str, seed: int, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    ts = when.strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{_safe_token(map_name)}_{_safe_token(opponent)}_seed{int(seed)}.jsonl"


class GameLog:
    """Incremental JSONL writer for one play-UI game."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.steps = 0
        self._closed = False
        self._complete: Optional[bool] = None
        self._fh: TextIO = self.path.open("w", encoding="utf-8")

    def write(self, record: dict) -> None:
        if self._closed:
            return
        self._fh.write(dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def complete(self) -> Optional[bool]:
        return self._complete


def start_record(session) -> dict:
    engine = session.engine
    try:
        map_data = json.loads(Path(session.map_path).read_text(encoding="utf-8"))
    except OSError:
        map_data = None
    rec = {
        "type": "game_start",
        "log_format": LOG_FORMAT_VERSION,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "map_name": session.map_name,
        "map_path": str(session.map_path),
        "map": map_data,
        "seed": int(session.seed),
        "max_turns": int(session.max_turns),
        "human_side": int(session.human_side),
        "opponent_side": 1 - int(session.human_side),
        "opponent": session.opponent,
        "opponent_label": session.opponent_label,
        "rules": rules_fingerprint(engine.kill_curve_k),
        "state": serialize_engine(engine),
        "complete": False,
    }
    if session._uses_llm():
        rec["llm"] = {
            "model": session.llm_model,
            "reasoning": session.llm_reasoning,
            "temperature": session.llm_temperature,
        }
    if session.checkpoint is not None:
        rec["checkpoint"] = str(session.checkpoint)
    return rec


def decision_record(
    session,
    action: Action,
    pid: int,
    *,
    legal: List[Action],
    state_before: dict,
    observations: Dict[str, dict],
    combat: Optional[dict] = None,
    llm: Optional[dict] = None,
    step: int,
) -> dict:
    rec = {
        "type": "decision",
        "log_format": LOG_FORMAT_VERSION,
        "step": int(step),
        "turn": int(session.engine.turn),
        "actor_id": int(pid),
        "actor": "human" if pid == session.human_side else "opponent",
        "action": action_record(action),
        "legal": [action_record(a) for a in legal],
        "state": state_before,
        "observation": observations.get(str(pid)) or observations.get(pid),
        "observations": observations,
        "state_after": serialize_engine(session.engine),
    }
    if combat:
        rec["combat"] = combat
    if llm:
        rec["llm"] = llm
    return rec


def end_record(session, *, complete: bool, reason: str) -> dict:
    rec = {
        "type": "game_end",
        "log_format": LOG_FORMAT_VERSION,
        "complete": bool(complete),
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "winner": session.engine.winner,
        "reason": reason,
        "result": session.result_text if complete else "Incomplete — game did not finish.",
        "turn": int(session.engine.turn),
        "max_turns": int(session.max_turns),
        "steps": int(getattr(session, "_log_steps", 0)),
        "state": serialize_engine(session.engine),
    }
    if session._uses_llm():
        rec["llm_totals"] = dict(session.llm_totals)
        rec["llm"] = {
            "model": session.llm_model,
            "reasoning": session.llm_reasoning,
        }
    return rec


def combat_result(action: Action, attack_meta, engine: gameEngine) -> Optional[dict]:
    if not isinstance(action, AttackAction) or attack_meta is None:
        if isinstance(action, EndTurnAction):
            return None
        return None
    aid, atype, apos, tid, ttype, tpos = attack_meta
    killed = tid not in engine.ships
    return {
        "attacker_id": int(aid),
        "attacker_type": atype.value if isinstance(atype, ShipType) else str(atype),
        "attacker_position": [int(apos[0]), int(apos[1])],
        "target_id": int(tid),
        "target_type": ttype.value if isinstance(ttype, ShipType) else str(ttype),
        "target_position": [int(tpos[0]), int(tpos[1])],
        "killed": bool(killed),
        "outcome": "SUNK" if killed else "MISSED",
    }


def llm_fields(meta: Optional[dict]) -> Optional[dict]:
    if not meta:
        return None
    summary = dict(meta.get("usage_summary") or {})
    out = {
        "fallback": bool(meta.get("fallback")),
        "passed": bool(meta.get("passed")),
        "latency_s": meta.get("latency_s"),
        "reasoning": meta.get("reasoning") or "",
        "replies": list(meta.get("replies") or []),
        "raw_reply": (meta.get("replies") or [None])[-1] if meta.get("replies") else None,
        "prompt": meta.get("prompt"),
        "usage": meta.get("usage"),
        "usage_summary": summary,
        "error": meta.get("error") or "",
    }
    if meta.get("retry_prompt") is not None:
        out["retry_prompt"] = meta["retry_prompt"]
    return out
