"""One local game: human vs random, scripted, or a policy checkpoint.

The human sees fog of war (own units, plus enemy ships and ports they have
sighted). Terrain is shown in full — coastline is not hidden by the
observation builder. Random and scripted opponents read the engine; the
scripted one is full-information and the page says so. A checkpoint opponent
uses the same masked policy as training.
"""

from __future__ import annotations

import html
import random
from pathlib import Path
from typing import List, Optional, Tuple

from battleboats.agents.random_agent import random_action
from battleboats.agents.scripted import scripted_action
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
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.skill_maps import write_full_map
from battleboats.training.acting import policy_action
from battleboats.training.checkpoints import load_checkpoint
from battleboats.training.device import resolve_device

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


def describe_action(action: Action, engine: gameEngine) -> str:
    """Short label for a button."""
    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        kind = _SHIP_CODE.get(ship.type, "?") if ship else "?"
        return f"Move {kind} #{action.ship_id} → {action.destination}"
    if isinstance(action, AttackAction):
        a = engine.ships.get(action.attacker_id)
        t = engine.ships.get(action.target_id)
        ak = _SHIP_CODE.get(a.type, "?") if a else "?"
        tk = _SHIP_CODE.get(t.type, "?") if t else "?"
        return f"Attack {ak} #{action.attacker_id} → {tk} #{action.target_id}"
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
    """Human seat vs one opponent. Not thread-safe; the server holds a lock."""

    def __init__(
        self,
        *,
        opponent: str,
        human_side: int,
        width: int,
        height: int,
        seed: int,
        max_turns: int,
        map_dir: Path,
        checkpoint: Optional[Path] = None,
        device: str = "auto",
        step_budget: int = 400,
    ) -> None:
        if opponent not in ("random", "scripted", "checkpoint"):
            raise ValueError(f"unknown opponent {opponent!r}")
        if human_side not in (0, 1):
            raise ValueError("human_side must be 0 or 1")
        self.opponent = opponent
        self.human_side = human_side
        self.max_turns = max_turns
        self.step_budget = step_budget
        self.rng = random.Random(seed)
        self.log: List[str] = []
        self.error = ""
        map_dir.mkdir(parents=True, exist_ok=True)
        self.map_path = write_full_map(map_dir / f"play_{seed}.json", width, height, seed=seed)
        self.engine = gameEngine(self.map_path)
        self.engine.reset(seed=seed)
        self.policy = None
        self.checkpoint = None
        if opponent == "checkpoint":
            if checkpoint is None or not Path(checkpoint).exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
            self.checkpoint = Path(checkpoint)
            dev = resolve_device(device)
            self.policy, _ = load_checkpoint(self.checkpoint, dev)
            self.policy.eval()
        self._play_opponent_until_human()

    @property
    def over(self) -> bool:
        return self.engine.winner is not None or self.engine.turn >= self.max_turns

    @property
    def result_text(self) -> str:
        if self.engine.winner is None and self.engine.turn >= self.max_turns:
            return "Draw — turn limit."
        if self.engine.winner is None:
            return ""
        if self.engine.winner == self.human_side:
            return "You win."
        return "You lose."

    def legal(self) -> List[Action]:
        if self.over or self.engine.current_player != self.human_side:
            return []
        return self.engine.enumerate_legal(self.human_side)

    def apply(self, index: int) -> None:
        self.error = ""
        legal = self.legal()
        if not legal:
            self.error = "No legal actions."
            return
        if index < 0 or index >= len(legal):
            self.error = f"Action {index} is out of range."
            return
        action = legal[index]
        self._apply(action, self.human_side)
        self._play_opponent_until_human()

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
            "result": self.result_text,
            "opponent": self.opponent,
            "human_side": self.human_side,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
        }

    def render_html(self) -> str:
        st = self.status()
        rows = []
        for row in self.board_cells():
            cells = []
            for c in row:
                cls = ["cell", c["terrain"]]
                if c["port"] == "own":
                    cls.append("port-own")
                elif c["port"] == "enemy":
                    cls.append("port-enemy")
                if c["stale"]:
                    cls.append("stale")
                label = ""
                if c["ship"]:
                    label = c["ship"]
                elif c["home"]:
                    label = "H"
                elif c["port"]:
                    label = "P"
                title = f"({c['x']},{c['y']})"
                cells.append(
                    f'<td class="{" ".join(cls)}" title="{title}">{html.escape(label)}</td>'
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        actions = []
        for i, action in enumerate(self.legal()):
            label = html.escape(describe_action(action, self.engine))
            actions.append(
                f'<form method="post" action="/act" class="act">'
                f'<input type="hidden" name="index" value="{i}"/>'
                f'<button type="submit">{i}. {label}</button></form>'
            )
        if not actions and not st["over"]:
            actions.append("<p>Waiting…</p>")
        log = "".join(f"<li>{html.escape(line)}</li>" for line in self.log[-12:])
        err = f'<p class="err">{html.escape(self.error)}</p>' if self.error else ""
        opp = {
            "random": "Random",
            "scripted": "Scripted (full information)",
            "checkpoint": f"Checkpoint {st['checkpoint']}",
        }[self.opponent]
        result = f'<p class="result">{html.escape(st["result"])}</p>' if st["result"] else ""
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Battleboats</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; background: #0f1419; color: #e7ecf1; }}
a {{ color: #8ecae6; }}
table.board {{ border-collapse: collapse; }}
td.cell {{ width: 36px; height: 36px; text-align: center; font-size: 11px; font-weight: 700;
  border: 1px solid #1b2836; }}
td.water {{ background: #1d4e89; }}
td.land {{ background: #2d6a4f; }}
td.port-own {{ background: #1d3557; outline: 2px solid #ffd400; }}
td.port-enemy {{ background: #9b2226; }}
td.stale {{ opacity: 0.55; }}
form.act {{ margin: 0.15rem 0; }}
button {{ background: #1d3557; color: white; border: 0; padding: 0.35rem 0.6rem; cursor: pointer; }}
.err {{ color: #ffb4a2; }}
.result {{ font-size: 1.3rem; color: #ffd400; }}
.layout {{ display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap; }}
</style></head><body>
<p><a href="/">New game</a></p>
<h1>You are player {st['human_side']}</h1>
<p>Opponent: {html.escape(opp)} · Turn {st['turn']}/{st['max_turns']} ·
Cash {st['cash']} · Ships {st['ships']} · Ports {st['ports']} ·
Enemy ships in sight {st['sighted_ships']}</p>
{result}{err}
<div class="layout">
<table class="board">{''.join(rows)}</table>
<div><h2>Actions</h2>{''.join(actions) or '<p>Game over.</p>'}
<h2>Recent</h2><ol>{log}</ol></div>
</div>
<p>Gold outline is your port. H is a home you can see. Enemy ships you have lost
sight of are faded. Unsighted enemy ports look like empty land.</p>
</body></html>"""

    def _apply(self, action: Action, pid: int) -> None:
        label = describe_action(action, self.engine)
        who = "You" if pid == self.human_side else "Opponent"
        self.engine.step(action)
        self.log.append(f"{who}: {label}")

    def _opponent_action(self) -> Action:
        pid = self.engine.current_player
        if self.opponent == "random":
            return random_action(self.engine, pid, self.rng)
        if self.opponent == "scripted":
            return scripted_action(self.engine, pid, self.rng)
        assert self.policy is not None
        return policy_action(self.policy, self.engine, pid)

    def _play_opponent_until_human(self) -> None:
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
            self.engine.step(EndTurnAction())
            self.log.append("Opponent: forced end turn (step budget)")
