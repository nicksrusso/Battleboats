"""Full-state visualization for MCTS debugging — no fog of war.

Renders the entire game state (terrain, ports, both fleets) plus a highlight
of the action just selected. Designed to be called immediately after action
selection (before engine.step), so you can drop a breakpoint right after
the call, inspect the figure, then continue.

A single persistent figure is reused across calls — successive plots update
in place instead of spawning new windows.

Visual conventions (per-cell fill, gridded):
    blue   = water
    green  = land
    red    = friendly port
    black  = enemy port
    white  = friendly ship
    grey   = enemy ship
    (ships override ports on the same tile.)

Action overlays:
    magenta ring on origin tile + magenta arrow for Move
    orange dashed line + orange X for Attack
    magenta ring on the target tile for Build / Capture / Load / Unload
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from battleboats.core.actions import (
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

# Cell-fill palette (RGB in 0-1).
_WATER = (0.20, 0.45, 0.78)
_LAND = (0.30, 0.69, 0.31)
_FRIENDLY_PORT = (0.84, 0.15, 0.16)
_ENEMY_PORT = (0.00, 0.00, 0.00)
_FRIENDLY_SHIP = (1.00, 1.00, 1.00)
_ENEMY_SHIP = (0.55, 0.55, 0.55)

# Action highlights — chosen to contrast against every fill color above.
_HL_ACTION = "#ff00ff"  # magenta: move arrows, target rings
_HL_ATTACK = "#ff8800"  # orange: attack line + target X

_FIG = None
_AX = None


def _ensure_figure():
    global _FIG, _AX
    if _FIG is None or not plt.fignum_exists(_FIG.number):
        plt.ion()
        _FIG, _AX = plt.subplots(figsize=(14, 7))
    return _FIG, _AX


def _describe_action(action, engine: gameEngine) -> str:
    if action is None:
        return "None"
    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        ty = ship.type.value if ship else "?"
        return f"Move({ty} #{action.ship_id}) -> {action.destination}"
    if isinstance(action, AttackAction):
        a = engine.ships.get(action.attacker_id)
        t = engine.ships.get(action.target_id)
        return f"Attack: {a.type.value if a else '?'} -> {t.type.value if t else '?'}"
    if isinstance(action, BuildShipAction):
        return f"BuildShip({action.ship_type.value}) at {action.port}"
    if isinstance(action, BuildPortAction):
        return f"BuildPort at {action.target}"
    if isinstance(action, CapturePortAction):
        return f"CapturePort at {action.target}"
    if isinstance(action, MerchantLoadAction):
        return f"MerchantLoad at {action.port}"
    if isinstance(action, MerchantUnloadAction):
        return f"MerchantUnload at {action.port}"
    if isinstance(action, EndTurnAction):
        return "EndTurn"
    return type(action).__name__


def _highlight_action(ax, action, engine: gameEngine) -> None:
    """Overlay action-specific markup. State is pre-step — ships still at origins."""
    if action is None or isinstance(action, EndTurnAction):
        return

    if isinstance(action, MoveAction):
        ship = engine.ships.get(action.ship_id)
        if ship is None:
            return
        x0, y0 = ship.position
        x1, y1 = action.destination
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=_HL_ACTION, lw=2.5, mutation_scale=20),
        )
        ax.scatter([x0], [y0], s=300, facecolors="none", edgecolors=_HL_ACTION, linewidths=2.5, zorder=5)
        return

    if isinstance(action, AttackAction):
        a = engine.ships.get(action.attacker_id)
        t = engine.ships.get(action.target_id)
        if a is None or t is None:
            return
        ax.plot(
            [a.position[0], t.position[0]],
            [a.position[1], t.position[1]],
            color=_HL_ATTACK, linestyle="--", linewidth=2, zorder=5,
        )
        ax.scatter([t.position[0]], [t.position[1]], s=350, marker="X",
                   c=_HL_ATTACK, edgecolors="white", linewidths=1.5, zorder=6)
        return

    # Build/Capture/Load/Unload — highlight the target tile.
    target = getattr(action, "target", None) or getattr(action, "port", None)
    if target is not None:
        ax.scatter([target[0]], [target[1]], s=400, facecolors="none",
                   edgecolors=_HL_ACTION, linewidths=2.5, zorder=5)


def plot_state(
    engine: gameEngine,
    action,
    actor: str,
    mcts_player_id: int,
    *,
    step: Optional[int] = None,
    value: Optional[float] = None,
    extra: str = "",
) -> None:
    """Render the full game state with the just-chosen action highlighted.

    Call this AFTER an action has been selected but BEFORE ``engine.step``
    has been applied — the highlight assumes pre-step positions.
    """
    fig, ax = _ensure_figure()
    ax.clear()

    W, H = engine.map.width, engine.map.height

    # Build per-cell RGB grid. Map.terrain is [x, y]; imshow expects [y, x, 3].
    # Layered render (later writes override): terrain -> ports -> ships.
    water_mask = engine.map.terrain.T == 0
    img = np.empty((H, W, 3), dtype=float)
    img[water_mask] = _WATER
    img[~water_mask] = _LAND

    for pos, port in engine.ports.items():
        x, y = pos
        img[y, x] = _FRIENDLY_PORT if port.owner == mcts_player_id else _ENEMY_PORT

    for ship in engine.ships.values():
        x, y = ship.position
        img[y, x] = _FRIENDLY_SHIP if ship.owner == mcts_player_id else _ENEMY_SHIP

    ax.imshow(
        img, origin="lower",
        extent=(-0.5, W - 0.5, -0.5, H - 0.5),
        interpolation="nearest",
    )

    # Cell-boundary gridlines (minor ticks on half-integers).
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3, alpha=0.5)
    # Sparse major ticks so axis labels stay readable on a 160x80 map.
    ax.set_xticks(np.arange(0, W, 10))
    ax.set_yticks(np.arange(0, H, 10))
    ax.tick_params(which="minor", length=0)

    _highlight_action(ax, action, engine)

    parts = []
    if step is not None:
        parts.append(f"step={step}")
    parts.append(f"turn={engine.turn}")
    parts.append(f"actor={actor}")
    parts.append(f"action={_describe_action(action, engine)}")
    if value is not None:
        parts.append(f"value={value:+.4f}")
    if extra:
        parts.append(extra)
    ax.set_title("  ".join(parts), fontsize=10)
    ax.set_xlabel(f"x  (MCTS=player_{mcts_player_id}; white=friendly ship, grey=enemy ship, red=friendly port, black=enemy port)")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")

    fig.canvas.draw()
    plt.pause(0.001)
