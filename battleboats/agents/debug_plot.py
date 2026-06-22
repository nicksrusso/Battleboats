"""Full-state visualization for MCTS debugging — no fog of war.

Renders the entire game state (terrain, ports, both fleets) plus a highlight
of the action just selected. Designed to be called immediately after action
selection (before engine.step), so you can drop a breakpoint right after
the call, inspect the figure, then continue.

A single persistent figure is reused across calls — successive plots update
in place instead of spawning new windows.

Visual conventions (color is by absolute owner: blue = player 0, red = player 1):
    light blue = water        (per-cell fill)
    green      = land         (per-cell fill)
    dark blue  = player-0 port,  red = player-1 port   (per-cell fill)
    gold X     = home port (the capture objective), drawn on top
    ships   = white round tokens stamped with a 2-letter type code (CV/BB/CA/...);
              the ring color marks owner (dark blue player 0 / red player 1).
              A ship-type + owner key is drawn to the right. (Ships sit on
              top of ports.) Owner coloring is fixed (not POV-relative), so games
              are visually comparable regardless of which seat the MCTS occupies.

Action overlays:
    magenta ring on origin tile + magenta arrow for Move
    orange dashed line + orange X for Attack
    magenta ring on the target tile for Build / Capture / Load / Unload
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

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
_WATER = (0.69, 0.85, 0.96)  # light blue
_LAND = (0.30, 0.69, 0.31)   # green

# Owner colors — fixed by absolute player id (NOT POV-relative), used
# consistently for ports (cell fill) and ship rings.
_P0 = (0.08, 0.24, 0.55)  # dark blue — player 0
_P1 = (0.84, 0.15, 0.16)  # red — player 1


def _owner_color(owner: int):
    return _P0 if owner == 0 else _P1
_SHIP_FILL = (1.00, 1.00, 1.00)  # neutral token fill so the type code stays legible
_HOME_MARK = "#ffd400"  # gold X marking each home port (the capture objective)

# Two-letter code stamped on each ship token (real naval hull classifications
# where they exist: CV carrier, BB battleship, CA cruiser, DD destroyer,
# SS submarine). The on-board glyph; the key spells out the full name.
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
        _FIG.subplots_adjust(right=0.82)  # reserve room for the ship-type key
    return _FIG, _AX


def _draw_ships(ax, engine: gameEngine, mcts_player_id: int) -> None:
    """Stamp each ship as an owner-colored token bearing its 2-letter type code.

    Radius is in data units so tokens scale with the board. Ring color marks the
    absolute owner (blue = player 0, red = player 1); type = the code text.
    """
    for ship in engine.ships.values():
        x, y = ship.position
        ring = _owner_color(ship.owner)
        ax.add_patch(Circle((x, y), 0.44, facecolor=_SHIP_FILL, edgecolor=ring,
                            linewidth=2.0, zorder=4))
        ax.text(x, y, _SHIP_CODE.get(ship.type, "?"), ha="center", va="center",
                fontsize=6, fontweight="bold", color="black", zorder=4)


def _mark_home_ports(ax, engine: gameEngine) -> None:
    """Stamp a gold X on each home port — the capture objective. Drawn on top of
    ships so it stays visible even when a ship sits on the home tile."""
    for pos, port in engine.ports.items():
        if getattr(port, "is_home", False):
            x, y = pos
            ax.scatter([x], [y], s=260, marker="X", c=_HOME_MARK,
                       edgecolors="black", linewidths=1.2, zorder=7)


def _ship_type_key(ax) -> None:
    """Legend mapping each code to its ship type, plus the owner fill colors."""
    type_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_SHIP_FILL,
               markeredgecolor="0.45", markersize=11, label=f"{code} — {st.value}")
        for st, code in _SHIP_CODE.items()
    ]
    allegiance_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_SHIP_FILL,
               markeredgecolor=_P0, markeredgewidth=2.2, markersize=11,
               label="player 0 ship (ring)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_SHIP_FILL,
               markeredgecolor=_P1, markeredgewidth=2.2, markersize=11,
               label="player 1 ship (ring)"),
        Patch(facecolor=_P0, edgecolor="black", label="player 0 port"),
        Patch(facecolor=_P1, edgecolor="black", label="player 1 port"),
        Line2D([0], [0], marker="X", linestyle="none", markerfacecolor=_HOME_MARK,
               markeredgecolor="black", markersize=12, label="home port (objective)"),
    ]
    ax.legend(handles=type_handles + allegiance_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.95,
              title="Ship types  &  owner", borderaxespad=0.0)


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
        img[y, x] = _owner_color(port.owner)

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

    _draw_ships(ax, engine, mcts_player_id)
    _mark_home_ports(ax, engine)
    _ship_type_key(ax)
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
    ax.set_xlabel(f"x  (MCTS = player_{mcts_player_id}; blue = player 0, red = player 1 — ports filled, ships ringed, gold X = home; type = code, see key)")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")

    fig.canvas.draw()
    plt.pause(0.001)
