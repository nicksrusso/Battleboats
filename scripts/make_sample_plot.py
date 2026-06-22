"""Generate an illustrative sample-game frame for the presentation.

Builds a small, hand-arranged board state (synthetic — NOT a real game) and
renders it with agents.debug_plot so the slide shows the token + allegiance
visuals clearly on a board small enough that the ships are legible.

Scene tells the core story: friendly fleet massing on the left, a Landing craft
(LC) pushing toward the enemy home, and a hostile Destroyer (DD) closing to
intercept it.

    poetry run python scripts/make_sample_plot.py
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np

from battleboats.agents import debug_plot
from battleboats.core.shipyard.ship_type import ShipType

W, H = 18, 11

# terrain: 0 = water, 1 = land. A little coastline at the corners for texture.
terrain = np.zeros((W, H), dtype=int)
for x, y in [(0, 0), (1, 0), (0, 10), (17, 0), (16, 10), (17, 10),
             (8, 0), (9, 10), (2, 9), (15, 1)]:
    terrain[x, y] = 1

home0, home1 = (2, 5), (15, 5)
ports = {
    home0: SimpleNamespace(owner=0),          # friendly home
    home1: SimpleNamespace(owner=1),          # hostile home
    (5, 9): SimpleNamespace(owner=0),         # forward friendly port
    (12, 2): SimpleNamespace(owner=1),        # forward hostile port
}


def _ship(x, y, t, owner):
    return SimpleNamespace(position=(x, y), type=t, owner=owner)


ships_list = [
    # player 0 (friendly) — fleet massing left, landing craft pushing center
    _ship(2, 4, ShipType.BUILDER, 0),
    _ship(3, 6, ShipType.MERCHANT, 0),
    _ship(4, 7, ShipType.SUBMARINE, 0),
    _ship(5, 5, ShipType.DESTROYER, 0),
    _ship(6, 4, ShipType.CRUISER, 0),
    _ship(7, 6, ShipType.BATTLESHIP, 0),
    _ship(9, 5, ShipType.LANDING, 0),         # the march to the enemy home
    # player 1 (hostile) — defending right; a destroyer closing to intercept
    _ship(15, 6, ShipType.CARRIER, 1),
    _ship(14, 5, ShipType.BATTLESHIP, 1),
    _ship(13, 4, ShipType.CRUISER, 1),
    _ship(13, 7, ShipType.SUBMARINE, 1),
    _ship(11, 5, ShipType.DESTROYER, 1),      # interceptor closing on the LC
]
ships = {i: s for i, s in enumerate(ships_list)}

engine = SimpleNamespace(
    map=SimpleNamespace(
        width=W, height=H, terrain=terrain,
        manhattan=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
    ),
    ports=ports,
    ships=ships,
    turn=14,
)

debug_plot.plot_state(
    engine, action=None, actor="player_0", mcts_player_id=0,
    step=37, extra="illustrative frame",
)
debug_plot._FIG.savefig("docs/sample_game_frame.png", dpi=160, bbox_inches="tight")
print("saved docs/sample_game_frame.png")
