"""ActionMasks stay aligned with enumerate_legal for bought-this-turn ships."""

import json

from battleboats.core.actions import BuildShipAction, EndTurnAction, MoveAction
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks


def _engine(tmp_path, w, h, ports, land=None):
    land = land if land is not None else [p for ps in ports.values() for p in ps]
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"width": w, "height": h, "seed": 0, "land": land, "ports": ports}))
    e = gameEngine(str(path))
    e.reset(seed=0)
    return e


def test_bought_ship_is_not_a_selectable_asset(tmp_path):
    """A ship bought this turn is in the token set but cannot be chosen as an asset."""
    e = _engine(tmp_path, 10, 4, {"0": [[1, 1]], "1": [[8, 2]]})
    e.players[0].cash = 1000
    action = next(
        a
        for a in e.enumerate_legal(0)
        if isinstance(a, BuildShipAction) and a.ship_type == ShipType.DESTROYER
    )
    before = set(e.ships)
    e.step(action)
    sid = next(iter(set(e.ships) - before))
    assert e.ships[sid].bought_this_turn

    masks = ActionMasks(e, 0)
    tok = masks._ship_tok[sid]
    assert not bool(masks.asset[0, tok])
    legal = e.enumerate_legal(0)
    assert not any(isinstance(a, MoveAction) and a.ship_id == sid for a in legal)
    # factor() must still be able to name the ship once it can act.
    e.step(EndTurnAction())
    e.step(EndTurnAction())
    masks_next = ActionMasks(e, 0)
    assert bool(masks_next.asset[0, masks_next._ship_tok[sid]])
    assert any(isinstance(a, MoveAction) and a.ship_id == sid for a in e.enumerate_legal(0))
