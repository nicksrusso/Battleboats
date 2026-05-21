"""Debug harness for the observation builder.

Sets up a game state with one of each interesting entity kind in scope —
friendly ship, friendly port, enemy ship sighting, enemy port sighting,
coastline tiles — then calls build_observation so you can drop a breakpoint
inside it and walk through populated data.

Run with: poetry run python main.py
Or under a debugger to hit your breakpoint inside build_observation.
"""
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import build_observation

MAP_JSON = "/home/nick/Desktop/repos/NavalCivGame/battleboats/core/config/map.json"


def _adjacent_water_tiles(engine, pos):
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cand = (pos[0] + dx, pos[1] + dy)
        if (
            engine.map.in_bounds(cand)
            and engine.map.is_water(cand)
            and not engine.map.is_occupied(cand)
        ):
            out.append(cand)
    return out


if __name__ == "__main__":
    engine = gameEngine(map_json_path=MAP_JSON)
    engine.reset(seed=0)

    # Pick any P1 port and place a P0 destroyer adjacent — yields a fresh
    # port sighting (and the destroyer itself becomes a friendly ship token).
    p1_port = next(iter(engine.players[1].owned_port_positions))
    water_tiles = _adjacent_water_tiles(engine, p1_port)
    assert len(water_tiles) >= 1, "no water adjacent to P1 port — pick another"
    p0_spawn = water_tiles[0]
    engine._spawn_ship(0, ShipType.DESTROYER, p0_spawn)

    # If a second adjacent water tile exists, drop a P1 sub there so we
    # also get a fresh enemy ship sighting in the observation.
    if len(water_tiles) >= 2:
        engine._spawn_ship(1, ShipType.SUBMARINE, water_tiles[1])

    engine._refresh_sightings()

    obs = build_observation(engine, player_id=0)
    print("entity_tokens.shape =", obs["entity_tokens"].shape)
    print("globals.shape       =", obs["globals"].shape)
    print("globals             =", obs["globals"])
