from battleboats.core.gameEngine import gameEngine

MAP_JSON = "/home/nick/Desktop/repos/NavalCivGame/battleboats/core/config/map.json"

if __name__ == "__main__":
    engine = gameEngine(map_json_path=MAP_JSON)
    state = engine.get_state()
    print(state)
