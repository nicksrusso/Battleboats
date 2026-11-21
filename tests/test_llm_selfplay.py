"""Self-play prompt and harness tests. The LLM client is a fake; no network."""

from __future__ import annotations

import json
import random
from pathlib import Path

from battleboats.core.actions import (AttackAction, BuildShipAction,
                                      EndTurnAction)
from battleboats.core.gameEngine import gameEngine
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.envs.observation import build_entity_refs
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.llm.play import (chat_completions_url, chat_payload,
                                  choose_move, message_text, parse_choice,
                                  play_env)
from battleboats.llm.prompt import (FORMAT_VERSION, STATIC_PREFIX, build_view,
                                    render_prompt)


def _write_map(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data))
    return str(path)


def _engine(path: str, ships, *, stock=None, cargo=None, cash=None):
    engine = gameEngine(path)
    engine.reset(seed=0)
    spawned = []
    for owner, ship_type, pos in ships:
        spawned.append(engine._spawn_ship(owner, ship_type, pos))
    if stock:
        for pos, amount in stock.items():
            engine.ports[pos].stockpile = amount
    if cargo:
        for ship, amount in cargo:
            ship.cargo = amount
    if cash:
        for pid, amount in cash.items():
            engine.players[pid].cash = amount
    engine._refresh_sightings()
    return engine, spawned


def _open_engine(tmp_path: Path):
    path = _write_map(
        tmp_path / "open.json",
        {
            "width": 8,
            "height": 6,
            "land": [[0, 0], [0, 5], [7, 5]],
            "ports": {"0": [[0, 0], [0, 5]], "1": [[7, 5]]},
        },
    )
    engine, ships = _engine(
        path,
        [
            (0, ShipType.DESTROYER, (3, 2)),
            (0, ShipType.MERCHANT, (1, 1)),
            (0, ShipType.LANDING, (6, 4)),
            (1, ShipType.CRUISER, (6, 2)),
            (1, ShipType.SUBMARINE, (7, 0)),
        ],
        stock={(0, 5): 50},
        cargo=[],
        cash={0: 250, 1: 424242},
    )
    ships[1].cargo = 40
    return engine, ships


def test_prompt_sections_and_interactions(tmp_path: Path) -> None:
    engine, ships = _open_engine(tmp_path)
    destroyer, merchant, landing, cruiser, sub = ships
    legal = engine.enumerate_legal(0)
    view = build_view(engine, 0, legal, horizon=4, max_edges=20)
    prompt = render_prompt(view)

    for section in ("FORMAT 3", "SHIPS", "PORTS", "LAND", "MOVES"):
        assert section in prompt
    for dropped in ("GRID", "FACTS", "EDGES", "ENTITIES"):
        assert dropped not in prompt
    assert view["format_version"] == FORMAT_VERSION == 3
    assert prompt.startswith(STATIC_PREFIX)
    assert prompt.count("8x6") == 1
    assert render_prompt(view) == prompt
    assert "424242" not in prompt
    assert "gold 250" in prompt
    assert "cargo=0" not in prompt
    assert "moved=" not in prompt
    assert "attacked=" not in prompt
    assert "can_attack" not in prompt
    assert "home_turns" not in prompt

    ids = {ent.get("id") for ent in view["ships"]}
    assert f"s{sub.id}" not in ids
    assert f"s{cruiser.id}" in ids
    assert f"s{destroyer.id}" in ids
    assert f"s{landing.id}" in ids
    cruiser_ent = next(
        ent for ent in view["ships"] if ent.get("id") == f"s{cruiser.id}"
    )
    assert cruiser_ent["owner"] == "enemy"
    assert cruiser_ent["pos"] == [6, 2]
    assert cruiser_ent["strength"] == int(BASE_STATS[ShipType.CRUISER].strength)
    assert cruiser_ent["range"] == BASE_STATS[ShipType.CRUISER].attack_range
    assert "fresh" not in cruiser_ent
    assert "stale" not in cruiser_ent
    assert "cargo" not in cruiser_ent
    destroyer_ent = next(
        ent for ent in view["ships"] if ent.get("id") == f"s{destroyer.id}"
    )
    assert destroyer_ent["owner"] == "me"
    assert destroyer_ent["strength"] == int(destroyer.stats.strength)
    assert destroyer_ent["speed"] == destroyer.stats.speed
    merchant_ent = next(
        ent for ent in view["ships"] if ent.get("id") == f"s{merchant.id}"
    )
    assert merchant_ent["cargo"] == 40
    home = next(ent for ent in view["ports"] if ent.get("id") == "p0_0")
    assert home["home"] == 1
    assert home["cash"] == 250
    assert "stock" not in home
    stocked = next(ent for ent in view["ports"] if ent.get("id") == "p0_5")
    assert stocked["stock"] == 50
    assert "home" not in stocked
    assert view["land"] == []
    assert "\nLAND\nMOVES\n" in prompt
    assert "(6,2)" in prompt

    assert len(view["moves"]) == len(legal)
    assert any(isinstance(action, AttackAction) for action in legal)
    attack_i = next(
        i for i, action in enumerate(legal) if isinstance(action, AttackAction)
    )
    assert view["moves"][attack_i].startswith("Attack ")
    end_i = next(
        i for i, action in enumerate(legal) if isinstance(action, EndTurnAction)
    )
    assert view["moves"][end_i] == "End"
    assert view["legal"][end_i] == {"type": "end"}

    build_lines = [line for line in prompt.splitlines() if line.startswith("B@")]
    assert len(build_lines) == 2
    assert "\n13 Build" not in prompt
    seen_builds = 0
    for line in build_lines:
        for part in line.split(": ", 1)[1].split(" | "):
            idx_s, type_name, cost_s = part.split()
            action = legal[int(idx_s)]
            assert isinstance(action, BuildShipAction)
            assert action.ship_type.value == type_name
            assert BASE_STATS[action.ship_type].cost == int(cost_s)
            seen_builds += 1
    assert seen_builds == sum(isinstance(a, BuildShipAction) for a in legal)

    engine.players[0].cash = 100
    poor = render_prompt(build_view(engine, 0, engine.enumerate_legal(0)))
    assert poor.startswith(STATIC_PREFIX)
    poor_moves = poor.split("\nMOVES\n", 1)[1]
    for banned in ("Carrier", "Battleship", "Cruiser", "Landing", "Builder"):
        assert banned not in poor_moves
    for kept in ("Destroyer 100", "Submarine 100", "Merchant 100"):
        assert kept in poor_moves


def test_stale_sighting_hides_true_position(tmp_path: Path) -> None:
    path = _write_map(
        tmp_path / "stale.json",
        {
            "width": 16,
            "height": 8,
            "land": [[0, 0], [15, 7]],
            "ports": {"0": [[0, 0]], "1": [[15, 7]]},
        },
    )
    engine, ships = _engine(
        path, [(0, ShipType.MERCHANT, (0, 1)), (1, ShipType.SUBMARINE, (3, 1))]
    )
    merchant, sub = ships
    engine._refresh_sightings()
    assert engine.players[0].sightings[sub.id].fresh
    engine.map.relocate_ship(sub.id, sub.position, (15, 6))
    sub.position = (15, 6)
    engine._refresh_sightings()
    sight = engine.players[0].sightings[sub.id]
    assert sight.fresh is False
    assert sight.position == (3, 1)

    legal = engine.enumerate_legal(0)
    view = build_view(engine, 0, legal, horizon=4, max_edges=20)
    prompt = render_prompt(view)
    ent = next(item for item in view["ships"] if item.get("id") == f"s{sub.id}")
    assert ent["pos"] == [3, 1]
    assert ent["fresh"] == 0
    assert ent["stale"] == engine.turn - sight.turn_seen
    assert "(15,6)" not in prompt
    assert "(3,1)" in prompt
    assert f"s{merchant.id}" in prompt
    assert all(pos != [15, 6] for pos in view["land"])
    assert [15, 7] not in view["land"]


def test_wall_blocks_capture_path(tmp_path: Path) -> None:
    path = _write_map(
        tmp_path / "wall.json",
        {
            "width": 6,
            "height": 3,
            "land": [[0, 0], [2, 0], [2, 1], [2, 2], [5, 0]],
            "ports": {"0": [[0, 0]], "1": [[5, 0]]},
        },
    )
    engine, _ships = _engine(path, [(0, ShipType.LANDING, (1, 1))])
    legal = engine.enumerate_legal(0)
    view = build_view(engine, 0, legal, horizon=4, max_edges=20)
    assert any(
        ent.get("id") == "p5_0" and ent.get("owner") == "enemy" and ent.get("home") == 1
        for ent in view["ports"]
    )
    assert view["land"] == [[2, 0], [2, 1], [2, 2]]
    assert "\n0: 2\n1: 2\n2: 2\n" in render_prompt(view)


def test_parse_choice_and_retry() -> None:
    assert parse_choice("4 I close on the cruiser.", 10) == (
        4,
        "I close on the cruiser",
    )
    assert parse_choice("I pick 2 because it loads.", 5) == (
        2,
        "I pick because it loads",
    )
    assert parse_choice("nope", 4) is None
    assert parse_choice("99 way off", 5) is None
    assert parse_choice("-1 nope", 5) is None
    assert parse_choice("0", 3) == (0, "")

    calls = {"n": 0}

    def flaky(prompt, temperature):
        calls["n"] += 1
        assert temperature == 0.3
        if calls["n"] == 1:
            return "let me think"
        return "1 sail north"

    picked = choose_move(flaky, "MOVES\n0 End\n1 Move", 2, random.Random(0), 0.3)
    assert picked["fallback"] is False
    assert picked["choice"] == 1
    assert picked["reasoning"] == "sail north"
    assert len(picked["replies"]) == 2
    assert picked["retry_prompt"].endswith("one short sentence.")

    def always_bad(prompt, temperature):
        return "hello"

    rng = random.Random(1)
    expected = random.Random(1).randrange(4)
    picked = choose_move(always_bad, "prompt", 4, rng, 0.2)
    assert picked["fallback"] is True
    assert picked["choice"] == expected
    assert picked["reasoning"] == ""
    assert len(picked["replies"]) == 2


def test_chat_payload_shape() -> None:
    assert (
        chat_completions_url("http://127.0.0.1:8080")
        == "http://127.0.0.1:8080/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://127.0.0.1:8080/v1/")
        == "http://127.0.0.1:8080/v1/chat/completions"
    )
    payload = chat_payload("hi", 0.3, "inkling", 48)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 48
    assert payload["stream"] is False
    assert payload["cache_prompt"] is True


def test_thinking_blocks_do_not_choose_a_move() -> None:
    both = {
        "choices": [
            {
                "message": {
                    "content": "4 I close on the cruiser.",
                    "reasoning_content": "maybe 9, or 1",
                }
            }
        ]
    }
    text = message_text(both)
    assert text == (
        "[Start thinking]\nmaybe 9, or 1\n[End thinking]\n4 I close on the cruiser."
    )
    assert parse_choice(text, 10) == (4, "I close on the cruiser")

    listed = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": "2 "}, "go"],
                    "reasoning_content": None,
                }
            }
        ]
    }
    assert message_text(listed) == "2 go"
    assert parse_choice(message_text(listed), 5) == (2, "go")

    reasoning_only = {
        "choices": [{"message": {"content": "", "reasoning_content": "3 sail"}}]
    }
    assert message_text(reasoning_only) == "[Start thinking]\n3 sail\n[End thinking]\n"
    assert parse_choice(message_text(reasoning_only), 5) is None

    assert parse_choice("<think>99</think>\n2 go", 5) == (2, "go")
    assert parse_choice("[Start thinking]\n7\n[End thinking]\n1 north", 5) == (
        1,
        "north",
    )
    assert parse_choice("<|content_thinking|>8<|content_text|>0 end", 3) == (0, "end")
    assert parse_choice("[Start thinking]\n4 I pick this", 10) is None
    assert parse_choice("<think>3", 5) is None
    assert parse_choice("<|content_thinking|>1 never closed", 4) is None


class _EndTurnClient:
    """First reply is garbage; the retry picks End from the menu."""

    def __init__(self) -> None:
        self.prompts = []

    def __call__(self, prompt: str, temperature: float) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) % 2 == 1:
            return "not a move"
        for line in prompt.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "End":
                return f"{parts[0]} ending the turn"
        raise AssertionError("menu has no End")


def test_fake_llm_plays_a_short_game(tmp_path: Path) -> None:
    map_path = write_two_home_map(tmp_path / "tiny.json", 6, 4)
    env = BattleboatsAEC(map_json_path=map_path, max_turns=1)
    env.reset(seed=0)
    client = _EndTurnClient()
    log_path = tmp_path / "game.jsonl"
    result = play_env(
        env,
        client,
        path=log_path,
        seed=0,
        game_index=0,
        temperature=0.3,
        horizon=4,
        max_edges=20,
        map_path=map_path,
    )
    assert result["winner"] is None
    assert result["reason"] == "max_turns"
    assert result["turn"] == 1
    assert result["steps"] == 2

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["type"] for row in rows] == [
        "game_start",
        "decision",
        "decision",
        "result",
    ]
    seats = []
    for row in rows:
        if row["type"] != "decision":
            continue
        prompt = row["prompt"]
        for section in ("FORMAT 3", "SHIPS", "PORTS", "LAND", "MOVES"):
            assert section in prompt
        assert prompt.startswith(STATIC_PREFIX)
        assert row["fallback"] is False
        assert row["action"] == {"type": "end"}
        assert row["reasoning"] == "ending the turn"
        assert len(row["replies"]) == 2
        assert row["latency_s"] >= 0
        assert render_prompt(row["state"]) == prompt
        seats.append(row["agent"])
    assert seats == ["player_0", "player_1"]


def test_fallback_random_still_finishes(tmp_path: Path) -> None:
    map_path = write_two_home_map(tmp_path / "tiny.json", 6, 4)
    env = BattleboatsAEC(map_json_path=map_path, max_turns=1)
    env.reset(seed=0)
    for player in env.engine.players:
        player.cash = 0
    agent = env.agent_selection
    env.infos[agent]["legal_actions"] = env.engine.enumerate_legal(
        env.engine.current_player
    )

    def always_bad(prompt, temperature):
        return "no number here"

    result = play_env(
        env,
        always_bad,
        path=tmp_path / "fallback.jsonl",
        seed=0,
        game_index=3,
        temperature=0.0,
        horizon=3,
        max_edges=5,
        map_path=map_path,
    )
    assert result["reason"] == "max_turns"
    assert result["steps"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "fallback.jsonl").read_text().splitlines()
    ]
    decisions = [row for row in rows if row["type"] == "decision"]
    assert len(decisions) == 2
    for row in decisions:
        assert row["fallback"] is True
        assert row["action"]["type"] == "end"
        assert row["choice"] == 0
        assert "SHIPS" in row["prompt"]
        assert "usage" not in row
        assert "timings" not in row


class _UsageClient:
    """Ends the turn and stamps llama-server usage on each call."""

    def __init__(self) -> None:
        self.last_usage = None
        self.last_timings = None
        self.n = 0

    def __call__(self, prompt: str, temperature: float) -> str:
        self.n += 1
        self.last_usage = {"prompt_tokens": 1000 + self.n, "completion_tokens": 4}
        self.last_timings = {"prompt_ms": 12.5, "predicted_n": 4}
        for line in prompt.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "End":
                return f"{parts[0]} ending the turn"
        raise AssertionError("menu has no End")


def test_decision_log_records_usage(tmp_path: Path) -> None:
    map_path = write_two_home_map(tmp_path / "tiny.json", 6, 4)
    env = BattleboatsAEC(map_json_path=map_path, max_turns=1)
    env.reset(seed=0)
    client = _UsageClient()
    result = play_env(
        env,
        client,
        path=tmp_path / "usage.jsonl",
        seed=0,
        game_index=0,
        temperature=0.0,
        horizon=4,
        max_edges=20,
        map_path=map_path,
    )
    assert result["steps"] == 2
    decisions = [
        json.loads(line)
        for line in (tmp_path / "usage.jsonl").read_text().splitlines()
        if json.loads(line)["type"] == "decision"
    ]
    assert [row["usage"]["prompt_tokens"] for row in decisions] == [1001, 1002]
    assert all(row["timings"]["prompt_ms"] == 12.5 for row in decisions)
    assert all(row["fallback"] is False for row in decisions)


def _token_estimate(text: str) -> int:
    """chars/4, or tiktoken cl100k when that package is installed."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return (len(text) + 3) // 4


def _standard_map(tmp_path: Path) -> str:
    """160x80 board. Prefer map.json when that file is already this size."""
    repo_map = (
        Path(__file__).resolve().parents[1]
        / "battleboats"
        / "core"
        / "config"
        / "map.json"
    )
    if repo_map.is_file():
        data = json.loads(repo_map.read_text())
        if data.get("width") == 160 and data.get("height") == 80:
            return str(repo_map)
    from battleboats.core.config.generate_map import generate_map_json

    yaml_path = (
        Path(__file__).resolve().parents[1]
        / "battleboats"
        / "core"
        / "config"
        / "mapConfigLarge.yaml"
    )
    out = tmp_path / "map_160x80.json"
    generate_map_json(str(yaml_path), str(out), seed=0)
    return str(out)


def _parse_land(prompt: str) -> set:
    body = prompt.split("\nLAND\n", 1)[1].split("\nMOVES\n", 1)[0]
    cells = set()
    for line in body.splitlines():
        if not line.strip():
            continue
        y_s, xs = line.split(": ", 1)
        y = int(y_s)
        for span in xs.split(","):
            if "-" in span:
                start_s, end_s = span.split("-", 1)
                cells.update((x, y) for x in range(int(start_s), int(end_s) + 1))
            else:
                cells.add((int(span), y))
    return cells


def test_sparse_prompt_is_much_smaller_than_the_grid(tmp_path: Path) -> None:
    """Opening and mid-game prompts stay far under the old 160x80 text grid."""
    path = _standard_map(tmp_path)
    engine = gameEngine(path)
    engine.reset(seed=0)
    assert (engine.map.width, engine.map.height) == (160, 80)
    initial_view = build_view(engine, 0, engine.enumerate_legal(0))
    initial = render_prompt(initial_view)

    waters = [
        (x, y)
        for y in range(engine.map.height)
        for x in range(engine.map.width)
        if engine.map.is_water((x, y))
    ]
    base = waters[len(waters) // 3]
    spots = [base]
    for dist in (2, 4, 8, 12):
        for x, y in waters:
            if abs(x - base[0]) + abs(y - base[1]) == dist:
                spots.append((x, y))
                break
    fleet = [
        (0, ShipType.DESTROYER),
        (0, ShipType.MERCHANT),
        (0, ShipType.LANDING),
        (1, ShipType.CRUISER),
        (1, ShipType.SUBMARINE),
        (1, ShipType.BATTLESHIP),
    ]
    spawned = []
    for (owner, ship_type), pos in zip(fleet, spots):
        spawned.append(engine._spawn_ship(owner, ship_type, pos))
    occupied = {ship.position for ship in spawned}
    far = next(pos for pos in reversed(waters) if pos not in occupied)
    sub = next(ship for ship in spawned if ship.type is ShipType.SUBMARINE)
    engine.map.relocate_ship(sub.id, sub.position, far)
    sub.position = far
    engine.turn = 12
    engine._refresh_sightings()
    for ship in engine.ships.values():
        if ship.owner == 0 and ship.type is ShipType.MERCHANT:
            ship.cargo = 30
        if ship.owner == 0 and ship.type is ShipType.DESTROYER:
            ship.tiles_moved_this_turn = 1
            ship.has_attacked = True
    for pos in sorted(engine.players[0].owned_port_positions):
        if pos != engine.players[0].home_port:
            engine.ports[pos].stockpile = 75
            break
    mid_view = build_view(engine, 0, engine.enumerate_legal(0))
    mid = render_prompt(mid_view)

    # The old GRID block alone was one character per cell plus a newline per row.
    grid_chars = engine.map.width * engine.map.height + engine.map.height
    for label, prompt in (("initial", initial), ("mid", mid)):
        assert prompt.count(f"{engine.map.width}x{engine.map.height}") == 1, label
        assert "GRID" not in prompt, label
        assert len(prompt) * 2 < grid_chars, (label, len(prompt), grid_chars)
        assert _token_estimate(prompt) < grid_chars // 6, (
            label,
            len(prompt),
            _token_estimate(prompt),
        )
    coast = {
        (ident[0], ident[1])
        for kind, ident in build_entity_refs(engine, 0)
        if kind == "coastline"
    }
    assert _parse_land(initial) == coast
    assert _parse_land(mid) == coast
    assert coast
    assert len(initial_view["land"]) == len(coast)
    assert "cargo=30" in mid
    assert "attacked=1" in mid
    assert "moved=1" in mid
    assert "stock=75" in mid
