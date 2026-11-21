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
from battleboats.envs.skill_maps import write_two_home_map
from battleboats.llm.play import (chat_completions_url, chat_payload,
                                  choose_move, parse_choice, play_env)
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


def _cell(grid, x, y, height):
    return grid[height - 1 - y][x]


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

    for section in ("FORMAT 2", "ENTITIES", "GRID", "FACTS", "EDGES", "MOVES"):
        assert section in prompt
    assert view["format_version"] == FORMAT_VERSION == 2
    assert prompt.startswith(STATIC_PREFIX)
    assert render_prompt(view) == prompt
    assert "424242" not in prompt
    assert "gold 250" in prompt
    assert all(ent.get("kind") not in ("gold", "turn") for ent in view["entities"])
    assert '"cargo":0' not in prompt
    assert '"moved"' not in prompt
    assert '"attacked"' not in prompt

    ids = {ent.get("id") for ent in view["entities"]}
    assert f"s{sub.id}" not in ids
    assert f"s{cruiser.id}" in ids
    cruiser_ent = next(
        ent for ent in view["entities"] if ent.get("id") == f"s{cruiser.id}"
    )
    assert cruiser_ent["owner"] == "enemy"
    assert cruiser_ent["pos"] == [6, 2]
    assert "fresh" not in cruiser_ent
    assert "seen" not in cruiser_ent
    assert "cargo" not in cruiser_ent
    merchant_ent = next(
        ent for ent in view["entities"] if ent.get("id") == f"s{merchant.id}"
    )
    assert merchant_ent["cargo"] == 40
    home = next(ent for ent in view["entities"] if ent.get("id") == "p0_0")
    assert home["home"] is True
    assert "stock" not in home
    stocked = next(ent for ent in view["entities"] if ent.get("id") == "p0_5")
    assert stocked["stock"] == 50
    assert "home" not in stocked
    assert _cell(view["grid"], 7, 0, 6) == "."
    assert _cell(view["grid"], 6, 2, 6) == "r"
    assert _cell(view["grid"], 3, 2, 6) == "D"
    assert _cell(view["grid"], 0, 0, 6) == "H"
    assert _cell(view["grid"], 0, 5, 6) == "P"
    assert _cell(view["grid"], 7, 5, 6) == "h"

    def edge(kind, src, dst):
        return next(
            item
            for item in view["edges"]
            if item["kind"] == kind and item["from"] == src and item["to"] == dst
        )

    assert edge("can_attack", f"s{destroyer.id}", f"s{cruiser.id}")["turns"] == 0
    assert edge("can_load", f"s{merchant.id}", "p0_5")["turns"] == 1
    assert edge("can_unload", f"s{merchant.id}", "p0_0")["turns"] == 0
    assert edge("can_capture", f"s{landing.id}", "p7_5")["turns"] == 0
    assert edge("threat", f"s{cruiser.id}", f"s{destroyer.id}")["turns"] == 0
    assert edge("threat", f"s{cruiser.id}", f"s{merchant.id}")["turns"] == 0
    assert edge("threat", f"s{cruiser.id}", f"s{landing.id}")["turns"] == 0
    assert all(f"s{sub.id}" not in (item["from"], item["to"]) for item in view["edges"])

    facts = {fact["id"]: fact for fact in view["facts"]}
    assert facts[f"s{destroyer.id}"]["home_turns"] == 0
    assert facts[f"s{destroyer.id}"]["cargo_turns"] == 0
    assert facts[f"s{destroyer.id}"]["in_range"] == [f"s{cruiser.id}"]
    assert facts[f"s{merchant.id}"]["home_turns"] == 0
    assert facts[f"s{merchant.id}"]["cargo_turns"] == 1
    assert facts[f"s{merchant.id}"]["in_range"] == []
    assert facts[f"s{landing.id}"]["home_turns"] == 2
    assert facts[f"s{landing.id}"]["cargo_turns"] == 1

    near = build_view(engine, 0, legal, horizon=0, max_edges=20)
    assert all(item["turns"] == 0 for item in near["edges"])
    assert all(item["kind"] != "can_load" for item in near["edges"])
    capped = build_view(engine, 0, legal, horizon=4, max_edges=1)
    assert len(capped["edges"]) == 1
    assert capped["edges"][0]["turns"] == 0
    assert capped["edges"][0]["kind"] == "can_attack"

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
    ent = next(item for item in view["entities"] if item.get("id") == f"s{sub.id}")
    assert ent["pos"] == [3, 1]
    assert ent["fresh"] is False
    assert "[15,6]" not in prompt
    assert _cell(view["grid"], 3, 1, 8) == "s"
    assert _cell(view["grid"], 15, 6, 8) == "."
    assert _cell(view["grid"], 15, 7, 8) == "#"
    threat = next(item for item in view["edges"] if item["kind"] == "threat")
    assert threat["from"] == f"s{sub.id}"
    assert threat["to"] == f"s{merchant.id}"
    assert threat["turns"] == 0


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
        ent.get("id") == "p5_0" and ent.get("owner") == "enemy"
        for ent in view["entities"]
    )
    assert all(item["kind"] != "can_capture" for item in view["edges"])


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
        for section in ("FORMAT 2", "ENTITIES", "GRID", "FACTS", "EDGES", "MOVES"):
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
        assert "ENTITIES" in row["prompt"]
