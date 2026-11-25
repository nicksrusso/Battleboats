"""HTTP play UI: tactical/scripted/checkpoint games, JSON API, fog of war, and full playouts."""

import json
import random
import re
import threading
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

import pytest

from battleboats.core.actions import AttackAction, EndTurnAction
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import TOKEN_DIM
from battleboats.training.checkpoints import save_checkpoint
from battleboats.training.policy import PolicyNetwork
import battleboats.ui.server as server
from battleboats.ui.maps import list_map_options, resolve_map
from battleboats.ui.server import Handler, ThreadingHTTPServer
from battleboats.ui.session import PlaySession


@pytest.fixture
def ui_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_init = PlaySession.__init__

    def _capped(self, *args, **kwargs):
        kwargs.setdefault("step_budget", 8)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(PlaySession, "__init__", _capped)
    server._SESSION = None
    server._CHECKPOINT_DIR = tmp_path
    server._MAP_DIR = tmp_path / "maps"
    server._DEVICE = "cpu"
    server._AUTO_PLAY = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        server._SESSION = None
        server._AUTO_PLAY = False


def _request(port: int, method: str, path: str, form: dict | None = None):
    conn = HTTPConnection("127.0.0.1", port, timeout=60)
    body = None
    headers = {}
    if form is not None:
        body = urlencode(form)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, dict(resp.headers), payload


def _json_request(port: int, method: str, path: str, payload: dict | None = None):
    conn = HTTPConnection("127.0.0.1", port, timeout=60)
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload)
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw.decode())


def _save_policy(path: Path) -> None:
    policy = PolicyNetwork(
        token_dim=TOKEN_DIM,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    save_checkpoint(policy, path)


def test_health_and_scripted_game(ui_server: int) -> None:
    status, _, body = _request(ui_server, "GET", "/health")
    assert status == 200
    assert body == b"ok"
    status, headers, _ = _request(
        ui_server,
        "POST",
        "/start",
        {
            "opponent": "scripted",
            "human_side": "1",
            "width": "4",
            "height": "4",
            "max_turns": "8",
            "seed": "0",
        },
    )
    assert status == 303
    assert headers["Location"] == "/play"
    status, _, page = _request(ui_server, "GET", "/play")
    text = page.decode()
    assert status == 200
    assert "Scripted baseline (full-information opponent)" in text
    assert "Opponent:" in text
    status, headers, _ = _request(ui_server, "POST", "/act", {"index": "0"})
    assert status == 303
    status, _, page = _request(ui_server, "GET", "/play")
    assert "You:" in page.decode()


def test_checkpoint_opponent(ui_server: int, tmp_path: Path) -> None:
    ckpt = tmp_path / "smoke.pt"
    _save_policy(ckpt)
    status, headers, body = _request(
        ui_server,
        "POST",
        "/start",
        {
            "opponent": "checkpoint",
            "checkpoint": str(ckpt),
            "human_side": "1",
            "width": "4",
            "height": "4",
            "max_turns": "6",
            "seed": "1",
        },
    )
    assert status == 303, body.decode()
    status, _, page = _request(ui_server, "GET", "/play")
    text = page.decode()
    assert status == 200
    assert "smoke.pt" in text
    assert "Opponent:" in text


def test_missing_checkpoint_is_a_form_error(ui_server: int) -> None:
    status, _, page = _request(
        ui_server,
        "POST",
        "/start",
        {
            "opponent": "checkpoint",
            "checkpoint": "runs/does_not_exist.pt",
            "human_side": "0",
            "width": "4",
            "height": "4",
            "max_turns": "6",
            "seed": "0",
        },
    )
    assert status == 400
    assert "not found" in page.decode().lower()


def test_json_api_tactical_session_step_build_and_move(ui_server: int) -> None:
    status, maps_data = _json_request(ui_server, "GET", "/api/maps")
    assert status == 200
    assert any(m["value"] == "16x8" for m in maps_data["maps"])

    status, state = _json_request(
        ui_server,
        "POST",
        "/api/start",
        {"opponent": "tactical", "map": "16x8", "human_side": 0, "seed": 2},
    )
    assert status == 200
    assert state["status"]["opponent"] == "tactical"
    assert state["status"]["width"] == 16
    assert state["status"]["height"] == 8
    assert state["status"]["cash"] == 250
    home_port = next(p for p in state["own_ports"] if p["is_home"])

    # Buy a Destroyer ($100) at home port
    status, state = _json_request(
        ui_server,
        "POST",
        "/api/action",
        {"type": "build_ship", "port": home_port["pos"], "ship_type": "Destroyer"},
    )
    assert status == 200
    assert state["error"] == ""
    assert state["status"]["cash"] == 150
    assert len(state["own_ships"]) == 1
    assert state["own_ships"][0]["bought_this_turn"] is True
    assert state["own_ships"][0]["moves_left"] == 0

    # End turn -> tactical agent plays turn 0, human gets turn 1
    status, state = _json_request(ui_server, "POST", "/api/action", {"type": "end_turn"})
    assert status == 200
    assert state["status"]["turn"] == 1
    dd = state["own_ships"][0]
    assert dd["bought_this_turn"] is False
    assert dd["moves_left"] == 5
    assert len(dd["reachable"]) > 0

    # Multi-step path move
    target = dd["reachable"][-1]
    start_pos = dd["pos"]
    status, state = _json_request(
        ui_server,
        "POST",
        "/api/action",
        {"type": "move_path", "ship_id": dd["id"], "to": target},
    )
    assert status == 200
    assert state["error"] == ""
    moved_dd = state["own_ships"][0]
    assert moved_dd["pos"] == target
    assert moved_dd["pos"] != start_pos
    assert moved_dd["moves_left"] < 5


def test_fog_of_war_masking_and_stale_sightings(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="tactical",
        human_side=0,
        map_spec="48x24",
        seed=3,
        max_turns=30,
        map_dir=tmp_path / "maps",
    )
    engine = session.engine
    # Spawn an enemy submarine well outside Player 0's port spotting range (port spots sub at 1).
    opp_home = engine.players[1].home_port
    sub_pos = next(
        (opp_home[0] + dx, opp_home[1] + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if engine.map.in_bounds((opp_home[0] + dx, opp_home[1] + dy))
        and engine.map.is_water((opp_home[0] + dx, opp_home[1] + dy))
    )
    sub = engine._spawn_ship(1, ShipType.SUBMARINE, sub_pos)
    engine._refresh_sightings()

    state = session.to_state_dict()
    assert state["enemy_ships"] == []
    # Any enemy port >16 Manhattan from all friendly ports must be hidden and masked as land
    unsighted_ports = [
        ep for ep in engine.players[1].owned_port_positions
        if all(engine.map.manhattan(mp, ep) > 16 for mp in engine.players[0].owned_port_positions)
    ]
    assert unsighted_ports, "Expected at least one unsighted enemy port on 48x24"
    seen_port_positions = {tuple(ep["pos"]) for ep in state["enemy_ports"]}
    for up in unsighted_ports:
        assert up not in seen_port_positions
        assert state["terrain"][up[1]][up[0]] == 1

    # End turn as human -> opponent moves sub out of sight, must not leak into log or state
    pre_log_len = len(session.entries)
    session.apply_spec({"type": "end_turn"})
    post_state = session.to_state_dict()
    if sub.id not in engine.players[0].sightings:
        assert all(s["id"] != sub.id for s in post_state["enemy_ships"])
        for entry in session.entries[pre_log_len:]:
            assert f"#{sub.id}" not in entry["text"]

    # Now place a human Destroyer right next to the enemy sub so it's spotted, then move DD away
    if sub.id in engine.ships:
        live_sub = engine.ships[sub.id]
        dd_spot = next(
            (live_sub.position[0] + dx, live_sub.position[1] + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if engine.map.in_bounds((live_sub.position[0] + dx, live_sub.position[1] + dy))
            and engine.map.is_water((live_sub.position[0] + dx, live_sub.position[1] + dy))
            and not engine.map.is_occupied((live_sub.position[0] + dx, live_sub.position[1] + dy))
        )
        dd = engine._spawn_ship(0, ShipType.DESTROYER, dd_spot)
        engine._refresh_sightings()
        seen_state = session.to_state_dict()
        seen_sub = next(s for s in seen_state["enemy_ships"] if s["id"] == sub.id)
        assert seen_sub["fresh"] is True
        assert seen_sub["pos"] == list(live_sub.position)

        # Unseen attacker combat log must not leak hidden attacker coordinates
        bb_pos = (engine.players[0].home_port[0] + 1, engine.players[0].home_port[1])
        if not engine.map.is_water(bb_pos) or engine.map.is_occupied(bb_pos):
            bb_pos = next(
                (engine.players[0].home_port[0] + dx, engine.players[0].home_port[1] + dy)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if engine.map.in_bounds((engine.players[0].home_port[0] + dx, engine.players[0].home_port[1] + dy))
                and engine.map.is_water((engine.players[0].home_port[0] + dx, engine.players[0].home_port[1] + dy))
                and not engine.map.is_occupied((engine.players[0].home_port[0] + dx, engine.players[0].home_port[1] + dy))
            )
        bb = engine._spawn_ship(0, ShipType.BATTLESHIP, bb_pos)
        # Put an enemy sub at distance 2 from BB (sub spots BB at 4, BB spots sub at 1!)
        ambush_pos = next(
            p
            for p in engine.map.tiles_within(bb.position, 2)
            if engine.map.manhattan(p, bb.position) == 2
            and engine.map.is_water(p)
            and not engine.map.is_occupied(p)
            and engine.map.manhattan(p, engine.players[0].home_port) >= 2
            and engine.map.manhattan(p, dd.position) >= 4
        )
        ambush_sub = engine._spawn_ship(1, ShipType.SUBMARINE, ambush_pos)
        engine._refresh_sightings()
        assert ambush_sub.id not in engine.players[0].sightings
        engine.current_player = 1
        session._apply(AttackAction(ambush_sub.id, bb.id), 1)
        last_log = session.entries[-1]["text"]
        assert "Unseen enemy" in last_log
        assert str(ambush_pos) not in last_log
        assert ("SUNK" in last_log) or ("MISSED" in last_log)


def test_full_game_vs_tactical_random_human_completes(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="tactical",
        human_side=0,
        map_spec="16x8",
        seed=7,
        max_turns=30,
        map_dir=tmp_path / "maps",
        step_budget=400,
    )
    rng = random.Random(123)
    steps = 0
    while not session.over and steps < 2500:
        legal = session.legal()
        assert legal, "Human turn had no legal actions before game over"
        idx = rng.randrange(len(legal))
        session.apply(idx)
        steps += 1
        if steps % 50 == 0 and not session.over:
            # Ensure human doesn't get stuck moving back and forth forever on one turn
            end_idx = next(
                (i for i, a in enumerate(session.legal()) if isinstance(a, EndTurnAction)),
                None,
            )
            if end_idx is not None:
                session.apply(end_idx)
                steps += 1
    assert session.over is True
    assert session.result_text != ""


def test_map_pool_presets_resolve(tmp_path: Path) -> None:
    opts = list_map_options(search_roots=[])
    names = {o["value"] for o in opts}
    assert {"16x8", "24x12", "32x16", "48x24"}.issubset(names)
    res = resolve_map("training", seed=1, map_dir=tmp_path / "maps")
    assert res.width == 24 and res.height == 12
    assert res.path.is_file()


_BANNED_LLM_FIELDS = ("reasoning", "replies", "raw_reply", "prompt", "retry_prompt")


def _assert_state_is_fog_safe(state: dict, engine, human_id: int, ever_seen: set[int]) -> None:
    """The browser payload must not contain unseen enemy units/positions or LLM text."""
    human = engine.players[human_id]
    own_ids = set(human.owned_ship_ids)
    last = (state.get("status") or {}).get("last_llm") or {}
    for key in _BANNED_LLM_FIELDS:
        assert not last.get(key), f"UI last_llm leaked {key}: {last.get(key)!r}"

    for es in state.get("enemy_ships") or []:
        sight = human.sightings.get(es["id"])
        assert sight is not None, f"unsighted enemy ship {es} in payload"
        assert list(es["pos"]) == [sight.position[0], sight.position[1]]
        assert bool(es["fresh"]) is bool(sight.fresh)

    for ep in state.get("enemy_ports") or []:
        assert tuple(ep["pos"]) in human.port_sightings

    st = state["status"]
    assert st["cash"] == human.cash
    assert st["ships"] == len(human.owned_ship_ids)
    if not st["over"]:
        assert st["winner"] is None
        assert not st.get("result")

    for entry in state.get("log") or []:
        text = entry.get("text", "")
        # Human-authored lines ("You: DD #4 attacked ...") use kind combat_miss
        # when a shot misses. Those IDs are the fire-er (possibly later sunk,
        # so no longer in owned_ship_ids) and a target the human just shot at.
        if text.startswith("You:"):
            continue
        if not (text.startswith("Opponent") or entry.get("kind") in {"enemy_move", "spot", "combat_loss", "combat_miss"}):
            continue
        for raw_id in re.findall(r"#(\d+)", text):
            sid = int(raw_id)
            if sid in own_ids:
                continue
            # "attacked your DD #13" — already skipped if the id is currently owned;
            # a sunk friendly ship may no longer be in owned_ship_ids.
            if "your " in text and f"#{sid}" in text.split("your", 1)[-1][:40]:
                continue
            assert sid in ever_seen, f"log mentions never-seen enemy #{sid}: {text}"

    for mv in state.get("recent_opponent_moves") or []:
        sid = mv.get("ship_id")
        if sid is not None:
            assert sid in ever_seen, f"trail for never-seen ship: {mv}"


def _start_ui_server(tmp_path: Path):
    server._SESSION = None
    server._CHECKPOINT_DIR = tmp_path
    server._MAP_DIR = tmp_path / "maps"
    server._DEVICE = "cpu"
    server._AUTO_PLAY = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _stop_ui_server(httpd, thread) -> None:
    httpd.shutdown()
    thread.join(timeout=5)
    session = server._SESSION
    if session is not None:
        try:
            session.close(complete=session.over)
        except Exception:
            pass
    server._SESSION = None
    server._AUTO_PLAY = False


def test_new_game_form_lists_scripted_opponents(ui_server: int) -> None:
    status, _, page = _request(ui_server, "GET", "/")
    text = page.decode()
    assert status == 200
    assert 'value="tactical"' in text
    assert 'value="scripted"' in text
    assert 'value="fog_commander"' in text
    assert "Fog Commander (fog of war)" in text
    assert "Tactical (fog of war)" in text
    assert "Scripted baseline (full-information opponent)" in text
    assert "you still have fog of war" in text.lower() or "You always play under fog of war" in text


@pytest.mark.parametrize("opponent", ("tactical", "scripted", "fog_commander"))
@pytest.mark.parametrize("human_side", (0, 1))
def test_full_game_via_api_vs_scripted_opponents_is_fog_safe(
    tmp_path: Path, opponent: str, human_side: int
) -> None:
    httpd, thread = _start_ui_server(tmp_path)
    try:
        port = httpd.server_address[1]
        status, state = _json_request(
            port,
            "POST",
            "/api/start",
            {
                "opponent": opponent,
                "map": "16x8",
                "human_side": human_side,
                "seed": 7 + human_side,
                "max_turns": 24,
            },
        )
        assert status == 200, state
        assert state["status"]["opponent"] == opponent
        session = server._SESSION
        assert session is not None
        human = session.engine.players[human_side]
        ever_seen = set(human.sightings)
        _assert_state_is_fog_safe(state, session.engine, human_side, ever_seen)

        rng = random.Random(123)
        steps = 0
        while not state["status"]["over"] and steps < 1500:
            legal = state["legal_actions"]
            assert legal, f"{opponent} human had no legal actions before game over"
            if steps % 6 == 5:
                end = next((a for a in legal if a.get("type") == "end_turn"), None)
                payload = {"index": end["index"]} if end is not None else {"index": rng.choice(legal)["index"]}
            else:
                payload = {"index": rng.choice(legal)["index"]}
            status, state = _json_request(port, "POST", "/api/action", payload)
            assert status == 200, state
            human = session.engine.players[human_side]
            ever_seen |= set(human.sightings)
            _assert_state_is_fog_safe(state, session.engine, human_side, ever_seen)
            steps += 1
            if steps % 11 == 0:
                status, polled = _json_request(port, "GET", "/api/state")
                assert status == 200
                _assert_state_is_fog_safe(polled, session.engine, human_side, ever_seen)
        assert state["status"]["over"] is True
        assert state["status"]["result"]
        _assert_state_is_fog_safe(state, session.engine, human_side, ever_seen)
    finally:
        _stop_ui_server(httpd, thread)
