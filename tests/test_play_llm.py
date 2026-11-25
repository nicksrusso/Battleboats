"""LLM / Inkling play-UI opponent: mocked client, no network."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
import pytest

from battleboats.core.actions import EndTurnAction
from battleboats.llm.openrouter import (
    INKLING_MODEL_ID,
    OpenRouterClient,
    decide_llm_action,
    format_usage_line,
    load_openrouter_api_key,
    openrouter_payload,
    resolve_model_id,
    resolve_reasoning,
    usage_summary,
)
from battleboats.llm.play import message_text
from battleboats.llm.prompt import FORMAT_VERSION
import battleboats.ui.server as server
from battleboats.ui.server import Handler, ThreadingHTTPServer
from battleboats.ui.session import PlaySession


def _end_turn_reply(prompt: str, temperature: float = 0.3) -> str:
    matches = re.findall(r"^(\d+) End\b", prompt, re.M)
    idx = matches[-1] if matches else "0"
    return f"{idx} end the turn"


class ScriptedLLM:
    def __init__(self, replies=None, usage=None, hold: threading.Event | None = None):
        self.replies = list(replies) if replies is not None else []
        self.prompts = []
        self.temperatures = []
        self.last_usage = usage or {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
            "cost": 0.00042,
            "completion_tokens_details": {"reasoning_tokens": 60},
        }
        self.last_summary = usage_summary(self.last_usage)
        self.hold = hold

    def __call__(self, prompt: str, temperature: float) -> str:
        if self.hold is not None:
            self.hold.wait(timeout=5)
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        if self.replies:
            return self.replies.pop(0)
        return _end_turn_reply(prompt, temperature)


def test_resolve_model_and_reasoning() -> None:
    assert resolve_model_id(None) == INKLING_MODEL_ID
    assert resolve_model_id("inkling") == INKLING_MODEL_ID
    assert resolve_model_id("thinkingmachines/inkling-2-mini") == "thinkingmachines/inkling-2-mini"
    assert resolve_reasoning("max") == "max"
    assert resolve_reasoning(None) == "max"
    with pytest.raises(ValueError):
        resolve_reasoning("ultra")


def test_openrouter_payload_uses_reasoning_max() -> None:
    payload = openrouter_payload(
        "hi",
        model=INKLING_MODEL_ID,
        temperature=0.3,
        max_tokens=32768,
        reasoning_effort="max",
    )
    assert payload["model"] == INKLING_MODEL_ID
    assert payload["reasoning"] == {"effort": "max"}
    assert payload["max_tokens"] == 32768
    assert payload["stream"] is False
    assert "cache_prompt" not in payload


def test_load_key_from_env_and_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        load_openrouter_api_key(environ={}, search_paths=[tmp_path / "missing.env"])

    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-test-from-file\n")
    assert load_openrouter_api_key(environ={}, search_paths=[env_file]) == "sk-test-from-file"
    # Env wins over the file.
    assert (
        load_openrouter_api_key(
            environ={"OPENROUTER_API_KEY": "sk-test-from-env"},
            search_paths=[env_file],
        )
        == "sk-test-from-env"
    )


def test_client_does_not_put_key_in_errors() -> None:
    def opener(req, timeout=None):
        raise RuntimeError("should not be called")

    # A bogus opener path isn't needed; the constructor stores the key privately.
    client = OpenRouterClient(
        model="inkling",
        reasoning_effort="max",
        api_key="sk-secret-do-not-leak",
        opener=opener,
    )
    assert "sk-secret" not in repr(client)
    assert client._api_key == "sk-secret-do-not-leak"
    dumped = json.dumps(openrouter_payload("x", model=client.model, temperature=0.3, max_tokens=8, reasoning_effort="max"))
    assert "sk-secret" not in dumped


def test_message_text_reads_openrouter_reasoning_field() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "3 buy a destroyer",
                    "reasoning": "maybe 9 first",
                }
            }
        ]
    }
    text = message_text(body)
    assert "3 buy a destroyer" in text
    parsed_visible_end = text.split("[End thinking]")[-1]
    assert "9" not in parsed_visible_end or "buy a destroyer" in parsed_visible_end


def test_decide_llm_action_format4_and_end_turn(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="random",
        human_side=0,
        width=4,
        height=4,
        seed=0,
        max_turns=8,
        map_dir=tmp_path / "maps",
    )
    client = ScriptedLLM()
    action, meta = decide_llm_action(session.engine, 0, random.Random(0), client, temperature=0.3)
    assert isinstance(action, EndTurnAction)
    assert meta["fallback"] is False
    assert client.prompts, "client was not called"
    assert f"FORMAT {FORMAT_VERSION}" in client.prompts[0] or "FORMAT 4" in client.prompts[0]
    assert "MOVES" in client.prompts[0]
    assert "seat player_0" in client.prompts[0]
    line = format_usage_line(client.last_summary, meta["latency_s"])
    assert "tok" in line
    assert "$0.0004" in line


def test_illegal_reply_retries_then_fallback(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="random",
        human_side=0,
        width=4,
        height=4,
        seed=1,
        max_turns=8,
        map_dir=tmp_path / "maps",
    )
    legal = session.engine.enumerate_legal(0)
    client = ScriptedLLM(["not a move", "still no"])
    rng = random.Random(7)
    expected = random.Random(7).randrange(len(legal))
    action, meta = decide_llm_action(session.engine, 0, rng, client, temperature=0.3)
    assert meta["fallback"] is True
    assert action == legal[expected]
    assert len(client.prompts) == 2


def test_api_error_retries_then_passes(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="random",
        human_side=0,
        width=4,
        height=4,
        seed=2,
        max_turns=8,
        map_dir=tmp_path / "maps",
    )

    class Boom:
        def __init__(self):
            self.n = 0

        def __call__(self, prompt: str, temperature: float) -> str:
            self.n += 1
            raise RuntimeError("OpenRouter HTTP 503: unavailable")

    boom = Boom()
    action, meta = decide_llm_action(session.engine, 0, random.Random(0), boom, temperature=0.3)
    assert isinstance(action, EndTurnAction)
    assert meta["passed"] is True
    assert meta["fallback"] is True
    assert boom.n == 2
    assert "503" in meta["error"]
    assert "Bearer" not in meta["error"]


def test_session_async_thinking_with_mocked_client(tmp_path: Path) -> None:
    hold = threading.Event()
    client = ScriptedLLM(hold=hold)
    session = PlaySession(
        opponent="inkling",
        human_side=1,
        width=4,
        height=4,
        seed=3,
        max_turns=8,
        map_dir=tmp_path / "maps",
        llm_client=client,
        llm_model="inkling",
        llm_reasoning="max",
    )
    # Player 1 is human, so the LLM starts player 0's opening turn in a worker.
    assert session.thinking is True
    state = session.to_state_dict()
    assert state["status"]["thinking"] is True
    assert "Inkling thinking" in state["status"]["thinking_message"]
    assert state["status"]["llm_model"] == INKLING_MODEL_ID
    assert state["status"]["llm_reasoning"] == "max"
    session.apply_spec({"type": "end_turn"})
    assert session.error.startswith("Wait")
    hold.set()
    if session._worker is not None:
        session._worker.join(timeout=5)
    assert session.thinking is False
    assert session.engine.current_player == 1
    assert session.last_llm.get("total_tokens") == 200
    assert any(e["kind"] == "llm" for e in session.entries)


@pytest.fixture
def ui_server(tmp_path: Path):
    server._SESSION = None
    server._CHECKPOINT_DIR = tmp_path
    server._MAP_DIR = tmp_path / "maps"
    server._DEVICE = "cpu"
    server._AUTO_PLAY = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], tmp_path
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        server._SESSION = None


def _json_request(port: int, method: str, path: str, payload: dict | None = None, timeout: float = 10):
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
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


def test_http_poll_returns_while_llm_thinks(ui_server, tmp_path: Path) -> None:
    port, maps = ui_server
    hold = threading.Event()
    client = ScriptedLLM(hold=hold)
    session = PlaySession(
        opponent="llm",
        human_side=1,
        width=4,
        height=4,
        seed=4,
        max_turns=8,
        map_dir=maps / "maps",
        llm_client=client,
        llm_reasoning="max",
    )
    server._SESSION = session
    started = time.monotonic()
    status, state = _json_request(port, "GET", "/api/state")
    elapsed = time.monotonic() - started
    assert status == 200
    assert elapsed < 1.0
    assert state["status"]["thinking"] is True
    # Action while thinking is rejected quickly, not queued behind the LLM call.
    started = time.monotonic()
    status, state = _json_request(port, "POST", "/api/action", {"type": "end_turn"})
    assert status == 200
    assert time.monotonic() - started < 1.0
    assert "thinking" in state["error"].lower()
    hold.set()
    if session._worker is not None:
        session._worker.join(timeout=5)
    status, state = _json_request(port, "GET", "/api/state")
    assert state["status"]["thinking"] is False
    assert state["status"]["to_move"] == 1


def _read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_human_game_log_records_actions_and_can_replay(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="random",
        human_side=0,
        width=4,
        height=4,
        seed=11,
        max_turns=6,
        map_dir=tmp_path / "maps",
    )
    assert session.log_path.parent == (tmp_path / "human_games")
    rng = random.Random(0)
    steps = 0
    while not session.over and steps < 80:
        legal = session.legal()
        assert legal
        session.apply(rng.randrange(len(legal)))
        steps += 1
        if steps % 8 == 0 and not session.over:
            end = next(i for i, a in enumerate(session.legal()) if isinstance(a, EndTurnAction))
            session.apply(end)
            steps += 1
    assert session.over
    assert session.log_path.is_file()
    rows = _read_jsonl(session.log_path)
    assert rows[0]["type"] == "game_start"
    assert rows[0]["complete"] is False
    assert rows[0]["seed"] == 11
    assert rows[0]["human_side"] == 0
    assert rows[0]["opponent"] == "random"
    assert rows[0]["map"]["width"] == 4
    assert "config_sha256" in rows[0]["rules"]
    assert rows[0]["rules"]["git_sha"]
    assert rows[0]["state"]["ships"] == []
    decisions = [r for r in rows if r["type"] == "decision"]
    assert decisions
    assert {r["actor"] for r in decisions} == {"human", "opponent"}
    first = decisions[0]
    assert first["action"]["type"] in {
        "move", "attack", "build_ship", "build_port", "capture", "load", "unload", "end",
    }
    assert first["legal"]
    assert "observation" in first and first["observation"]["seat_id"] == first["actor_id"]
    assert "state" in first and "state_after" in first
    assert "players" in first["state"]
    end = rows[-1]
    assert end["type"] == "game_end"
    assert end["complete"] is True
    assert end["reason"] in {"capture", "turn_limit"}
    assert end["turn"] >= 0

    # Replay seed + action sequence through a fresh engine.
    from battleboats.core.actions import (
        AttackAction,
        BuildPortAction,
        BuildShipAction,
        CapturePortAction,
        MerchantLoadAction,
        MerchantUnloadAction,
        MoveAction,
    )
    from battleboats.core.shipyard.ship_type import ShipType

    def _rehydrate(rec):
        typ = rec["type"]
        if typ == "end":
            return EndTurnAction()
        if typ == "move":
            return MoveAction(rec["ship"], tuple(rec["to"]))
        if typ == "attack":
            return AttackAction(rec["ship"], rec["target"])
        if typ == "build_ship":
            return BuildShipAction(tuple(rec["port"]), tuple(rec["spawn"]), ShipType(rec["ship_type"]))
        if typ == "build_port":
            return BuildPortAction(rec["ship"], tuple(rec["at"]))
        if typ == "capture":
            return CapturePortAction(rec["ship"], tuple(rec["port"]))
        if typ == "load":
            return MerchantLoadAction(rec["ship"], tuple(rec["port"]))
        if typ == "unload":
            return MerchantUnloadAction(rec["ship"], tuple(rec["port"]))
        raise AssertionError(typ)

    replay = session.engine.__class__(str(session.map_path))
    replay.reset(seed=11)
    for row in decisions:
        action = _rehydrate(row["action"])
        replay.step(action)
    assert replay.turn == session.engine.turn
    assert replay.winner == session.engine.winner
    assert replay.current_player == session.engine.current_player
    assert replay.players[0].cash == session.engine.players[0].cash
    assert set(replay.ships) == set(session.engine.ships)


def test_incomplete_game_log_is_marked(tmp_path: Path) -> None:
    session = PlaySession(
        opponent="random",
        human_side=0,
        width=4,
        height=4,
        seed=5,
        max_turns=20,
        map_dir=tmp_path / "maps",
    )
    legal = session.legal()
    session.apply(0 if legal else 0)
    session.close(complete=False)
    rows = _read_jsonl(session.log_path)
    assert rows[0]["type"] == "game_start"
    assert rows[-1]["type"] == "game_end"
    assert rows[-1]["complete"] is False
    assert rows[-1]["reason"] == "incomplete"
    # Closing again is a no-op (no second game_end).
    session.close(complete=False)
    again = _read_jsonl(session.log_path)
    assert sum(1 for r in again if r["type"] == "game_end") == 1


def test_llm_decision_fields_are_logged(tmp_path: Path) -> None:
    hold = threading.Event()
    hold.set()
    client = ScriptedLLM(hold=hold)
    session = PlaySession(
        opponent="inkling",
        human_side=1,
        width=4,
        height=4,
        seed=8,
        max_turns=8,
        map_dir=tmp_path / "maps",
        llm_client=client,
        llm_reasoning="max",
    )
    if session._worker is not None:
        session._worker.join(timeout=5)
    assert session.thinking is False
    rows = _read_jsonl(session.log_path)
    assert rows[0]["llm"]["model"] == INKLING_MODEL_ID
    assert rows[0]["llm"]["reasoning"] == "max"
    llm_decisions = [r for r in rows if r["type"] == "decision" and r.get("llm")]
    assert llm_decisions
    rec = llm_decisions[0]
    assert rec["actor"] == "opponent"
    assert rec["llm"]["fallback"] is False
    assert rec["llm"]["replies"]
    assert rec["llm"]["latency_s"] is not None
    assert rec["llm"]["usage_summary"]["total_tokens"] == 200
    assert rec["llm"]["usage_summary"]["cost_usd"] == 0.00042
    assert rec["llm"]["prompt"]
    assert "FORMAT 4" in rec["llm"]["prompt"] or "MOVES" in rec["llm"]["prompt"]
