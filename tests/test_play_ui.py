"""HTTP play UI: scripted game, checkpoint game, and a missing file."""

import threading
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

import pytest

from battleboats.training.checkpoints import save_checkpoint
from battleboats.training.policy import PolicyNetwork
from battleboats.ui.server import Handler, ThreadingHTTPServer
import battleboats.ui.server as server
from battleboats.ui.session import PlaySession


@pytest.fixture
def ui_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_init = PlaySession.__init__

    def _capped(self, *args, **kwargs):
        kwargs["step_budget"] = 8
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(PlaySession, "__init__", _capped)
    server._SESSION = None
    server._CHECKPOINT_DIR = tmp_path
    server._MAP_DIR = tmp_path / "maps"
    server._DEVICE = "cpu"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        server._SESSION = None


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


def _save_policy(path: Path) -> None:
    policy = PolicyNetwork(
        token_dim=28,
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
    assert "Scripted (full information)" in text
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
