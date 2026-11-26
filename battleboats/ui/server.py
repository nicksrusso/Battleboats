"""Stdlib HTTP UI. Bind to 127.0.0.1 and open the printed URL.

Supports one active local game at a time behind a thread lock.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from battleboats.llm.openrouter import INKLING_MODEL_ID
from battleboats.ui.maps import list_map_options
from battleboats.ui.session import PlaySession

_LOCK = threading.Lock()
_SESSION: Optional[PlaySession] = None
_CHECKPOINT_DIR = Path("runs")
_DEVICE = "auto"
_MAP_DIR = Path("runs/play_maps")
_AUTO_PLAY = False


def list_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pt") if p.is_file())


def _page(body: str) -> bytes:
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Battleboats</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 1.5rem; background: #0b1118; color: #e7ecf1; max-width: 44rem; }}
label {{ display: block; margin: 0.65rem 0 0.2rem; }}
input, select {{ padding: 0.35rem 0.5rem; background: #131c26; color: #e7ecf1; border: 1px solid #233243; border-radius: 4px; }}
button {{ margin-top: 1rem; background: #0f766e; color: white; border: 1px solid #14b8a6; border-radius: 4px; padding: 0.5rem 0.9rem; cursor: pointer; font-weight: 600; }}
button:hover {{ background: #115e59; }}
.err {{ color: #fecaca; background: #451a1a; border: 1px solid #b91c1c; padding: 0.5rem 0.75rem; border-radius: 4px; }}
.row {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
a {{ color: #38bdf8; }}
</style></head><body>
<h1>Battleboats</h1>
{body}
</body></html>"""
    return doc.encode()


def _index(error: str = "") -> bytes:
    ckpts = list_checkpoints(_CHECKPOINT_DIR)
    ckpt_options = "".join(
        f'<option value="{html.escape(str(p))}"></option>' for p in ckpts
    )
    map_options_html = []
    for opt in list_map_options():
        val = html.escape(opt["value"])
        lbl = html.escape(opt["label"])
        map_options_html.append(f'<option value="{val}">{lbl}</option>')
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    default_ckpt = str(ckpts[-1]) if ckpts else ""
    resume_link = (
        '<p><a href="/play">Return to current game</a></p>'
        if _SESSION is not None
        else ""
    )
    body = f"""{err}{resume_link}
<form method="post" action="/start">
<label>Opponent</label>
<label><input type="radio" name="opponent" value="tactical" checked> Tactical (fog of war) — scripted agent limited to what it can see</label>
<label><input type="radio" name="opponent" value="scripted_agent"> Scripted agent (fog of war) — doctrine-driven, limited to what it can see</label>
<label><input type="radio" name="opponent" value="scripted"> Scripted baseline (full-information opponent) — you still have fog of war</label>
<label><input type="radio" name="opponent" value="inkling"> Inkling (OpenRouter, high thinking)</label>
<label><input type="radio" name="opponent" value="llm"> LLM (OpenRouter, choose a model)</label>
<label><input type="radio" name="opponent" value="random"> Random</label>
<label><input type="radio" name="opponent" value="checkpoint"> Trained checkpoint</label>
<label>LLM model (OpenRouter id; default is the Inkling flagship)</label>
<input name="llm_model" value="{html.escape(INKLING_MODEL_ID)}" size="58">
<label>Reasoning
  <select name="llm_reasoning">
    <option value="max">max</option>
    <option value="high" selected>high</option>
    <option value="medium">medium</option>
    <option value="low">low</option>
    <option value="minimal">minimal</option>
    <option value="none">none</option>
  </select>
</label>
<label>Checkpoint path (optional unless Checkpoint selected)</label>
<input name="checkpoint" list="ckpts" value="{html.escape(default_ckpt)}" size="58">
<datalist id="ckpts">{ckpt_options}</datalist>
<label>Map (from map pool or presets)</label>
<select name="map">{''.join(map_options_html)}</select>
<label>Your side</label>
<label><input type="radio" name="human_side" value="0" checked> Player 0 (first)</label>
<label><input type="radio" name="human_side" value="1"> Player 1</label>
<div class="row">
<label>Seed <input name="seed" type="number" value="0"></label>
<label>Max turns (blank = auto) <input name="max_turns" type="number" placeholder="auto" min="5" max="500"></label>
</div>
<button type="submit">Start game</button>
</form>
<p>Checkpoints are listed from <code>{html.escape(str(_CHECKPOINT_DIR))}</code>.
Maps come from <code>battleboats/envs/map_pool.py</code> presets or discovered pool JSON files.
You always play under fog of war. The tactical and scripted agents are also fog-limited; the scripted baseline may use the full board.
An Inkling / LLM game needs <code>OPENROUTER_API_KEY</code> in the environment or a repo-root <code>.env</code> file (never committed).</p>
"""
    return _page(body)


def _int_or_none(val: str) -> Optional[int]:
    val = str(val).strip()
    return int(val) if val else None


def _start_from_params(params: dict) -> PlaySession:
    global _SESSION
    opponent = str(params.get("opponent", "tactical")).strip()
    human_side = int(params.get("human_side", 0))
    seed = int(params.get("seed", 0))
    max_turns_raw = params.get("max_turns", "")
    max_turns = _int_or_none(max_turns_raw) if max_turns_raw is not None else None
    map_spec = params.get("map")
    map_spec = str(map_spec).strip() if map_spec else None
    width_raw = params.get("width")
    height_raw = params.get("height")
    width = int(width_raw) if width_raw not in (None, "") else None
    height = int(height_raw) if height_raw not in (None, "") else None
    ckpt_raw = str(params.get("checkpoint", "") or "").strip()
    checkpoint = Path(ckpt_raw) if ckpt_raw else None
    if _SESSION is not None:
        try:
            _SESSION.close(complete=_SESSION.over)
        except Exception:
            pass
    llm_model = str(params.get("llm_model") or params.get("model") or "").strip() or None
    llm_reasoning = str(params.get("llm_reasoning") or params.get("reasoning") or "").strip() or None
    _SESSION = PlaySession(
        opponent=opponent,
        human_side=human_side,
        width=width,
        height=height,
        map_spec=map_spec,
        seed=seed,
        max_turns=max_turns,
        map_dir=_MAP_DIR,
        checkpoint=checkpoint,
        device=_DEVICE,
        llm_model=llm_model,
        llm_reasoning=llm_reasoning,
    )
    return _SESSION


def _start(form: dict) -> PlaySession:
    flat = {k: v[0] for k, v in form.items() if v}
    return _start_from_params(flat)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def _form(self) -> dict:
        raw = self._read_body().decode()
        return parse_qs(raw)

    def _json_body(self) -> dict:
        raw = self._read_body().decode()
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/health":
            self._send(200, b"ok", "text/plain")
            return
        if path == "/api/maps":
            self._send_json(200, {"maps": list_map_options()})
            return
        if path == "/api/state":
            with _LOCK:
                session = _SESSION
            if session is None:
                self._send_json(404, {"error": "No active game."})
                return
            with session.lock:
                self._send_json(200, session.to_state_dict())
            return
        if path == "/play":
            with _LOCK:
                session = _SESSION
            if session is None:
                self._redirect("/")
                return
            with session.lock:
                body = session.render_html().encode()
            self._send(200, body)
            return
        if path == "/" and _AUTO_PLAY and qs.get("new") != ["1"]:
            with _LOCK:
                session = _SESSION
            if session is not None:
                with session.lock:
                    body = session.render_html().encode()
                self._send(200, body)
                return
        self._send(200, _index())

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        ctype = self.headers.get("Content-Type", "")
        is_json = "application/json" in ctype

        if path == "/api/start":
            try:
                params = self._json_body() if is_json else {k: v[0] for k, v in self._form().items() if v}
                with _LOCK:
                    session = _start_from_params(params)
                    state = session.to_state_dict()
                self._send_json(200, state)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
            return

        if path == "/api/action":
            try:
                payload = self._json_body()
                with _LOCK:
                    session = _SESSION
                if session is None:
                    self._send_json(404, {"error": "No active game."})
                    return
                with session.lock:
                    session.apply_spec(payload)
                    state = session.to_state_dict()
                self._send_json(200, state)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
            return

        form = self._form()
        try:
            if path == "/start":
                with _LOCK:
                    _start(form)
                self._redirect("/play")
                return
            if path == "/act":
                with _LOCK:
                    session = _SESSION
                if session is None:
                    self._redirect("/")
                    return
                with session.lock:
                    index = int(form.get("index", ["-1"])[0])
                    session.apply(index)
                self._redirect("/play")
                return
        except Exception as exc:
            self._send(400, _index(str(exc)))
            return
        self._send(404, _page("<p>Not found.</p>"))


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    checkpoint_dir: Path = Path("runs"),
    device: str = "auto",
    *,
    initial_session: Optional[PlaySession] = None,
    auto_play: bool = False,
) -> None:
    global _CHECKPOINT_DIR, _DEVICE, _MAP_DIR, _SESSION, _AUTO_PLAY
    _CHECKPOINT_DIR = Path(checkpoint_dir)
    _DEVICE = device
    _MAP_DIR = _CHECKPOINT_DIR / "play_maps"
    _AUTO_PLAY = auto_play
    if initial_session is not None:
        _SESSION = initial_session
    httpd = ThreadingHTTPServer((host, port), Handler)
    actual_port = httpd.server_address[1]
    extra = ""
    if _SESSION is not None and getattr(_SESSION, "log_path", None):
        extra = f"  (game log {_SESSION.log_path})"
    print(f"Battleboats UI at http://{host}:{actual_port}  (checkpoints in {_CHECKPOINT_DIR}){extra}", flush=True)
    httpd.serve_forever()
