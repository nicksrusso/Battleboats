"""Stdlib HTTP UI. Bind to 127.0.0.1 and open the printed URL.

One game at a time. Restart the process to drop it.
"""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from battleboats.ui.session import PlaySession

_SESSION: Optional[PlaySession] = None
_CHECKPOINT_DIR = Path("runs")
_DEVICE = "auto"
_MAP_DIR = Path("runs/play_maps")


def list_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pt") if p.is_file())


def _page(body: str) -> bytes:
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Battleboats</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; background: #0f1419; color: #e7ecf1; max-width: 42rem; }}
label {{ display: block; margin: 0.7rem 0 0.2rem; }}
input, select {{ padding: 0.3rem; }}
button {{ margin-top: 1rem; background: #1d3557; color: white; border: 0; padding: 0.5rem 0.8rem; }}
.err {{ color: #ffb4a2; }}
</style></head><body>
<h1>Battleboats</h1>
{body}
</body></html>"""
    return doc.encode()


def _index(error: str = "") -> bytes:
    ckpts = list_checkpoints(_CHECKPOINT_DIR)
    options = "".join(
        f'<option value="{html.escape(str(p))}"></option>' for p in ckpts
    )
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    default_ckpt = str(ckpts[-1]) if ckpts else ""
    body = f"""{err}
<form method="post" action="/start">
<label>Opponent</label>
<label><input type="radio" name="opponent" value="scripted" checked> Scripted (full information)</label>
<label><input type="radio" name="opponent" value="random"> Random</label>
<label><input type="radio" name="opponent" value="checkpoint"> Trained checkpoint</label>
<label>Checkpoint path</label>
<input name="checkpoint" list="ckpts" value="{html.escape(default_ckpt)}" size="60">
<datalist id="ckpts">{options}</datalist>
<label>Your side</label>
<label><input type="radio" name="human_side" value="0" checked> Player 0 (first)</label>
<label><input type="radio" name="human_side" value="1"> Player 1</label>
<label>Map width <input name="width" type="number" value="6" min="4" max="32"></label>
<label>Map height <input name="height" type="number" value="6" min="4" max="32"></label>
<label>Max turns <input name="max_turns" type="number" value="80" min="5" max="500"></label>
<label>Seed <input name="seed" type="number" value="0"></label>
<button type="submit">Start game</button>
</form>
<p>Checkpoints are listed from <code>{html.escape(str(_CHECKPOINT_DIR))}</code>.
A checkpoint game needs a <code>.pt</code> file from the curriculum or self-play run.</p>
"""
    return _page(body)


def _start(form: dict) -> None:
    global _SESSION
    opponent = form.get("opponent", ["scripted"])[0]
    human_side = int(form.get("human_side", ["0"])[0])
    width = int(form.get("width", ["6"])[0])
    height = int(form.get("height", ["6"])[0])
    max_turns = int(form.get("max_turns", ["80"])[0])
    seed = int(form.get("seed", ["0"])[0])
    ckpt_raw = form.get("checkpoint", [""])[0].strip()
    checkpoint = Path(ckpt_raw) if ckpt_raw else None
    _SESSION = PlaySession(
        opponent=opponent,
        human_side=human_side,
        width=width,
        height=height,
        seed=seed,
        max_turns=max_turns,
        map_dir=_MAP_DIR,
        checkpoint=checkpoint,
        device=_DEVICE,
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _form(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode() if length else ""
        return parse_qs(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, b"ok", "text/plain")
            return
        if path == "/play":
            if _SESSION is None:
                self._redirect("/")
                return
            self._send(200, _SESSION.render_html().encode())
            return
        self._send(200, _index())

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        form = self._form()
        try:
            if path == "/start":
                _start(form)
                self._redirect("/play")
                return
            if path == "/act":
                if _SESSION is None:
                    self._redirect("/")
                    return
                index = int(form.get("index", ["-1"])[0])
                _SESSION.apply(index)
                self._redirect("/play")
                return
        except Exception as exc:  # show setup errors on the form instead of a stack trace
            self._send(400, _index(str(exc)))
            return
        self._send(404, _page("<p>Not found.</p>"))


def serve(host: str = "127.0.0.1", port: int = 8765, checkpoint_dir: Path = Path("runs"), device: str = "auto") -> None:
    global _CHECKPOINT_DIR, _DEVICE, _MAP_DIR
    _CHECKPOINT_DIR = Path(checkpoint_dir)
    _DEVICE = device
    _MAP_DIR = _CHECKPOINT_DIR / "play_maps"
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Battleboats UI at http://{host}:{port}  (checkpoints in {_CHECKPOINT_DIR})")
    httpd.serve_forever()
