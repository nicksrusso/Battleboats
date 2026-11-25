"""OpenRouter client for the Battleboats LLM play harness.

Reuses FORMAT 4 prompts and ``choose_move`` (retry once on a bad reply, then
a random legal fallback). API/network errors are retried once, then the
seat passes (EndTurn) if that is legal.

The API key is read from ``OPENROUTER_API_KEY`` or a repo/cwd ``.env`` file.
It is never logged, printed, or included in exceptions.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from battleboats.core.actions import Action, EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.llm.play import choose_move, message_text
from battleboats.llm.prompt import build_view, render_prompt

INKLING_MODEL_ID = "thinkingmachines/inkling"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REASONING = "max"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TIMEOUT = 600.0
DEFAULT_TEMPERATURE = 0.3
SUPPORTED_EFFORTS = ("max", "high", "medium", "low", "minimal", "none")
MODEL_ALIASES = {
    "inkling": INKLING_MODEL_ID,
    "inkling-max": INKLING_MODEL_ID,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KEY_ENV = "OPENROUTER_API_KEY"


def resolve_model_id(model: Optional[str], *, opponent: str = "llm") -> str:
    """Map aliases like ``inkling`` to the OpenRouter slug."""
    raw = (model or "").strip()
    if not raw:
        return INKLING_MODEL_ID
    return MODEL_ALIASES.get(raw.lower(), raw)


def resolve_reasoning(effort: Optional[str]) -> str:
    raw = (effort or DEFAULT_REASONING).strip().lower()
    if raw not in SUPPORTED_EFFORTS:
        raise ValueError(
            f"unknown reasoning effort {effort!r}; "
            f"expected one of {', '.join(SUPPORTED_EFFORTS)}"
        )
    return raw


def _env_search_paths() -> List[Path]:
    paths = [Path.cwd() / ".env", _REPO_ROOT / ".env"]
    # Deduplicate while preserving order.
    out: List[Path] = []
    seen = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_openrouter_api_key(
    *,
    environ: Optional[Dict[str, str]] = None,
    search_paths: Optional[List[Path]] = None,
) -> str:
    """Return the OpenRouter key, or raise if it is missing.

    Looks at ``OPENROUTER_API_KEY`` first, then ``.env`` in the current
    working directory and the repo root. The value is never logged.
    """
    env = os.environ if environ is None else environ
    key = str(env.get(_KEY_ENV, "") or "").strip()
    if key:
        return key
    for path in search_paths if search_paths is not None else _env_search_paths():
        if not path.is_file():
            continue
        found = _parse_dotenv(path).get(_KEY_ENV, "").strip()
        if found:
            return found
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Export it in the environment, or put "
        "OPENROUTER_API_KEY=... in a .env file in the repo root (never commit it)."
    )


def safe_error(exc: BaseException) -> str:
    """Short error text with no credentials or bearer tokens."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    for token in ("bearer ", "api_key", "authorization"):
        if token in lowered:
            return f"{type(exc).__name__}: OpenRouter request failed"
    return text[:300]


def usage_summary(usage: Optional[dict]) -> dict:
    """Pick token and cost fields out of an OpenRouter ``usage`` object."""
    if not usage:
        return {}
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    cost = usage.get("cost", usage.get("total_cost"))
    if isinstance(cost, dict):
        cost = cost.get("total") or cost.get("amount") or cost.get("usd")
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    out = {}
    if prompt is not None:
        out["prompt_tokens"] = int(prompt)
    if completion is not None:
        out["completion_tokens"] = int(completion)
    if reasoning is not None:
        out["reasoning_tokens"] = int(reasoning)
    if total is not None:
        out["total_tokens"] = int(total)
    if cost is not None and cost != "":
        try:
            out["cost_usd"] = float(cost)
        except (TypeError, ValueError):
            pass
    return out


def format_usage_line(summary: dict, latency_s: Optional[float] = None) -> str:
    parts: List[str] = []
    if latency_s is not None:
        parts.append(f"{float(latency_s):.1f}s")
    tokens = summary.get("total_tokens") or summary.get("completion_tokens")
    if tokens is not None:
        bit = f"{int(tokens):,} tok"
        reasoning = summary.get("reasoning_tokens")
        if reasoning:
            bit += f" ({int(reasoning):,} reasoning)"
        parts.append(bit)
    if "cost_usd" in summary:
        parts.append(f"${summary['cost_usd']:.4f}")
    return " · ".join(parts)


def openrouter_payload(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_effort == "none":
        payload["reasoning"] = {"effort": "none"}
    return payload


class OpenRouterClient:
    """POST /api/v1/chat/completions. One user message, no tools.

    Compatible with ``choose_move`` / ``CompleteFn``: ``client(prompt, temperature)``.
    """

    def __init__(
        self,
        *,
        model: str = INKLING_MODEL_ID,
        reasoning_effort: str = DEFAULT_REASONING,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model = resolve_model_id(model)
        self.reasoning_effort = resolve_reasoning(reasoning_effort)
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self._api_key = api_key if api_key is not None else load_openrouter_api_key()
        self._opener = opener or urllib.request.urlopen
        self.last_usage: Optional[dict] = None
        self.last_timings: Optional[dict] = None
        self.last_summary: dict = {}

    def __call__(self, prompt: str, temperature: float) -> str:
        payload = openrouter_payload(
            prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "X-Title": "Battleboats",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            # Do not include the request or headers (they carry the key).
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"OpenRouter unreachable: {reason}") from None
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenRouter returned a non-JSON body") from exc
        self.last_usage = body.get("usage")
        self.last_timings = body.get("timings")
        self.last_summary = usage_summary(self.last_usage)
        self.last_summary["latency_s"] = round(time.perf_counter() - started, 4)
        return message_text(body)


class _SafeComplete:
    """Turn transport errors into empty replies so ``choose_move`` can retry."""

    def __init__(self, inner: Callable[[str, float], str]) -> None:
        self.inner = inner
        self.errors: List[str] = []
        self.successes = 0
        self.last_usage: Optional[dict] = None
        self.last_summary: dict = {}

    def __call__(self, prompt: str, temperature: float) -> str:
        try:
            text = self.inner(prompt, temperature)
        except Exception as exc:
            self.errors.append(safe_error(exc))
            return ""
        self.successes += 1
        self.last_usage = getattr(self.inner, "last_usage", None)
        self.last_summary = dict(getattr(self.inner, "last_summary", {}) or usage_summary(self.last_usage))
        return text if isinstance(text, str) else str(text)


def decide_llm_action(
    engine: gameEngine,
    player_id: int,
    rng: random.Random,
    client: Callable[[str, float], str],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[Action, dict]:
    """One FORMAT 4 decision. Always returns a currently legal action."""
    legal = engine.enumerate_legal(player_id)
    if not legal:
        raise RuntimeError(f"decide_llm_action: no legal actions for player {player_id}")
    view = build_view(engine, player_id, list(legal))
    prompt = render_prompt(view)
    safe = _SafeComplete(client)
    picked = choose_move(safe, prompt, len(legal), rng, temperature)
    if picked["fallback"] and safe.errors and safe.successes == 0:
        end_idx = next(
            (i for i, action in enumerate(legal) if isinstance(action, EndTurnAction)),
            picked["choice"],
        )
        picked = dict(picked)
        picked["choice"] = end_idx
        picked["error"] = safe.errors[-1]
        picked["passed"] = True
    meta = dict(picked)
    meta["usage"] = safe.last_usage
    meta["usage_summary"] = safe.last_summary
    meta["errors"] = list(safe.errors)
    meta["prompt"] = prompt
    return legal[picked["choice"]], meta
