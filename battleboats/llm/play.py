"""Play full games with a local OpenAI-compatible llama-server and log JSONL.

The server is not started here. Point ``--server-url`` at a running
``llama-server`` (default http://127.0.0.1:8080).
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from battleboats.core.actions import Action
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.llm.prompt import (FORMAT_VERSION, action_record, build_view,
                                    render_prompt)

CompleteFn = Callable[[str, float], str]

_LEADING_NUM = re.compile(r"\s*(\d+)\b")
_ANY_NUM = re.compile(r"(?<![\d-])(\d+)")
RETRY_NOTE = "Invalid reply. Answer with a MOVES number and one short sentence."

# Backstop so a seat that never ends the turn cannot loop forever.
# A normal turn is well under this many decisions.
_STEPS_PER_TURN = 2000


def default_map_path() -> str:
    path = Path(__file__).resolve().parents[1] / "core" / "config" / "map.json"
    if not path.is_file():
        from battleboats.envs.skill_maps import write_dev_map

        write_dev_map(path)
    return str(path)


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def chat_payload(prompt: str, temperature: float, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }


class LlamaClient:
    """POST /v1/chat/completions. One user message, no tools."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        model: str = "inkling",
        timeout: float = 180.0,
        max_tokens: int = 48,
    ) -> None:
        self.url = chat_completions_url(base_url)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_usage = None
        self.last_timings = None

    def __call__(self, prompt: str, temperature: float) -> str:
        payload = chat_payload(prompt, temperature, self.model, self.max_tokens)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-server unreachable at {self.url}: {exc.reason}"
            ) from exc
        self.last_usage = body.get("usage")
        self.last_timings = body.get("timings")
        return message_text(body)


def _field_text(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts).strip()
    return str(value).strip()


def message_text(body: dict) -> str:
    message = body["choices"][0]["message"]
    content = _field_text(message.get("content"))
    reasoning = _field_text(message.get("reasoning_content"))
    if reasoning and content:
        return f"[Start thinking]\n{reasoning}\n[End thinking]\n{content}"
    if content:
        return content
    if reasoning:
        return f"[Start thinking]\n{reasoning}\n[End thinking]\n"
    return ""


_THINK_BLOCKS = (
    re.compile(r"(?is)\[Start thinking\].*?\[End thinking\]"),
    re.compile(r"(?is)<think>.*?</think>"),
    re.compile(r"(?is)<\|content_thinking\|>.*?(?:<\|end_message\|>|<\|content_text\|>)"),
)


def _strip_thinking(text: str) -> str:
    visible = text
    for pat in _THINK_BLOCKS:
        visible = pat.sub("", visible)
    for start, end in (
        ("[Start thinking]", "[End thinking]"),
        ("<think>", "</think>"),
        ("<|content_thinking|>", "<|end_message|>"),
    ):
        idx = visible.find(start)
        if idx != -1 and end not in visible[idx:]:
            visible = visible[:idx]
    return visible.strip()


def parse_choice(text: str, n_legal: int) -> Optional[Tuple[int, str]]:
    """First usable move number, plus the rest of the reply as reasoning.

    A leading integer wins. Otherwise the first integer that is not part of
    a negative number. Out of range is a failure (caller retries).
    Numbers inside a thinking block are ignored.
    """
    if not text or n_legal <= 0:
        return None
    text = _strip_thinking(text)
    if not text:
        return None
    match = _LEADING_NUM.match(text)
    if match is None:
        match = _ANY_NUM.search(text)
    if match is None:
        return None
    idx = int(match.group(1))
    if idx < 0 or idx >= n_legal:
        return None
    reasoning = (text[: match.start(1)] + text[match.end(1) :]).strip()
    reasoning = re.sub(r"\s+", " ", reasoning).strip(" \t-:.")
    return idx, reasoning


def choose_move(
    complete: CompleteFn,
    prompt: str,
    n_legal: int,
    rng: random.Random,
    temperature: float,
) -> dict:
    """Ask once, retry once on a bad reply, then sample a legal index."""
    started = time.perf_counter()
    replies: List[str] = []
    retry_prompt = None
    for attempt in range(2):
        text_in = prompt if attempt == 0 else prompt + "\n" + RETRY_NOTE
        if attempt == 1:
            retry_prompt = text_in
        raw = complete(text_in, temperature)
        replies.append(raw if isinstance(raw, str) else str(raw))
        parsed = parse_choice(replies[-1], n_legal)
        if parsed is not None:
            idx, reasoning = parsed
            return {
                "choice": idx,
                "reasoning": reasoning,
                "fallback": False,
                "replies": replies,
                "retry_prompt": retry_prompt,
                "latency_s": round(time.perf_counter() - started, 4),
            }
    return {
        "choice": rng.randrange(n_legal),
        "reasoning": "",
        "fallback": True,
        "replies": replies,
        "retry_prompt": retry_prompt,
        "latency_s": round(time.perf_counter() - started, 4),
    }


def run_games(
    *,
    games: int,
    server_url: str = "http://127.0.0.1:8080",
    output_dir: str = "runs/llm_selfplay",
    seed: int = 0,
    max_turns: int = 200,
    temperature: float = 0.3,
    map_path: Optional[str] = None,
    horizon: int = 4,
    max_edges: int = 20,
    model: str = "inkling",
    timeout: float = 180.0,
    max_tokens: int = 48,
    complete: Optional[CompleteFn] = None,
) -> List[dict]:
    """Play ``games`` episodes. Seeds are ``seed + game_index``."""
    if games < 1:
        raise ValueError("games must be >= 1")
    client = complete or LlamaClient(
        server_url, model=model, timeout=timeout, max_tokens=max_tokens
    )
    map_json = map_path or default_map_path()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(games):
        game_seed = seed + index
        path = out / f"game_{index:04d}_seed_{game_seed}.jsonl"
        env = BattleboatsAEC(map_json_path=map_json, max_turns=max_turns)
        env.reset(seed=game_seed)
        result = play_env(
            env,
            client,
            path=path,
            seed=game_seed,
            game_index=index,
            temperature=temperature,
            horizon=horizon,
            max_edges=max_edges,
            map_path=map_json,
        )
        results.append(result)
        print(
            f"game {index} seed {game_seed} winner {result['winner']} "
            f"reason {result['reason']} steps {result['steps']} -> {path}",
            flush=True,
        )
    return results


def play_env(
    env: BattleboatsAEC,
    complete: CompleteFn,
    *,
    path: Path,
    seed: int,
    game_index: int,
    temperature: float,
    horizon: int,
    max_edges: int,
    map_path: str,
    max_steps: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> dict:
    """Drive one already-reset env. Both seats call ``complete``."""
    assert env.engine is not None
    rng = rng or random.Random(seed)
    step_budget = (
        max_steps if max_steps is not None else max(1, env.max_turns) * _STEPS_PER_TURN
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 0
    reason = "max_turns"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_game_start(env, seed, game_index, map_path)) + "\n")
        handle.flush()
        for agent in env.agent_iter():
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            if steps >= step_budget:
                reason = "step_budget"
                break
            assert env.engine is not None
            legal: Sequence[Action] = list(env.infos[agent]["legal_actions"])
            if not legal:
                reason = "no_legal"
                break
            player_id = env._player_id(agent)
            view = build_view(
                env.engine, player_id, list(legal), horizon=horizon, max_edges=max_edges
            )
            prompt = render_prompt(view)
            picked = choose_move(complete, prompt, len(legal), rng, temperature)
            choice = picked["choice"]
            action = legal[choice]
            record = {
                "type": "decision",
                "format_version": FORMAT_VERSION,
                "game_index": game_index,
                "seed": seed,
                "step": steps,
                "turn": env.engine.turn,
                "agent": agent,
                "state": view,
                "prompt": prompt,
                "raw_reply": picked["replies"][-1],
                "replies": picked["replies"],
                "choice": choice,
                "action": action_record(action),
                "reasoning": picked["reasoning"],
                "fallback": picked["fallback"],
                "latency_s": picked["latency_s"],
            }
            if getattr(complete, "last_usage", None) is not None:
                record["usage"] = complete.last_usage
            if getattr(complete, "last_timings", None) is not None:
                record["timings"] = complete.last_timings
            if picked["retry_prompt"] is not None:
                record["retry_prompt"] = picked["retry_prompt"]
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            env.step(action)
            steps += 1
        result = _result(env, seed, game_index, steps, reason)
        handle.write(json.dumps(result) + "\n")
        handle.flush()
    return result


def _game_start(env: BattleboatsAEC, seed: int, game_index: int, map_path: str) -> dict:
    assert env.engine is not None
    mp = env.engine.map
    land = [
        [x, y] for x in range(mp.width) for y in range(mp.height) if mp.is_land((x, y))
    ]
    return {
        "type": "game_start",
        "format_version": FORMAT_VERSION,
        "game_index": game_index,
        "seed": seed,
        "max_turns": env.max_turns,
        "map": map_path,
        "width": mp.width,
        "height": mp.height,
        "land": land,
    }


def _result(
    env: BattleboatsAEC, seed: int, game_index: int, steps: int, reason: str
) -> dict:
    assert env.engine is not None
    winner = env.engine.winner
    if winner is not None:
        reason = "capture"
    elif reason == "max_turns" and env.engine.turn < env.max_turns:
        reason = "stopped"
    return {
        "type": "result",
        "format_version": FORMAT_VERSION,
        "game_index": game_index,
        "seed": seed,
        "winner": winner,
        "winner_agent": None if winner is None else f"player_{winner}",
        "reason": reason,
        "turn": env.engine.turn,
        "steps": steps,
        "rewards": {
            agent: env._cumulative_rewards.get(agent, 0.0)
            for agent in env.possible_agents
        },
    }
