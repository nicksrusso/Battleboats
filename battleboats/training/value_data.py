"""Value-net data: harvest shards -> memmap position cache -> batches.

Harvest shards (``scripts/harvest_value.py``) are ``np.savez_compressed``
files, so they can't be memory-mapped and decompressing ~2,000 rows per draw
would dominate training. ``build_value_cache`` reads each shard ONCE, keeps a
seeded pool of ``k_pool`` positions per game, and writes:

    cache_dir/
      tokens.f16      raw float16 [T, TOKEN_DIM]   (np.memmap)
      rows.npz        row_splits int64 [R+1], globals f32 [R, 8], outcome,
                      returns, seat, turn, turn_frac, game (index into games)
      games.json      per-game: game_idx, seed, split, winner, reason, turns
      cache_meta.json build settings + counts

Split is by game (seed hash), so no game's positions land in two splits.
Each epoch ``ValueDataset`` draws ``k_per_game`` of a game's pooled rows, so
every game weighs the same no matter how long it ran.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from battleboats.envs.observation import (
    COASTLINE_TYPE_INDEX,
    GLOBALS_DIM,
    OWNER_OFFSET,
    PORT_STATE_OFFSET,
    PORT_TYPE_INDEX,
    SIGHTING_STATE_OFFSET,
    STATS_OFFSET,
    TOKEN_DIM,
)

SPLITS = ("train", "val", "test")
PHASE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)


# --------------------------------------------------------------------------- split


def split_for_seed(seed: int, val_frac: float = 0.05, test_frac: float = 0.05, salt: str = "bb-value-v0.5.0") -> str:
    """Deterministic game-level split from the game's seed (stable across runs)."""
    h = int(hashlib.sha1(f"{salt}:{int(seed)}".encode()).hexdigest()[:12], 16) / float(16**12)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


# --------------------------------------------------------------------------- shards


def list_harvest_games(harvest_dir) -> List[Dict]:
    """Completed games (progress.jsonl, else each shard's meta), sorted by game_idx."""
    harvest_dir = Path(harvest_dir)
    prog = harvest_dir / "progress.jsonl"
    games: List[Dict] = []
    if prog.exists():
        with prog.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line while a harvest runs
                    continue
                if rec.get("status", "ok") != "ok" or "path" not in rec:
                    continue
                if (harvest_dir / rec["path"]).exists():
                    games.append(rec)

    if not games:
        for p in sorted((harvest_dir / "games").glob("game_*.npz")):
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
            meta["path"] = str(p.relative_to(harvest_dir))
            games.append(meta)
    games.sort(key=lambda g: int(g["game_idx"]))
    return games


def position_weight(turn_frac, floor: float = 1.0, power: float = 1.0):
    """Per-position weight from game progress (turn / game length).

    ``w = floor + (1 - floor) * clip(turn_frac, 0, 1) ** power``.
    ``floor=1`` is uniform (backward compatible). ``floor=0.25, power=1`` is
    linear from 0.25 early to 1.0 at the end, so later positions count more
    while early ones are still seen. Accepts numpy arrays or torch tensors.
    """
    floor, power = float(floor), float(power)
    if isinstance(turn_frac, torch.Tensor):
        f = turn_frac.clamp(0, 1)
        return floor + (1.0 - floor) * torch.pow(f, power)
    f = np.clip(np.asarray(turn_frac, dtype=np.float64), 0.0, 1.0)
    return floor + (1.0 - floor) * np.power(f, power)


def _pool_rows(n_rows: int, k_pool: int, rng: np.random.Generator, weights=None) -> np.ndarray:
    if k_pool <= 0 or n_rows <= k_pool:
        return np.arange(n_rows)
    p = None
    if weights is not None:
        w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        s = w.sum()
        if s > 0 and np.isfinite(s):
            p = w / s
    return np.sort(rng.choice(n_rows, size=k_pool, replace=False, p=p))


def _extract_game(args) -> Optional[Dict]:
    path, game_rec, k_pool, base_seed, weight_floor, weight_power = args
    try:
        with np.load(path, allow_pickle=False) as z:
            rs = z["row_splits"].astype(np.int64)
            n = len(rs) - 1
            rng = np.random.default_rng([int(base_seed), int(game_rec["game_idx"])])
            turns = max(1, int(game_rec.get("turns", 1)))
            frac = np.clip(z["turn"].astype(np.float64) / turns, 0.0, 1.0)
            idx = _pool_rows(n, k_pool, rng, weights=position_weight(frac, weight_floor, weight_power))
            tok = z["tokens_data"]
            pieces = [tok[rs[i] : rs[i + 1]] for i in idx]
            lens = np.array([len(p) for p in pieces], dtype=np.int64)
            out = {
                "tokens": np.concatenate(pieces, axis=0).astype(np.float16) if len(pieces) else np.zeros((0, TOKEN_DIM), np.float16),
                "lens": lens,
                "globals": z["globals"][idx].astype(np.float32),
                "outcome": z["outcome"][idx].astype(np.float32),
                "returns": z["returns"][idx].astype(np.float32),
                "seat": z["seat"][idx].astype(np.int8),
                "turn": z["turn"][idx].astype(np.int16),
            }
    except Exception as exc:  # corrupt / partial shard: skip, report
        return {"error": f"{path}: {exc!r}"}
    return out


def build_value_cache(
    harvest_dir,
    cache_dir,
    k_pool: int = 64,
    seed: int = 0,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    workers: int = 2,
    max_games: Optional[int] = None,
    log_every: int = 500,
    weight_floor: float = 1.0,
    weight_power: float = 1.0,
) -> Dict:
    """Read every completed shard once and write the memmap cache. Returns cache_meta."""
    harvest_dir, cache_dir = Path(harvest_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    games = list_harvest_games(harvest_dir)
    if max_games is not None:
        games = games[: int(max_games)]
    jobs = [(str(harvest_dir / g["path"]), g, k_pool, seed, float(weight_floor), float(weight_power)) for g in games]

    tok_path = cache_dir / "tokens.f16.tmp"
    lens_all, glob_all, out_all, ret_all, seat_all, turn_all, frac_all, game_all = ([] for _ in range(8))
    game_recs: List[Dict] = []
    errors: List[str] = []
    n_tok = 0

    def _iter():
        if workers <= 1:
            for j in jobs:
                yield _extract_game(j)
        else:
            with get_context("spawn").Pool(workers) as pool:
                yield from pool.imap(_extract_game, jobs, chunksize=4)

    with tok_path.open("wb") as fh:
        for (path, g, *_), res in zip(jobs, _iter()):
            if res is None or "error" in res:
                errors.append(res["error"] if res else path)
                continue
            gi = len(game_recs)
            turns = max(1, int(g.get("turns", 1)))
            game_recs.append(
                {
                    "game_idx": int(g["game_idx"]),
                    "seed": int(g["seed"]),
                    "kind": g.get("kind"),
                    "split": split_for_seed(int(g["seed"]), val_frac, test_frac),
                    "winner": g.get("winner"),
                    "reason": g.get("reason"),
                    "turns": turns,
                    "num_decisions": int(g.get("num_decisions", 0)),
                    "path": g["path"],
                }
            )
            fh.write(res["tokens"].tobytes())
            n_tok += len(res["tokens"])
            lens_all.append(res["lens"])
            glob_all.append(res["globals"])
            out_all.append(res["outcome"])
            ret_all.append(res["returns"])
            seat_all.append(res["seat"])
            turn_all.append(res["turn"])
            frac_all.append(np.clip(res["turn"].astype(np.float32) / turns, 0.0, 1.0))
            game_all.append(np.full(len(res["lens"]), gi, dtype=np.int32))
            if log_every and len(game_recs) % log_every == 0:
                print(f"[cache] {len(game_recs)}/{len(jobs)} games, {n_tok:,} tokens", flush=True)

    os.replace(tok_path, cache_dir / "tokens.f16")
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    lens = cat(lens_all, np.int64)
    row_splits = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=row_splits[1:])
    np.savez(
        cache_dir / "rows.npz",
        row_splits=row_splits,
        globals=np.concatenate(glob_all).astype(np.float32) if glob_all else np.zeros((0, GLOBALS_DIM), np.float32),
        outcome=cat(out_all, np.float32),
        returns=cat(ret_all, np.float32),
        seat=cat(seat_all, np.int8),
        turn=cat(turn_all, np.int16),
        turn_frac=cat(frac_all, np.float32),
        game=cat(game_all, np.int32),
    )
    (cache_dir / "games.json").write_text(json.dumps(game_recs))
    split_counts = {s: sum(1 for g in game_recs if g["split"] == s) for s in SPLITS}
    meta = {
        "harvest_dir": str(harvest_dir),
        "k_pool": k_pool,
        "weight_floor": float(weight_floor),
        "weight_power": float(weight_power),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "games": len(game_recs),
        "rows": int(len(lens)),
        "tokens": int(n_tok),
        "token_dim": TOKEN_DIM,
        "split_games": split_counts,
        "decisive_games": sum(1 for g in game_recs if g["winner"] is not None),
        "errors": errors[:20],
        "n_errors": len(errors),
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- cache


class ValueCache:
    """Read-only view of a built cache (tokens are memory-mapped)."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "cache_meta.json").read_text())
        self.games: List[Dict] = json.loads((self.dir / "games.json").read_text())
        r = np.load(self.dir / "rows.npz")
        self.row_splits = r["row_splits"]
        self.globals = r["globals"]
        self.outcome = r["outcome"]
        self.returns = r["returns"]
        self.seat = r["seat"]
        self.turn = r["turn"]
        self.turn_frac = r["turn_frac"]
        self.game = r["game"]
        n_tok = int(self.row_splits[-1])
        self.tokens = (
            np.memmap(self.dir / "tokens.f16", dtype=np.float16, mode="r", shape=(n_tok, TOKEN_DIM))
            if n_tok
            else np.zeros((0, TOKEN_DIM), np.float16)
        )
        self.game_split = np.array([g["split"] for g in self.games])
        self.game_decisive = np.array([g["winner"] is not None for g in self.games], dtype=bool)

    def __len__(self) -> int:
        return len(self.outcome)

    def row_tokens(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[self.row_splits[i] : self.row_splits[i + 1]], dtype=np.float32)

    def split_rows(self, split: str, include_draws: bool = False) -> np.ndarray:
        ok = self.game_split[self.game] == split
        if not include_draws:
            ok &= self.game_decisive[self.game]
        return np.nonzero(ok)[0]

    def rows_by_game(self, rows: np.ndarray) -> Dict[int, np.ndarray]:
        g = self.game[rows]
        order = np.argsort(g, kind="stable")
        rows, g = rows[order], g[order]
        cuts = np.nonzero(np.diff(g))[0] + 1
        return {int(gg[0]): rr for gg, rr in zip(np.split(g, cuts), np.split(rows, cuts)) if len(rr)}


class ValueDataset(Dataset):
    """Rows of one split. ``k_per_game`` > 0: equal-weight draws per game per epoch
    (call ``set_epoch``); 0: every pooled row (use for eval)."""

    def __init__(self, cache: ValueCache, split: str, include_draws: bool = False, k_per_game: int = 0, seed: int = 0,
                 weight_floor: float = 1.0, weight_power: float = 1.0):
        self.cache = cache
        self.split = split
        self.k = int(k_per_game)
        self.seed = int(seed)
        self.weight_floor = float(weight_floor)
        self.weight_power = float(weight_power)
        self.all_rows = cache.split_rows(split, include_draws=include_draws)
        self._by_game = cache.rows_by_game(self.all_rows) if self.k > 0 else None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.k <= 0:
            self.rows = self.all_rows
            return
        rng = np.random.default_rng([self.seed, int(epoch)])
        picks = []
        for _, rr in sorted(self._by_game.items()):
            if len(rr) <= self.k:
                picks.append(rr)
            else:
                w = position_weight(self.cache.turn_frac[rr], self.weight_floor, self.weight_power)
                w = w / w.sum()
                picks.append(rng.choice(rr, size=self.k, replace=False, p=w))
        self.rows = np.sort(np.concatenate(picks)) if picks else np.zeros(0, np.int64)

    @property
    def n_games(self) -> int:
        return int(len(np.unique(self.cache.game[self.all_rows])))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        c = self.cache
        return (
            c.row_tokens(r),
            c.globals[r],
            float(c.outcome[r]),
            float(c.returns[r]),
            float(c.turn_frac[r]),
            int(c.seat[r]),
            r,
        )


def collate_value(batch):
    toks, glob, y, ret, frac, seat, rows = zip(*batch)
    B = len(toks)
    N = max(1, max(len(t) for t in toks))
    out = torch.zeros(B, N, TOKEN_DIM, dtype=torch.float32)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for b, t in enumerate(toks):
        if len(t):
            out[b, : len(t)] = torch.from_numpy(t)
            mask[b, : len(t)] = True
    return {
        "tokens": out,
        "pad_mask": mask,
        "globals": torch.as_tensor(np.stack(glob), dtype=torch.float32),
        "target": torch.tensor(y, dtype=torch.float32),
        "returns": torch.tensor(ret, dtype=torch.float32),
        "turn_frac": torch.tensor(frac, dtype=torch.float32),
        "seat": torch.tensor(seat, dtype=torch.long),
        "row": torch.tensor(rows, dtype=torch.long),
    }


# --------------------------------------------------------------------------- baseline features

FEATURE_NAMES_SIMPLE = ("strength_diff", "count_diff", "port_diff", "own_cash", "turn", "turn_x_strength_diff", "seat")


def _strength_col() -> int:
    return STATS_OFFSET + 1


def hand_features(tokens: np.ndarray, globals_vec: np.ndarray, seat: int) -> np.ndarray:
    """Fog-fair hand features from one observation (what the seat could see)."""
    t = tokens
    is_ship = t[:, :PORT_TYPE_INDEX].sum(axis=1) > 0.5
    is_port = t[:, PORT_TYPE_INDEX] > 0.5
    friendly = t[:, OWNER_OFFSET] > 0.5
    enemy = t[:, OWNER_OFFSET + 1] > 0.5
    fresh = t[:, SIGHTING_STATE_OFFSET] > 0.5
    strength = t[:, _strength_col()]
    own_ship = is_ship & friendly
    en_ship = is_ship & enemy
    own_type = t[own_ship, :PORT_TYPE_INDEX].sum(axis=0)
    en_type = t[en_ship & fresh, :PORT_TYPE_INDEX].sum(axis=0)
    own_str = strength[own_ship].sum()
    en_str_fresh = strength[en_ship & fresh].sum()
    en_str_any = strength[en_ship].sum()
    own_ports = float((is_port & friendly).sum())
    en_ports = float((is_port & enemy).sum())
    en_home_seen = float((is_port & enemy & (t[:, PORT_STATE_OFFSET + 1] > 0.5)).any())
    own_stock = t[is_port & friendly, PORT_STATE_OFFSET].sum()
    n_coast = float((t[:, COASTLINE_TYPE_INDEX] > 0.5).sum())
    return np.concatenate(
        [
            np.asarray(globals_vec, dtype=np.float32),
            own_type,
            en_type,
            np.array(
                [own_str, en_str_fresh, en_str_any, own_str - en_str_fresh, own_ports, en_ports, own_ports - en_ports,
                 en_home_seen, own_stock, n_coast / 100.0, float(seat)],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def simple_features(tokens: np.ndarray, globals_vec: np.ndarray, seat: int) -> np.ndarray:
    """Material/cash difference: the 'obvious' heuristic (task.md gate baseline 2).
    Uses the turn number, not turn fraction: game length is unknown mid-game."""
    f = hand_features(tokens, globals_vec, seat)
    g = GLOBALS_DIM + 2 * PORT_TYPE_INDEX
    own_cnt = f[GLOBALS_DIM : GLOBALS_DIM + PORT_TYPE_INDEX].sum()
    en_cnt = f[GLOBALS_DIM + PORT_TYPE_INDEX : g].sum()
    sd = f[g + 3]
    turn = float(globals_vec[5])
    return np.array([sd, own_cnt - en_cnt, f[g + 6], globals_vec[0], turn, turn * sd, float(seat)], dtype=np.float32)


def features_for_rows(cache: ValueCache, rows: np.ndarray, kind: str = "rich") -> np.ndarray:
    fn = hand_features if kind == "rich" else simple_features
    return np.stack([fn(cache.row_tokens(int(r)), cache.globals[r], int(cache.seat[r])) for r in rows]) if len(rows) else np.zeros((0, 1))


# --------------------------------------------------------------------------- metrics


def value_metrics(v: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> Dict[str, float]:
    """v: predicted value in [-1,1]; y: outcome in {-1,0,+1}. Win-prob p=(v+1)/2.
    Sign accuracy / Brier / log-loss are on decisive rows (y != 0)."""
    v = np.asarray(v, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out: Dict[str, float] = {"n": int(len(y))}
    if len(y) == 0:
        return out
    out["mse"] = float(np.mean((v - y) ** 2))
    d = y != 0
    out["n_decisive"] = int(d.sum())
    if d.any():
        p = np.clip((v[d] + 1) / 2, eps, 1 - eps)
        w = (y[d] > 0).astype(np.float64)
        out["sign_acc"] = float(np.mean((v[d] > 0) == (y[d] > 0)))
        out["brier"] = float(np.mean((p - w) ** 2))
        out["logloss"] = float(-np.mean(w * np.log(p) + (1 - w) * np.log(1 - p)))
    return out


def calibration_curve(v: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Dict]:
    """Reliability table on decisive rows: predicted win-prob bin vs observed win rate."""
    v = np.asarray(v, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = y != 0
    p = (v[d] + 1) / 2
    w = (y[d] > 0).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()), "win_rate": float(w[m].mean())})
    return out


def phase_metrics(v: np.ndarray, y: np.ndarray, frac: np.ndarray, edges: Sequence[float] = PHASE_EDGES) -> List[Dict]:
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (frac >= lo) & (frac < hi)
        r = value_metrics(v[m], y[m])
        r["phase"] = f"{lo:.1f}-{min(hi, 1.0):.1f}"
        res.append(r)
    return res
