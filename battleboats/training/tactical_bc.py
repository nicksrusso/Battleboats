"""Behavior-cloning data and helpers for the tactical scripted agent.

Harvest rows are the acting player's *policy* observation (entity tokens
and globals, the same tensors ``ActionMasks`` / ``build_observation``
produce) plus the tactical action encoded as an ``(asset, verb, target)``
triple in the pointer policy's action space. Only legal, encodable
actions are kept; failures are counted for the harvest report.

Storage is packed per-game ``.npz`` (float16 tokens) so a few thousand
games stay a few gigabytes, not tens. Returns are discounted terminal
outcomes (win +1, loss -1, draw 0) with ``gamma=0.99`` per decision, for
an optional value-head loss so a BC checkpoint can warm-start PPO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from battleboats.core.actions import Action
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.observation import (
    GLOBALS_DIM,
    TOKEN_DIM,
    build_entity_tokens,
    build_globals,
)
from battleboats.training.policy import VERB_NAMES, VERB_TO_IDX

GAMMA_DEFAULT = 0.99
# Grouped report names for agreement / harvest tallies.
VERB_GROUP = {
    "build_ship": "buy",
    "move": "move",
    "attack": "attack",
    "capture": "capture",
    "build_port": "build",
    "load": "load",
    "unload": "unload",
    "endturn": "end_turn",
}
V0_3_0_ARCH = {
    "token_dim": TOKEN_DIM,
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "skill_count": 0,
    "globals_dim": GLOBALS_DIM,
}


@dataclass
class EncodeResult:
    ok: bool
    action_triple: Optional[List[int]] = None  # [asset, verb, target]
    tokens: Optional[np.ndarray] = None
    globals: Optional[np.ndarray] = None
    verb: Optional[str] = None
    error: Optional[str] = None


def _actions_equivalent(a: Action, b: Action) -> bool:
    """True if two factored-legal actions are the same decision.

    Frozen dataclasses compare field-for-field. BuildShip spawn tiles are
    canonicalized by ``enumerate_legal``, so equality holds when the
    policy table returns the same port and type.
    """
    return a == b


def encode_tactical_action(engine, player_id: int, action: Action) -> EncodeResult:
    """Encode one tactical action in the pointer policy space.

    The observation is built from the same functions the PPO policy uses
    (``build_entity_tokens`` / ``build_globals``). The action is factored
    with ``ActionMasks.factor`` and checked against the legal (a, v, t)
    table derived from ``enumerate_legal``.
    """
    legal = engine.enumerate_legal(player_id)
    if action not in legal:
        return EncodeResult(ok=False, error="not_in_legal_set")
    masks = ActionMasks(engine, player_id)
    try:
        asset_idx, verb_idx, target_idx = masks.factor(action)
    except Exception as exc:  # KeyError when a referent is missing from tokens
        return EncodeResult(ok=False, error=f"unencodable:{type(exc).__name__}:{exc}")
    masks._ensure_built()
    key = (int(asset_idx), int(verb_idx), int(target_idx))
    if key not in masks._table:
        return EncodeResult(ok=False, error="not_in_policy_mask")
    try:
        roundtrip = masks.to_action(*key)
    except Exception as exc:
        return EncodeResult(ok=False, error=f"to_action:{type(exc).__name__}:{exc}")
    if not _actions_equivalent(roundtrip, action):
        return EncodeResult(ok=False, error="roundtrip_mismatch")
    tokens = build_entity_tokens(engine, player_id)
    globals_vec = build_globals(engine, player_id)
    verb = VERB_NAMES[int(verb_idx)]
    return EncodeResult(
        ok=True,
        action_triple=[int(asset_idx), int(verb_idx), int(target_idx)],
        tokens=tokens,
        globals=globals_vec,
        verb=verb,
    )


def discounted_returns(
    n: int, outcome: float, gamma: float = GAMMA_DEFAULT
) -> np.ndarray:
    """``G_t = outcome * gamma**(n-1-t)`` for ``t = 0..n-1`` (per decision)."""
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    expo = np.arange(n - 1, -1, -1, dtype=np.float32)
    return np.float32(outcome) * (np.float32(gamma) ** expo)


def pack_game_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pack variable-length token rows into a compact npz-friendly dict."""
    n = len(rows)
    if n == 0:
        return {
            "tokens_data": np.zeros((0, TOKEN_DIM), dtype=np.float16),
            "row_splits": np.zeros((1,), dtype=np.int32),
            "actions": np.zeros((0, 3), dtype=np.int16),
            "globals": np.zeros((0, GLOBALS_DIM), dtype=np.float32),
            "returns": np.zeros((0,), dtype=np.float32),
            "verbs": np.zeros((0,), dtype=np.int8),
        }
    lengths = [int(r["tokens"].shape[0]) for r in rows]
    splits = np.zeros(n + 1, dtype=np.int32)
    splits[1:] = np.cumsum(lengths, dtype=np.int32)
    data = np.zeros((int(splits[-1]), TOKEN_DIM), dtype=np.float16)
    actions = np.zeros((n, 3), dtype=np.int16)
    globals_arr = np.zeros((n, GLOBALS_DIM), dtype=np.float32)
    returns = np.zeros((n,), dtype=np.float32)
    verbs = np.zeros((n,), dtype=np.int8)
    for i, row in enumerate(rows):
        sl = slice(int(splits[i]), int(splits[i + 1]))
        data[sl] = np.asarray(row["tokens"], dtype=np.float16)
        actions[i] = np.asarray(row["action"], dtype=np.int16)
        globals_arr[i] = np.asarray(row["globals"], dtype=np.float32)
        returns[i] = float(row["return"])
        verbs[i] = int(VERB_TO_IDX[row["verb"]])
    return {
        "tokens_data": data,
        "row_splits": splits,
        "actions": actions,
        "globals": globals_arr,
        "returns": returns,
        "verbs": verbs,
    }


def write_game_npz(
    path: Path, rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_game_rows(rows)
    payload = {**packed, "meta_json": np.asarray(json.dumps(meta))}
    np.savez_compressed(path, **payload)


def load_game_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"]))
    return arrays, meta


# Verbs kept in full when downsampling a train set (rare / high-signal).
KEEP_ALL_VERBS = {
    "build_ship",
    "build_port",
    "capture",
    "attack",
    "load",
    "unload",
}


class TacticalBCDataset(Dataset):
    """Packed harvest. Returns (tokens, globals, action, return).

    ``preload=True`` (default) loads every selected game's arrays into
    memory once so a shuffled DataLoader does not re-read an npz per row.
    """

    def __init__(
        self,
        harvest_dir: str | Path,
        game_idxs: Optional[Set[int]] = None,
        preload: bool = True,
    ):
        self.root = Path(harvest_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json in {self.root}")
        self.manifest = json.loads(manifest_path.read_text())
        index: List[Tuple[int, int, Path]] = []  # (game_idx, row_in_game, path)
        for rec in self.manifest["games"]:
            gid = int(rec["game_idx"])
            if game_idxs is not None and gid not in game_idxs:
                continue
            path = self.root / rec["path"]
            if not path.is_file():
                continue
            n = int(rec["n_labels"])
            for row in range(n):
                index.append((gid, row, path))
        self._index = index
        self._store: Dict[Path, Dict[str, np.ndarray]] = {}
        self._cache_path: Optional[Path] = None
        self._cache_arrays: Optional[Dict[str, np.ndarray]] = None
        if preload:
            self.preload()

    def preload(self) -> None:
        for path in {p for _, _, p in self._index}:
            if path not in self._store:
                arrays, _meta = load_game_npz(path)
                self._store[path] = arrays

    def __len__(self) -> int:
        return len(self._index)

    def unique_game_ids(self) -> List[int]:
        return sorted({gid for gid, _, _ in self._index})

    def verb_name(self, idx: int) -> str:
        _gid, row, path = self._index[idx]
        arr = self._arrays(path)
        return VERB_NAMES[int(arr["verbs"][row])]

    def _arrays(self, path: Path) -> Dict[str, np.ndarray]:
        if path in self._store:
            return self._store[path]
        if self._cache_path != path:
            arrays, _meta = load_game_npz(path)
            self._cache_path = path
            self._cache_arrays = arrays
        assert self._cache_arrays is not None
        return self._cache_arrays

    def __getitem__(self, idx: int):
        _gid, row, path = self._index[idx]
        arr = self._arrays(path)
        lo = int(arr["row_splits"][row])
        hi = int(arr["row_splits"][row + 1])
        tokens = np.asarray(arr["tokens_data"][lo:hi], dtype=np.float32)
        globals_vec = np.asarray(arr["globals"][row], dtype=np.float32)
        action = np.asarray(arr["actions"][row], dtype=np.int64)
        ret = np.float32(arr["returns"][row])
        return (
            torch.from_numpy(tokens),
            torch.from_numpy(globals_vec),
            torch.from_numpy(action),
            torch.tensor(ret, dtype=torch.float32),
        )


def collate_tactical_bc(batch):
    """Pad tokens; stack globals, actions, returns."""
    tokens_list, globals_list, action_list, return_list = zip(*batch)
    B = len(tokens_list)
    max_n = max(t.shape[0] for t in tokens_list)
    token_dim = tokens_list[0].shape[1] if max_n else TOKEN_DIM
    tokens = torch.zeros(B, max_n, token_dim, dtype=torch.float32)
    pad_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, t in enumerate(tokens_list):
        n = t.shape[0]
        if n:
            tokens[i, :n] = t
            pad_mask[i, :n] = True
    globals_t = torch.stack(globals_list, dim=0)
    actions = torch.stack(action_list, dim=0)
    returns = torch.stack(return_list, dim=0)
    return tokens, pad_mask, globals_t, actions, returns


def stratified_row_indices(
    dataset: TacticalBCDataset,
    max_rows: Optional[int],
    seed: int,
    keep_all: Optional[Set[str]] = None,
) -> List[int]:
    """Keep all rare verbs; downsample move/endturn to a row budget."""
    n = len(dataset)
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return list(range(n))
    keep_all = set(keep_all) if keep_all is not None else set(KEEP_ALL_VERBS)
    by_verb: Dict[str, List[int]] = {}
    for i in range(n):
        by_verb.setdefault(dataset.verb_name(i), []).append(i)
    rng = np.random.RandomState(seed)
    reserved: List[int] = []
    rest: Dict[str, List[int]] = {}
    for verb, idxs in by_verb.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        if verb in keep_all:
            reserved.extend(shuffled)
        else:
            rest[verb] = shuffled
    if len(reserved) >= max_rows:
        # Still include a floor of move/endturn so those heads stay trained.
        floor = min(8192, max(512, max_rows // 8))
        extra: List[int] = []
        if rest:
            total = sum(len(v) for v in rest.values())
            used = 0
            items = list(rest.items())
            for i, (verb, idxs) in enumerate(items):
                take = (
                    floor - used
                    if i == len(items) - 1
                    else int(round(floor * len(idxs) / total))
                )
                take = max(0, min(len(idxs), take))
                extra.extend(idxs[:take])
                used += take
        return reserved + extra
    budget = max_rows - len(reserved)
    chosen = list(reserved)
    if not rest or budget <= 0:
        return chosen
    total = sum(len(v) for v in rest.values())
    used = 0
    items = list(rest.items())
    for i, (verb, idxs) in enumerate(items):
        take = (
            budget - used
            if i == len(items) - 1
            else int(round(budget * len(idxs) / total))
        )
        take = max(0, min(len(idxs), take))
        chosen.extend(idxs[:take])
        used += take
    return chosen


def write_split(
    harvest_dir: Path, seed: int = 0, val_frac: float = 0.1
) -> Dict[str, Any]:
    harvest_dir = Path(harvest_dir)
    manifest = json.loads((harvest_dir / "manifest.json").read_text())
    ids = [int(g["game_idx"]) for g in manifest["games"] if int(g["n_labels"]) > 0]
    ids.sort()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if ids else 0
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    payload = {
        "seed": seed,
        "val_frac": val_frac,
        "train": train,
        "val": val,
        "n_train": len(train),
        "n_val": len(val),
    }
    path = harvest_dir / "split.json"
    path.write_text(json.dumps(payload, indent=2))
    return payload
