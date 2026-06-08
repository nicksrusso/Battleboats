"""HarvestDataset — PyTorch Dataset over harvest output.

Reads harvest data produced by `scripts/harvest.py` and exposes
(phi_vector | tokens, target) pairs as torch tensors. Feature ordering is
taken from `FEATURE_KEYS` in heuristics.py, so the column layout always
matches the regression baseline — drop-in comparable.

Accepts either form of harvest output:
  - a DIRECTORY of per-game shards (`game_<idx>.jsonl`, current format) —
    each shard ends with a `_type=game_footer` line carrying the winner, from
    which the flat ±1 target is derived at load time; shards outside the
    split are skipped without opening them.
  - a single .jsonl FILE (legacy format) — every row carries its own target.

Memory model: the file is loaded once into a contiguous float32 numpy
array (rows × features) at construction. For the ~660k-row harvests
this is well under 100 MB.

Optional `game_idxs` filter selects only rows whose `game_idx` is in
the given set — that's how a caller builds by-game train/val splits
without writing extra wiring here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from battleboats.agents.heuristics import FEATURE_KEYS


def _shard_game_idx(path: Path) -> Optional[int]:
    """Parse the game index out of a `game_<idx>.jsonl` shard filename, so a
    split can skip shards it doesn't reference without opening them."""
    stem = path.stem  # "game_<idx>"
    if stem.startswith("game_"):
        try:
            return int(stem[len("game_") :])
        except ValueError:
            return None
    return None


class HarvestDataset(Dataset):
    """One row per (state, perspective). Returns (phi[F], target[]) tensors."""

    def __init__(
        self,
        jsonl_path: str | Path,
        game_idxs: Optional[Set[int]] = None,
        target_key: str = "target",
        load_tokens: bool = False,
        bc: bool = False,
        index_path: str | Path | None = None,
    ):
        """bc=True switches this from a VALUE dataset to a BEHAVIOR-CLONING one:
        instead of a scalar regression target, each kept row carries the expert's
        `action` = [asset_idx, verb_idx, target_idx] (target_idx -1 = no target).
        Keep ONLY rows with a non-null `action` (the acting/expert perspective).
        bc implies token mode — pass load_tokens=True so the encoder input is
        present; the scalar target_key machinery is bypassed for these rows.

        target_key selects the regression label per row:
          - "target":          flat terminal outcome (±1), constant per game.
          - "mcts_root_value":  search's value estimate at that state, which
                                varies within a game and carries signal even
                                in the opening.

        load_tokens=False (default): __getitem__ returns the fixed `phi`
        vector — the SimpleMLP input. load_tokens=True: __getitem__ returns
        the variable-length `(N, TOKEN_DIM)` entity-token tensor — the
        transformer-encoder input. Rows whose `tokens` field is null/empty
        (e.g. harvested without emit_tokens) are skipped in token mode.

        Memory note: token mode is LAZY. Token arrays are NOT held in RAM — at
        construction we record only a (shard_path, byte_offset) locator per kept
        row and re-read that single line from disk in __getitem__. RAM stays flat
        regardless of split size (the eager version was ~150 MB/game in BC mode,
        i.e. tens of GB for a full split). The only eager structures are the small
        phi/target/action/id/step arrays. See [[project-harvest-oom]].

        index_path: optional path to a prebuilt index (.npz from
        scripts/build_token_index.py). When given and present, the per-line JSON
        scan is SKIPPED — the (shard, offset, label, game_id, step) arrays are
        loaded directly (seconds, vs ~40 min to rescan 140 GB of JSON). The index
        is built over the WHOLE harvest; game_idxs still filters it in memory here,
        so one index serves any split. Tokens are still read lazily from the
        original shards via _read_tokens. A mismatched harvest/mode fails loudly.
        """
        path = Path(jsonl_path)
        self._source = str(path)
        self.load_tokens = load_tokens
        self.bc = bc
        self._target_key = target_key
        self.feature_keys: Tuple[str, ...] = FEATURE_KEYS
        want_tokens = load_tokens or bc

        # Fast path: load a prebuilt index instead of rescanning the JSON.
        if index_path is not None and Path(index_path).exists():
            if not want_tokens:
                raise ValueError("index_path is only supported in token mode (load_tokens or bc).")
            self._load_index(Path(index_path), game_idxs)
            return
        phi_rows: list[list[float]] = []
        targets: list[float] = []
        game_ids: list[int] = []
        steps: list[int] = []
        actions: list[list[int]] = []  # bc mode: [asset_idx, verb_idx, target_idx] per row
        # Lazy token loading: instead of materializing every (N, TOKEN_DIM) array
        # (~150 MB/game in BC mode -> tens of GB for a full split), record only a
        # (shard_path, byte_offset) locator per kept row and re-read that one line
        # in __getitem__. token_dim is sniffed once from the first kept token row.
        token_index: list[Tuple[str, int]] = []
        token_dim: Optional[int] = None

        def _process_row(row: dict, winner: Optional[int], shard_str: str, offset: int) -> None:
            nonlocal token_dim
            gid = row["game_idx"]
            if game_idxs is not None and gid not in game_idxs:
                return
            if bc:
                act = row.get("action")
                if act is None:
                    return
                tok = row.get("tokens")
                if not tok:
                    return
                if token_dim is None:
                    token_dim = len(tok[0])  # sniff width without keeping the array
                token_index.append((shard_str, offset))
                actions.append(act)
                game_ids.append(gid)
                steps.append(row["step"])
                return
            if target_key == "target":
                # Legacy single-file rows carry the flat target directly.
                # Sharded rows don't (winner only known at game end) — derive
                # it from the footer's winner: +1 if this perspective won,
                # else -1. None for truncated games (no flat target).
                if row.get("target") is not None:
                    target_val = row["target"]
                elif winner is not None:
                    target_val = 1.0 if winner == row["perspective"] else -1.0
                else:
                    target_val = None
            else:
                # mcts_root_value etc. — null on the non-acting perspective.
                target_val = row.get(target_key)
            if target_val is None:
                return  # skip rather than poison the tensor with NaN
            if load_tokens:
                tok = row.get("tokens")
                if not tok:  # None (no emit_tokens) or empty board -> unusable
                    return
                if token_dim is None:
                    token_dim = len(tok[0])
                token_index.append((shard_str, offset))
            phi = row["phi"]
            # Force canonical feature order; missing keys would silently shift
            # columns otherwise — the nastiest possible data bug.
            phi_rows.append([phi[k] for k in FEATURE_KEYS])
            targets.append(target_val)
            game_ids.append(gid)
            steps.append(row["step"])

        def _iter_offsets(fh):
            """Yield (byte_offset, raw_line) via an explicit readline loop. A
            `for line in fh` loop read-ahead-buffers, which makes fh.tell()
            meaningless — but we need offsets valid for a later fh.seek()."""
            while True:
                off = fh.tell()
                line = fh.readline()
                if not line:
                    return
                yield off, line

        if path.is_dir():
            # Sharded harvest: one game_<idx>.jsonl per game + a
            # `_type=game_footer` line carrying the winner.
            # BC labels from `action`, never the winner, so it must NOT take the
            # buffer-whole-game path (that path holds a full game of parsed
            # JSON-with-tokens in RAM, spiking the allocator high-water mark for
            # nothing). Only the flat ±1 value target genuinely needs the footer.
            need_winner = target_key == "target" and not bc
            for shard in sorted(path.glob("game_*.jsonl")):
                # Skip shards outside the split without opening them.
                gid = _shard_game_idx(shard)
                if game_idxs is not None and gid is not None and gid not in game_idxs:
                    continue
                shard_str = str(shard)
                if need_winner:
                    # Footer is at end of file -> buffer this one game's (offset,
                    # obj) so targets can be assigned once the winner is known.
                    # Only metadata is buffered; tokens stay on disk.
                    buf, winner = [], None
                    with open(shard) as f:
                        for off, line in _iter_offsets(f):
                            obj = json.loads(line)
                            if obj.get("_type") == "game_footer":
                                winner = obj.get("winner")
                            else:
                                buf.append((off, obj))
                    for off, obj in buf:
                        _process_row(obj, winner, shard_str, off)
                else:
                    # No winner needed -> stream rows, skip footer, no buffering.
                    with open(shard) as f:
                        for off, line in _iter_offsets(f):
                            obj = json.loads(line)
                            if obj.get("_type") != "game_footer":
                                _process_row(obj, None, shard_str, off)
        else:
            # Legacy single-file harvest: every row carries its own target.
            shard_str = str(path)
            with open(path) as f:
                for off, line in _iter_offsets(f):
                    _process_row(json.loads(line), None, shard_str, off)

        self.phi = np.asarray(phi_rows, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.game_ids = np.asarray(game_ids, dtype=np.int32)
        # Move index within each game — used to stratify metrics by game
        # phase (early/mid/late). Not returned by __getitem__.
        self.steps = np.asarray(steps, dtype=np.int32)
        self.feature_keys: Tuple[str, ...] = FEATURE_KEYS

        self.load_tokens = load_tokens
        self.bc = bc
        # Per-row (shard_path, byte_offset) locators for lazy token reads. None in
        # phi-only mode; aligned row-for-row with actions (bc) / targets (value).
        self._token_index: Optional[list[Tuple[str, int]]] = token_index if want_tokens else None
        self.token_dim: Optional[int] = token_dim
        # Tokens are NOT materialized anymore — kept as an attribute (None) only for
        # backward-compat with callers that probed ds.tokens.
        self.tokens = None

        # BC mode: (M, 3) int64 expert action triples, aligned row-for-row with
        # _token_index. None in value mode. __len__/getitem branch on self.bc.
        self.actions: Optional[np.ndarray] = np.asarray(actions, dtype=np.int64) if bc else None

    def _read_tokens(self, idx: int) -> np.ndarray:
        """Lazily read one row's (N, TOKEN_DIM) token array from its shard: open,
        seek to the recorded offset, read+parse that single line. O(row), not
        O(dataset). Re-opens the file each call so it's fork-safe under DataLoader
        workers (no shared file descriptor / no shared seek position)."""
        shard_str, offset = self._token_index[idx]
        with open(shard_str) as f:
            f.seek(offset)
            row = json.loads(f.readline())
        return np.asarray(row["tokens"], dtype=np.float32)

    def save_index(self, out_path: str | Path) -> None:
        """Serialize the scan result (the slow part) to a compact .npz so future
        runs skip the JSON rescan. Stores per-row (shard, offset, label, game_id,
        step) + a stamp (source harvest + mode) validated on load. Token bytes are
        NOT copied — they stay in the original shards, read lazily by _read_tokens.
        Distinct shard paths are deduped (store unique list + int32 ids) so the
        file stays tiny even at millions of rows."""
        if self._token_index is None:
            raise ValueError("save_index requires token mode (load_tokens or bc).")
        offsets = np.fromiter((o for _, o in self._token_index), dtype=np.int64, count=len(self._token_index))
        uniq, shard_ids = np.unique(np.asarray([s for s, _ in self._token_index]), return_inverse=True)
        np.savez(
            out_path,
            shards=uniq,
            shard_ids=shard_ids.astype(np.int32),
            offsets=offsets,
            game_ids=self.game_ids,
            steps=self.steps,
            targets=self.targets,
            actions=self.actions if self.actions is not None else np.empty((0, 3), dtype=np.int64),
            token_dim=np.int64(self.token_dim if self.token_dim is not None else -1),
            source=np.array(self._source),
            bc=np.array(self.bc),
            load_tokens=np.array(self.load_tokens),
            target_key=np.array(self._target_key),
        )

    def _load_index(self, index_path: Path, game_idxs: Optional[Set[int]]) -> None:
        """Reconstruct the dataset from a prebuilt .npz (see save_index), skipping
        the JSON scan. The index covers the whole harvest; filter to game_idxs in
        memory here so one index serves any split. Fails loudly on a harvest/mode
        mismatch rather than silently seeking to garbage offsets."""
        z = np.load(index_path, allow_pickle=False)
        src, bc, lt, tk = str(z["source"]), bool(z["bc"]), bool(z["load_tokens"]), str(z["target_key"])
        if src != self._source:
            raise ValueError(f"index built for harvest {src!r}, but loading {self._source!r}")
        if (bc, lt, tk) != (self.bc, self.load_tokens, self._target_key):
            raise ValueError(
                f"index mode mismatch: index(bc={bc}, load_tokens={lt}, target_key={tk!r}) != "
                f"requested(bc={self.bc}, load_tokens={self.load_tokens}, target_key={self._target_key!r})"
            )
        shard_ids, offsets = z["shard_ids"], z["offsets"]
        game_ids, steps, targets, actions = z["game_ids"], z["steps"], z["targets"], z["actions"]
        if game_idxs is not None:
            keep = np.isin(game_ids, np.fromiter(game_idxs, dtype=game_ids.dtype, count=len(game_idxs)))
            shard_ids, offsets, game_ids, steps = shard_ids[keep], offsets[keep], game_ids[keep], steps[keep]
            targets = targets[keep] if targets.size else targets
            actions = actions[keep] if actions.size else actions
        self.game_ids, self.steps, self.targets = game_ids, steps, targets
        td = int(z["token_dim"])
        self.token_dim = td if td >= 0 else None
        self.actions = actions if self.bc else None
        self.phi = np.empty((0, len(FEATURE_KEYS)), dtype=np.float32)
        self.tokens = None
        shards = [str(s) for s in z["shards"]]
        self._token_index = [(shards[sid], int(off)) for sid, off in zip(shard_ids, offsets)]

    def __len__(self) -> int:
        return len(self.actions) if self.bc else len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bc:
            # BC sample: (tokens (N, TOKEN_DIM), action_triple (3,) long).
            # Pair with a collate that pads tokens + pad_mask and stacks the
            # triples into (B, 3) — see collate_bc_batch.
            return torch.from_numpy(self._read_tokens(idx)), torch.from_numpy(self.actions[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.load_tokens:
            # (N, TOKEN_DIM) — variable N; batched via collate_token_batch.
            return torch.from_numpy(self._read_tokens(idx)), target
        return torch.from_numpy(self.phi[idx]), target

    def unique_game_ids(self) -> np.ndarray:
        """Distinct game_idx values present after filtering — used to
        validate by-game splits don't leak between train and val."""
        return np.unique(self.game_ids)


def collate_token_batch(batch: list[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (tokens, target) samples into a rectangular batch.


    Returns:
        tokens   (B, maxN, TOKEN_DIM) float32 — zero-padded
        pad_mask (B, maxN) bool                — True = real token, False = pad
        targets  (B,) float32

    Convention: pad_mask True = real. The model inverts it for torch's
    src_key_padding_mask (which wants True = ignore). Keep the polarity here
    consistent or everything downstream silently breaks.
    """
    token_list, target_list = zip(*batch)
    B = len(token_list)
    maxN = max(t.shape[0] for t in token_list)
    token_dim = token_list[0].shape[1]

    tokens = torch.zeros(B, maxN, token_dim, dtype=torch.float32)
    pad_mask = torch.zeros(B, maxN, dtype=torch.bool)
    for i, t in enumerate(token_list):
        n = t.shape[0]
        tokens[i, :n] = t  # rows [n:] stay zero (padding)
        pad_mask[i, :n] = True  # mark the n real rows

    targets = torch.stack(target_list)  # (B,)
    return tokens, pad_mask, targets


def collate_bc_batch(batch: list[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (tokens, action_triple) BC samples into a rectangular batch.

    Returns:
        tokens   (B, maxN, TOKEN_DIM) float32 — zero-padded
        pad_mask (B, maxN) bool                — True = real token, False = pad
        actions  (B, 3) int64                  — [asset_idx, verb_idx, target_idx]

    The token padding is IDENTICAL to collate_token_batch (same maxN logic, same
    pad_mask polarity). One alignment invariant matters: pad ONLY by appending, so
    each board's real tokens keep indices 0..n-1 — otherwise the stored asset_idx /
    attack target_idx (which index into the original token order) would point at the
    wrong row. pad_mask therefore doubles as the asset/attack legality mask the
    BatchedMasks (#4) will feed evaluate_actions.

    TODO (fill the body):
      - token_list, action_list = zip(*batch)
      - build tokens + pad_mask exactly as collate_token_batch does
      - actions = torch.stack(action_list)   # (B, 3) long  (NOT a scalar)
      - return tokens, pad_mask, actions
    """
    token_list, action_list = zip(*batch)

    B = len(token_list)
    maxN = max(t.shape[0] for t in token_list)
    token_dim = token_list[0].shape[1]

    tokens = torch.zeros(B, maxN, token_dim, dtype=torch.float32)
    pad_mask = torch.zeros(B, maxN, dtype=torch.bool)
    for i, t in enumerate(token_list):
        n = t.shape[0]
        tokens[i, :n] = t  # rows [n:] stay zero (padding)
        pad_mask[i, :n] = True  # mark the n real rows

    actions = torch.stack(action_list)
    return tokens, pad_mask, actions
