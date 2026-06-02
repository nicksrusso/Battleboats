"""HarvestDataset — PyTorch Dataset over the harvest JSONL.

Reads a harvest file produced by `scripts/harvest.py` and exposes
(phi_vector, target) pairs as torch tensors. Feature ordering is taken
from `FEATURE_KEYS` in heuristics.py, so the column layout always
matches the regression baseline — drop-in comparable.

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


class HarvestDataset(Dataset):
    """One row per (state, perspective). Returns (phi[F], target[]) tensors."""

    def __init__(
        self,
        jsonl_path: str | Path,
        game_idxs: Optional[Set[int]] = None,
        target_key: str = "target",
        load_tokens: bool = False,
    ):
        """target_key selects the regression label per row:
          - "target":          flat terminal outcome (±1), constant per game.
          - "mcts_root_value":  search's value estimate at that state, which
                                varies within a game and carries signal even
                                in the opening.

        load_tokens=False (default): __getitem__ returns the fixed `phi`
        vector — the SimpleMLP input. load_tokens=True: __getitem__ returns
        the variable-length `(N, TOKEN_DIM)` entity-token tensor — the
        transformer-encoder input. Rows whose `tokens` field is null/empty
        (e.g. harvested without emit_tokens) are skipped in token mode.

        Memory note: token mode holds one ragged float32 array per row in a
        Python list (variable N), which is far heavier than the phi matrix.
        Watch RAM on large harvests — see [[project-harvest-oom]].
        """
        path = Path(jsonl_path)
        phi_rows: list[list[float]] = []
        targets: list[float] = []
        game_ids: list[int] = []
        steps: list[int] = []
        token_rows: list[np.ndarray] = []

        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if game_idxs is not None and row["game_idx"] not in game_idxs:
                    continue
                # Some targets (notably mcts_root_value) are null on rows
                # where no value was recorded — e.g. the non-search
                # perspective. Skip those rather than poisoning the tensor
                # with NaN. Rows are dropped wholesale so phi/target/
                # game_id/step/tokens stay aligned.
                target_val = row[target_key]
                if target_val is None:
                    continue
                if load_tokens:
                    tok = row.get("tokens")
                    if not tok:  # None (no emit_tokens) or empty board -> unusable
                        continue
                    token_rows.append(np.asarray(tok, dtype=np.float32))
                phi = row["phi"]
                # Force canonical feature order; missing keys would
                # silently shift columns otherwise, which is the
                # nastiest possible data bug.
                phi_rows.append([phi[k] for k in FEATURE_KEYS])
                targets.append(target_val)
                game_ids.append(row["game_idx"])
                steps.append(row["step"])

        self.phi = np.asarray(phi_rows, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.game_ids = np.asarray(game_ids, dtype=np.int32)
        # Move index within each game — used to stratify metrics by game
        # phase (early/mid/late). Not returned by __getitem__.
        self.steps = np.asarray(steps, dtype=np.int32)
        self.feature_keys: Tuple[str, ...] = FEATURE_KEYS

        self.load_tokens = load_tokens
        # Ragged: list of (N_i, TOKEN_DIM) arrays. None in phi mode.
        self.tokens: Optional[list[np.ndarray]] = token_rows if load_tokens else None
        self.token_dim: Optional[int] = (
            token_rows[0].shape[1] if load_tokens and token_rows else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.load_tokens:
            # (N, TOKEN_DIM) — variable N; batched via collate_token_batch.
            return torch.from_numpy(self.tokens[idx]), target
        return torch.from_numpy(self.phi[idx]), target

    def unique_game_ids(self) -> np.ndarray:
        """Distinct game_idx values present after filtering — used to
        validate by-game splits don't leak between train and val."""
        return np.unique(self.game_ids)


def collate_token_batch(
    batch: list[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (tokens, target) samples into a rectangular batch.

    This is THE new mechanic versus the MLP. Each sample's `tokens` is
    (N_i, TOKEN_DIM) with a DIFFERENT N_i, so they can't just stack. We pad
    every sample up to the batch's max N with zero rows, and emit a boolean
    `pad_mask` marking which rows are real so the model can ignore the padding
    in both attention and pooling.

    Pass this as `collate_fn=collate_token_batch` to the DataLoader.

    Returns:
        tokens   (B, maxN, TOKEN_DIM) float32 — zero-padded
        pad_mask (B, maxN) bool                — True = real token, False = pad
        targets  (B,) float32

    Convention: pad_mask True = real. The model inverts it for torch's
    src_key_padding_mask (which wants True = ignore). Keep the polarity here
    consistent or everything downstream silently breaks.
    """
    # TODO: token_list, target_list = zip(*batch)
    # TODO: B = len(token_list)
    # TODO: maxN = max(t.shape[0] for t in token_list)
    # TODO: token_dim = token_list[0].shape[1]
    #
    # TODO: tokens = torch.zeros(B, maxN, token_dim, dtype=torch.float32)
    # TODO: pad_mask = torch.zeros(B, maxN, dtype=torch.bool)
    # TODO: for i, t in enumerate(token_list):
    # TODO:     n = t.shape[0]
    # TODO:     tokens[i, :n] = t           # rows [n:] stay zero (padding)
    # TODO:     pad_mask[i, :n] = True      # mark the n real rows
    #
    # TODO: targets = torch.stack(target_list)   # (B,)
    # TODO: return tokens, pad_mask, targets
    raise NotImplementedError("Fill in collate_token_batch — see TODOs.")
