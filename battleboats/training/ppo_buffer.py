"""PPO rollout buffer and StoredMasks.

Bridges ``collect_trajectory`` (a flat list of GAE'd ``Transition``s carrying
per-decision legality masks) and the PPO update (which re-scores those decisions
in batches via ``PolicyNetwork.evaluate_actions``).

``RolloutBuffer`` accumulates transitions across many games and serves them back
as shuffled minibatches for the K update epochs.

``StoredMasks`` is the PPO analog of ``BatchedMasks``: it satisfies the same
contract ``evaluate_actions`` calls (``.asset``, ``.verbs_for``, ``.target_for``)
but, rather than recomputing legality from engine state that no longer exists, it
replays the masks each decision was sampled under at rollout. This keeps the same
legality on both sides of the ratio ``exp(log pi_new - log pi_old)``; relaxing the
masks (as BC's ``BatchedMasks`` may) would silently corrupt the ratio.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from battleboats.training.policy import (
    NUM_DIRECTIONS,
    NUM_SHIP_TYPES,
    NUM_VERBS,
    VERB_TO_IDX,
    VERBS_WITH_TARGET,
)
from battleboats.training.rollout import Transition

# Per-verb target cardinality K. attack is the odd one out — it's a POINTER over
# the N entity tokens, so its K is the (padded) token count N_max of the batch, not
# a fixed constant. The fixed-width verbs are listed here; attack is handled by N.
TARGET_WIDTH = {
    "move": NUM_DIRECTIONS,
    "build_port": NUM_DIRECTIONS,
    "build_ship": NUM_SHIP_TYPES,
    # "attack": N_max  -> computed at collate time, NOT a constant
}


class StoredMasks:
    """Replay the rollout-time legality masks for a collated minibatch.

    Exposes the contract ``PolicyNetwork.evaluate_actions`` calls (``.asset``,
    ``verbs_for``, ``target_for``) over tensors that ``collate_transitions``
    already padded and moved to the update device. The ``asset_idx`` /
    ``verb_name`` arguments are ignored when choosing which mask to return (the
    mask for the chosen path was stored at rollout); they exist only to match
    the ``BatchedMasks`` / live ``ActionMasks`` signature.

    Args:
        asset_mask: ``(B, N)`` bool, legal assets per row (pad slots False).
        verb_mask: ``(B, NUM_VERBS)`` bool, legal verbs given each row's asset.
        target_masks: dict mapping each target-verb to its ``(B, K)`` bool mask,
            where ``K`` is ``N`` for "attack" else ``TARGET_WIDTH[verb]``. Only
            rows whose stored verb matches are meaningful; others are filler that
            ``evaluate_actions`` slices away.
    """

    def __init__(self, asset_mask, verb_mask, target_masks):
        self.asset = asset_mask
        self._verb_mask = verb_mask
        self._target_masks = target_masks
        self.B, self.N = asset_mask.shape

    def verbs_for(self, asset_idx):
        """Return the ``(B, NUM_VERBS)`` stored verb-legality mask."""
        return self._verb_mask

    def target_for(self, verb_name, asset_idx):
        """Return the ``(B, K)`` stored target-legality mask for ``verb_name``."""
        return self._target_masks[verb_name]


def collate_transitions(
    transitions: List[Transition],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], StoredMasks]:
    """Pad and stack ragged transitions into one batch plus its StoredMasks.

    Pads the pointer-head tensors (``tokens``, ``asset_mask``) to the batch's
    ``N_max``, derives ``pad_mask`` from each row's token count, stacks the
    fixed-width fields (verb mask, action indices, log-probs, value, advantage,
    return), and builds one ``(B, K)`` target mask per target-verb (attack width
    is ``N_max``; rows whose stored verb differs are harmless filler that
    ``evaluate_actions`` slices away). When ``device`` is given, every tensor is
    moved there once so the K-epoch update never re-moves per minibatch.

    Note that ``pad_mask`` (real tokens, for encoder pooling) is distinct from
    ``asset_mask`` (legal-to-act tokens, a subset); both are needed.

    Returns:
        ``(batch, masks)`` where ``batch`` is a dict of "tokens", "pad_mask",
        "asset_idx", "verb_idx", "target_idx", "logp_old", "value_old",
        "advantage", "ret", and ``masks`` is the matching ``StoredMasks``.
    """
    B = len(transitions)
    N_max = max([tr.tokens.shape[0] for tr in transitions])

    tokens = pad_sequence([tr.tokens for tr in transitions], batch_first=True)  # (B, N_max, TOKEN_DIM)
    asset_mask = pad_sequence([tr.asset_mask for tr in transitions], batch_first=True, padding_value=False)
    lengths = torch.tensor([tr.tokens.shape[0] for tr in transitions])  # (B,) each row's N_i
    pad_mask = torch.arange(N_max)[None, :] < lengths[:, None]  # (B, N_max) bool

    verb_mask = torch.stack([tr.verb_mask for tr in transitions])  # (B, NUM_VERBS) bool
    asset_idx = torch.tensor([tr.asset_idx for tr in transitions])  # (B,) long
    verb_idx = torch.tensor([tr.verb_idx for tr in transitions])  # (B,) long
    target_idx = torch.tensor([tr.target_idx for tr in transitions])  # (B,) long (-1 sentinels kept)
    logp_old = torch.tensor([tr.logp for tr in transitions])  # (B,) float
    value_old = torch.tensor([tr.value for tr in transitions])  # (B,) float
    advantage = torch.tensor([tr.advantage for tr in transitions])  # (B,) float
    ret = torch.tensor([tr.ret for tr in transitions])  # (B,) float

    target_masks = {}
    for verb_name in VERBS_WITH_TARGET:
        if verb_name == "attack":
            K = N_max
        else:
            K = TARGET_WIDTH[verb_name]
        m = torch.ones((B, K), dtype=bool)
        for i, tr in enumerate(transitions):
            if tr.verb_idx != VERB_TO_IDX[verb_name]:
                continue
            src = tr.target_mask
            if verb_name == "attack":
                m[i] = False
                m[i, : src.shape[0]] = src
            else:
                m[i] = src
        target_masks[verb_name] = m
    batch = {
        "tokens": tokens,
        "pad_mask": pad_mask,
        "asset_idx": asset_idx,
        "verb_idx": verb_idx,
        "target_idx": target_idx,
        "logp_old": logp_old,
        "value_old": value_old,
        "advantage": advantage,
        "ret": ret,
    }

    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
        asset_mask = asset_mask.to(device)
        verb_mask = verb_mask.to(device)
        target_masks = {k: v.to(device) for k, v in target_masks.items()}
    masks = StoredMasks(asset_mask, verb_mask, target_masks)
    return batch, masks


class RolloutBuffer:
    """Accumulate GAE'd transitions and serve them as shuffled minibatches.

    Deliberately thin: ``collate_transitions`` does the tensor work; this stores,
    shuffles, and slices. The caller must run ``compute_gae`` on each trajectory
    before ``add`` (GAE is per-trajectory and must not cross game boundaries);
    once added, transitions are flattened into one pool. Advantages are kept raw
    here and normalized in the PPO update.
    """

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, traj: List[Transition]):
        """Flatten an already-GAE'd trajectory into the pool."""
        self.transitions.extend(traj)

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        """Empty the buffer; PPO is on-policy, so data is stale after an update."""
        self.transitions.clear()

    def iter_minibatches(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], StoredMasks]]:
        """Yield ``(batch, masks)`` chunks; reshuffles on each call (one epoch)."""
        n = len(self.transitions)
        idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            chunk = [self.transitions[i] for i in sel.tolist()]
            yield collate_transitions(chunk, device=device)
