"""Action ↔ factored-index adapter — the bridge between the engine's flat
`enumerate_legal()` and the policy network's (asset, verb, target) world.

The engine speaks `Action` dataclasses; the policy speaks factored integer indices
into the entity-token order (see `observation.build_entity_refs`). This adapter
translates both directions and produces the legality masks the heads consume:

  - factor(action)            -> (asset_idx, verb_idx, target_idx)   [BC labels]
  - to_action(a, v, t)        -> Action                              [PPO rollout]
  - .asset / .verbs_for / .target_for                                [head masks]

All masks are derived from ONE call to `enumerate_legal`, so the policy can only
ever pick something the engine agrees is legal. Built for a single state (B=1):
`.asset` etc. carry a leading batch axis of 1 so they plug straight into
`PolicyNetwork.act`. Batched masks for BC *training* are a separate, downstream
collation of many of these.

THREE CONVENTIONS (reversible — change here, not in the network):
  1. endturn's asset is the HOME PORT token. EndTurnAction has no actor; rather
     than add a sentinel token we attribute it to the always-present home port.
  2. build_ship's target is SHIP TYPE only. The Action also needs a spawn tile,
     but the policy has one target factor — so same-type-different-spawn actions
     collide on one triple and we keep the first (spawn isn't policy-controlled).
  3. capture / load / unload have NO target sub-head (target_idx = -1). If a ship
     is adjacent to several ports, those collide and we keep the first.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import torch

from battleboats.core.actions import (
    AttackAction,
    BuildPortAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MerchantLoadAction,
    MerchantUnloadAction,
    MoveAction,
)
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.observation import build_entity_refs, build_entity_tokens
from battleboats.training.policy import NUM_DIRECTIONS, NUM_SHIP_TYPES, NUM_VERBS, VERB_TO_IDX

# Direction order (dx, dy) — fixed; index IS the move/build_port target. Movement
# is one tile, 4-directional, so a destination is always one of these offsets.
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}

# ShipType enum order = build_ship target index.
SHIP_TYPE_ORDER = list(ShipType)
SHIP_TYPE_TO_IDX = {t: i for i, t in enumerate(SHIP_TYPE_ORDER)}

# Per-verb target width (K). attack is a pointer over ALL N entity tokens (mask
# restricts to legal enemies); the rest are categorical over fixed sets.
_NO_TARGET = -1  # sentinel target_idx for verbs with no sub-head


class ActionMasks:
    """Legality masks + Action↔index translation for one engine state / player."""

    def __init__(self, engine, player_id: int, device="cpu"):
        self.player_id = player_id
        # All tensors this object produces (eager `asset`/`pad_mask` AND the lazy
        # `tokens`/`verbs_for`/`target_for`) are built on this device, so the
        # rollout's net.act() — which pulls masks mid-forward — never mixes
        # cpu/cuda. Defaults to cpu so the harvest factor() hot path is unchanged.
        self.device = torch.device(device)
        self.refs = build_entity_refs(engine, player_id)
        self.n = len(self.refs)

        # token-index lookups by identity
        self._ship_tok = {}
        self._port_tok = {}
        self._enemy_tok = {}
        for i, (kind, ident) in enumerate(self.refs):
            if kind == "friendly_ship":
                self._ship_tok[ident] = i
            elif kind == "friendly_port":
                self._port_tok[ident] = i
            elif kind == "enemy_ship":
                self._enemy_tok[ident] = i

        # friendly-ship positions, for move/build_port direction math in factor()
        self._ship_pos = {s.id: s.position for s in engine.ships.values() if s.owner == player_id}
        self._home_port = engine.players[player_id].home_port

        # Lazy: the legal-action table, mask sets, and the token tensor are only
        # built on first access. The harvest's hot path uses factor() ONLY, which
        # needs just the lookups above — so labeling a 3M-step harvest never pays
        # for enumerate_legal a second time, builds no torch tensors, and stays
        # memory-lean. Rollout/PPO (act, to_action, masks) triggers the build.
        self._engine = engine
        self._built = False

    # ------------------------------------------------------------------ build
    def _ensure_built(self):
        """Derive masks + the (a,v,t)->Action table from enumerate_legal. Lazy +
        idempotent — only the mask/translation paths need it, not factor()."""
        if self._built:
            return
        self._assets = set()
        self._verbs = defaultdict(set)  # asset_idx -> {verb_idx}
        self._targets = defaultdict(set)  # (asset_idx, verb_idx) -> {target_idx}
        self._table = {}  # (a, v, t) -> Action  (first wins on collision)
        for action in self._engine.enumerate_legal(self.player_id):
            a, v, t = self.factor(action)
            self._assets.add(a)
            self._verbs[a].add(v)
            if t != _NO_TARGET:
                self._targets[(a, v)].add(t)
            self._table.setdefault((a, v, t), action)
        self._built = True

    @property
    def tokens(self) -> "torch.Tensor":
        """(1, N, TOKEN_DIM) — entity tokens for this state (lazy; rollout only)."""
        toks = build_entity_tokens(self._engine, self.player_id)
        return torch.from_numpy(toks).unsqueeze(0).to(self.device)

    @property
    def pad_mask(self) -> "torch.Tensor":
        """(1, N) bool — all real (a single live state never pads)."""
        return torch.ones(1, self.n, dtype=torch.bool, device=self.device)

    # -------------------------------------------------------------- translate
    def factor(self, action) -> Tuple[int, int, int]:
        """Action -> (asset_idx, verb_idx, target_idx). target_idx = -1 if none."""
        if isinstance(action, MoveAction):
            a = self._ship_tok[action.ship_id]
            t = self._direction_idx(self._ship_pos[action.ship_id], action.destination)
            return a, VERB_TO_IDX["move"], t
        if isinstance(action, AttackAction):
            return self._ship_tok[action.attacker_id], VERB_TO_IDX["attack"], self._enemy_tok[action.target_id]
        if isinstance(action, BuildShipAction):
            return self._port_tok[action.port], VERB_TO_IDX["build_ship"], SHIP_TYPE_TO_IDX[action.ship_type]
        if isinstance(action, BuildPortAction):
            a = self._ship_tok[action.builder_ship_id]
            t = self._direction_idx(self._ship_pos[action.builder_ship_id], action.target)
            return a, VERB_TO_IDX["build_port"], t
        if isinstance(action, CapturePortAction):
            return self._ship_tok[action.landing_ship_id], VERB_TO_IDX["capture"], _NO_TARGET
        if isinstance(action, MerchantLoadAction):
            return self._ship_tok[action.merchant_id], VERB_TO_IDX["load"], _NO_TARGET
        if isinstance(action, MerchantUnloadAction):
            return self._ship_tok[action.merchant_id], VERB_TO_IDX["unload"], _NO_TARGET
        if isinstance(action, EndTurnAction):
            return self._port_tok[self._home_port], VERB_TO_IDX["endturn"], _NO_TARGET
        raise TypeError(f"unknown action type: {type(action).__name__}")

    def to_action(self, asset_idx, verb_idx, target_idx=None):
        """(asset_idx, verb_idx, target_idx) -> the legal Action. Accepts ints or
        0-d/1-elem tensors. target_idx None or -1 means no target."""
        self._ensure_built()
        a, v = int(asset_idx), int(verb_idx)
        t = _NO_TARGET if target_idx is None else int(target_idx)
        return self._table[(a, v, t)]

    def _direction_idx(self, src, dst) -> int:
        return DIR_TO_IDX[(dst[0] - src[0], dst[1] - src[1])]

    # ------------------------------------------------------------------ masks
    @property
    def asset(self) -> torch.Tensor:
        """(1, N) bool — token indices that may be selected as the acting asset."""
        self._ensure_built()
        m = torch.zeros(1, self.n, dtype=torch.bool, device=self.device)
        for a in self._assets:
            m[0, a] = True
        return m

    def verbs_for(self, asset_idx) -> torch.Tensor:
        """(1, NUM_VERBS) bool — legal verbs given the chosen asset."""
        self._ensure_built()
        a = int(asset_idx)
        m = torch.zeros(1, NUM_VERBS, dtype=torch.bool, device=self.device)
        for v in self._verbs.get(a, ()):
            m[0, v] = True
        return m

    def target_for(self, verb_name: str, asset_idx) -> torch.Tensor:
        """(1, K) bool — legal targets for (asset, verb). K = N for attack (pointer
        over entity tokens), else the categorical width for that verb."""
        self._ensure_built()
        a = int(asset_idx)
        v = VERB_TO_IDX[verb_name]
        k = self.n if verb_name == "attack" else (NUM_SHIP_TYPES if verb_name == "build_ship" else NUM_DIRECTIONS)
        m = torch.zeros(1, k, dtype=torch.bool, device=self.device)
        for t in self._targets.get((a, v), ()):
            m[0, t] = True
        return m


class BatchedMasks:
    """Pad-aware, all-True-except-padding masks for a BATCH of stored BC states —
    the training counterpart to ActionMasks.

    BC didn't store per-state legality, so instead of real masks we mask ONLY
    padding. The expert only ever chose legal moves, so the labels live on legal
    options anyway; the policy learns legality implicitly. (PPO later enforces TRUE
    legality with live ActionMasks.) Satisfies the same contract
    PolicyNetwork.evaluate_actions calls: `.asset`, `.verbs_for`, `.target_for`.

    Built from a collate_bc_batch `pad_mask` (B, N), True = real token. Because the
    collate pads by appending, pad_mask IS the asset/attack legality mask — a
    pointer must never select a pad slot. Categorical factors (verb / direction /
    ship type) have no padding, so they're fully open (all-True).

    Keep every mask on pad_mask.device so it lines up with the (GPU) tokens.
    """

    def __init__(self, pad_mask: torch.Tensor):
        """pad_mask: (B, N) bool."""
        self.pad_mask = pad_mask
        self.B, self.N = pad_mask.shape
        self.device = pad_mask.device

    @property
    def asset(self) -> torch.Tensor:
        """(B, N) bool — any real token may act"""
        return self.pad_mask

    def verbs_for(self, asset_idx) -> torch.Tensor:
        """(B, NUM_VERBS) bool — verbs unconstrained in unmasked BC."""
        return torch.ones((self.B, NUM_VERBS), dtype=torch.bool, device=self.device)

    def target_for(self, verb_name: str, asset_idx) -> torch.Tensor:
        """(B, K) bool — legal targets for the batch under `verb_name`.
        All on pad_mask.device. evaluate_actions slices the returned (B, K) by the
        rows whose verb == verb_name, so it must cover the whole batch.
        """
        if verb_name == "attack":
            return self.pad_mask
        elif verb_name == "build_ship":
            return torch.ones((self.B, NUM_SHIP_TYPES), dtype=torch.bool, device=self.pad_mask.device)
        elif verb_name == "move" or verb_name == "build_port":
            return torch.ones((self.B, NUM_DIRECTIONS), dtype=torch.bool, device=self.pad_mask.device)

        raise ValueError(f"{verb_name} has no target sub-head")
