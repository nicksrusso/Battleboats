"""Policy network — encoder + autoregressive (asset, verb, target) heads + value.

The action-producing successor to `TransformerValueModel` (value-only). Same
entity-token encoder (`policy_architecture.md` §2), but instead of pooling
straight to a scalar we keep the PER-ENTITY embeddings and feed three
autoregressive heads (§3 Head A, §4 Head V, §5 Head T) plus a non-autoregressive
value head (§6).

Why this shape (the RL intuition, not just the wiring):
  - PPO needs, for every action it took, the policy's log-prob of that action and
    a state value V(s). The heads produce the log-prob; the value head the baseline.
  - The action is a FACTORED triple (asset, verb, target) with prefix-dependent
    legality. We decompose the joint as
        log pi(a,v,t | s) = log pi(a|s) + log pi(v|s,a) + log pi(t|s,a,v)
    (§7). Each head sees the SAMPLED upstream choices (prefix-feeding), not the
    logits — that is what makes the factorization correct.
  - Pointer head for asset/attack because their candidate count is variable
    (ships come and go); categorical for verb/move/ship-type because those domains
    are fixed and small. See §9.

TWO CALL PATHS — keep them straight, they share the heads but differ in control flow:
  - `act(...)`   — rollout/inference. STRICTLY SEQUENTIAL: sample asset, feed its
                   embedding to the verb head, sample verb, route to a target
                   sub-head, sample target. Three dependent forward passes. Returns
                   the sampled triple + joint log-prob + value, for the rollout buffer.
  - `evaluate_actions(...)` — PPO update. The triple was ALREADY sampled and stored;
                   re-run all heads in PARALLEL on the stored prefix to get the
                   CURRENT policy's log-prob + entropy + value of those same actions.
                   No sampling here. This is the ratio numerator in PPO's clip.

MASKING DISCIPLINE (the silent-bug minefield):
  - Mask BEFORE softmax by setting illegal logits to -inf (§ "mask before softmax").
    Illegal actions then get exactly zero probability AND zero gradient.
  - Masks come from the ENGINE/ENV (legal-action queries), passed in here — this
    module does NOT know the rules. Contract for each mask is in the docstrings.
  - Fog of war is enforced UPSTREAM: enemies with no sighting are not even tokens,
    so the attack pointer physically cannot point at them (§5a).
  - At least one legal option must exist per active head, or softmax over all -inf
    gives NaNs. `endturn` is the always-legal escape hatch (see VERB convention).

STATUS: implemented. Smoke-tested — act() (both target and no-target verbs) and
evaluate_actions() (mixed-verb batch + backward) run end-to-end against a fake
masks object. STILL TODO before training: (1) the real `masks` object, built by
the env from engine legal-action queries (contract below), and (2) a numerical
unit test of the joint log-prob against a hand-computed example (§7 warns the
factor sum is easy to mis-add — the smoke test only checks shape/finiteness/grad).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical

# --- Action factor vocabularies (mirror core/actions.py + ship_type.py) -------
# Verb order is load-bearing: it IS the verb_idx used everywhere. Keep in sync
# with the 8 Action dataclasses in core/actions.py.
VERB_NAMES = [
    "move",  # MoveAction          -> target: direction (categorical)
    "attack",  # AttackAction        -> target: enemy entity (pointer)
    "build_ship",  # BuildShipAction     -> target: ship type (categorical)
    "build_port",  # BuildPortAction     -> target: direction (categorical)
    "capture",  # CapturePortAction   -> target: implicit (adjacent port)
    "load",  # MerchantLoadAction  -> target: implicit
    "unload",  # MerchantUnloadAction-> target: implicit
    "endturn",  # EndTurnAction       -> target: none; always-legal escape hatch
]
VERB_TO_IDX = {name: i for i, name in enumerate(VERB_NAMES)}
NUM_VERBS = len(VERB_NAMES)

# Verbs whose target needs a sub-head. The rest (capture/load/unload/endturn) have
# an implicit-or-no target, so Head T is SKIPPED and contributes log pi(t|...) = 0.
VERBS_WITH_TARGET = ("attack", "move", "build_ship", "build_port")

NUM_DIRECTIONS = 4  # N/E/S/W — Manhattan movement is 4-directional (move + build_port)
NUM_SHIP_TYPES = 8  # ShipType enum cardinality (build_ship target)


# =============================================================================
# Reusable head primitives
# =============================================================================
class PointerHead(nn.Module):
    """Variable-cardinality selector: score each candidate embedding by dot
    product with a learned query, mask, softmax. Used by Head A (assets) and the
    attack target sub-head (enemies). Output width = number of candidates = adapts
    to the board, no fixed cap (§3).

    cond_dim is the width of the conditioning vector (Head A: d_model; attack
    sub-head: context + e_asset + e_verb, so 3*d_model).
    """

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        # Project the conditioning vector down to a query in the embedding space,
        # so score_i = q . e_i is a dot product in R^d_model.
        self.query_mlp = nn.Sequential(nn.Linear(cond_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, cond: torch.Tensor, candidate_emb: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """cond: (B, cond_dim)   candidate_emb: (B, N, d_model)   candidate_mask: (B, N) bool.
        Returns masked logits (B, N) — one score per candidate entity.

        - Use `self.query_mlp(cond)` to produce a query q of shape (B, d_model), so
          that the head can phrase a different "what entity do I want?" question per
          board while its weights stay fixed.
        - Use `torch.einsum('bd,bnd->bn', q, candidate_emb)` (or bmm with
          q.unsqueeze(1)) to dot the query against every entity, so that output
          element i becomes the compatibility score q · eᵢ. This is the step whose
          WIDTH equals N — it adapts to the board with no fixed output layer.
        - Use `scores.masked_fill(~candidate_mask, float('-inf'))` so that illegal
          entities get exactly zero probability AND zero gradient once softmaxed
          (masking BEFORE the softmax is what makes both true).

        Return the raw masked logits, NOT a softmax — hand them to
        `masked_categorical` so ONE distribution is reused for sample / log_prob /
        entropy (single source of truth → no train-vs-rollout drift).
        """
        q = self.query_mlp(cond)  # (B, d_model) — the "what am I looking for?" query
        scores = torch.einsum("bd,bnd->bn", q, candidate_emb)  # (B, N) dot per entity
        return scores.masked_fill(~candidate_mask, float("-inf"))


class CategoricalHead(nn.Module):
    """Fixed-width selector over a small static domain (verbs, directions, ship
    types). MLP from the conditioning vector to `num_options` logits, masked (§4).
    """

    def __init__(self, cond_dim: int, num_options: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_options))

    def forward(self, cond: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """cond: (B, cond_dim)   legal_mask: (B, num_options) bool.
        Returns masked logits (B, num_options).

        - Use `self.mlp(cond)` to map the conditioning vector straight to one logit
          per fixed option (verb / direction / ship type), so that — unlike the
          pointer head — the output width is FIXED and learned. That's correct here
          because this domain (e.g. the 8 verbs) never changes size board to board.
        - Use `logits.masked_fill(~legal_mask, float('-inf'))` so that illegal
          options drop out, same discipline as the pointer head.

        Return raw masked logits; let `masked_categorical` build the distribution.
        """
        logits = self.mlp(cond)  # (B, num_options)
        return logits.masked_fill(~legal_mask, float("-inf"))


class TargetHead(nn.Module):
    """Head T — routes to a target sub-head by the sampled verb (§5).

    Sub-heads:
      attack     -> PointerHead over enemy entity embeddings (variable count)
      move       -> CategoricalHead over NUM_DIRECTIONS
      build_ship -> CategoricalHead over NUM_SHIP_TYPES
      build_port -> CategoricalHead over NUM_DIRECTIONS
      capture/load/unload/endturn -> no sub-head; target is implicit/none.

    cond for every sub-head is [context, e_asset, e_verb] (3*d_model): the global
    summary plus the embeddings of the chosen asset and a learned verb embedding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        cond_dim = 3 * d_model
        self.attack = PointerHead(cond_dim=cond_dim, d_model=d_model)
        self.move = CategoricalHead(cond_dim=cond_dim, num_options=NUM_DIRECTIONS)
        self.build_ship = CategoricalHead(cond_dim=cond_dim, num_options=NUM_SHIP_TYPES)
        self.build_port = CategoricalHead(cond_dim=cond_dim, num_options=NUM_DIRECTIONS)

    def forward(self, verb_name: str, cond: torch.Tensor, target_mask: torch.Tensor, candidate_emb=None):
        """Route to ONE sub-head by verb_name and return its masked logits.

        Called the same way from both paths — the only difference is who supplies
        the rows: act() passes the whole (B=1) batch; evaluate_actions() passes one
        verb's GROUP of rows at a time (see its loop). So this stays a clean scalar
        dispatch; the batching lives in the caller.

        - `attack` -> the pointer sub-head scores `candidate_emb` (the entity
          embeddings) against the query; target_mask (B, N) keeps only legal enemies.
        - move / build_port / build_ship -> the matching categorical sub-head over a
          fixed option set; candidate_emb is ignored.
        Returns logits (B, N) for attack or (B, K) for the categorical verbs.
        """
        if verb_name == "attack":
            return self.attack(cond, candidate_emb, target_mask)
        if verb_name == "move":
            return self.move(cond, target_mask)
        if verb_name == "build_ship":
            return self.build_ship(cond, target_mask)
        if verb_name == "build_port":
            return self.build_port(cond, target_mask)
        raise ValueError(f"{verb_name} has no target sub-head (not in VERBS_WITH_TARGET)")


# =============================================================================
# Full policy network
# =============================================================================
class PolicyNetwork(nn.Module):
    """tokens (B, N, token_dim) + pad_mask (B, N) -> sampled (asset, verb, target),
    joint log-prob, and V(s).

    Shares the encoder design of TransformerValueModel but exposes the per-entity
    embeddings (needed by the pointer heads), not just the pooled context.
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Encoder (identical recipe to TransformerValueModel) ---
        self.proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # --- Heads ---
        # Head A conditions on the pooled context only (d_model).
        self.head_asset = PointerHead(cond_dim=d_model, d_model=d_model)
        # Head V conditions on [context, e_asset] (2*d_model).
        self.head_verb = CategoricalHead(cond_dim=2 * d_model, num_options=NUM_VERBS)
        # Learned verb embedding, so Head T can be fed e_verb (the sampled verb)
        # rather than a raw index — gives the target heads a dense prefix signal.
        self.verb_embedding = nn.Embedding(NUM_VERBS, d_model)
        # Head T conditions on [context, e_asset, e_verb] (3*d_model), routed.
        self.head_target = TargetHead(d_model=d_model)

        # Value head: NON-autoregressive, pooled context only -> scalar in [-1,1].
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    # --- Encoder ---------------------------------------------------------------
    def encode(self, tokens: torch.Tensor, pad_mask: torch.Tensor):
        """tokens: (B, N, token_dim)   pad_mask: (B, N) bool, True = real token.

        Returns (embeddings (B, N, d_model), context (B, d_model)).
        Same as TransformerValueModel.forward up to the pool, but return BOTH:
          - embeddings: proj -> encoder(src_key_padding_mask = ~pad_mask).
            Remember the polarity flip (torch wants True = ignore).
          - context: masked mean-pool of embeddings over real tokens only
            (zero the padding rows, sum, divide by pad_mask.sum, clamp >=1).
        The value head and the categorical heads consume `context`; the pointer
        heads consume `embeddings`.
        """
        x = self.proj(tokens)  # lift 1x27 tokens to 64d vector, size model expects as input
        x = self.encoder(x, src_key_padding_mask=~pad_mask)  # attend to vectors

        m = pad_mask.unsqueeze(-1)  # add trailing axis
        summed = (x * m).sum(dim=1)  # zero out padding rows
        counts = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)  # count how many real tokens the board has
        pooled = summed / counts  # take mean of embedded tokens
        return x, pooled

    def value(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, d_model) -> V(s): (B,) in [-1, 1].

        - Use `self.value_head(context)` to map the board summary to one raw score
          of shape (B, 1).
        - Use `torch.tanh(...)` on it so that the output is squashed into [-1, 1],
          matching the mcts_root_value label range (see [[project-value-target]]).
          Do NOT stack a ReLU before the tanh — that one-sided saturation was the
          SimpleMLP bug.
        - Use `.squeeze(-1)` so that the shape is (B,), not (B, 1) — otherwise
          MSELoss against (B,) targets silently broadcasts to a wrong (B, B) loss.
        """

        raw_score = self.value_head(context)
        return torch.tanh(raw_score).squeeze(-1)

    # --- Rollout (sequential sampling) -----------------------------------------
    @torch.no_grad()
    def act(self, tokens: torch.Tensor, pad_mask: torch.Tensor, masks):
        """One action for ONE state (rollout). `masks` carries the engine's
        legality info; suggested contract (an object/dict the env builds):
            masks.asset            : (1, N) bool  — selectable assets (Head A)
            masks.verbs_for(asset) : (1, NUM_VERBS) bool — legal verbs given asset
            masks.target_for(asset, verb) : sub-head mask (directions / ship types
                                            / enemy candidates (1, N_enemy) + enemy_emb)

        Sequence (§7 inference):
          1. embeddings, context = encode(...)
          2. asset: logits = head_asset(context, embeddings, masks.asset);
             dist = Categorical(logits=...); asset_idx = dist.sample();
             lp_a = dist.log_prob(asset_idx). e_asset = embeddings[:, asset_idx].
          3. verb: cond = cat([context, e_asset]); logits = head_verb(cond, verb_mask);
             sample -> verb_idx, lp_v. e_verb = verb_embedding(verb_idx).
          4. target: if VERB_NAMES[verb_idx] in VERBS_WITH_TARGET:
                 cond = cat([context, e_asset, e_verb]); route via head_target;
                 sample -> target_idx, lp_t.
             else: target_idx = None, lp_t = 0.
          5. joint_logprob = lp_a + lp_v + lp_t;  v = value(context).
        Return whatever the rollout buffer needs: the triple, joint_logprob, v.
        (no_grad: rollout collection doesn't backprop; PPO backprops through
        evaluate_actions on the stored data instead.)

        ROLLOUT GOTCHAS (these are where it actually breaks):
        - dist comes from `masked_categorical(logits)`, NOT from a head class. The
          heads make logits; masked_categorical makes the distribution.
        - Everything is batch-first with B=1 here, so tensors carry a leading
          (1, ...) axis and asset_idx is shape (B,), not a scalar.
        - Pull e_asset by a BATCH GATHER, not Python indexing:
              e_asset = embeddings[torch.arange(B), asset_idx]   # (B, d_model)
          `embeddings[:, asset_idx]` is the wrong axis and will bite you at B>1.
        - cond plumbing is just concatenation on the feature axis:
              verb_cond   = torch.cat([context, e_asset], dim=-1)          # (B, 2*d)
              e_verb      = self.verb_embedding(verb_idx)                  # (B, d)
              target_cond = torch.cat([context, e_asset, e_verb], dim=-1)  # (B, 3*d)
        - Routing needs a Python str: `VERB_NAMES[verb_idx.item()]` (verb_idx is a
          tensor). Only call head_target when that name is in VERBS_WITH_TARGET.
        - For the no-target verbs, make lp_t a TENSOR so the sum stays a tensor:
          `lp_t = torch.zeros(B, device=...)`, not the int 0, and target_idx = None.
        - act() returns INDICES (asset_idx, verb_idx, target_idx), not an Action
          object. The env owns the index→ship_id / index→tile mapping and builds
          the actual core/actions.py dataclass — see [[project-battleboats]] (env
          owns masking + translation, engine owns rules).
        """
        # B = how many boards at once. Live rollout is B=1, but we keep the batch
        # axis explicit so the exact same code path also runs batched.
        B = tokens.shape[0]

        # One encoder pass yields BOTH things the heads need: per-entity embeddings
        # (the pointer head scores these) and the pooled board summary `context`
        # (the categorical + value heads consume this). Never re-encode mid-action.
        embeddings, context = self.encode(tokens, pad_mask)

        # ----- Factor 1: ASSET — "which of my entities acts?" --------------------
        # Pointer head scores every entity by q·eᵢ; masks.asset sets illegal ones to
        # -inf so they can't be chosen.
        asset_logits = self.head_asset(context, embeddings, masks.asset)
        asset_dist = masked_categorical(asset_logits)  # logits -> distribution
        asset_idx = asset_dist.sample()  # (B,) chosen entity index
        lp_a = asset_dist.log_prob(asset_idx)  # (B,) log π(asset | s)
        # Gather the chosen entity's embedding to FEED the next head (prefix-feeding).
        # arange(B) pairs each board b with ITS choice asset_idx[b] -> row b, col idx.
        e_asset = embeddings[torch.arange(B), asset_idx]  # (B, d_model)

        # ----- Factor 2: VERB — "what should that entity do?" --------------------
        # Concatenating e_asset onto the context is what makes this log π(verb | s,
        # asset) rather than log π(verb | s): the verb logits now depend on WHICH
        # asset got picked. That conditioning is the whole point of an autoregressive
        # policy.
        verb_cond = torch.cat([context, e_asset], dim=-1)  # (B, 2*d_model)
        verb_logits = self.head_verb(verb_cond, masks.verbs_for(asset_idx))
        verb_dist = masked_categorical(verb_logits)
        verb_idx = verb_dist.sample()  # (B,) chosen verb index
        lp_v = verb_dist.log_prob(verb_idx)  # (B,) log π(verb | s, asset)
        e_verb = self.verb_embedding(verb_idx)  # (B, d_model) verb as a vector

        # ----- Factor 3: TARGET — "the object of the verb" -----------------------
        # The target's TYPE depends on the verb (enemy / direction / ship type / none),
        # so Head T routes to a per-verb sub-head. Rollout samples one action at a
        # time, so we can branch on the scalar verb name (verb_idx is a (B,)=(1,)
        # tensor here -> .item()).
        verb_name = VERB_NAMES[verb_idx.item()]
        if verb_name in VERBS_WITH_TARGET:
            # Condition on the FULL prefix (asset AND verb): log π(target | s, a, v).
            target_cond = torch.cat([context, e_asset, e_verb], dim=-1)  # (B, 3*d_model)
            # One legality mask per sub-head. For `attack` it's an (B, N) mask over the
            # entity embeddings (True only for visible enemies in range — the same
            # pointer pattern as Head A); for move/build_* it's an (B, K) mask over the
            # fixed option set. Signature matches evaluate_actions: (verb_name, asset_idx).
            target_mask = masks.target_for(verb_name, asset_idx)
            # head_target dispatches on verb_name: the attack sub-head points over
            # `embeddings`; the categorical sub-heads ignore them and use target_cond.
            target_logits = self.head_target(verb_name, target_cond, target_mask, embeddings)
            target_dist = masked_categorical(target_logits)
            target_idx = target_dist.sample()  # (B,)
            lp_t = target_dist.log_prob(target_idx)  # (B,) log π(target | s, a, v)
        else:
            # capture/load/unload/endturn have an implicit-or-no target: nothing to
            # sample, and this factor contributes log-prob 0. Keep lp_t a TENSOR of
            # zeros (not the int 0) so the sum below stays a (B,) tensor.
            target_idx = None
            lp_t = torch.zeros(B, device=tokens.device)

        # Joint log-prob is the SUM of the three factor log-probs — this is exactly
        #   log π(a,v,t | s) = log π(a|s) + log π(v|s,a) + log π(t|s,a,v)
        # and it's the single number PPO treats as "the log-prob of this action."
        joint_lp = lp_a + lp_v + lp_t  # (B,)

        # Return INDICES (the env maps them back to ship_id / tile and constructs the
        # real core/actions.py dataclass), plus the joint log-prob and V(s) for the
        # rollout buffer. target_idx is None for no-target verbs — store a sentinel
        # like -1 if your buffer needs everything tensor-shaped.
        value = self.value(context)  # (B,)
        return asset_idx, verb_idx, target_idx, joint_lp, value

    # --- PPO update (parallel re-evaluation of stored actions) -----------------
    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        asset_idx: torch.Tensor,
        verb_idx: torch.Tensor,
        target_idx: torch.Tensor,
        masks,
    ):
        """Re-score a BATCH of stored (asset, verb, target) triples under the
        CURRENT policy. Returns (joint_logprob (B,), entropy (B,), value (B,)).

        This is the PPO workhorse: joint_logprob feeds the probability ratio
        r = exp(logprob_new - logprob_old) in the clipped objective; entropy feeds
        the exploration bonus; value feeds the value loss + advantage. NO sampling
        — use the stored indices.

        Build it as the parallel mirror of act():
          - encode once; gather e_asset via the stored asset_idx; e_verb via
            verb_embedding(verb_idx).
          - run head_asset / head_verb on the whole batch -> Categorical ->
            log_prob(stored_idx) and .entropy().
          - target factor is the hard part: rows have DIFFERENT verbs routing to
            DIFFERENT sub-heads. Group rows by verb and evaluate each sub-head on
            its group, scatter results back; rows whose verb has no target
            contribute log-prob 0 and entropy 0. (See TargetHead.forward note.)
          - joint_logprob = lp_a + lp_v + lp_t (sum the per-factor entropies too).
        Mask polarity / -inf handling identical to the rollout path — reuse the
        same masked-logits helpers so train and rollout can't silently diverge.

        `target_idx` is the STORED target index per row; for no-target verbs the
        buffer stores a sentinel (e.g. -1). We never read those rows (their verb
        isn't in VERBS_WITH_TARGET), so the sentinel is harmless.
        """
        B = tokens.shape[0]
        embeddings, context = self.encode(tokens, pad_mask)

        # --- Asset factor: score the batch, then read off the STORED idx's log-prob
        # (and entropy). No sampling — the action already happened during rollout.
        asset_logits = self.head_asset(context, embeddings, masks.asset)
        asset_dist = masked_categorical(asset_logits)
        lp_a = asset_dist.log_prob(asset_idx)  # (B,)
        ent_a = asset_dist.entropy()  # (B,)
        e_asset = embeddings[torch.arange(B), asset_idx]  # (B, d_model)

        # --- Verb factor (conditioned on the stored asset).
        verb_cond = torch.cat([context, e_asset], dim=-1)
        verb_logits = self.head_verb(verb_cond, masks.verbs_for(asset_idx))
        verb_dist = masked_categorical(verb_logits)
        lp_v = verb_dist.log_prob(verb_idx)  # (B,)
        ent_v = verb_dist.entropy()  # (B,)
        e_verb = self.verb_embedding(verb_idx)  # (B, d_model)

        # --- Target factor: THE batching wrinkle. Rows carry different verbs, which
        # route to different sub-heads with different output widths — so we can't run
        # one head over the whole batch. Strategy: GROUP BY VERB. For each target-verb
        # present, select its rows, run that sub-head on the group, and scatter the
        # per-row log-prob / entropy back. Rows whose verb has no target keep 0.
        target_cond = torch.cat([context, e_asset, e_verb], dim=-1)  # (B, 3*d_model)
        lp_t = torch.zeros(B, device=tokens.device)
        ent_t = torch.zeros(B, device=tokens.device)
        for verb_name in VERBS_WITH_TARGET:
            rows = (verb_idx == VERB_TO_IDX[verb_name]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            sub_mask = masks.target_for(verb_name, asset_idx)[rows]  # (R, K)
            sub_emb = embeddings[rows] if verb_name == "attack" else None
            logits = self.head_target(verb_name, target_cond[rows], sub_mask, sub_emb)
            dist = masked_categorical(logits)
            lp_t[rows] = dist.log_prob(target_idx[rows])  # (R,)
            ent_t[rows] = dist.entropy()  # (R,)

        # Joint log-prob and entropy are SUMS over the three independent factors
        # (independent → joint entropy is additive too). Value is state-only.
        joint_logprob = lp_a + lp_v + lp_t  # (B,)
        entropy = ent_a + ent_v + ent_t  # (B,)
        value = self.value(context)  # (B,)
        return joint_logprob, entropy, value


def masked_categorical(logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """logits: (B, K)   mask: optional (B, K) bool, True = legal.
    Returns a torch.distributions.Categorical over K options.

    - If `mask` is given, use `logits.masked_fill(~mask, float('-inf'))` so that
      illegal entries become zero-probability — so neither sampling nor log_prob
      can ever pick or credit an illegal action.
    - Use `assert mask.any(dim=-1).all()` FIRST (when masking) so that a row with
      no legal option fails LOUDLY here, instead of silently producing NaNs (a
      softmax over an all-`-inf` row is NaN). `endturn` is meant to keep every row
      non-empty; this assert is the tripwire for when it doesn't.
    - Return `torch.distributions.Categorical(logits=masked_logits)`.

    Centralizing here means act() (which calls `.sample()`) and evaluate_actions()
    (which calls `.log_prob(stored_idx)` + `.entropy()`) share ONE masking path —
    the usual source of train-vs-rollout divergence.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    # Tripwire that works whether the head pre-masked or we just did: every row must
    # keep at least one finite logit, else softmax over all -inf gives silent NaNs.
    assert torch.isfinite(logits).any(dim=-1).all(), "row with no legal option -> NaN"
    return Categorical(logits=logits)
