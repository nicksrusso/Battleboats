"""Debug harness for the behavior-cloning pipeline.

Run it to SEE every tensor flowing through
    HarvestDataset(bc=True) -> collate_bc_batch -> BatchedMasks -> evaluate_actions
so you can implement BatchedMasks (and later train_bc.py) by inspection.

    poetry run python scripts/debug_bc.py

It tolerates the not-yet-filled BatchedMasks methods: each mask call is wrapped, so
before you implement them you still get the inputs + a reminder of the expected
shape; once filled, it runs the full BC loss + backward to prove the step works.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from battleboats.envs.action_masks import BatchedMasks
from battleboats.training.dataset import HarvestDataset, collate_bc_batch
from battleboats.training.policy import NUM_VERBS, VERB_NAMES, PolicyNetwork

HARVEST = "runs/harvests/harvest_20260605_111638"


def show(name: str, t: torch.Tensor) -> None:
    print(f"    {name:26s} shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")


def main() -> None:
    # --- 1. Dataset (a few games, BC mode) --------------------------------------
    print("== HarvestDataset(bc=True), games {0,1,2} ==")
    ds = HarvestDataset(HARVEST, game_idxs={0, 1, 2}, bc=True, load_tokens=True)
    print(f"    samples={len(ds)}   token_dim={ds.token_dim}")
    tok0, act0 = ds[0]
    show("item[0].tokens", tok0)
    show("item[0].action", act0)
    print(f"    action triple [asset, verb, target] = {act0.tolist()}  (verb='{VERB_NAMES[int(act0[1])]}')")

    # --- 2. One collated batch --------------------------------------------------
    print("\n== one collate_bc_batch (B=4) ==")
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_bc_batch)
    tokens, pad_mask, actions = next(iter(dl))
    show("tokens", tokens)
    show("pad_mask", pad_mask)
    show("actions", actions)
    print(f"    real-token counts per row: {pad_mask.sum(1).tolist()}")
    print(f"    actions (asset, verb, target):\n{actions}")
    print(f"    verbs in batch: {[VERB_NAMES[v] for v in actions[:, 1].tolist()]}")

    # --- 3. BatchedMasks --------------------------------------------------------
    print("\n== BatchedMasks(pad_mask) ==")
    m = BatchedMasks(pad_mask)
    print(f"    B={m.B}  N={m.N}  device={m.device}")
    asset_idx = actions[:, 0]

    def try_mask(label, fn, expected):
        try:
            show(label, fn())
        except NotImplementedError:
            print(f"    {label:26s} NOT IMPLEMENTED — should return {expected}")

    try_mask("asset", lambda: m.asset, "pad_mask, shape (B, N)")
    try_mask("verbs_for(asset_idx)", lambda: m.verbs_for(asset_idx), f"all-True (B, {NUM_VERBS})")
    for vn in ("attack", "move", "build_ship", "build_port"):
        try_mask(
            f"target_for('{vn}')", lambda vn=vn: m.target_for(vn, asset_idx), "pad_mask (B,N) if attack else all-True (B,K)"
        )

    # --- 4. Full BC loss path (works once BatchedMasks is filled) ---------------
    print("\n== policy.evaluate_actions -> BC loss (needs BatchedMasks filled) ==")
    try:
        net = PolicyNetwork(token_dim=ds.token_dim)
        verb_idx, target_idx = actions[:, 1], actions[:, 2]
        logp, ent, val = net.evaluate_actions(tokens, pad_mask, asset_idx, verb_idx, target_idx, m)
        show("joint_logprob", logp)
        show("entropy", ent)
        show("value", val)
        loss = -logp.mean()
        print(f"    BC loss (-joint_logprob.mean) = {loss.item():.4f}")
        loss.backward()
        print("    backward OK — full BC step runs end to end.")
    except NotImplementedError:
        print("    blocked on a NotImplementedError above — fill BatchedMasks, then re-run.")


if __name__ == "__main__":
    main()
