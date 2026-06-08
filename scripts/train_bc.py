"""Behavior-cloning trainer for the pointer-net policy (PolicyNetwork).

The POLICY sibling of train_transformer.py. Same scaffolding (by-game split,
overfit-batch sanity check, per-epoch val, checkpoint save), but the task is
IMITATION, not regression:
  - dataset is HarvestDataset(bc=True): each row is (tokens, expert action triple
    [asset, verb, target]), batched via collate_bc_batch -> (tokens, pad_mask,
    actions[B,3]). See [[project-battleboats]].
  - model is PolicyNetwork; we score the stored expert action with
    evaluate_actions(...) -> (joint_logprob, entropy, value).
  - loss = -joint_logprob.mean()  (maximize the log-prob the policy assigns to the
    expert's move). Optional entropy bonus via --ent-coef.
  - masks are BatchedMasks(pad_mask): pad-only, NO true legality. BC learns legality
    implicitly from legal labels; real masking returns in PPO with live ActionMasks.
  - the VALUE head is NOT trained here — BC is policy-only. evaluate_actions still
    returns a value, we just don't put a loss on it. The value head gets its signal
    in PPO (or you can warm it separately against mcts_root_value later).

The val metric to watch is action-match ACCURACY (how often a greedy decode picks
the expert's asset/verb/target), which is far more readable than the loss number.

STATUS: skeleton. Plumbing + train loop + val-loss are wired (the exact core
scripts/debug_bc.py already proved end-to-end). TWO TODO stubs remain:
  - bc_accuracy(): greedy-decode accuracy (genuinely new code).
  - save_checkpoint() / log_checkpoint_artifact(): mirror train_transformer.py.

Usage (after filling the stubs):
    poetry run python scripts/train_bc.py \\
        --harvest runs/harvests/<dir> --split runs/splits/<file>.json
    poetry run python scripts/train_bc.py ... --overfit-batch --steps 300
    poetry run python scripts/train_bc.py ... --wandb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from battleboats.envs.action_masks import BatchedMasks
from battleboats.training.dataset import HarvestDataset, collate_bc_batch
from battleboats.training.policy import VERB_TO_IDX, VERBS_WITH_TARGET, PolicyNetwork
from battleboats.training.split import load_split

import wandb


def bc_loss(model, tokens, pad_mask, actions, ent_coef: float):
    """One forward pass -> (loss, joint_logprob mean). Shared by train + val so the
    two can't drift. actions: (B, 3) = [asset_idx, verb_idx, target_idx].

    Splits the triple into the three index tensors, builds the pad-only BatchedMasks
    for this batch, scores the stored action, and returns the BC loss:
        loss = -joint_logprob.mean() - ent_coef * entropy.mean()
    The entropy term (ent_coef > 0) discourages the policy from collapsing to
    near-deterministic too early; for pure imitation leave it 0.
    """
    asset_idx, verb_idx, target_idx = actions[:, 0], actions[:, 1], actions[:, 2]
    masks = BatchedMasks(pad_mask)
    logp, ent, _value = model.evaluate_actions(tokens, pad_mask, asset_idx, verb_idx, target_idx, masks)
    loss = -logp.mean() - ent_coef * ent.mean()
    return loss, logp.mean()


@torch.no_grad()
def evaluate_bc(model, loader, device, ent_coef: float) -> float:
    """Mean BC loss over a loader, no gradients. model.eval() for clean dropout-off
    numbers; flip back to train() before returning (mirrors evaluate() in
    train_transformer.py)."""
    model.eval()
    total, n = 0.0, 0
    for tokens, pad_mask, actions in loader:
        tokens, pad_mask, actions = tokens.to(device), pad_mask.to(device), actions.to(device)
        loss, _ = bc_loss(model, tokens, pad_mask, actions, ent_coef)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def bc_accuracy(model, loader, device) -> dict:
    """Greedy-decode action-match accuracy on a loader. The human-readable "how
    often does the policy pick the expert's move?" metric.

    STUB — fill the body. Per batch, GREEDILY decode (argmax, not sample) the same
    autoregressive chain act() walks, but teacher-FREE — each head conditions on the
    head's OWN argmax, not the stored label:
      1. encode(tokens, pad_mask) -> embeddings, context
      2. asset:  argmax(head_asset(context, embeddings, masks.asset))      -> a_hat
                 e_asset = embeddings[arange(B), a_hat]
      3. verb:   argmax(head_verb(cat([context,e_asset]), masks.verbs_for))-> v_hat
                 e_verb = verb_embedding(v_hat)
      4. target: per-verb routing like evaluate_actions; for no-target verbs t_hat
                 is the sentinel and counts as correct iff the label is the sentinel.
    Compare a_hat/v_hat/t_hat to actions[:,0/1/2] and accumulate matches.
    Return e.g. {"asset": .., "verb": .., "target": .., "joint": ..} where joint =
    all three correct (target trivially correct for no-target verbs). Use
    BatchedMasks(pad_mask) for the masks, same as training. model.eval()/.train().

    Tip: this is basically act() with sample()->argmax() and no log-prob/value
    bookkeeping. Consider a small greedy_decode() on PolicyNetwork that both share,
    so rollout and accuracy can't diverge — but a local copy here is fine to start.
    """

    model.eval()
    correct = {"asset": 0, "verb": 0, "target": 0, "joint": 0}
    total = 0
    for tokens, pad_mask, actions in loader:
        tokens, pad_mask, actions = tokens.to(device), pad_mask.to(device), actions.to(device)
        B = tokens.shape[0]
        masks = BatchedMasks(pad_mask)
        embeddings, context = model.encode(tokens, pad_mask)

        # --- Asset: greedy = argmax of the masked logits (no sampling). -inf on
        # illegal/pad slots means argmax can never land on one.
        asset_logits = model.head_asset(context, embeddings, masks.asset)
        a_hat = asset_logits.argmax(dim=-1)  # (B,)
        e_asset = embeddings[torch.arange(B, device=device), a_hat]  # (B, d_model)

        # --- Verb: condition on the PREDICTED asset (teacher-free), then argmax.
        verb_cond = torch.cat([context, e_asset], dim=-1)
        verb_logits = model.head_verb(verb_cond, masks.verbs_for(a_hat))
        v_hat = verb_logits.argmax(dim=-1)  # (B,)
        e_verb = model.verb_embedding(v_hat)  # (B, d_model)

        # --- Target: per-verb routing, mirroring evaluate_actions but argmax instead
        # of log_prob. Default to the no-target sentinel (-1); only rows whose
        # PREDICTED verb has a sub-head get overwritten.
        target_cond = torch.cat([context, e_asset, e_verb], dim=-1)  # (B, 3*d_model)
        t_hat = torch.full((B,), -1, dtype=torch.long, device=device)
        for verb_name in VERBS_WITH_TARGET:
            rows = (v_hat == VERB_TO_IDX[verb_name]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            sub_mask = masks.target_for(verb_name, a_hat)[rows]  # (R, K)
            sub_emb = embeddings[rows] if verb_name == "attack" else None
            logits = model.head_target(verb_name, target_cond[rows], sub_mask, sub_emb)
            t_hat[rows] = logits.argmax(dim=-1)

        # --- Compare to the expert triple. Per-factor matches + joint (all three).
        # For no-target verbs the stored label is the -1 sentinel, so t_hat == t_lab
        # is True exactly when both are -1 — "trivially correct, nothing to predict."
        a_lab, v_lab, t_lab = actions[:, 0], actions[:, 1], actions[:, 2]
        asset_ok = a_hat == a_lab
        verb_ok = v_hat == v_lab
        target_ok = t_hat == t_lab
        correct["asset"] += int(asset_ok.sum())
        correct["verb"] += int(verb_ok.sum())
        correct["target"] += int(target_ok.sum())
        correct["joint"] += int((asset_ok & verb_ok & target_ok).sum())
        total += B

    model.train()
    return {k: round(v / max(total, 1), 4) for k, v in correct.items()}


def save_checkpoint(model, args, metrics: dict, save_dir: Path, tag: Optional[str] = None) -> Path:
    """Persist the policy and return the path. ONE self-describing torch.save dict:
      - "model_state":  model.state_dict()
      - "model_config": kwargs to rebuild PolicyNetwork — token_dim, d_model, nhead,
        num_layers, dim_feedforward, dropout. (state_dict has NO architecture info.)
      - "train_args":   vars(args)
      - "metrics":      metrics  (val_loss + accuracies + epoch)

    tag gives a STABLE filename (bc_policy_<tag>.pt) so per-epoch saves OVERWRITE in
    place — used for "latest" (crash recovery every epoch) and "best" (best val_loss
    so far). With tag=None it timestamps instead. The PPO warm-start reloads via
    PolicyNetwork.load_state_dict, so it must round-trip cleanly.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    name = tag if tag else time.strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"bc_policy_{name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            # token_dim/d_model are recoverable from the net; the rest aren't, so
            # carry them from args. Together = everything PolicyNetwork.__init__ needs.
            "model_config": {
                "token_dim": model.proj.in_features,
                "d_model": model.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
            "train_args": vars(args),
            "metrics": metrics,
        },
        path,
    )
    return path


def log_checkpoint_artifact(ckpt_path: Path, metrics: dict, args) -> None:
    """Upload the checkpoint to W&B as a versioned Artifact. STUB — mirror
    train_transformer.log_checkpoint_artifact, but use a STABLE name like
    "bc-policy" so runs become v0, v1, ... of the same artifact. Only reached
    when --wandb is on (wandb.init already ran)."""
    # Stringify args — vars(args) holds Path objects, which W&B's JSON metadata
    # can't serialize. metrics are plain floats and pass through.
    metadata = {**metrics, **{k: str(v) for k, v in vars(args).items()}}
    art = wandb.Artifact(name="bc-policy", type="model", metadata=metadata)
    art.add_file(str(ckpt_path))
    wandb.log_artifact(art)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Prebuilt token index (.npz from build_token_index.py). Skips the ~40 min JSON scan.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus weight (0 = pure imitation).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overfit-batch", action="store_true", help="Sanity check on one batch.")
    parser.add_argument("--steps", type=int, default=300, help="Steps in --overfit-batch mode.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (ROCm exposes the 7800 XT as cuda).")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/checkpoints"))
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--wandb-entity", default="nicksrusso")
    parser.add_argument("--wandb-project", default="battleboats-data")
    args = parser.parse_args()

    use_wandb = args.wandb

    # --- Data (BC mode: tokens + expert action triple) ---
    train_ids, val_ids, meta = load_split(args.split)
    train_ds = HarvestDataset(args.harvest, game_idxs=train_ids, bc=True, load_tokens=True, index_path=args.index)
    val_ds = HarvestDataset(args.harvest, game_idxs=val_ids, bc=True, load_tokens=True, index_path=args.index)
    print(f"token_dim = {train_ds.token_dim}   train rows = {len(train_ds)}   val rows = {len(val_ds)}")

    # --- Model / optimizer ---
    model = PolicyNetwork(
        token_dim=train_ds.token_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))
        wandb.config.update({"token_dim": train_ds.token_dim, "train_rows": len(train_ds), "val_rows": len(val_ds)})

    # --- DataLoaders (collate_bc_batch pads + builds pad_mask + stacks the triples) ---
    # 12 worker procs parse the lazy per-row token reads (open+seek+json) in parallel,
    # ahead of the GPU, so the small/fast model isn't starved by data IO. persistent_
    # workers keeps them alive across epochs (spawn is costly); prefetch_factor stages
    # batches ahead. _read_tokens re-opens files per call, so forked workers are safe.
    NUM_WORKERS = 12
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bc_batch,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bc_batch,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4,
    )

    # ------------------------------------------------------------------
    # Overfit-a-batch sanity check (run FIRST after any change). Loss should
    # drive toward 0 — if it can't fit ONE batch, training won't fit the set.
    # ------------------------------------------------------------------
    if args.overfit_batch:
        tokens, pad_mask, actions = next(iter(loader))
        tokens, pad_mask, actions = tokens.to(args.device), pad_mask.to(args.device), actions.to(args.device)
        t0 = time.perf_counter()
        for step in range(args.steps):
            loss, mean_lp = bc_loss(model, tokens, pad_mask, actions, args.ent_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print(
                    f"    step{step:4d} loss={loss.item():.6f}  mean_logp={mean_lp.item():.4f}  t={time.perf_counter()-t0:6.1f}s"
                )
                if use_wandb:
                    wandb.log({"overfit/loss": loss.item(), "overfit/step": step})
        if use_wandb:
            wandb.finish()
        return

    # ------------------------------------------------------------------
    # Normal training: epochs x batches. Same core debug_bc proved:
    #   bc_loss -> backward -> step.  Log train loss every --log-every,
    #   eval val loss + accuracy each epoch.
    # ------------------------------------------------------------------
    global_step = 0
    best_val = float("inf")
    best_path = None
    best_metrics = None
    for epoch in range(args.epochs):
        running, n = 0.0, 0
        for tokens, pad_mask, actions in loader:
            tokens, pad_mask, actions = tokens.to(args.device), pad_mask.to(args.device), actions.to(args.device)
            loss, _ = bc_loss(model, tokens, pad_mask, actions, args.ent_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                avg = running / n
                print(f"epoch={epoch}  step={global_step}  train_loss={avg:.4f}")
                if use_wandb:
                    wandb.log({"train/loss": avg, "epoch": epoch}, step=global_step)
                running, n = 0.0, 0

        val_loss = evaluate_bc(model, val_loader, args.device, args.ent_coef)
        acc = bc_accuracy(model, val_loader, args.device)
        print(f"epoch={epoch}  VAL  val_loss={val_loss:.4f}  acc={acc}")
        if use_wandb:
            wandb.log({"val/loss": val_loss, **{f"val_acc/{k}": v for k, v in acc.items()}, "epoch": epoch}, step=global_step)

        # Per-epoch checkpointing: "latest" is overwritten every epoch (crash/early-stop
        # recovery — a mid-run GPU hang or Ctrl-C still leaves a usable warm-start), and
        # "best" tracks the lowest val_loss seen (the one PPO should bootstrap from).
        metrics = {"val_loss": val_loss, "epoch": epoch, **{f"acc_{k}": v for k, v in acc.items()}}
        save_checkpoint(model, args, metrics, args.save_dir, tag="latest")
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = metrics
            best_path = save_checkpoint(model, args, metrics, args.save_dir, tag="best")
            print(f"  new best val_loss={val_loss:.4f} -> {best_path}")

    # --- Final: the BEST checkpoint is the PPO warm-start; upload it as the artifact. ---
    print(f"done. best val_loss={best_val:.4f} at {best_path}")
    if use_wandb:
        if best_path is not None:
            log_checkpoint_artifact(best_path, best_metrics, args)
        wandb.finish()


if __name__ == "__main__":
    main()
