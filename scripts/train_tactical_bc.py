#!/usr/bin/env python3
"""Behavior-clone the tactical scripted agent into a pointer policy.

Initializes a v0.3.0-architecture ``PolicyNetwork`` from scratch and trains
masked cross-entropy on harvested tactical actions, plus an MSE value-head
loss on discounted terminal returns (gamma=0.99, win +1 / loss -1 / draw 0)
so the checkpoint can warm-start PPO.

    PYTHONPATH=<repo> python scripts/train_tactical_bc.py \
        --harvest runs/bc_tactical_v040/harvest \
        --save-dir runs/bc_tactical_v040/train
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from battleboats.envs.action_masks import BatchedMasks
from battleboats.envs.observation import TOKEN_DIM
from battleboats.training.checkpoints import model_config_of, save_checkpoint
from battleboats.training.device import resolve_device
from battleboats.training.manifest import ManifestSession, canonical_run_dir
from battleboats.training.policy import (
    VERB_NAMES,
    VERB_TO_IDX,
    VERBS_WITH_TARGET,
    PolicyNetwork,
)
from battleboats.training.tactical_bc import (
    V0_3_0_ARCH,
    VERB_GROUP,
    CompactBCDataset,
    TacticalBCDataset,
    collate_tactical_bc,
    stratified_row_indices,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_policy(arch: Dict[str, Any], device: torch.device) -> PolicyNetwork:
    """Fresh PolicyNetwork with the v0.3.0 architecture. Default PyTorch init."""
    cfg = dict(V0_3_0_ARCH)
    cfg.update(arch)
    # token_dim must match the live observation width.
    cfg["token_dim"] = int(TOKEN_DIM)
    net = PolicyNetwork(**cfg)
    return net.to(device)


def bc_loss(model, tokens, pad_mask, globals_t, actions, returns, vf_coef: float):
    asset_idx, verb_idx, target_idx = actions[:, 0], actions[:, 1], actions[:, 2]
    masks = BatchedMasks(pad_mask)
    logp, _ent, value = model.evaluate_actions(
        tokens,
        pad_mask,
        asset_idx,
        verb_idx,
        target_idx,
        masks,
        skill_idx=0,
        globals=globals_t,
    )
    policy_loss = -logp.mean()
    value_loss = torch.nn.functional.mse_loss(value, returns)
    loss = policy_loss + vf_coef * value_loss
    return loss, policy_loss, value_loss, logp.mean()


@torch.no_grad()
def greedy_decode(model, tokens, pad_mask, globals_t):
    """Teacher-free greedy (asset, verb, target), same chain as act()."""
    B = tokens.shape[0]
    device = tokens.device
    masks = BatchedMasks(pad_mask)
    embeddings, context = model.encode(tokens, pad_mask, globals_t)
    asset_logits = model.head_asset(context, embeddings, masks.asset)
    a_hat = asset_logits.argmax(dim=-1)
    e_asset = embeddings[torch.arange(B, device=device), a_hat]
    verb_logits = model.head_verb(
        torch.cat([context, e_asset], dim=-1), masks.verbs_for(a_hat)
    )
    v_hat = verb_logits.argmax(dim=-1)
    e_verb = model.verb_embedding(v_hat)
    target_cond = torch.cat([context, e_asset, e_verb], dim=-1)
    t_hat = torch.full((B,), -1, dtype=torch.long, device=device)
    for verb_name in VERBS_WITH_TARGET:
        rows = (v_hat == VERB_TO_IDX[verb_name]).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        sub_mask = masks.target_for(verb_name, a_hat)[rows]
        sub_emb = embeddings[rows] if verb_name == "attack" else None
        logits = model.head_target(verb_name, target_cond[rows], sub_mask, sub_emb)
        t_hat[rows] = logits.argmax(dim=-1)
    return a_hat, v_hat, t_hat


@torch.no_grad()
def evaluate(model, loader, device, vf_coef: float) -> Dict[str, Any]:
    model.eval()
    tot_loss = tot_pol = tot_val = tot_logp = 0.0
    n_batches = 0
    n = 0
    correct = {"asset": 0, "verb": 0, "target": 0, "joint": 0}
    by_group = {g: {"n": 0, "joint": 0} for g in sorted(set(VERB_GROUP.values()))}
    by_verb = {v: {"n": 0, "joint": 0} for v in VERB_NAMES}
    for tokens, pad_mask, globals_t, actions, returns in loader:
        tokens = tokens.to(device)
        pad_mask = pad_mask.to(device)
        globals_t = globals_t.to(device)
        actions = actions.to(device)
        returns = returns.to(device)
        loss, pol, val, logp = bc_loss(
            model, tokens, pad_mask, globals_t, actions, returns, vf_coef
        )
        tot_loss += float(loss.detach())
        tot_pol += float(pol.detach())
        tot_val += float(val.detach())
        tot_logp += float(logp.detach())
        n_batches += 1
        a_hat, v_hat, t_hat = greedy_decode(model, tokens, pad_mask, globals_t)
        a_lab, v_lab, t_lab = actions[:, 0], actions[:, 1], actions[:, 2]
        asset_ok = a_hat == a_lab
        verb_ok = v_hat == v_lab
        target_ok = t_hat == t_lab
        joint_ok = asset_ok & verb_ok & target_ok
        correct["asset"] += int(asset_ok.sum())
        correct["verb"] += int(verb_ok.sum())
        correct["target"] += int(target_ok.sum())
        correct["joint"] += int(joint_ok.sum())
        n += int(actions.shape[0])
        for i in range(actions.shape[0]):
            verb = VERB_NAMES[int(v_lab[i])]
            grp = VERB_GROUP.get(verb, verb)
            by_verb[verb]["n"] += 1
            by_group[grp]["n"] += 1
            if bool(joint_ok[i]):
                by_verb[verb]["joint"] += 1
                by_group[grp]["joint"] += 1
    model.train()

    def _rate(k, d):
        return (k / d) if d else None

    return {
        "loss": tot_loss / n_batches if n_batches else None,
        "policy_loss": tot_pol / n_batches if n_batches else None,
        "value_loss": tot_val / n_batches if n_batches else None,
        "logp": tot_logp / n_batches if n_batches else None,
        "n": n,
        "agreement": {
            "asset": _rate(correct["asset"], n),
            "verb": _rate(correct["verb"], n),
            "target": _rate(correct["target"], n),
            "joint": _rate(correct["joint"], n),
        },
        "agreement_by_group": {
            g: {"n": rec["n"], "joint": _rate(rec["joint"], rec["n"])}
            for g, rec in by_group.items()
            if rec["n"]
        },
        "agreement_by_verb": {
            v: {"n": rec["n"], "joint": _rate(rec["joint"], rec["n"])}
            for v, rec in by_verb.items()
            if rec["n"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            "/home/nick/Desktop/repos/Battleboats/runs/bc_tactical_v040/train"
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, cuda:N, or auto (ROCm GPU shows up as cuda).")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Load only the selected rows (CompactBCDataset). Needed for 48x24 harvests that do not fit in RAM.",
    )
    parser.add_argument(
        "--full-val-rows",
        type=int,
        default=0,
        help="With --compact: natural-mix val sample for the final held-out pass (0 = all val rows).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="With --compact: memory-map the selected rows from .npy files here (keeps RSS low).",
    )
    parser.add_argument(
        "--row-mix",
        choices=("stratified", "natural"),
        default="stratified",
        help="With --compact: stratified keeps rare verbs in full (shifts the verb prior toward them); "
        "natural samples train rows uniformly so the clone keeps the teacher's verb frequencies.",
    )
    parser.add_argument("--log-every", type=int, default=0, help="Print running loss every N batches (0 = off).")
    parser.add_argument("--source-tag", type=str, default="tactical_bc", help="Checkpoint extra['source'] label.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=250000,
        help="Stratified cap on train rows (0 = use all). Rare verbs are kept in full.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=16000,
        help="Val rows used each epoch (0 = all). A full val pass runs at the end.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    device = resolve_device(args.device)

    split_path = args.split or (args.harvest / "split.json")
    split = json.loads(split_path.read_text())
    train_ids = set(int(i) for i in split["train"])
    val_ids = set(int(i) for i in split["val"])
    if args.compact:
        cache = args.cache_dir
        train_full = CompactBCDataset(
            args.harvest, train_ids, args.max_train_rows, args.seed, mode=args.row_mix,
            cache_dir=cache and cache / f"train_{args.row_mix}",
        )
        val_epoch_full = CompactBCDataset(
            args.harvest, val_ids, args.max_val_rows, args.seed + 1, cache_dir=cache and cache / "val_epoch"
        )
        val_full = CompactBCDataset(
            args.harvest, val_ids, args.full_val_rows, args.seed + 2, mode="natural",
            cache_dir=cache and cache / "val_full",
        )
        train_idx = list(range(len(train_full)))
        val_epoch_idx = list(range(len(val_epoch_full)))
        n_full_train_rows = train_full.n_rows_full
    else:
        train_full = TacticalBCDataset(args.harvest, train_ids, preload=True)
        val_full = TacticalBCDataset(args.harvest, val_ids, preload=True)
        val_epoch_full = val_full
        train_idx = stratified_row_indices(train_full, args.max_train_rows, args.seed)
        val_epoch_idx = stratified_row_indices(val_full, args.max_val_rows, args.seed + 1)
        n_full_train_rows = len(train_full)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_epoch_full, val_epoch_idx)

    def _verb_hist(base, indices):
        c = Counter(base.verb_name(i) for i in indices)
        return dict(sorted(c.items()))

    train_verb_hist = _verb_hist(train_full, train_idx)
    val_verb_hist = _verb_hist(val_epoch_full, val_epoch_idx)
    print(
        f"train games={len(train_ids)} rows={len(train_ds)}/{n_full_train_rows} "
        f"val games={len(val_ids)} rows={len(val_ds)}/{len(val_full)} device={device}",
        flush=True,
    )
    print("train verbs:", json.dumps(train_verb_hist), flush=True)
    print("val-epoch verbs:", json.dumps(val_verb_hist), flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tactical_bc,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tactical_bc,
    )
    full_val_loader = DataLoader(
        val_full,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tactical_bc,
    )

    arch = dict(V0_3_0_ARCH)
    arch["dropout"] = float(args.dropout)
    model = new_policy(arch, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    session = ManifestSession(
        canonical_run_dir(args.save_dir),
        kind="bc",
        config=vars(args),
    ).start()

    curve: List[Dict[str, Any]] = []
    best_joint = -1.0
    best_path: Optional[Path] = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_b = 0
        epoch_t0 = time.time()
        for tokens, pad_mask, globals_t, actions, returns in train_loader:
            tokens = tokens.to(device)
            pad_mask = pad_mask.to(device)
            globals_t = globals_t.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            loss, _pol, _val, _logp = bc_loss(
                model, tokens, pad_mask, globals_t, actions, returns, args.vf_coef
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            running += float(loss.detach())
            n_b += 1
            if args.log_every and n_b % args.log_every == 0:
                print(f"  epoch {epoch} batch {n_b}/{len(train_loader)} loss={running / n_b:.4f} "
                      f"t={time.time() - epoch_t0:.0f}s", flush=True)
        train_metrics = {
            "loss_running": running / n_b if n_b else None,
        }
        val_metrics = evaluate(model, val_loader, device, args.vf_coef)
        rec = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_sec": time.time() - epoch_t0,
        }
        curve.append(rec)
        vj = (val_metrics.get("agreement") or {}).get("joint")
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={train_metrics['loss_running']:.4f} val_loss={val_metrics['loss']:.4f}  "
            f"val_joint={vj:.4f}  "
            f"val_value_loss={val_metrics['value_loss']:.4f}  "
            f"sec={rec['epoch_sec']:.1f}",
            flush=True,
        )
        print(
            "  val by group:",
            json.dumps(val_metrics["agreement_by_group"], default=str),
            flush=True,
        )

        latest = args.save_dir / "latest.pt"
        save_checkpoint(
            model,
            latest,
            extra={
                "source": args.source_tag,
                "epoch": epoch,
                "metrics": {"train": train_metrics, "val": val_metrics},
            },
        )
        if vj is not None and vj > best_joint:
            best_joint = vj
            best_path = args.save_dir / "best.pt"
            save_checkpoint(
                model,
                best_path,
                extra={
                    "source": args.source_tag,
                    "epoch": epoch,
                    "metrics": {"train": train_metrics, "val": val_metrics},
                },
            )
            print(f"  new best val joint {best_joint:.4f} -> {best_path}", flush=True)

    summary = {
        "init": "from scratch (PyTorch default)",
        "arch": model_config_of(model),
        "value_head": (
            "MSE on discounted terminal returns; G_t = z * 0.99**(K-1-t) "
            "per decision, z = +1 win / -1 loss / 0 draw for the acting "
            "tactical player. vf_coef="
            f"{args.vf_coef}. Value head is in the official checkpoint "
            "state_dict so load_checkpoint / PPO warm-start works."
        ),
        "harvest": str(args.harvest),
        "split": str(split_path),
        "n_train_rows": len(train_ds),
        "n_train_rows_full": n_full_train_rows,
        "n_val_rows_epoch": len(val_ds),
        "n_val_rows_full": len(val_full),
        "n_train_games": len(train_ids),
        "n_val_games": len(val_ids),
        "train_verb_counts": train_verb_hist,
        "val_epoch_verb_counts": val_verb_hist,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "vf_coef": args.vf_coef,
        "seed": args.seed,
        "best_val_joint": best_joint,
        "best_checkpoint": str(best_path) if best_path else None,
        "elapsed_sec": time.time() - t0,
        "curve": curve,
    }
    # Honest held-out agreement: full val set, best checkpoint.
    if best_path is not None:
        from battleboats.training.checkpoints import load_checkpoint

        best_model, _ = load_checkpoint(best_path, device)
        best_model.eval()
        full_val = evaluate(best_model, full_val_loader, device, args.vf_coef)
        summary["full_val"] = full_val
        summary["best_val_joint_full"] = (full_val.get("agreement") or {}).get("joint")
        print(
            "FULL VAL (best ckpt):",
            json.dumps(full_val, indent=2, default=str),
            flush=True,
        )
    (args.save_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "curve"}, indent=2, default=str
        )
    )
    final = best_path if best_path is not None else (args.save_dir / "latest.pt")
    session.complete(
        final_checkpoint=final if Path(final).is_file() else None,
        updates_completed=args.epochs,
    )
    print(f"wrote {args.save_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
