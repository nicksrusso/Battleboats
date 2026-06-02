"""Training loop for the transformer encoder + value head.

The token-input sibling of train_mlp.py — SAME task (regress mcts_root_value),
SAME scaffolding (by-game split, overfit-batch sanity check, per-epoch val,
predict-mean / label-variance baselines, phase-stratified val MSE). The only
differences:
  - input is variable-length entity tokens, batched via collate_token_batch
    (padding + pad_mask), not a fixed phi vector;
  - model is TransformerValueModel(tokens, pad_mask) -> value;
  - default target is mcts_root_value (the learnable target proven on the MLP).

Optional Weights & Biases logging (--wandb). Import is guarded so the script
runs whether or not wandb is installed; install with `poetry add wandb` and
`wandb login` to enable.

STATUS: skeleton. Setup/plumbing is done; the loops + eval helpers are TODO
stubs for you to fill in (mirrors how train_mlp.py started). Each batch from
the loaders is a 3-tuple (tokens, pad_mask, targets) — NOT (x, y) — because
collate_token_batch adds the mask. Remember model.eval()/.train() now matters:
the transformer has dropout.

Usage (after filling the stubs):
    poetry run python scripts/train_transformer.py \\
        --harvest runs/harvests/<file>.jsonl --split runs/splits/<file>.json
    poetry run python scripts/train_transformer.py ... --overfit-batch --steps 300
    poetry run python scripts/train_transformer.py ... --wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from battleboats.training.dataset import HarvestDataset, collate_token_batch
from battleboats.training.split import load_split
from battleboats.training.transformer import TransformerValueModel

import wandb


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    """Mean loss over a token loader, no gradients.

    model.eval() matters here — the transformer has dropout, which must be OFF
    for a clean val number; flip back to model.train() before returning.
    """
    # TODO: model.eval()
    # TODO: total_loss, n_batches = 0.0, 0
    # TODO: for tokens, pad_mask, targets in loader:
    # TODO:     tokens, pad_mask, targets = tokens.to(device), pad_mask.to(device), targets.to(device)
    # TODO:     pred = model(tokens, pad_mask)
    # TODO:     total_loss += loss_fn(pred, targets).item()
    # TODO:     n_batches += 1
    # TODO: model.train()
    # TODO: return total_loss / max(n_batches, 1)
    raise NotImplementedError("Fill in evaluate — see TODOs.")


@torch.no_grad()
def evaluate_by_phase(model, ds, loader, device, n_bins: int = 3) -> dict:
    """Val MSE split by game phase (early/mid/late). Same diagnostic as the MLP.

    Gather predictions by iterating `loader` (MUST be shuffle=False) so the
    concatenated order matches ds row order, then line each squared error up
    with ds.steps / ds.game_ids for phase bucketing. See [[project-value-target]].
    """
    # TODO: model.eval()
    # TODO: preds = []
    # TODO: for tokens, pad_mask, _ in loader:
    # TODO:     preds.append(model(tokens.to(device), pad_mask.to(device)).cpu())
    # TODO: pred = torch.cat(preds).numpy()
    # TODO: sq_err = (pred - ds.targets) ** 2
    #
    # TODO: normalize step -> progress fraction in [0,1] per game:
    # TODO:   steps = ds.steps.astype(np.float64); max_step = np.zeros_like(steps)
    # TODO:   for gid in np.unique(ds.game_ids):
    # TODO:       mask = ds.game_ids == gid; m = steps[mask].max()
    # TODO:       max_step[mask] = m if m > 0 else 1.0
    # TODO:   progress = steps / max_step
    #
    # TODO: bucket into n_bins, print mse per bucket, return {label: mse}
    # TODO: model.train()
    raise NotImplementedError("Fill in evaluate_by_phase — see TODOs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overfit-batch", action="store_true", help="Sanity check on one batch.")
    parser.add_argument("--steps", type=int, default=300, help="Steps in --overfit-batch mode.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (ROCm exposes the 7800 XT as cuda).")
    parser.add_argument(
        "--target",
        choices=["target", "mcts_root_value"],
        default="mcts_root_value",
        help="Regression label. Default mcts_root_value (the learnable one).",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--wandb-project", default="battleboats-value")
    args = parser.parse_args()

    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("[warn] --wandb passed but wandb not installed; continuing without logging. " "`poetry add wandb` to enable.")

    # --- Data (token mode) ---
    train_ids, val_ids, meta = load_split(args.split)
    train_ds = HarvestDataset(args.harvest, game_idxs=train_ids, target_key=args.target, load_tokens=True)
    val_ds = HarvestDataset(args.harvest, game_idxs=val_ids, target_key=args.target, load_tokens=True)
    print(
        f"target = {args.target}   token_dim = {train_ds.token_dim}   "
        f"train rows = {len(train_ds)}   val rows = {len(val_ds)}"
    )

    # --- Model / optimizer / loss ---
    model = TransformerValueModel(
        token_dim=train_ds.token_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # --- Naive baselines (the bar to clear; pure label properties) ---
    train_mean = float(train_ds.targets.mean())
    baseline_val_mse = float(((val_ds.targets - train_mean) ** 2).mean())
    val_label_var = float(val_ds.targets.var())
    print(f"baseline  predict train-mean ({train_mean:+.4f})   val_mse={baseline_val_mse:.4f}")
    print(f"baseline  val label variance (best constant)   val_mse={val_label_var:.4f}")

    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.config.update(
            {
                "token_dim": train_ds.token_dim,
                "train_rows": len(train_ds),
                "val_rows": len(val_ds),
                "baseline_val_mse": baseline_val_mse,
                "val_label_var": val_label_var,
            }
        )

    # --- DataLoaders (collate_token_batch pads + builds pad_mask) ---
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_token_batch)
    # Val loader: shuffle=False so evaluate_by_phase can align preds to row order.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_token_batch)

    # ------------------------------------------------------------------
    # Overfit-a-batch sanity check (run FIRST after any change).
    # ------------------------------------------------------------------
    # TODO: if args.overfit_batch:
    # TODO:     tokens, pad_mask, targets = next(iter(loader))
    # TODO:     tokens, pad_mask, targets = tokens.to(args.device), pad_mask.to(args.device), targets.to(args.device)
    # TODO:     for step in range(args.steps):
    # TODO:         pred = model(tokens, pad_mask)
    # TODO:         loss = loss_fn(pred, targets)
    # TODO:         optimizer.zero_grad(); loss.backward(); optimizer.step()
    # TODO:         if step % 20 == 0:
    # TODO:             print(f"  step={step:4d}  loss={loss.item():.6f}")
    # TODO:             if use_wandb: wandb.log({"overfit/loss": loss.item(), "overfit/step": step})
    # TODO:     if use_wandb: wandb.finish()
    # TODO:     return

    # ------------------------------------------------------------------
    # Normal training: epochs x batches. Same 4-line core as the MLP, but
    # pred = model(tokens, pad_mask). Log train loss every --log-every, eval
    # on val each epoch, run the phase breakdown at the end.
    # ------------------------------------------------------------------
    # TODO: global_step = 0
    # TODO: for epoch in range(args.epochs):
    # TODO:     running_loss, n_batches = 0.0, 0
    # TODO:     for tokens, pad_mask, targets in loader:
    # TODO:         tokens, pad_mask, targets = tokens.to(args.device), pad_mask.to(args.device), targets.to(args.device)
    # TODO:         pred = model(tokens, pad_mask)
    # TODO:         loss = loss_fn(pred, targets)
    # TODO:         optimizer.zero_grad(); loss.backward(); optimizer.step()
    # TODO:         running_loss += loss.item(); n_batches += 1; global_step += 1
    # TODO:         if global_step % args.log_every == 0:
    # TODO:             avg = running_loss / n_batches
    # TODO:             print(f"epoch={epoch}  step={global_step}  train_loss={avg:.4f}")
    # TODO:             if use_wandb: wandb.log({"train/loss": avg, "epoch": epoch}, step=global_step)
    # TODO:             running_loss, n_batches = 0.0, 0
    # TODO:     val_loss = evaluate(model, val_loader, loss_fn, args.device)
    # TODO:     print(f"epoch={epoch}  VAL  val_loss={val_loss:.4f}")
    # TODO:     if use_wandb: wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
    #
    # TODO: phase_mse = evaluate_by_phase(model, val_ds, val_loader, args.device)
    # TODO: if use_wandb:
    # TODO:     wandb.log({f"val_phase/{k}": v for k, v in phase_mse.items()}, step=global_step)
    # TODO:     wandb.finish()


if __name__ == "__main__":
    main()
