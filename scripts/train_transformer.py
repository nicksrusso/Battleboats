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
import time
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

    model.eval()
    total_loss = 0.0
    n_batches = 0
    for tokens, pad_mask, targets in loader:
        tokens = tokens.to(device)
        pad_mask = pad_mask.to(device)
        targets = targets.to(device)
        pred = model(tokens, pad_mask)
        total_loss += loss_fn(pred, targets).item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_by_phase(model, ds, loader, device, n_bins: int = 3) -> dict:
    """Val MSE split by game phase (early/mid/late). Same diagnostic as the MLP.

    Gather predictions by iterating `loader` (MUST be shuffle=False) so the
    concatenated order matches ds row order, then line each squared error up
    with ds.steps / ds.game_ids for phase bucketing. See [[project-value-target]].
    """
    model.eval()
    preds = []
    for tokens, pad_mask, _ in loader:
        # .cpu() before the cat: model output lives on `device` (cuda), and
        # .numpy() on a cuda tensor throws. Loader MUST be shuffle=False so this
        # concatenation lines up row-for-row with ds.targets / ds.steps.
        preds.append(model(tokens.to(device), pad_mask.to(device)).cpu())
    pred = torch.cat(preds).numpy()
    sq_err = (pred - ds.targets) ** 2

    # step -> progress fraction in [0, 1] within each game, so games of
    # different lengths bucket comparably.
    steps = ds.steps.astype(np.float64)
    max_step = np.zeros_like(steps)
    for gid in np.unique(ds.game_ids):
        mask = ds.game_ids == gid
        m = steps[mask].max()
        max_step[mask] = m if m > 0 else 1.0
    progress = steps / max_step

    out = {}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    print("phase-stratified val MSE (lower = signal present):")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin inclusive of 1.0; others half-open so no row is double-counted.
        in_bin = (progress >= lo) & (progress <= hi) if i == n_bins - 1 else (progress >= lo) & (progress < hi)
        n = int(in_bin.sum())
        mse = float(sq_err[in_bin].mean()) if n else float("nan")
        label = ("early", "mid", "late")[i] if n_bins == 3 else f"bin{i}"
        out[label] = mse
        print(f"  {label:5s}  progress[{lo:.2f},{hi:.2f}]  n={n:6d}  mse={mse:.4f}")
    model.train()
    return out


def save_checkpoint(model, args, metrics: dict, save_dir: Path) -> Path:
    """Persist the trained net to disk and return the checkpoint path.

    STUB — fill the body. Save ONE self-describing dict via torch.save, so the
    file alone is enough to rebuild and evaluate the model later:
      - "model_state":  model.state_dict()   — the actual weights
      - "model_config": the kwargs needed to re-instantiate TransformerValueModel
        (token_dim, d_model, nhead, num_layers). state_dict has NO architecture
        info — without this you can't reload.
      - "train_args":   vars(args)           — lr, target, split, harvest, ...
      - "metrics":      metrics              — final val_loss, phase MSE, baselines
    Name the file by timestamp/run (e.g. f"transformer_{ts}.pt"); derive ts from
    time.strftime here. Path.mkdir(parents=True, exist_ok=True) on save_dir first.
    Return the written path so the caller can hand it to log_checkpoint_artifact.
    """
    raise NotImplementedError


def log_checkpoint_artifact(ckpt_path: Path, metrics: dict, args) -> None:
    """Upload the checkpoint to W&B as a versioned Artifact (lineage to this run).

    STUB — fill the body. Only reached when --wandb is on (wandb.init already ran).
    Pattern:
        art = wandb.Artifact(name="transformer-value", type="model", metadata={...})
        art.add_file(str(ckpt_path))
        wandb.log_artifact(art)
    Use a STABLE name so successive runs become v0, v1, v2... of the SAME artifact
    rather than unrelated blobs — that versioning + run-lineage is the whole point.
    Put final val_loss / phase MSE in metadata so the W&B UI shows the best version.

    HuggingFace alternative (final PUBLISH step, not every run): huggingface_hub
    upload_file / push_to_hub the checkpoint + a short model card. Leave it out
    until there's a finished agent worth sharing — HF is a registry, W&B Artifacts
    is the experiment-lineage store. See [[project-value-target]].
    """
    raise NotImplementedError


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
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/checkpoints"),
        help="Directory to torch.save the trained checkpoint into (TransformerValueModel weights + config).",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--wandb-entity", default="nicksrusso", help="W&B entity (user/team).")
    parser.add_argument("--wandb-project", default="battleboats-data")
    args = parser.parse_args()

    use_wandb = args.wandb

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
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))
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
    if args.overfit_batch:
        tokens, pad_mask, targets = next(iter(loader))
        tokens = tokens.to(args.device)
        pad_mask = pad_mask.to(args.device)
        targets = targets.to(args.device)
        t0 = time.perf_counter()
        for step in range(args.steps):
            pred = model(tokens, pad_mask)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    step{step:4d} loss={loss.item():.6f}  t={elapsed:6.1f}s")
                if use_wandb:
                    wandb.log({"overfit/loss": loss.item(), "overfit/step": step})
        if use_wandb:
            wandb.finish()
        return

    # ------------------------------------------------------------------
    # Normal training: epochs x batches. Same 4-line core as the MLP, but
    # pred = model(tokens, pad_mask). Log train loss every --log-every, eval
    # on val each epoch, run the phase breakdown at the end.
    # ------------------------------------------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        n_batches = 0
        for (
            tokens,
            pad_mask,
            targets,
        ) in loader:
            tokens = tokens.to(args.device)
            pad_mask = pad_mask.to(args.device)
            targets = targets.to(args.device)
            pred = model(tokens, pad_mask)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % args.log_every == 0:
                avg = running_loss / n_batches
                print(f"epoch={epoch}  step={global_step}  train_loss={avg:.4f}")
                if use_wandb:
                    wandb.log({"train/loss": avg, "epoch": epoch}, step=global_step)
                running_loss = 0.0
                n_batches = 0
        val_loss = evaluate(model, val_loader, loss_fn, args.device)
        print(f"epoch={epoch}  VAL  val_loss={val_loss:.4f}")
        if use_wandb:
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

    phase_mse = evaluate_by_phase(model, val_ds, val_loader, args.device)
    if use_wandb:
        wandb.log({f"val_phase/{k}": v for k, v in phase_mse.items()}, step=global_step)

    # --- Persist weights (fill save_checkpoint / log_checkpoint_artifact above) ---
    metrics = {
        "val_loss": val_loss,
        "baseline_val_mse": baseline_val_mse,
        "val_label_var": val_label_var,
        **{f"phase_{k}": v for k, v in phase_mse.items()},
    }
    ckpt_path = save_checkpoint(model, args, metrics, args.save_dir)
    print(f"saved checkpoint -> {ckpt_path}")
    if use_wandb:
        log_checkpoint_artifact(ckpt_path, metrics, args)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
