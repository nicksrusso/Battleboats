"""Smallest-thing-first training loop for the trivial MLP.

Goals for this script:
  - Confirm the full pipeline (Dataset -> DataLoader -> model -> loss ->
    backward -> optimizer step) runs end-to-end without errors.
  - Print training loss over time so you can eyeball "is it going down."
  - Support an --overfit-batch mode that trains on a single batch for K
    steps; loss should drop to near zero. If it doesn't, something
    fundamental is broken (data shape, label sign, gradient flow, loss).
    Run this FIRST every time you change the model or data pipeline.

Deliberately NOT included yet (will be added in later steps):
  - Validation set evaluation (step 6)
  - W&B logging (step 7)
  - Linear regression baseline comparison (step 8)
  - Checkpointing

Usage:
    poetry run python scripts/train_mlp.py \\
        --harvest runs/harvests/<file>.jsonl \\
        --split   runs/splits/<file>.json
    poetry run python scripts/train_mlp.py \\
        --harvest ... --split ... --overfit-batch --steps 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from battleboats.training.dataset import HarvestDataset
from battleboats.training.model import SimpleMLP, compute_normalization_stats
from battleboats.training.split import load_split


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    """Mean loss over a loader, no gradients. Returns a plain float.

    model.eval() / model.train() toggle inference-vs-training behavior for
    layers like dropout and batchnorm. SimpleMLP has neither yet, so it's a
    no-op here — but it's the correct habit, and the transformer WILL have
    them. The @torch.no_grad() decorator skips building the autograd graph,
    so eval is faster and can't accidentally leak val data into a gradient.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        total_loss += loss_fn(pred, batch_y).item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_by_phase(model, ds, device, n_bins: int = 3) -> None:
    """Print val MSE split by game phase (early/mid/late).

    The whole question: is the ~1.0 val loss the irreducible noise floor of
    an unpredictable target, or is there signal the model is missing? Early
    states genuinely can't predict the eventual winner (≈coin flip → MSE≈1).
    But LATE states are near-decided — if the label carried real positional
    signal, late-state MSE should drop well below 1.0. If late states ALSO
    sit at ≈1.0, the flat terminal target is worthless even where the
    outcome is obvious, which indicts the target, not the model.

    Phase = step normalized to [0, 1] within each game (so games of
    different lengths are comparable), then bucketed into n_bins.
    """
    model.eval()
    x = torch.from_numpy(ds.phi).to(device)
    y = torch.from_numpy(ds.targets).to(device)
    pred = model(x)
    sq_err = ((pred - y) ** 2).cpu().numpy()

    # Per-game max step, to normalize step -> progress fraction in [0, 1].
    steps = ds.steps.astype(np.float64)
    max_step = np.zeros_like(steps)
    for gid in np.unique(ds.game_ids):
        mask = ds.game_ids == gid
        m = steps[mask].max()
        max_step[mask] = m if m > 0 else 1.0  # single-move game -> avoid /0
    progress = steps / max_step  # 0.0 = opening, 1.0 = final move

    print("phase-stratified val MSE (lower = signal present):")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is inclusive of 1.0; others are half-open.
        in_bin = (progress >= lo) & (progress <= hi) if i == n_bins - 1 else (progress >= lo) & (progress < hi)
        n = int(in_bin.sum())
        mse = float(sq_err[in_bin].mean()) if n else float("nan")
        label = ("early", "mid", "late")[i] if n_bins == 3 else f"bin{i}"
        print(f"  {label:5s}  progress[{lo:.2f},{hi:.2f}]  n={n:6d}  mse={mse:.4f}")
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50, help="Print loss every N batches.")
    parser.add_argument(
        "--overfit-batch",
        action="store_true",
        help="Sanity check: grab one batch, train on it for --steps iterations. "
        "Loss should drop to near zero. If it doesn't, something is broken.",
    )
    parser.add_argument("--steps", type=int, default=300, help="Steps to run in --overfit-batch mode.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (or cuda:0 etc).")
    parser.add_argument(
        "--target",
        choices=["target", "mcts_root_value"],
        default="target",
        help="Regression label: 'target' = flat terminal outcome (constant per game); "
        "'mcts_root_value' = search's per-state value estimate (varies within a game).",
    )
    args = parser.parse_args()

    train_ids, val_ids, meta = load_split(args.split)
    train_ds = HarvestDataset(args.harvest, game_idxs=train_ids, target_key=args.target)
    val_ds = HarvestDataset(args.harvest, game_idxs=val_ids, target_key=args.target)
    print(f"target = {args.target}")

    phi_train = torch.from_numpy(train_ds.phi)
    mean, std = compute_normalization_stats(phi_train)

    model = SimpleMLP(in_features=phi_train.shape[1], hidden=args.hidden, mean=mean, std=std)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # ------------------------------------------------------------------
    # Naive baselines — the bar the model must clear to be worth anything.
    # ------------------------------------------------------------------
    # No training involved; these are pure properties of the labels.
    #   (1) DEPLOYABLE baseline: a model with no inputs can still predict
    #       the train-set mean label for every state. Its MSE on val is the
    #       real bar — if the trained model can't beat THIS, it learned
    #       nothing a constant couldn't.
    #   (2) THEORETICAL FLOOR for any constant predictor on val = the
    #       variance of the val labels (achieved by predicting val's own
    #       mean, which you don't get to peek at in practice).
    # If val_loss lands near these, the bottleneck is the data/labels, not
    # the architecture — and swapping in a transformer won't move it.
    train_mean = float(train_ds.targets.mean())
    baseline_val_mse = float(((val_ds.targets - train_mean) ** 2).mean())
    val_label_var = float(val_ds.targets.var())
    print(f"baseline  predict train-mean ({train_mean:+.4f})   val_mse={baseline_val_mse:.4f}")
    print(f"baseline  val label variance (best constant)   val_mse={val_label_var:.4f}")

    # ------------------------------------------------------------------
    # 4. Build DataLoader.
    # ------------------------------------------------------------------
    # shuffle=True is important — without it, every batch is from one
    # game (since rows are contiguous per game) and the gradients are
    # absurdly correlated. num_workers=0 is fine for now; the dataset
    # is in RAM and __getitem__ is cheap.
    # TODO: loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # Val loader: shuffle=False — order is irrelevant for evaluation, and
    # we never backprop through it, so correlated batches don't matter.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 5a. Overfit-a-batch sanity check (skip if not requested).
    # ------------------------------------------------------------------
    # Grab ONE batch out of the loader. Train on just that batch for
    # args.steps iterations. Loss should drop to near zero. The point
    # is to prove the model CAN fit data; if it can't fit 256 examples
    # by overfitting, it definitely can't generalize.
    #
    # TODO: if args.overfit_batch:
    # TODO:     batch_x, batch_y = next(iter(loader))
    # TODO:     batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
    # TODO:     for step in range(args.steps):
    # TODO:         pred = model(batch_x)
    # TODO:         loss = loss_fn(pred, batch_y)
    # TODO:         optimizer.zero_grad()
    # TODO:         loss.backward()
    # TODO:         optimizer.step()
    # TODO:         if step % 20 == 0:
    # TODO:             print(f"  step={step:4d}  loss={loss.item():.6f}")
    # TODO:     return
    if args.overfit_batch:
        batch_x, batch_y = next(iter(loader))
        batch_x = batch_x.to(args.device)
        batch_y = batch_y.to(args.device)
        for step in range(args.steps):
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print(f"  step={step:4d}  loss={loss.item():.6f}")
        return

    # ------------------------------------------------------------------
    # 5b. Normal training: iterate epochs, iterate batches.
    # ------------------------------------------------------------------
    # The four-line core of supervised training. Every pipeline in ML
    # is some elaboration of these four lines:
    #     pred = model(x)
    #     loss = loss_fn(pred, y)
    #     optimizer.zero_grad(); loss.backward(); optimizer.step()
    #
    # Print rolling-mean training loss every --log-every batches so you
    # can see it trend down. A flat or rising loss = something is wrong;
    # stop, don't keep training, debug first.
    #
    # TODO: global_step = 0
    # TODO: for epoch in range(args.epochs):
    # TODO:     running_loss = 0.0
    # TODO:     n_batches = 0
    # TODO:     for batch_x, batch_y in loader:
    # TODO:         batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
    # TODO:         pred = model(batch_x)
    # TODO:         loss = loss_fn(pred, batch_y)
    # TODO:         optimizer.zero_grad()
    # TODO:         loss.backward()
    # TODO:         optimizer.step()
    # TODO:         running_loss += loss.item()
    # TODO:         n_batches += 1
    # TODO:         global_step += 1
    # TODO:         if global_step % args.log_every == 0:
    # TODO:             print(f"epoch={epoch}  step={global_step}  train_loss={running_loss/n_batches:.4f}")
    # TODO:             running_loss, n_batches = 0.0, 0
    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch={epoch}  step={global_step}  train_loss={running_loss/n_batches:.4f}")
                running_loss, n_batches = 0.0, 0

        # End of epoch: evaluate on held-out val games. This is the number
        # that actually matters — train loss only says "did it memorize."
        # If val flattens or rises while train keeps dropping, that gap is
        # overfitting made visible.
        val_loss = evaluate(model, val_loader, loss_fn, args.device)
        print(f"epoch={epoch}  VAL  val_loss={val_loss:.4f}")

    # After the final epoch: where does the error live? If late-game states
    # are no more predictable than openings, the flat terminal target is the
    # bottleneck — not the model.
    evaluate_by_phase(model, val_ds, args.device)


if __name__ == "__main__":
    main()
