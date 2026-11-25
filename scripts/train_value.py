#!/usr/bin/env python3
"""Train the v0.5.0 value net on a value-harvest cache (CPU).

    poetry run python scripts/train_value.py --config configs/value_net_v0.5.0.yaml
    poetry run python scripts/train_value.py --resume runs/value_net_v0.5.0/<timestamp>

Writes runs/value_net_v0.5.0/<timestamp>/: config.yaml, metrics.csv (per epoch),
train_log.csv (every log_every steps), last.pt, best.pt (lowest val Brier),
baselines_val.json, eval_test.json + eval_test.txt at the end. TensorBoard
events too if tensorboard is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from battleboats.training.value_data import ValueCache, ValueDataset, build_value_cache, collate_value, value_metrics  # noqa: E402
from battleboats.training.value_eval import Baselines, evaluate_split, format_report, predict  # noqa: E402
from battleboats.training.value_net import ValueNet, count_params, save_value_checkpoint  # noqa: E402

try:  # tensorboard is optional (not in the poetry env today)
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None


def _set(cfg, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    old = d.get(keys[-1])
    if isinstance(old, bool):
        value = str(value).lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        value = int(value)
    elif isinstance(old, float):
        value = float(value)
    d[keys[-1]] = value


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/value_net_v0.5.0.yaml")
    ap.add_argument("--resume", help="run dir to resume from (uses its config.yaml + last.pt)")
    ap.add_argument("--run-dir", help="explicit output dir (default out_root/<timestamp>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override config, e.g. train.epochs=2")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop training after this wall time (0 = off)")
    ap.add_argument("--skip-test", action="store_true", help="don't run the final test-split eval")
    a = ap.parse_args(argv)

    if a.resume:
        run_dir = Path(a.resume)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    else:
        cfg = yaml.safe_load(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        _set(cfg, k, v)
    if not a.resume:
        stamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(a.run_dir) if a.run_dir else Path(cfg.get("out_root", "runs/value_net_v0.5.0")) / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dc, mc, tc, ec = cfg["data"], cfg["model"], cfg["train"], cfg.get("eval", {})
    torch.set_num_threads(int(tc.get("torch_threads", 4)))
    torch.manual_seed(int(tc.get("seed", 0)))

    cache_dir = Path(dc["cache_dir"])
    if not (cache_dir / "cache_meta.json").exists():
        print(f"[data] building cache {cache_dir} from {dc['harvest_dir']}", flush=True)
        build_value_cache(dc["harvest_dir"], cache_dir, k_pool=dc["k_pool"], seed=dc.get("seed", 0),
                          val_frac=dc["val_frac"], test_frac=dc["test_frac"], workers=dc.get("cache_workers", 2))
    cache = ValueCache(cache_dir)
    inc = bool(dc.get("include_draws", False))
    train_ds = ValueDataset(cache, "train", include_draws=inc, k_per_game=int(dc["k_per_game"]), seed=int(dc.get("seed", 0)))
    val_ds = ValueDataset(cache, "val", include_draws=inc, k_per_game=0)
    print(f"[data] cache {cache.meta['games']} games / {len(cache):,} pooled rows; split games {cache.meta['split_games']}; "
          f"train {train_ds.n_games} games x<= {dc['k_per_game']} = {len(train_ds):,} rows/epoch; val {val_ds.n_games} games / {len(val_ds):,} rows", flush=True)

    model = ValueNet(**mc)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    steps_per_epoch = math.ceil(len(train_ds) / int(tc["batch_size"]))
    if int(tc.get("max_steps_per_epoch", 0)) > 0:
        steps_per_epoch = min(steps_per_epoch, int(tc["max_steps_per_epoch"]))
    total_steps = steps_per_epoch * int(tc["epochs"])
    start_epoch, step, best = 0, 0, {"val_brier": float("inf"), "epoch": -1}
    if a.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        print(f"[resume] from epoch {ck['epoch']} step {step}; best {best}", flush=True)
    print(f"[model] ValueNet {n_params:,} params {mc}; {steps_per_epoch} steps/epoch x {tc['epochs']} epochs; threads {torch.get_num_threads()}", flush=True)

    baselines = None
    bl_path = run_dir / "baselines_val.json"
    t_bl = time.time()
    baselines = Baselines(cache, cache.split_rows("train", include_draws=False), max_fit_rows=int(ec.get("baseline_fit_rows", 60000)), seed=int(dc.get("seed", 0)))
    if len(val_ds) and not bl_path.exists():
        vb = baselines.predict_all(val_ds.rows)
        bl = {k: value_metrics(v, cache.outcome[val_ds.rows]) for k, v in vb.items()}
        bl_path.write_text(json.dumps(bl, indent=2))
        print(f"[baselines] fit on {baselines.n_fit:,} train rows in {time.time()-t_bl:.1f}s; val: " +
              "; ".join(f"{k} acc {m.get('sign_acc', float('nan')):.3f} brier {m.get('brier', float('nan')):.4f}" for k, m in bl.items()), flush=True)

    tb = SummaryWriter(str(run_dir / "tb")) if SummaryWriter is not None else None
    aux_w = float(tc.get("aux_weight", 0.0))
    t_start = time.time()
    stopped_early = False
    for epoch in range(start_epoch, int(tc["epochs"])):
        train_ds.set_epoch(epoch)
        g = torch.Generator().manual_seed(int(tc.get("seed", 0)) * 1000 + epoch)
        dl = DataLoader(train_ds, batch_size=int(tc["batch_size"]), shuffle=True, generator=g, collate_fn=collate_value,
                        num_workers=int(tc.get("num_workers", 0)), persistent_workers=False)
        model.train()
        t_ep = time.time()
        sums = {"loss": 0.0, "mse": 0.0, "aux": 0.0, "n": 0}
        for i, b in enumerate(dl):
            if i >= steps_per_epoch:
                break
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total_steps, float(tc["lr"]), int(tc.get("warmup_steps", 0)))
            v, aux = model.forward_all(b["tokens"], b["pad_mask"], b["globals"])
            mse = torch.mean((v - b["target"]) ** 2)
            loss = mse
            aux_l = torch.zeros(())
            if aux is not None and aux_w > 0:
                aux_l = torch.mean((aux - b["returns"]) ** 2)
                loss = loss + aux_w * aux_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(tc.get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            opt.step()
            n = len(b["target"])
            sums["loss"] += loss.item() * n
            sums["mse"] += mse.item() * n
            sums["aux"] += aux_l.item() * n
            sums["n"] += n
            step += 1
            if int(tc.get("log_every", 0)) and step % int(tc["log_every"]) == 0:
                rate = (i + 1) / max(1e-9, time.time() - t_ep)
                row = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"], "loss": loss.item(), "mse": mse.item(), "aux": aux_l.item(), "steps_per_s": round(rate, 2)}
                append_csv(run_dir / "train_log.csv", row)
                print(f"  ep {epoch} step {step} ({i+1}/{steps_per_epoch}) loss {loss.item():.4f} mse {mse.item():.4f} {rate:.2f} it/s", flush=True)
                if tb:
                    tb.add_scalar("train/loss", loss.item(), step)
            if a.max_minutes and (time.time() - t_start) > a.max_minutes * 60:
                stopped_early = True
                break
        ep_s = time.time() - t_ep
        t_ev = time.time()
        vm = {}
        if len(val_ds):
            pv = predict(model, val_ds, int(tc.get("eval_batch_size", 512)))
            vm = value_metrics(pv["v"], pv["y"])
        ev_s = time.time() - t_ev
        n = max(1, sums["n"])
        row = {"epoch": epoch, "step": step, "train_loss": sums["loss"] / n, "train_mse": sums["mse"] / n, "train_aux": sums["aux"] / n,
               "val_mse": vm.get("mse"), "val_acc": vm.get("sign_acc"), "val_brier": vm.get("brier"), "val_logloss": vm.get("logloss"),
               "epoch_s": round(ep_s, 1), "eval_s": round(ev_s, 1), "time": datetime.now(ZoneInfo("America/Chicago")).strftime("%H:%M:%S %Z")}
        append_csv(run_dir / "metrics.csv", row)
        if tb:
            for k in ("train_mse", "val_mse", "val_acc", "val_brier", "val_logloss"):
                if row[k] is not None:
                    tb.add_scalar(k, row[k], epoch)
        print(f"[epoch {epoch}] train mse {row['train_mse']:.4f} | val mse {vm.get('mse', float('nan')):.4f} acc {vm.get('sign_acc', float('nan')):.3f} "
              f"brier {vm.get('brier', float('nan')):.4f} logloss {vm.get('logloss', float('nan')):.4f} | {ep_s:.0f}s train + {ev_s:.0f}s eval", flush=True)
        if vm.get("brier", float("inf")) < best["val_brier"]:
            best = {"val_brier": vm["brier"], "epoch": epoch, "val": vm}
            save_value_checkpoint(run_dir / "best.pt", model, epoch=epoch, step=step, val=vm, n_params=n_params)
        save_value_checkpoint(run_dir / "last.pt", model, optimizer=opt.state_dict(), epoch=epoch, step=step, best=best, n_params=n_params)
        if stopped_early:
            print(f"[stop] --max-minutes {a.max_minutes} reached", flush=True)
            break

    summary = {"run_dir": str(run_dir), "n_params": n_params, "best": best, "steps": step, "train_minutes": round((time.time() - t_start) / 60, 2)}
    if not a.skip_test:
        from battleboats.training.value_net import load_value_checkpoint

        ck_path = run_dir / ("best.pt" if (run_dir / "best.pt").exists() else "last.pt")
        best_model, _ = load_value_checkpoint(ck_path)
        for split in ("val", "test"):
            res = evaluate_split(best_model, cache, split, baselines=baselines, include_draws=False,
                                 batch_size=int(tc.get("eval_batch_size", 512)), gate_cfg=ec)
            (run_dir / f"eval_{split}.json").write_text(json.dumps(res, indent=2))
            txt = format_report(res)
            (run_dir / f"eval_{split}.txt").write_text(txt + "\n")
            print(txt, flush=True)
            summary[f"{split}_gate"] = res.get("gate", {}).get("pass")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] {json.dumps(summary, default=str)}", flush=True)
    if tb:
        tb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
