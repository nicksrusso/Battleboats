"""Held-out evaluation for the value net: metrics, baselines, gate (docs/task.md §1.2)."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from battleboats.training.value_data import (
    ValueCache,
    ValueDataset,
    calibration_curve,
    collate_value,
    features_for_rows,
    phase_metrics,
    value_metrics,
)


def resolve_device(spec: str = "cpu") -> torch.device:
    """Map a config string to a torch device.

    ``cpu`` stays on CPU. ``cuda`` / ``cuda:N`` / ``auto`` use the GPU when
    ``torch.cuda.is_available()`` (NVIDIA CUDA or AMD ROCm). If a GPU was
    requested but is missing, fall back to CPU and print a note.
    """
    spec = str(spec or "cpu").strip().lower()
    want = spec in ("auto", "cuda") or spec.startswith("cuda:")
    if not want:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda" if spec in ("auto", "cuda") else spec)
    if spec != "auto":
        print(f"[device] {spec} requested but torch.cuda.is_available() is False; using cpu", flush=True)
    return torch.device("cpu")


def move_batch(batch: dict, device: Union[str, torch.device]) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(model, ds: ValueDataset, batch_size: int = 256, device=None) -> Dict[str, np.ndarray]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)
    vs, ys, fr, seats, rows = [], [], [], [], []
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_value):
        b = move_batch(b, device)
        v = model(b["tokens"], b["pad_mask"], b["globals"]).detach().cpu()
        vs.append(v.numpy())
        ys.append(b["target"].cpu().numpy())
        fr.append(b["turn_frac"].cpu().numpy())
        seats.append(b["seat"].cpu().numpy())
        rows.append(b["row"].cpu().numpy())
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    return {"v": cat(vs), "y": cat(ys), "turn_frac": cat(fr), "seat": cat(seats), "row": cat(rows)}


def seat_bias(v: np.ndarray, y: np.ndarray, seat: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    d = y != 0
    for s in (0, 1):
        m = d & (seat == s)
        if m.any():
            out[str(s)] = {"mean_pred_winprob": float(((v[m] + 1) / 2).mean()), "win_rate": float((y[m] > 0).mean()), "n": int(m.sum())}
    return out


class Baselines:
    """Fit on train rows; predict value in [-1,1] for any rows."""

    def __init__(self, cache: ValueCache, train_rows: np.ndarray, max_fit_rows: int = 60000, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.cache = cache
        rng = np.random.default_rng(seed)
        rows = train_rows[cache.outcome[train_rows] != 0]
        if len(rows) > max_fit_rows:
            rows = np.sort(rng.choice(rows, size=max_fit_rows, replace=False))
        y = (cache.outcome[rows] > 0).astype(int)
        seat = cache.seat[rows]
        self.prior = float(y.mean()) if len(y) else 0.5
        self.seat_prior = {s: float(y[seat == s].mean()) if (seat == s).any() else self.prior for s in (0, 1)}
        self.models = {}
        self.n_fit = int(len(rows))
        if len(np.unique(y)) == 2:
            for kind in ("simple", "rich"):
                X = features_for_rows(cache, rows, kind)
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                m.fit(X, y)
                self.models[kind] = m

    def predict_all(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        seat = self.cache.seat[rows]
        out = {
            "coin_flip": np.zeros(len(rows)),
            "constant_prior": np.full(len(rows), 2 * self.prior - 1),
            "seat_prior": np.array([2 * self.seat_prior[int(s)] - 1 for s in seat], dtype=np.float64),
        }
        for kind, m in self.models.items():
            p = m.predict_proba(features_for_rows(self.cache, rows, kind))[:, 1] if len(rows) else np.zeros(0)
            out[f"logistic_{kind}"] = 2 * p - 1
        return out


def evaluate_split(
    model,
    cache: ValueCache,
    split: str,
    baselines: Optional[Baselines] = None,
    include_draws: bool = False,
    batch_size: int = 256,
    gate_cfg: Optional[Dict] = None,
    device=None,
) -> Dict:
    ds = ValueDataset(cache, split, include_draws=include_draws, k_per_game=0)
    pred = predict(model, ds, batch_size=batch_size, device=device)
    v, y = pred["v"], pred["y"]
    res: Dict = {
        "split": split,
        "games": ds.n_games,
        "net": value_metrics(v, y),
        "net_by_phase": phase_metrics(v, y, pred["turn_frac"]),
        "calibration": calibration_curve(v, y),
        "seat_bias": seat_bias(v, y, pred["seat"]),
    }
    if baselines is not None:
        bp = baselines.predict_all(pred["row"].astype(np.int64))
        res["baselines"] = {k: value_metrics(bv, y) for k, bv in bp.items()}
        res["baselines_by_phase"] = {k: phase_metrics(bv, y, pred["turn_frac"]) for k, bv in bp.items()}
        res["gate"] = gate(res["net"], res["baselines"], gate_cfg or {})
    return res


TASK_GATE_BASELINES = ("coin_flip", "constant_prior", "seat_prior", "logistic_simple")


def gate(net: Dict, baselines: Dict[str, Dict], cfg: Dict) -> Dict:
    """task.md §1.2 gate: net must beat coin flip AND the simple baselines on held-out
    decisive games (``cfg['gate_baselines']``, default TASK_GATE_BASELINES). The richer
    hand-feature logistic is reported as ``beats_rich_logistic`` (informational).

    Pass = sign_acc > 0.5 + min_acc_margin over coin, sign_acc >= best baseline acc,
    Brier <= best baseline Brier - brier_margin, logloss < best baseline logloss.
    """
    acc_margin = float(cfg.get("min_acc_margin", 0.02))
    brier_margin = float(cfg.get("brier_margin", 0.005))
    if "sign_acc" not in net:
        return {"pass": False, "reason": "no decisive rows"}
    names = [n for n in cfg.get("gate_baselines", TASK_GATE_BASELINES) if n in baselines] or list(baselines)
    rich = baselines.get("logistic_rich")
    baselines = {n: baselines[n] for n in names}
    best_acc = max(b.get("sign_acc", 0.0) for b in baselines.values())
    best_brier = min(b.get("brier", 1.0) for b in baselines.values())
    best_ll = min(b.get("logloss", 9.9) for b in baselines.values())
    checks = {
        "acc_vs_coin": net["sign_acc"] > 0.5 + acc_margin,
        "brier_vs_coin": net["brier"] < 0.25 - brier_margin,
        "acc_vs_best_baseline": net["sign_acc"] >= best_acc,
        "brier_vs_best_baseline": net["brier"] <= best_brier - brier_margin,
        "logloss_vs_best_baseline": net["logloss"] < best_ll,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "best_baseline_acc": best_acc,
        "best_baseline_brier": best_brier,
        "best_baseline_logloss": best_ll,
        "gate_baselines": names,
        "beats_rich_logistic": None if rich is None else bool(net["brier"] < rich.get("brier", 1.0) and net["sign_acc"] >= rich.get("sign_acc", 0.0)),
    }


def format_report(res: Dict) -> str:
    lines = [f"== {res['split']} ({res['games']} games, {res['net'].get('n_decisive', 0)} decisive rows) =="]
    hdr = f"{'model':<18}{'MSE':>8}{'acc':>8}{'Brier':>8}{'logloss':>9}"
    lines.append(hdr)
    rows = [("value_net", res["net"])] + list(res.get("baselines", {}).items())
    for name, m in rows:
        lines.append(
            f"{name:<18}{m.get('mse', float('nan')):>8.4f}{m.get('sign_acc', float('nan')):>8.3f}"
            f"{m.get('brier', float('nan')):>8.4f}{m.get('logloss', float('nan')):>9.4f}"
        )
    lines.append("acc by phase (turn fraction):  " + "  ".join(f"{p['phase']}:{p.get('sign_acc', float('nan')):.3f}" for p in res["net_by_phase"]))
    if "baselines_by_phase" in res and "logistic_simple" in res["baselines_by_phase"]:
        lines.append("  logistic_simple by phase:    " + "  ".join(f"{p['phase']}:{p.get('sign_acc', float('nan')):.3f}" for p in res["baselines_by_phase"]["logistic_simple"]))
    if "gate" in res:
        gt = res["gate"]
        lines.append(f"GATE (task.md, vs {gt.get('gate_baselines')}): {'PASS' if gt['pass'] else 'FAIL'}  {gt.get('checks')}")
        lines.append(f"beats logistic_rich: {gt.get('beats_rich_logistic')}")
    return "\n".join(lines)
