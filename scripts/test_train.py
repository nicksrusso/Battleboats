"""Train/test pipeline for V_θ linear regression from harvested self-play data.

Loads JSONL output(s) from `scripts/harvest.py`, performs a **game-level**
train/test split (rows from the same game never cross the train/test boundary
— see the docstring on `game_level_split` for why row-level splits leak
in-game state correlation), then fits a linear value function using
scikit-learn's OLS.

CLI:
    poetry run python scripts/test_train.py runs/harvests/harvest_<ts>.jsonl
    poetry run python scripts/test_train.py harvest_a.jsonl harvest_b.jsonl \\
        --test-frac 0.1 --seed 42 --output runs/weights/v1.json

Library use:
    from test_train import (
        load_rows, feature_keys_from, game_level_split,
        to_xy, fit_linear_v, weights_dict,
    )
    rows = load_rows(["runs/harvests/harvest_<ts>.jsonl"])
    fkeys = feature_keys_from(rows)
    train, test = game_level_split(rows, test_frac=0.1, seed=42)
    X_tr, y_tr = to_xy(train, fkeys)
    X_te, y_te = to_xy(test, fkeys)
    model, metrics = fit_linear_v(X_tr, y_tr, X_te, y_te)
    print(metrics)
    print(weights_dict(model, fkeys))
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default regression feature set: the 7 hand-designed heuristic features plus
# combat_coverage_min. These are the features that carry independent signal —
# the 16 per-type ship counts are perfectly collinear with material_diff (it's
# their weighted sum), and combat_total_overmatch / combat_uncovered_count are
# redundant with combat_balance + coverage_min. See VIF analysis. Override with
# --features if you want to refit a different set.
DEFAULT_FEATURES: List[str] = [
    "material_diff",
    "home_pressure_diff",
    "combat_balance",
    "econ_value_self",
    "merchant_count_value_self",
    "has_landing_self",
    "landing_pressure_self",
    "landing_danger_self",
    "combat_coverage_min",
]


# --------------------------------------------------------------------- loading
def load_rows(jsonl_paths: List[Path], decisive_only: bool = True) -> List[Dict[str, Any]]:
    """Load harvest rows into a flat list, normalizing the regression target.

    Accepts JSONL files and/or directories (a directory is expanded to its
    ``*.jsonl`` shards — pass a harvest run dir directly).

    Handles both harvest formats:
      - sharded self-play: each shard ends with a ``_type=game_footer`` line
        carrying ``winner``; data rows carry ``mcts_root_value`` (null on the
        non-acting perspective) and ``phi``.
      - legacy single-file: every data row carries its own scalar ``target``.

    Each kept row gets a normalized ``target`` = ``mcts_root_value`` (preferred,
    the search's value estimate at that state) falling back to legacy ``target``.
    Rows with no usable value (e.g. the non-acting perspective) are dropped.

    decisive_only (default True): drop every row belonging to a TRUNCATED game
    (footer ``winner is None``). Tuning the guidance heuristic on stalemate-heavy
    data biases the fit toward states that never resolve; decisive games give a
    value signal anchored to actual wins. No-op (with a warning) on legacy files
    that have no footers.
    """
    files: List[Path] = []
    for p in jsonl_paths:
        p = Path(p)
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])

    raw: List[Dict[str, Any]] = []
    decisive_games: set = set()        # (seed, game_idx) with a real winner
    has_footer = False
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("_type") == "game_footer":
                    has_footer = True
                    if r.get("winner") is not None:
                        decisive_games.add((r.get("seed"), r.get("game_idx")))
                    continue
                raw.append(r)

    rows: List[Dict[str, Any]] = []
    all_games: set = set()
    for r in raw:
        if "phi" not in r:
            continue
        val = r.get("mcts_root_value")
        if val is None:
            val = r.get("target")
        if val is None:
            continue  # non-acting perspective / unlabeled row
        key = (r.get("seed"), r.get("game_idx"))
        all_games.add(key)
        if decisive_only and has_footer and key not in decisive_games:
            continue  # truncated game
        rows.append({**r, "target": val})

    if decisive_only and not has_footer:
        print("  WARNING: --decisive-only requested but no game_footer lines found "
              "(legacy format?) — keeping all rows.")
    elif decisive_only and has_footer:
        kept = all_games & decisive_games
        print(f"  decisive-only: kept {len(kept)} decisive games, "
              f"dropped {len(all_games) - len(kept)} truncated "
              f"(of {len(all_games)} labeled games).")
    return rows


def feature_keys_from(rows: List[Dict[str, Any]]) -> List[str]:
    """Canonical feature ordering, derived from the first row's phi.

    Python dicts preserve insertion order, and `heuristics.features()` always
    emits the same key order, so this gives a stable column layout for the
    design matrix. If `harvest.py` is later changed to emit different
    feature sets across rows, this function will need a stricter check.
    """
    if not rows:
        raise ValueError("Cannot extract feature keys from an empty row list.")
    return list(rows[0]["phi"].keys())


# ------------------------------------------------------------ game-level split
def game_level_split(
    rows: List[Dict[str, Any]],
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows by GAME, not by row.

    Why this matters: consecutive states within one game differ by one
    action and share the same MC target — they're nearly-duplicate samples
    from the regression's POV. If you split row-wise, rows from the same
    game appear in both train and test, and the model gets to "peek" at
    almost-identical states in the test set. Held-out R² becomes wildly
    optimistic.

    Game identity here is (seed, game_idx) — unique across the harvest
    when combining multiple JSONLs from different seed starting points.
    """
    games: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games[(row["seed"], row["game_idx"])].append(row)

    keys = list(games.keys())
    np.random.default_rng(seed).shuffle(keys)

    n_test = max(1, int(round(test_frac * len(keys))))
    test_keys = set(keys[:n_test])

    train_rows = [r for k, gs in games.items() if k not in test_keys for r in gs]
    test_rows = [r for k in test_keys for r in games[k]]
    return train_rows, test_rows


# ---------------------------------------------------------- design matrix prep
def to_xy(
    rows: List[Dict[str, Any]],
    feature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack rows into a (X, y) numpy pair for sklearn.

    X has shape (n_rows, n_features) with columns ordered by `feature_keys`.
    y has shape (n_rows,) with the MC return target.
    """
    X = np.array([[row["phi"][k] for k in feature_keys] for row in rows], dtype=np.float64)
    y = np.array([row["target"] for row in rows], dtype=np.float64)
    return X, y


# --------------------------------------------------------------------- fitting
def fit_linear_v(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[LinearRegression, Dict[str, Any]]:
    """Fit V_θ(s) = w · φ(s) + b via sklearn's OLS.

    Returns the trained model plus a metrics dict with train/test R²,
    MSE, and MAE. The intercept `b` is included as `model.intercept_`.
    For integration back into `heuristic_eval`, you can either add `b`
    as a constant inside tanh or set `fit_intercept=False` (slightly
    worse fit, simpler integration with the existing zero-intercept
    heuristic).
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "test_mse": float(mean_squared_error(y_test, y_pred_test)),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
    }
    return model, metrics


def weights_dict(model: LinearRegression, feature_keys: List[str]) -> Dict[str, float]:
    """Return learned coefficients as `{feature_name: weight}` for direct
    transcription back into `heuristics.DEFAULT_WEIGHTS`.
    """
    return {k: float(w) for k, w in zip(feature_keys, model.coef_)}


# -------------------------------------------------------------------- CLI main
def _print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"  n_train_rows  : {metrics['n_train_rows']}")
    print(f"  n_test_rows   : {metrics['n_test_rows']}")
    print(f"  train_r2      : {metrics['train_r2']:+.4f}")
    print(f"  test_r2       : {metrics['test_r2']:+.4f}  <-- the honest number")
    print(f"  train_mse     : {metrics['train_mse']:.4f}")
    print(f"  test_mse      : {metrics['test_mse']:.4f}")
    print(f"  train_mae     : {metrics['train_mae']:.4f}")
    print(f"  test_mae      : {metrics['test_mae']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit V_θ linear regression from harvest JSONL(s).")
    parser.add_argument(
        "jsonl_paths",
        type=Path,
        nargs="+",
        help="Harvest JSONL file(s) and/or run director(ies). Concatenated before split.",
    )
    parser.add_argument(
        "--include-truncated",
        action="store_true",
        help="Include truncated games (winner=None). Default: train on decisive games only.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=f"Feature keys to fit on. Default: the 8 non-collinear features {DEFAULT_FEATURES}. "
             "Pass 'all' to use every key in phi (collinear; not recommended).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of GAMES (not rows) to hold out for the test set. Default 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/test split — keep fixed across iterations for honest comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write learned weights + metrics to this JSON path for use by the orchestrator / heuristics.py.",
    )
    args = parser.parse_args()

    rows = load_rows(args.jsonl_paths, decisive_only=not args.include_truncated)
    if not rows:
        raise SystemExit("No rows loaded — check your JSONL paths.")
    print(f"Loaded {len(rows)} rows from {len(args.jsonl_paths)} path(s).")

    available = feature_keys_from(rows)
    if args.features == ["all"]:
        fkeys = available
    else:
        fkeys = args.features or DEFAULT_FEATURES
        missing = [k for k in fkeys if k not in available]
        if missing:
            raise SystemExit(f"Requested features not present in phi: {missing}\nAvailable: {available}")
    print(f"Features ({len(fkeys)}): {fkeys}")

    train_rows, test_rows = game_level_split(rows, test_frac=args.test_frac, seed=args.seed)
    n_train_games = len({(r["seed"], r["game_idx"]) for r in train_rows})
    n_test_games = len({(r["seed"], r["game_idx"]) for r in test_rows})
    print(
        f"Game-level split: train={n_train_games} games / {len(train_rows)} rows  "
        f"test={n_test_games} games / {len(test_rows)} rows"
    )

    X_tr, y_tr = to_xy(train_rows, fkeys)
    X_te, y_te = to_xy(test_rows, fkeys)
    model, metrics = fit_linear_v(X_tr, y_tr, X_te, y_te)

    print()
    print("Metrics:")
    _print_metrics(metrics)

    weights = weights_dict(model, fkeys)
    intercept = float(model.intercept_)

    print()
    print("Learned weights:")
    for k, v in weights.items():
        print(f"  {k:30s}: {v:+.6e}")
    print(f"  {'(intercept)':30s}: {intercept:+.6e}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": weights,
            "intercept": intercept,
            "metrics": metrics,
            "feature_keys": fkeys,
            "source_files": [str(p) for p in args.jsonl_paths],
            "split": {"test_frac": args.test_frac, "seed": args.seed},
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWeights + metrics written to {args.output}")


if __name__ == "__main__":
    main()
