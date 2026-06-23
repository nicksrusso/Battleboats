"""Plot PPO self-play training curves for the PPO slide.

Two composed panels from one PPO run (default: the most recent run in the
battleboats-ppo project):

  LEFT  — game outcomes (win / draw / loss rate) over updates: the learning
          curve. Pinned at draw=1.0 / win=0 means no terminal reward ever
          reached the policy (every game truncates to a draw).
  RIGHT — policy entropy (left axis) + approx-KL and clip-fraction (right axis):
          update health. Entropy drifting up while outcomes stay flat = the
          policy keeps changing on noise without ever learning to win.

This is the PPO analog of the BC figure, but for an RL run "loss" is not the
story (the clipped surrogate is re-zeroed each update) — outcomes + stability
are. History is pulled live from the W&B API (read-only).

    poetry run python scripts/plot_ppo_training.py
    poetry run python scripts/plot_ppo_training.py --run nicksrusso/battleboats-ppo/<id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "nicksrusso/battleboats-ppo"


def _resolve_run(api, run_path):
    """An explicit --run wins; otherwise pick the most recently created run."""
    if run_path:
        return api.run(run_path if "/" in run_path else f"{PROJECT}/{run_path}")
    runs = list(api.runs(PROJECT))
    live = [r for r in runs if r.state in ("running", "finished")] or runs
    return max(live, key=lambda r: r.created_at)


def _fetch(run):
    rows = [x for x in run.scan_history() if x.get("update") is not None]
    rows.sort(key=lambda x: x["update"])
    return rows


def _col(rows, key):
    """(updates, values) for one metric, skipping steps where it's absent."""
    xs = [r["update"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="W&B run path or id (default: latest in battleboats-ppo).")
    parser.add_argument("--out", type=Path, default=Path("docs/ppo_training.png"))
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=30)
    run = _resolve_run(api, args.run)
    rows = _fetch(run)
    cmap = plt.get_cmap("tab10")

    fig, (ax_out, ax_stab) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- LEFT: game outcomes over updates (the learning curve) ----
    for key, label, ci in [("rollout/win_rate", "win", 2),
                           ("rollout/draw_rate", "draw", 0),
                           ("rollout/loss_rate", "loss", 3)]:
        xs, ys = _col(rows, key)
        if ys:
            ax_out.plot(xs, ys, color=cmap(ci), lw=2.2, marker="o", ms=3, label=f"{label}  ({ys[-1]:.2f})")
    ax_out.set_ylim(-0.02, 1.02)
    ax_out.set_xlabel("PPO update")
    ax_out.set_ylabel("rollout outcome rate")
    ax_out.set_title("Game Outcomes (win / draw / loss)", fontsize=13, fontweight="bold")
    ax_out.legend(loc="center right", fontsize=10, title="outcome (final)")
    ax_out.grid(True, alpha=0.25)

    # ---- RIGHT: entropy (left axis) + KL & clip-frac (right axis) ----
    xs_e, ys_e = _col(rows, "ppo/entropy")
    ax_stab.plot(xs_e, ys_e, color=cmap(1), lw=2.4, marker="o", ms=3, label="entropy")
    ax_stab.set_xlabel("PPO update")
    ax_stab.set_ylabel("policy entropy", color=cmap(1))
    ax_stab.tick_params(axis="y", labelcolor=cmap(1))
    ax_stab.set_title("Policy Entropy & Update Health", fontsize=13, fontweight="bold")
    ax_stab.grid(True, alpha=0.25)

    ax2 = ax_stab.twinx()
    for key, label, ci in [("ppo/approx_kl", "approx KL", 4), ("ppo/clip_frac", "clip frac", 5)]:
        xs, ys = _col(rows, key)
        if ys:
            ax2.plot(xs, ys, color=cmap(ci), lw=1.6, ls="--", label=label)
    ax2.set_ylabel("approx KL / clip fraction")
    h1, l1 = ax_stab.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_stab.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)

    n_up = int(rows[-1]["update"]) + 1 if rows else 0
    cfg = run.config
    scen = str(cfg.get("scenarios", "?")).split("/")[-1]
    warm = Path(str(cfg.get("bc_checkpoint", ""))).name or "random-init"
    fig.suptitle(f"PPO Self-Play from BC Warm-Start — {run.name}", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.005,
             f"{n_up} updates · scenarios={scen} · opponent={cfg.get('opponent', '?')} · warm-start={warm}",
             ha="center", fontsize=8, color="0.4")

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    final_win = next((r["rollout/win_rate"] for r in reversed(rows) if r.get("rollout/win_rate") is not None), None)
    print(f"saved {args.out}  (run {run.name}/{run.id}, updates={n_up}, final_win={final_win})")


if __name__ == "__main__":
    main()
