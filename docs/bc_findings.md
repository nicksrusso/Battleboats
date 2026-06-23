# Behavior-Cloning Findings (presentation prep)

Analysis of the two behavior-cloning (BC) runs that clone the god-mode MCTS
expert's `(asset, verb, target)` action. Captured so the numbers behind the
slides aren't lost. Figures: [`bc_training.png`](bc_training.png) (64×32, good)
and [`bc_training_160x80.png`](bc_training_160x80.png) (160×80, plateaued).

## The two runs

| run | W&B id | board | token_dim | harvest | epochs | final joint acc |
|---|---|---|---|---|---|---|
| **efficient-oath-3** | `btt0dicp` | 160×80 | 27 | `harvest_20260605_111638` | killed @ 3 | **0.26** |
| **lucky-music-7** | `s0ga84ri` | 64×32 | 28 | `harvest_20260608_040843` | finished 15 | **0.73** |

Project: `nicksrusso/battleboats-data`. Same architecture (d_model=64, 2 layers,
4 heads). The 160×80→64×32 board shrink + cash-token (tok 27→28) happened
between them (the 2026-06-07 "map substrate pivot").

## Finding 1 — accuracy plateaus within epoch 0 on BOTH boards

Per-epoch validation accuracy:

**efficient-oath-3 (160×80)** — `val_loss` 2.84→2.74; train≈val throughout (no overfit):

| epoch | joint | asset | verb | target |
|---|---|---|---|---|
| 0 | 0.247 | 0.381 | 0.975 | 0.570 |
| 1 | 0.255 | 0.387 | 0.975 | 0.580 |
| 2 | 0.259 | 0.389 | 0.975 | 0.580 |

**lucky-music-7 (64×32)** — reached 0.729 joint in epoch 0, ended 0.735 after 14 more:

| epoch | joint | asset | verb | target |
|---|---|---|---|---|
| 0 | 0.729 | 0.764 | 0.963 | 0.822 |
| 14 | 0.735 | 0.767 | 0.963 | 0.829 |

(+0.006 joint over 14 epochs.) Train loss on 160×80 fell 3.37→2.78 in the first
half, then only 2.74→2.72 over the entire second half.

**Read:** the model hits its ceiling in <1 epoch (2.4M rows ÷ batch 256 ≈ 9.5k
steps/epoch — plenty for a small model). Killing efficient-oath-3 early was the
right call; the 14 extra epochs on lucky-music-7 were wasted compute. Because
train≈val even at the plateau (never overfits), the model is **capacity/feature-
limited, not training-limited** — the lever to go higher is more capacity / better
features, not more epochs.

## Finding 2 — verb accuracy is a majority-class mirage (the expert barely acts)

Expert action labels — verb distribution (200k labeled decisions sampled per
harvest; sequential sample over leading shards, not uniform random — but the
dominance is extreme enough that it won't move materially):

| verb | 160×80 | 64×32 |
|---|---|---|
| **move** | **95.7%** | 50.5% |
| endturn | 2.6% | 46.7% |
| build_ship | 1.0% | 1.8% |
| **attack** | **0.6%** | **1.0%** |
| load | 0.1% | <0.1% |
| unload | 0.1% | <0.1% |
| capture | 0.0% | 0.0% |
| build_port | 0.0% | 0.0% |

**Majority-class baseline for the verb head ("always predict move"):**

- **160×80:** baseline = **95.7%**, head got **97.5%** → **+1.8 pts**. The verb
  head learned essentially nothing; its high accuracy is pure class imbalance.
- **64×32:** baseline = **50.5%**, head got **~96%** → **+46 pts**. Here it learned
  real structure — but only the easy **move-vs-endturn** distinction.

On both boards the strategically interesting verbs (attack, build, capture,
load/unload) are <3% combined, so the head learns almost nothing about them.

**Consequence:** the headline "73% joint accuracy" is partly inflated — joint is
propped up by move/endturn being trivially predictable. The cloned policy is
**passive**: MCTS attacks only 0.6–1.0% of the time, and BC reproduces that.
(Confirms the "lack of attacks" seen watching games back, and ties to the earlier
heuristic finding that combat is undervalued.)

## Why this sank PPO (mechanism)

One root cause threads through the whole pipeline:

> combat-undervalued heuristic → MCTS rarely attacks (0.6–1% of decisions) →
> BC clones a passive policy → no captures → games can't end decisively →
> ~100% truncation → all draws → terminal reward = 0.

The trap is *why no-signal was destructive rather than benign*:

1. Reward is sparse terminal-only; a draw is a flat 0 for the whole trajectory.
2. Advantage = return − V(s) = 0 − V(s) = **−V(s)**.
3. The **value head was never trained in BC**, so V(s) is noise.
4. PPO **normalizes advantages to unit variance per minibatch** → rescales that
   noise up to full-size gradients.
5. Policy random-walks away from the BC init → **entropy explodes (0.04→2.57),
   KL≈0.18, clip-frac≈0.64** in a single update.

**Caveats (keep the slides honest):**
- The entropy/KL/clip numbers are from a **stdout smoke test** (2026-06-08, 64×32),
  not a logged W&B run — there is no PPO wandb plot. A short re-run with `--wandb`
  on 64×32 would produce a real divergence curve.
- The "advantage = −V(s) → normalized noise" step is **mechanism inference** from
  the GAE math + untrained value head + per-minibatch normalization (all confirmed
  in code), not measured on that run.

## Asset-pointer width per state (why asset is the hard head)

The asset head is a pointer that scores **all N entity tokens** in the state, then
masks to legal ones. So the bigger N, the harder the selection. Measured entity
count per state (40k states sampled per harvest, leading shards):

| board | mean | std | p50 | range |
|---|---|---|---|---|
| **160×80** | **144.6** | 16.0 | 148 | 109–173 |
| **64×32** | **55.9** | 32.1 | 51 | 15–147 |

The big board's asset pointer is **~2.6× wider** (≈145 vs ≈56 candidates), which
lines up with asset accuracy collapsing 0.77 (64×32) → 0.39 (160×80). Caveat: N
counts total entity tokens per perspective (own + visible enemy + ports = the
pointer width / encoder context); the legal-to-act subset is smaller and wasn't
measured separately.

## MCTS truncation by board size — PLACEHOLDER (verify before slides)

The expert (god-mode MCTS) resolves games on the big board but stalemates on the
small one — opposite of the "shrink → decisive" intuition:

| board | iters | max_turns | decisive | truncated |
|---|---|---|---|---|
| 160×80 (`harvest_20260605_111638`) | 50 | 500 | **223 / 336 (66%)** | 113 (34%) |
| 24×12 (recent harvests) | 250–500 | 25–50 | **0 (0%)** | 10/10 (100%) |

**Hypothesis (NOT yet isolated):** the small board is far denser in assets, so
defenders saturate the short corridor between the (close) home ports; a passive,
attack-averse policy can't clear the chokepoint → stalemate. On the sparse big
board the homes are far apart with open lanes, so a landing routes *around* the
few defenders and captures — the lack of attacks doesn't bite. Consistent with
attacks being ~0.6% (160×80) vs ~1.0% (64×32): low everywhere, only fatal when
the board is clogged.

**Confounds — why this isn't conclusive yet:** the runs differ in `max_turns`
(500 vs 25–50) and `iterations` (50 vs 250–500), not just board size. (Prior
note: raising max_turns 25→32 didn't reduce truncation, so turns likely aren't
the binding constraint; and stronger search on the small board would *sharpen*
defense, consistent with the clog story.) To isolate sparsity: run MCTS on both
boards with matched iters + max_turns scaled to traversal distance, and measure
defenders within N tiles of the landing's path.

## How to regenerate the data

```bash
# per-epoch accuracy / loss (pulls live from W&B):
poetry run python scripts/plot_bc_training.py --run nicksrusso/battleboats-data/s0ga84ri --out docs/bc_training.png
poetry run python scripts/plot_bc_training.py --run nicksrusso/battleboats-data/btt0dicp --out docs/bc_training_160x80.png
# verb label distribution: sampled from the harvest `action` field (factored
# [asset_idx, verb_idx, target_idx]); verb idx map in battleboats/training/policy.py VERB_TO_IDX.
```
