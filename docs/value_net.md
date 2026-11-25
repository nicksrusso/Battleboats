# Value net v0.5.0

Plan step 1.2 in `docs/task.md`: a value net on fogged observations, gated against
simple baselines before any MCTS is built. Next steps after the gate: MCTS with
this net as the leaf evaluator (fog determinization), then MCTS games as a better
behavior-cloning warm start.

## Data

* Source: `scripts/harvest_value.py` shards (fog_commander self-play on 48x24,
  `runs/value_harvest_48x24_fc_v0.5.0`). Each decision has one row per seat; the
  label is that seat's final outcome (+1 / -1 / 0 draw) plus the discounted
  return `outcome * 0.99^(decisions_left)`.
* Shards are compressed `.npz` (not memory-mappable), so
  `scripts/build_value_cache.py` reads each shard once and keeps a seeded pool of
  `k_pool=64` positions per game in a memmap cache (`tokens.f16` + `rows.npz` +
  `games.json`). About 42 tokens per row; ~0.4 MB per game.
* Split: by game, `sha1("bb-value-v0.5.0:<seed>")` -> 90 / 5 / 5 train / val /
  test. A game's positions never cross splits (tested).
* Sampling: each epoch draws `k_per_game` (32) of a train game's pooled rows, so
  long and short games weigh the same. Decisive games only by default
  (`include_draws: false`).

## Model

`battleboats/training/value_net.py` `ValueNet` subclasses `TransformerValueModel`
(76-dim token projection, 2-layer 4-head encoder with `d_model=64`, masked mean-pool,
MLP + tanh) and adds a prepended globals token (like `PolicyNetwork`) and an
auxiliary tanh head on the discounted return (loss weight 0.25). 80,898 parameters.
`value_of_observation(model, tokens, globals)` gives V(s) for a single
`build_observation` output (the MCTS leaf call).

## Train / eval

```bash
poetry run python scripts/build_value_cache.py --config configs/value_net_v0.5.0.yaml
poetry run python scripts/train_value.py --config configs/value_net_v0.5.0.yaml
poetry run python scripts/train_value.py --resume runs/value_net_v0.5.0/<timestamp>
poetry run python scripts/eval_value.py --checkpoint runs/value_net_v0.5.0/<timestamp>/best.pt \
    --cache runs/value_harvest_48x24_fc_v0.5.0/value_cache_k64 --split test
```

A run dir holds `config.yaml`, `metrics.csv`, `train_log.csv`, `best.pt` (lowest val
Brier), `last.pt` (with optimizer, used by `--resume`), `baselines_val.json`,
`eval_{val,test}.{json,txt}` and `summary.json`. TensorBoard logs are written only
if tensorboard is installed. Override any config key with `--set key=value`.

Metrics, on decisive held-out games: MSE, sign accuracy, Brier and log-loss on
`p = (v + 1) / 2`, a 10-bin calibration table, seat bias, and accuracy by game phase
(turn / game length, 5 buckets).

Baselines: coin flip, constant prior, per-seat prior, `logistic_simple`
(strength / ship-count / port differences, own cash, turn, turn x strength diff,
seat) and `logistic_rich` (all globals, per-type counts, sighted strength, enemy
home seen, ...). All features are fog-fair.

**Gate** (`task.md` §1.2): on the test split, accuracy > 0.52, Brier < 0.245, and
accuracy, Brier (by 0.005) and log-loss all better than the best of coin flip,
constant prior, seat prior and `logistic_simple`. Beating `logistic_rich` is
reported separately.

## Large model (GPU)

`configs/value_net_v0.5.0_large.yaml` is the same data and 90/5/5 split, with a
bigger encoder: `d_model=128`, 4 layers, 8 heads, `dim_feedforward=256`
(574,210 parameters). Training uses `train.device: cuda` (works on NVIDIA CUDA
or AMD ROCm via `torch.cuda`). LR is 3e-4 with 200-step warmup and cosine decay;
batch size 512; early stopping on val Brier with patience 8 and `min_delta` 0.0005.
Best checkpoint is still `best.pt` (lowest val Brier). `train_value.py` falls
back to CPU and prints a note if a GPU was requested but is not available.

```bash
HIP_VISIBLE_DEVICES=0 poetry run python scripts/train_value.py --config configs/value_net_v0.5.0_large.yaml
```

(`HIP_VISIBLE_DEVICES=0` pins an AMD box to the discrete GPU when an iGPU is also visible.)

## Mixed-doctrine harvest and later-position weighting

`scripts/harvest_value.py --games-per-matchup 2000 --doctrine-schedule cycle`
plays 72,000 Fog Commander self-play games on 48x24: exactly 2,000 games for
each of the 36 ordered doctrine pairs (Raider Blitz, Economic Boom, Wolfpack
Stealth, Capital Armada, Island Hopper, Balanced Adaptive, both seats). Each
shard records the doctrine used by each seat. No scripted or random opponent.

`configs/value_net_v0.5.0_mixed.yaml` is the default training recipe for that
harvest. It uses the **large** encoder (574,210 params, `d_model=128`, 4 layers,
8 heads, dropout 0.2) on GPU (`train.device: cuda`, launch with
`HIP_VISIBLE_DEVICES=0` on AMD ROCm). Point `--config` at
`configs/value_net_v0.5.0.yaml` (and shrink the `model:` block) to train the
small 80,898-param net on the same data.

Later positions are weighted more heavily, three ways, all configurable:

* Cache pool (`data.sample_weight_floor` / `sample_weight_power`, default in
  this recipe 0.25 / 1.0): of each game's rows, the 128 kept positions are
  drawn with probability `0.25 + 0.75 * (turn / game_turns)`.
* Per-epoch sample (`k_per_game: 64`): the same curve, without replacement.
* Training loss: the same curve (or `train.loss_weight_*` if set), then
  renormalized so the mean weight in a batch is 1, so the learning rate scale
  does not change. Early positions are still trained, just with less weight.

`floor=1` recovers uniform sampling and unweighted loss (the older configs).
Weight decay is `train.weight_decay` (0.02 here); dropout is `model.dropout`.
