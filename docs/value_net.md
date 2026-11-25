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
