# Model versions

Tagged weights live outside git (``runs/`` is gitignored) in
``runs/checkpoints/<version>_<commit>.pt`` on Nick's PC. ``<commit>`` is the
code the weights were trained with, and the annotated git tag ``<version>``
points at that commit. v0.1.0 also keeps its run config
(``<version>_<commit>.config.yaml``) and metadata (``<version>_<commit>.json``)
next to the weights, as do later versions.

Versions use periods: `vMAJOR.MINOR.PATCH` (v0.1.1, not v011). Run
directories for versioned models carry the same name, e.g.
`runs/selfplay_16x8_v0.2.0`. A minor bump marks a rules change, so models
from different minor versions were trained under different game rules.

Experiment run folders such as `runs/selfplay_mixed_v0.2.0` and
`runs/selfplay_mixed_v0.2.1` are named for the starting checkpoint or
experiment, not model versions; there is no tagged v0.2.1 model.
`runs/selfplay_mixed_v0.2.2` produced v0.3.0 (update 800), not a v0.2.2 model.

| Version | Weights | Commit | What |
|---|---|---|---|
| v0.0.1 | `runs/checkpoints/v0.0.1_f4f7bef.pt` | f4f7bef | Fresh sandbox warm-up: Fourier positions, globals token, live goal gates, cheap typed builds only, ent_coef 0.01. Used for the LLM games. |
| v0.1.0 | `runs/checkpoints/v0.1.0_cb0a0c7.pt` | cb0a0c7 | **Baseline.** 12x6 self-play smoke v7, update 150. |
| v0.1.1 | `runs/checkpoints/v0.1.1_cb0a0c7.pt` | cb0a0c7 | 16x8 self-play from v0.1.0, 700 updates. Old rules. |
| v0.2.0 | `runs/checkpoints/v0.2.0_955f221.pt` | 955f221 | 16x8 self-play from v0.1.0, 700 updates, under the #38 rules (bought ships wait a turn; captured/built ports wait a turn). |
| v0.3.0 | `runs/checkpoints/v0.3.0_26348ee.pt` | 26348ee | Mixed 16x8+20x10 self-play from v0.2.0, tagged at update 800 (the run's final u1000 collapsed and is not this version). Same #38 rules; ship_lost / draw penalty / distance-scaled landing_progress. |

## v0.1.0 (baseline)

- Source: `runs/selfplay_12x6_smoke_v7/latest.pt` (same weights as `update_150.pt`), 150 updates, 936 episodes, trained 2026-11-23 04:12-04:19 CST.
- sha256 `03e04eadec3939e3b228b2a244d0f0339638eb4340c9d84d4c65f98897efa004`.
- Warm start: `runs/sandboxes_v006/latest.pt` (merchant progress shaping and greedy-eval gate, #34).
- Self-play: 12x6 pool (30 maps, 2 ports a side) with `map_gen.min_port_enemy_home_water: 7` (#36), `ppo.hull_ent_coef: 0.05` (#35), ent_coef 0.01, dual-clip 3.0, hull exploration eps 0.15 (policy log-probs), reward shaping on, 2048 decisions per update.
- Training games: 829, W/L/D 402/252/175, mean turn 31.2. Updates 141-150: 65/30/5%, mean turn 22.6, hulls sub 29% / merchant 26% / destroyer 21% / landing 18% / cruiser 5%, hull-head entropy 0.89.
- Rush probe (303 games vs its last-4 snapshot pool): earliest win turn 2 (none on turns 0-1), median 12, turn-2 wins 23% of games.
- 16x8 smoke (5 games vs its own snapshot, alternating seats): learner 3-2, all home captures, 0 illegal moves.

## v0.1.1

- Source: `runs/selfplay_16x8_v0.1.1/latest.pt` (formerly `runs/selfplay_16x8_v010`; same weights as `update_700.pt`), 700 updates, 5250 episodes, trained 2026-11-23 04:29-04:47 CST.
- sha256 `6dd51a1ebf4d85e218555aafbe57498f137ee01c41f4b51f9359dfe79c42cf6e`.
- Lineage: warm start from v0.1.0.
- Code: cb0a0c7 (launched at 6da2cdd, which only adds docs). Old rules: a bought ship could act on its build turn, and a captured or built port could build at once.
- Self-play: 16x8 pool (30 maps, 2 ports a side), `map_gen.min_port_enemy_home_water: 7`, hull_ent_coef 0.05, hull exploration eps 0.15, fleet starts prob 0.333, reward shaping on, 7 workers x 8 games.
- Training games: 4971, W/L/D 2586/1800/585, mean turn 30.1. Updates 601-700: 587/406/75, mean turn 26.3, entropy 1.50, hull-head entropy 0.75, landing 31-35% of buys.
- Rush probe (update 700, 153 games): 135 wins; one turn-0 win (forward-start chain), 33 wins by turn 3, median win turn 18.
- Head-to-head vs v0.1.0 (old rules, 200 games, alternating seats): 121/58/21, win rate 60.5% (Wilson 95% [53.6, 67.0]).

## v0.2.0

- Source: `runs/selfplay_16x8_v0.2.0/latest.pt` (formerly `runs/selfplay_16x8_v011`; same weights as `update_700.pt`), 700 updates, 5159 episodes, trained 2026-11-23 05:00-05:19 CST.
- sha256 `46c884db86984a36b3cbdd14a46e528de326c6bdc794c6fce12f2f00a665fbbd`.
- Lineage: warm start from v0.1.0 (trained under the old rules). Same config as v0.1.1 otherwise.
- Code: 955f221 (#38). Rules: a ship bought this turn takes no actions until its owner's next turn (fleet-start and scenario ships are not bought); a port captured or built this turn can't build ships until its owner's next turn. Earliest direct landing capture in the pool is turn 3 (was 2). No observation-shape change.
- Training games: 4768, W/L/D 2535/1717/516, mean turn 29.1. Updates 601-700: 626/395/74, mean turn 24.4, entropy 1.56, hull-head entropy 0.78, landing 32.4% of buys.
- Rush probe (update 700, 160 games): 154 wins, 6 draws; one turn-2 win (a fleet-start landing), 22 wins by turn 3, median win turn 8, P0 49% of wins.
- Head-to-heads under the new rules (200 games each, alternating seats; opponents trained under the old rules):
  - vs v0.1.0: 108/70/22, win rate 54.0% (Wilson 95% [47.1, 60.8]), score 0.595.
  - vs v0.1.1: 68/124/8, win rate 34.0% (Wilson 95% [27.8, 40.8]), score 0.360. v0.1.1 is stronger even under the new rules.

## v0.3.0

- Source: `runs/selfplay_mixed_v0.2.2/update_800.pt`, 800 updates (of a 1000-update run), 4684 finished games through update 800, trained 2026-11-23 06:21:56-06:45:32 CST. The run continued after this checkpoint and collapsed by updates 901-1000 (draws 45%, landing 7%, mean turn 60); `latest.pt` / `update_1000.pt` is not v0.3.0.
- sha256 `e5ba7164392d8b90b18c77b0893413351b038e6fe34ee891b5e57cc81d61d285`.
- Lineage: warm start from v0.2.0. Same #38 rules as v0.2.0.
- Code: 26348ee (#41). Training changes vs v0.2.0: 40-map pool (16x8 x20 ports 2 + 20x10 x20 ports 2, no 24x12), `reward_shaping.ship_lost` 0.02 (cap 0.5), kill 0.02 (cap 0.3), `reward.draw` -0.15, `landing_progress_distance_scale` on (a full home-to-adjacent landing trip pays exactly `landing_progress` on every map size).
- Self-play: `map_gen.min_port_enemy_home_water: 7`, land_fraction 0.02, hull_ent_coef 0.05, hull exploration eps 0.15, fleet starts prob 0.333, 7 workers x 8 games, 2048 decisions per update.
- Training through update 800: 4684 games, W/L/D 2404/1727/553, mean turn 34.7, draw 11.8%, landing 27.9% of buys. Updates 701-800: 522 games, W/L/D 291/189/42, mean turn 32.1, draw 8.0%, entropy 2.28, hull-head entropy 0.80, landing 23.2% of buys.
- Head-to-head vs v0.2.0 (200 games, alternating seats, on this run's 16x8+20x10 pool): 129/67/4, win rate 64.5% (Wilson 95% [57.7, 70.8]), score 0.655. As seat 0: 72/26/2; as seat 1: 57/41/2. Per size: 16x8 67/34/3, 20x10 62/33/1.
