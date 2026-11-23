# Model versions

Tagged weights live outside git (``runs/`` is gitignored) in
``runs/checkpoints/<version>_<commit>.pt`` on Nick's PC. ``<commit>`` is the
code the weights were trained with, and the annotated git tag ``<version>``
points at that commit. v0.1.0 also keeps its run config
(``<version>_<commit>.config.yaml``) and metadata (``<version>_<commit>.json``)
next to the weights.

| Version | Weights | Commit | What |
|---|---|---|---|
| v0.0.1 | `runs/checkpoints/v0.0.1_f4f7bef.pt` | f4f7bef | Fresh sandbox warm-up: Fourier positions, globals token, live goal gates, cheap typed builds only, ent_coef 0.01. Used for the LLM games. |
| v0.1.0 | `runs/checkpoints/v0.1.0_cb0a0c7.pt` | cb0a0c7 | **Baseline.** 12x6 self-play smoke v7, update 150. |

## v0.1.0 (baseline)

- Source: `runs/selfplay_12x6_smoke_v7/latest.pt` (same weights as `update_150.pt`), 150 updates, 936 episodes, trained 2026-11-23 04:12-04:19 CST.
- sha256 `03e04eadec3939e3b228b2a244d0f0339638eb4340c9d84d4c65f98897efa004`.
- Warm start: `runs/sandboxes_v006/latest.pt` (merchant progress shaping and greedy-eval gate, #34).
- Self-play: 12x6 pool (30 maps, 2 ports a side) with `map_gen.min_port_enemy_home_water: 7` (#36), `ppo.hull_ent_coef: 0.05` (#35), ent_coef 0.01, dual-clip 3.0, hull exploration eps 0.15 (policy log-probs), reward shaping on, 2048 decisions per update.
- Training games: 829, W/L/D 402/252/175, mean turn 31.2. Updates 141-150: 65/30/5%, mean turn 22.6, hulls sub 29% / merchant 26% / destroyer 21% / landing 18% / cruiser 5%, hull-head entropy 0.89.
- Rush probe (303 games vs its last-4 snapshot pool): earliest win turn 2 (none on turns 0-1), median 12, turn-2 wins 23% of games.
- 16x8 smoke (5 games vs its own snapshot, alternating seats): learner 3-2, all home captures, 0 illegal moves.
