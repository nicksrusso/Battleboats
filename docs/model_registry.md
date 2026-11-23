# Model registry

This is the catalog of tagged Battleboats checkpoints: what each one is, how it
was trained, how it plays, and what to use it for. Version naming and the
mapping from tags to weight files are in [model_versions.md](model_versions.md);
this page does not repeat those rules.

Weights live outside git in `runs/checkpoints/` on Nick's PC. Every headline
number below is taken from a file (checkpoint `.json` metadata,
`docs/model_versions.md`, a `runs/eval/*/summary.json`, or
`runs/eval/behavior_scripted_v_v0.3.0/REPORT.md`). sha256 prefixes were
recomputed from the `.pt` files and match the metadata. Anything not found in
those files is called out as unverified or omitted.

W/L/D is wins/losses/draws for the named side. Wilson intervals are 95% on the
win rate. "Score" counts a draw as half a win.

## Summary

| Version | Tag commit | What it is | Best known result |
|---|---|---|---|
| v0.1.0 | cb0a0c7 | 12x6 self-play baseline, old rules | Beaten by v0.1.1, 58/121/21 (200 games, old rules) |
| v0.1.1 | cb0a0c7 | 16x8 self-play from v0.1.0, old rules | vs v0.1.0 121/58/21 (old rules); also beat v0.2.0 124/68/8 under the #38 rules |
| v0.2.0 | 955f221 | First 16x8 self-play under the #38 rules, from v0.1.0 | vs v0.1.0 108/70/22; lost to v0.1.1 68/124/8 (both under #38 rules) |
| v0.3.0 | 26348ee | Mixed 16x8+20x10 self-play from v0.2.0, tagged at update 800 | vs v0.2.0 129/67/4 (current best tagged trained policy) |
| v0.4.0 | 8ad2530 | Behavior clone of the #46 tactical scripted agent | Sampled vs tactical 24/28/48 (score 0.480); vs v0.3.0 11/50/39 (score 0.305). Use sampled play, not greedy. |
| v0.4.1 | docs commit (tag); weights 79754fb | v0.4.0 BC + buy-diversity sandbox fine-tune from both seats, anchor 0.05 | Sampled vs v0.4.0 14/11/25 (score 0.530); vs tactical 10/25/15 (score 0.350); self-play 36 decisive / 14 draws (72%). Use sampled play. |

An earlier sandbox-warmup tag, [v0.0.1](model_versions.md), exists at f4f7bef
(`runs/checkpoints/v0.0.1_f4f7bef.pt`). It is not part of this self-play series.

---

## v0.1.0

- **Tag / commit:** annotated tag `v0.1.0` → cb0a0c7 (`cb0a0c7447ce8e7b94155bd0e264981b92e0f07e`).
- **Weights:** `runs/checkpoints/v0.1.0_cb0a0c7.pt`, sha256 prefix `03e04ead`. Config and metadata sit beside it (`.config.yaml`, `.json`).
- **How it was produced:** 12x6 self-play in `runs/selfplay_12x6_smoke_v7`, update 150 (same weights as `latest.pt` / `update_150.pt`), 936 episodes, trained 2026-11-23 04:12–04:19 CST. Warm start from `runs/sandboxes_v006/latest.pt` (merchant progress shaping and greedy-eval gate, PR #34). **Old rules** (pre-#38): a bought ship could act on its build turn, and a captured or built port could build immediately.
- **Map pool / training setup:** 30 maps at 12x6, 2 ports a side, `land_fraction` 0.02, `map_gen.min_port_enemy_home_water` 7. Opponent pool is the last 4 snapshots (saved every 5 updates) plus 15% random opponents. Fleet starts at probability 0.333, off below 16x8. PPO on CPU, `ent_coef` 0.01, `hull_ent_coef` 0.05, dual-clip 3.0, hull exploration eps 0.15, 2048 decisions per update, 4 workers.
- **Reward shaping:** on. Weights: build 0.01, load 0.01, unload 0.02, port_build 0.05, port_capture 0.1, kill 0.02 (cap 0.3), landing_progress 0.2 (cap 0.3, home-to-home normalized, no distance scaling), landing_buy 0.1 (decay 0.5). No `ship_lost`. No `reward.draw` (timeout terminal reward is 0).
- **Headline eval:**
  - Training: 829 games, W/L/D 402/252/175, mean turn 31.2. Updates 141–150: 65/30/5%, mean turn 22.6; hull mix sub 29% / merchant 26% / destroyer 21% / landing 18% / cruiser 5%.
  - Rush probe, 303 games vs its last-4 snapshot pool: earliest win on turn 2 (none on turns 0–1), median win turn 12, turn-2 wins 23% of games.
  - 16x8 smoke, 5 games vs its own snapshot, alternating seats: learner 3–2, all home captures, 0 illegal moves.
  - Head-to-head as the *opponent*: v0.1.1 beat it 121/58/21 under the old rules; v0.2.0 beat it 108/70/22 under the #38 rules (200 games each). There is no tagged-checkpoint match in which v0.1.0 is the newer side.
  - Sources: `runs/checkpoints/v0.1.0_cb0a0c7.json`, `docs/model_versions.md`, and the later checkpoints' head-to-head entries.
- **Known behavior / weaknesses:** Trained only on 12x6. The only 16x8 result is a 5-game smoke. Under the old rules it could still win on turn 2 (23% of the rush-probe games). No behavior audit comparable to the v0.3.0 report exists for this checkpoint.
- **Recommended use:** Historical baseline and the warm start for v0.1.1 and v0.2.0. Do not use it as a current-rules opponent without noting the #38 change, and do not treat it as a 16x8 or mixed-map policy.

## v0.1.1

- **Tag / commit:** annotated tag `v0.1.1` → cb0a0c7 (same code as v0.1.0; only the weights differ). The run was launched at 6da2cdd, which only adds docs.
- **Weights:** `runs/checkpoints/v0.1.1_cb0a0c7.pt`, sha256 prefix `6dd51a1e`.
- **How it was produced:** 16x8 self-play in `runs/selfplay_16x8_v0.1.1` (formerly `runs/selfplay_16x8_v010`), update 700 (same weights as `latest.pt` / `update_700.pt`), 5250 episodes, trained 2026-11-23 04:29–04:47 CST. Warm start from v0.1.0. **Old rules** (pre-#38), same as v0.1.0.
- **Map pool / training setup:** 30 maps at 16x8, 2 ports a side, `land_fraction` 0.02, `min_port_enemy_home_water` 7. Same snapshot pool (last 4 + 15% random), fleet starts probability 0.333 (min 16x8), same PPO and shaping as v0.1.0, 7 workers × 8 games, 8 PPO threads, 700 updates.
- **Headline eval:**
  - Training: 4971 games, W/L/D 2586/1800/585, mean turn 30.1. Updates 601–700: 587/406/75, mean turn 26.3, landing 31–35% of buys.
  - Rush probe, update 700, 153 games: 135 wins; one turn-0 win (forward-start chain), 33 wins by turn 3, median win turn 18.
  - vs v0.1.0, old rules, 200 games, alternating seats: **121/58/21**, win rate 60.5% (Wilson 95% [53.6, 67.0]).
  - vs v0.2.0, under the #38 rules, 200 games: **124/68/8** from v0.1.1's side (v0.2.0 recorded 68/124/8). v0.1.1 is stronger even under the new rules it was not trained on.
  - Sources: `runs/checkpoints/v0.1.1_cb0a0c7.json`, `runs/checkpoints/v0.2.0_955f221.json`, `docs/model_versions.md`.
- **Known behavior / weaknesses:** Old rules, including a rush-probe turn-0 win from a forward-start chain. No written behavior study of buys, port play, or fight-or-run decisions.
- **Recommended use:** Strongest old-rules 16x8 checkpoint, and still a tough reference opponent under the #38 rules. Not a current-rules trained policy.

## v0.2.0

- **Tag / commit:** annotated tag `v0.2.0` → 955f221 (`955f221f1ba4c2e80e282456ae75e2d7a263ac2d`, PR #38).
- **Weights:** `runs/checkpoints/v0.2.0_955f221.pt`, sha256 prefix `46c884db`.
- **How it was produced:** 16x8 self-play in `runs/selfplay_16x8_v0.2.0` (formerly `runs/selfplay_16x8_v011`), update 700 (same weights as `latest.pt` / `update_700.pt`), 5159 episodes, trained 2026-11-23 05:00–05:19 CST. Warm start from v0.1.0 (which was trained under the old rules). First model trained under the **#38 rules**: a ship bought this turn takes no actions until its owner's next turn (fleet-start and scenario ships are not treated as bought); a port captured or built this turn cannot build until its owner's next turn. Earliest direct landing capture in this pool is turn 3 (was 2). No observation-shape change.
- **Map pool / training setup:** Same 16x8 ×30 pool and PPO / shaping / opponent-pool settings as v0.1.1 (kill 0.02 cap 0.3, landing_progress 0.2 without distance scaling, no `ship_lost`, no `reward.draw`).
- **Headline eval:**
  - Training: 4768 games, W/L/D 2535/1717/516, mean turn 29.1. Updates 601–700: 626/395/74, mean turn 24.4, landing 32.4% of buys.
  - Rush probe, update 700, 160 games: 154 wins, 6 draws; one turn-2 win (a fleet-start landing), 22 wins by turn 3, median win turn 8.
  - Head-to-heads under the #38 rules, 200 games each, alternating seats (opponents were trained under the old rules):
    - vs v0.1.0: **108/70/22**, win rate 54.0% (Wilson 95% [47.1, 60.8]), score 0.595.
    - vs v0.1.1: **68/124/8**, win rate 34.0% (Wilson 95% [27.8, 40.8]), score 0.360.
  - Sources: `runs/checkpoints/v0.2.0_955f221.json`, `docs/model_versions.md`.
- **Known behavior / weaknesses:** It is the first current-rules policy, but it lost a 200-game match to old-rules v0.1.1 even after the rule change. No written behavior study of buys or engagements. Superseded on 16x8+20x10 by v0.3.0.
- **Recommended use:** First current-rules tagged baseline, and the warm start for v0.3.0. Use it as a weaker 16x8 opponent or as a historical comparison, not as the default learned agent.

## v0.3.0

- **Tag / commit:** annotated tag `v0.3.0` → 26348ee (`26348ee58304d3aab32b6bf820928871df8f5e6d`, PR #41).
- **Weights:** `runs/checkpoints/v0.3.0_26348ee.pt`, sha256 prefix `e5ba7164` (same bytes as `runs/selfplay_mixed_v0.2.2/update_800.pt`). The saved `.config.yaml` is the run config and still says `updates: 1000`; the tag is update 800, not the run's final checkpoint.
- **How it was produced:** Mixed-map self-play in `runs/selfplay_mixed_v0.2.2`, tagged at **update 800 of a 1000-update run**, 4684 finished games through update 800, trained 2026-11-23 06:21:56–06:45:32 CST. Warm start from v0.2.0. Same **#38 rules** as v0.2.0. The run continued after this checkpoint and collapsed by updates 901–1000 (draws 45%, landing 7%, mean turn 60); `latest.pt` / `update_1000.pt` is **not** v0.3.0.
- **Map pool / training setup:** 40 maps, 16x8 ×20 ports 2 + 20x10 ×20 ports 2 (no 24x12), `land_fraction` 0.02, `min_port_enemy_home_water` 7, seed 0. Same snapshot pool (last 4 + 15% random), fleet starts probability 0.333 (min 16x8), 7 workers × 8 games, 2048 decisions per update, PPO same as v0.2.0.
- **Reward shaping:** same base weights as v0.2.0, plus `ship_lost` 0.02 (cap 0.5; one-sided, cost-scaled), `reward.draw` −0.15 on timeout, and `landing_progress_distance_scale` on (a full home-to-adjacent landing trip pays exactly `landing_progress` 0.2 on every map size). Kill stays 0.02 (cap 0.3).
- **Headline eval:**
  - Training through update 800: 4684 games, W/L/D 2404/1727/553, mean turn 34.7, draw 11.8%, landing 27.9% of buys. Updates 701–800: 522 games, 291/189/42, mean turn 32.1, draw 8.0%, landing 23.2% of buys.
  - vs v0.2.0, 200 games, alternating seats, on this run's 16x8+20x10 pool: **129/67/4**, win rate 64.5% (Wilson 95% [57.7, 70.8]), score 0.655. As seat 0: 72/26/2; as seat 1: 57/41/2. By size: 16x8 67/34/3, 20x10 62/33/1.
  - vs the latest tactical scripted agent (any-port, fog, uniform queue; see below): the scripted side went 29/46/25 in 100 games, so v0.3.0 scored 46/29/25 (score 0.585). An earlier fog + home-first scripted match was closer (scripted 40/49/11; v0.3.0 score 0.545).
  - Sources: `runs/checkpoints/v0.3.0_26348ee.json`, `docs/model_versions.md`, `runs/eval/scripted_v_v0.3.0_anyport/summary.json`, `runs/eval/scripted_v_v0.3.0_fog/summary.json`.
- **Known behavior / weaknesses** (from 100 instrumented games vs the fog + uniform-queue scripted agent, commit ad2ce3f, reported in `runs/eval/behavior_scripted_v_v0.3.0/REPORT.md`; that match was 37/49/14 for the scripted side, a fresh sample, not a replay of the fog smoke):
  - Turn-1 landing rush: first buy is a landing ship in 100/100 games (median time to first landing: turn 1).
  - After that it only buys destroyers (38.6%), subs (35.3%), merchants (13.2%), and landing ships (12.9%). It never bought a builder or a capital ship (n=1875 buys).
  - It takes fights it is losing: when the local force ratio is below 0.5 it still attacks 64.7% of the time (the scripted agent 1.7%) and dies within two turns in 21.2% of engagements (scripted 14.2%). When a legal shot exists it fires 73.7% of the time.
  - It is focused on winning: landings close on the enemy home at +1.22 tiles/turn, and it captures 0.75 enemy ports per game (P(capture | able) 98.7%). It does not expand with builders.
- **Recommended use:** Current best tagged trained policy under the #38 rules, on 16x8 and 20x10. Use it as the default learned opponent and as a warm start. Do not use `runs/selfplay_mixed_v0.2.2/latest.pt` or `update_1000.pt` in its place.

## v0.4.0

- **Tag / commit:** annotated tag `v0.4.0` → 8ad2530 (`8ad25302417e8ea7810914853a8a44a5c9c980f7`, PR #48).
- **Weights:** `runs/checkpoints/v0.4.0_8ad2530.pt`, sha256 prefix `101c699c`. Config and metadata sit beside it (`.config.yaml`, `.json`). Same bytes as `runs/bc_tactical_v040/train/best.pt`.
- **How it was produced:** Behavior clone of `tactical_scripted_action` at c1b9124 (#46: uniform queue, fog of war, landings target the nearest known capturable port). Initialized **from scratch** with the v0.3.0 pointer architecture. Harvested 2600 games on the v0.2.2 16x8+20x10 pool (2000 tactical self-play, 400 vs v0.3.0, 200 vs random); only tactical-side actions were labels (1,482,329 kept; 29 MerchantLoadAction roundtrip mismatches dropped, 0.002%). Trained 2026-11-23 08:43–09:15 CST, 8 epochs, Adam 1e-3, batch 256, CPU. Same **#38 rules** as v0.3.0.
- **Value head / PPO warm start:** the official checkpoint includes the value head, trained with `vf_coef` 0.5 MSE on discounted terminal returns `G_t = z * 0.99**(K-1-t)` (`z` = +1 / −1 / 0 for the acting tactical player). `load_checkpoint` reconstructs the full `PolicyNetwork`, so PPO can set `init_checkpoint` to these weights and Adam will see every parameter, value head included. Saved `model_config` dropout is 0.1 (the training value); `model.eval()` disables it, and PPO can `set_dropout` as it did for v0.3.0.
- **Held-out agreement** (full val, 150,295 rows, natural verb mix, from `runs/bc_tactical_v040/train/metrics.json`): joint 44.8%, asset 78.3%, verb 84.2%, target 50.5%. By group: capture 99.7%, unload 99.9%, load 91.1%, build 78.5%, attack 76.0%, end-turn 58.1%, move 39.9%, **buy 17.4%**. Buy agreement is low because the teacher's uniform per-game build queue is hidden state, not in the observation. Value loss 0.0548.
- **Headline eval** (v0.2.2 pool, 25 games per BC seat × size at N=100, 40 self-play, seed 0):
  - **Greedy (argmax):** vs tactical **1/52/47**, score 0.245; vs v0.3.0 **4/24/72**, score 0.400; self-play **0/0/40**. Buys collapse to a modal hull (vs v0.3.0: destroyer 3043, landing 27). This is expected: argmax on a nearly uniform teacher buy policy picks one type.
  - **Sampled (`policy.act`, `torch.manual_seed(0)` — how PPO and the opponent pool will use it):** vs tactical **24/28/48**, win rate 24.0% (Wilson 95% [16.7, 33.2]), score 0.480. vs v0.3.0 **11/50/39**, win rate 11.0% (Wilson 95% [6.3, 18.6]), score 0.305 (the #46 teacher was 29/46/25, score 0.415). Self-play 9/6/25 (seat 0), score 0.538, 25 timeouts.
  - Sampled buy mix vs tactical (2,872 buys): destroyer 764, merchant 698, submarine 665, cruiser 312, battleship 135, carrier 101, builder 99, landing 98. All eight types appear, but destroyer / merchant / submarine are overweight relative to the uniform teacher. Landings: 98 bought, 27 killed, 71 survived, 45 port captures (24 home). vs v0.3.0: 95 landings bought, 20 port captures (11 home).
  - Sources: `runs/checkpoints/v0.4.0_8ad2530.json`, `runs/eval/bc_v0.4.0/summary.json`, `runs/eval/bc_v0.4.0_sampled/summary.json`, `runs/bc_tactical_v040/train/metrics.json`.
- **Recommended use:** PPO warm start and a sampled opponent-pool member, not a stronger policy than v0.3.0 or the #46 teacher. Always use sampled `policy.act` (or PPO rollouts), not greedy `argmax`.

---


## v0.4.1

- **Tag / commit:** annotated tag `v0.4.1` is applied to this documentation commit after merge. The weight filename uses `79754fb` (`79754fb4c50979702ae3c18d7f18b2664fe48e43`, PR #51), the code the sandbox was trained with.
- **Weights:** `runs/checkpoints/v0.4.1_79754fb.pt`, sha256 prefix `1694863d`. Config and metadata sit beside it (`.config.yaml`, `.json`). Same bytes as `runs/sandbox_buydiv_v0.4.1/update_300.pt`.
- **How it was produced:** Fine-tune of v0.4.0 (`runs/checkpoints/v0.4.0_8ad2530.pt`) in the buy-diversity sandbox (PR #50, both-seats fix PR #51 / BAT-23). The learner seat alternates each episode and the 2–3 port economy is mirrored onto that seat's side, so the policy is not fit only to the left (player-0) half of the board. Non-buy decisions are anchored with MSE (`anchor_coef` 0.05, chosen over 0.15 in a 40-update probe) toward frozen v0.4.0. Reward is the potential-based Δ of `max(0, 1 - std/mean)` over all eight hull types, plus a training-only coverage bonus (`0.25 * unique/8`). Cheap types (destroyer, submarine, merchant) are intermittently masked with annealed probability 0.7→0.0; landing is never masked. 300 updates, 1024 decisions, 50-turn horizon, 16x8+20x10, CPU, lr 1e-4, dropout 0.0, trained 2026-11-23 12:31–12:47 CST (15.7 min). Same **#38 rules** as v0.3.0 / v0.4.0.
- **Value head / PPO warm start:** official `save_checkpoint` format, value head included. After this fine-tune the head is fit to diversity potential, not terminal win/loss (v0.4.0 had trained it on discounted +1/−1/0). A later win/loss PPO run should watch value-head loss at the start. Dropout in the saved `model_config` is 0.0.
- **Headline eval** (v0.2.2 pool, sampled `policy.act`, seed 0; sources under `runs/sandbox_buydiv_v0.4.1/`):
  - **Unmasked sandbox**, 128 episodes, p=0: diversity mean **0.481**, Wald 95% CI [0.458, 0.504], `roughly_even=True`. Type totals: carrier 332, battleship 543, cruiser 510, destroyer 487, submarine 534, merchant 558, landing 321, builder 318; max/min 1.75. Landing gold share 0.137. Ahead of frozen v0.4.0 in the same protocol (0.185 [0.159, 0.211], max/min 6.12) and of the earlier single-seat sandbox update_300 (0.362 [0.333, 0.391]).
  - **Self-play**, 50 games: 36 decisive / 14 turn-cap draws, decisive rate **72.0%** (Wilson 95% [58.3, 82.5]). All 36 wins were home captures. Seat 0: **13/23/14** (13 home caps, 64 landings bought). Seat 1: **23/13/14** (23 home caps, 84 landings). Landing share 5.16% of 2,868 buys. Decisive turns mean 53.1, median 49.5, range 7–138. The single-seat BAT-21 update_300 self-play was 16 decisive / 34 draws and **0 home-capture attempts as seat 0**.
  - **50-game sampled matchups** (seat split 26/24):
    - vs tactical **10/25/15**, win rate 20.0% (Wilson 95% [11.2, 33.0]), score 0.350. Seat 0: 5/13/8; seat 1: 5/12/7. Landing share 5.5%.
    - vs v0.3.0 **10/19/21**, win rate 20.0% (Wilson 95% [11.2, 33.0]), score 0.410. Seat 0: 3/9/14; seat 1: 7/10/7. Landing share 5.6%.
    - vs frozen v0.4.0 **14/11/25**, win rate 28.0% (Wilson 95% [17.5, 41.7]), score 0.530. Seat 0: 7/6/13; seat 1: 7/5/12. Landing share 6.1%.
  - vs the BAT-21 single-seat update_300 (3/34/13 vs tactical, 2/38/10 vs v0.3.0, 3/12/35 vs v0.4.0, and 0/26 as seat 0 in each) this is a large recovery, especially as seat 0. vs v0.4.0's own sampled record (24/28/48 vs tactical over 100, score 0.480; 11/50/39 vs v0.3.0, score 0.305) the win-rate CI vs tactical overlaps, but the score is lower because this policy loses more often rather than drawing; it is ahead of v0.4.0 vs v0.3.0 and slightly ahead of frozen v0.4.0 head-to-head. Full-game landing share is 5–6%, above v0.4.0's ~2–3%.
  - Sources: `runs/checkpoints/v0.4.1_79754fb.json`, `runs/sandbox_buydiv_v0.4.1/eval_unmasked/summary.json`, `runs/sandbox_buydiv_v0.4.1/eval_selfplay/summary.json`, `runs/sandbox_buydiv_v0.4.1/eval_final/summary.json`.
- **Recommended use:** PPO warm start and a sampled opponent-pool member when you want a more even eight-type buy mix than v0.4.0 and can tolerate a quality drop vs the #46 tactical teacher. Always use sampled `policy.act` (or PPO rollouts), not greedy `argmax`. Not a stronger tagged trained policy than v0.3.0.

---

## Scripted baselines

These are heuristic agents, not tagged checkpoints. They play under the current
(#38) rules. Eval is on the v0.2.2 mixed pool (20 maps at 16x8 and 20 at 20x10),
seed 0, seats alternating, vs `runs/checkpoints/v0.3.0_26348ee.pt` unless noted.
Self-play smokes are 40 games; matches vs v0.3.0 are 100 games. Landing ships
are not supposed to attack other ships; `ship_attacks` was 0 in every file
below.

| Version | PR / commit | Buy and vision | vs v0.3.0 (100 games) | Self-play (40 games, seat 0) |
|---|---|---|---|---|
| Inverse-cost random buys, full vision | #43 / bd7d5a1 | Buy weighted by inverse cost (almost never a landing ship: 30 of 4,401 buys vs v0.3.0). Sees the whole map. | **11/18/71**, wr 11.0% (Wilson 95% [6.3, 18.6]), score 0.465. 71 timeouts. Landings: 30 bought, 2 killed, 22 port captures (11 home). | **3/3/34**, 34 timeouts. Each side bought 11 landing ships. Also, in the original 20-game checks in the same eval dir: vs itself 0/2/18, vs a random agent 4/0/16. |
| Uniform build queue, full vision | #44 / 0703c0f | Per-game queue sampled uniformly over the 8 ship types; buy the next ship once cash allows. Still full vision. | **37/48/15**, wr 37.0% (Wilson 95% [28.2, 46.8]), score 0.445. Landings: 212 bought, 65% survived, 52 port captures (37 home). | **16/16/8**, median turn 37 (was 80). 32 of 40 ended in a home capture. |
| Same queue, fog of war | #45 / ad2ce3f | Same queue. Vision matches the policy observation; remembers last-seen enemies for 4 turns. | **40/49/11**, wr 40.0% (Wilson 95% [30.9, 49.8]), score 0.455. Landing survival fell to 43%; 54 port captures (40 home). | **17/16/7**, 7 timeouts. |
| Same queue and fog; landings go to any port | #46 / c1b9124 | **Latest teacher.** After a sighting, a landing ship targets the nearest known capturable port, not the enemy home first. | **29/46/25**, wr 29.0% (Wilson 95% [21.0, 38.5]), score 0.415. Non-home captures rose (26 vs 14 in the fog/home-first match); home captures fell (29 vs 40); 25 timeouts, up from 11. Landing survival 56%. | **13/14/13**, 13 timeouts. Still even, but less decisive than #45. |

Sources: `runs/eval/scripted_v_v0.3.0/summary.json`,
`runs/eval/scripted_v_v0.3.0_queue/summary.json`,
`runs/eval/scripted_v_v0.3.0_fog/summary.json`,
`runs/eval/scripted_v_v0.3.0_anyport/summary.json`, and the matching
`runs/eval/scripted_selfplay_smoke{,_queue,_fog,_anyport}/summary.json` files.
The behavior study in `runs/eval/behavior_scripted_v_v0.3.0/` is a fresh 100-game
sample of the #45 agent (not #46) and went 37/49/14.

The #45 study is the best written read of scripted vs v0.3.0 strategy. The
scripted agent buys each type about equally, often waits on an expensive queue
head (first landing at median turn 10; no landing at all in 13/100 games),
fights carefully (retreats when outgunned), and actually runs merchant routes,
but it is slow to put a landing ship on the enemy home. After #46 it also tends
to sit on a nearby captured port instead of pushing on, which is why draws
went up. Builders are bought often and usually build when they reach a legal
tile (88%), but only about 11% of them ever get there.

**Recommended use:** #46 is the current scripted teacher and the agent cloned
as v0.4.0. For a closer match to v0.3.0, the #45 home-first fog agent is the
stronger measured opponent.

---

## Not found in files

- No `runs/eval` artifact records a head-to-head of
  `runs/selfplay_mixed_v0.2.2/update_1000.pt` against v0.2.0, so that result is
  omitted. The collapse itself is documented in
  `runs/checkpoints/v0.3.0_26348ee.json` and `docs/model_versions.md`.
- There is no behavior audit of v0.1.0, v0.1.1, or v0.2.0 comparable to
  `runs/eval/behavior_scripted_v_v0.3.0/REPORT.md`.
