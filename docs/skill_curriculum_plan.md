# Skill Curriculum Plan (High Level)

## Goal

Train a full-game agent **without the heuristic** (no φ tuning, no MCTS leaf-eval dependence for learning).

**Bet:** dense, ship-local sandbox skills create behavioral priors so full-game exploration reaches terminals; sparse win/loss can then fine-tune a single 3-headed policy.

## Constraints

- Heuristic stays out of the training loop.
- Sub-agents reuse the **existing 3-headed `PolicyNetwork`** (transformer + asset/verb/target + value).
- Final product: one flat policy at inference (scaffolding removed).

## Stages

### 1. Specialist sandboxes

- One skill / ship role at a time (start: **merchant**).
- Tiny maps (e.g. 2×3: port + home + merchant).
- Bootstrap from random; **dense reward local to the skill** (e.g. deliver materials).
- Same network architecture; illegal verbs masked.

### 2. Curriculum within a skill

- Grow map / add threats (e.g. 3×4 + enemy destroyer).
- Multi-objective rewards (e.g. deliver + survive).
- Gate: skill still succeeds at a target rate before advancing.

### 3. Cover missing roles

Skills needed before full games can end, at least:

- Merchant logistics  
- Combat  
- Landing / capture  
- Build / port economy (or script builds early)

### 4. Compose full games (no learned manager yet)

- Fixed composition rule: which ship net acts when; scripted or simple build/endturn.
- Measure: delivery rate, attack rate, **% decisive games**.
- If decisive % ≈ 0, do not train a manager yet.

### 5. Manager + takeover

- Manager can be the existing 3-headed net (possibly modified).
- Optional: mix skill policy vs manager by a confidence/gate weight.
- Prefer path: **BC / distill from composed rollouts → PPO on full game** (terminal ±1; small event bonuses only if needed — still not φ).
- Anneal mixture → pure 3-headed policy.

### 6. Drop scaffolding

- Inference: only the final 3-headed agent.

## Explicit non-goals (for now)

- Tuning the existing heuristic / regression weights.
- God-mode MCTS as the learning signal.
- Permanent multi-policy feudal control at deploy time.

## Open decisions

- Skill obs: local crop vs full fog tokens  
- Distinct weights per skill vs one multi-task net  
- Runtime mixture vs distill-only  
- Composition rule for step 4 (build, endturn, multi-ship)

## Success signals

| Stage | Signal |
|-------|--------|
| Sandbox | High success rate on skill reward |
| Compose | Non-trivial decisive game % |
| Manager / PPO | Stable entropy/KL; wins above random / composed baseline |
| Final | Scaffolding off; still wins |
