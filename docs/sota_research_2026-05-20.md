# SOTA Research — Architectures, Orchestration, and Training Methods

**Date:** 2026-05-20
**Scope:** Survey of state-of-the-art in deep RL relevant to battleboats (2-player turn-based naval strategy, fog of war, variable-cardinality entities, sparse zero-sum terminal reward, planned: PPO self-play → league).
**Method:** Four parallel research agents (architectures / hierarchy / training / imperfect-info), synthesized below.

---

## Headline finding

The closest published analog to battleboats is **DeepNash on Stratego** (Perolat et al., *Science* 2022) — 2-player zero-sum, fog of war, no chance nodes, long horizon. DeepNash beat human experts using a **flat** network + model-free RL with **R-NaD (Regularized Nash Dynamics)**. No explicit belief tracker, no search, no opponent modeling, no temporal hierarchy.

Reinforcing this: Rudolph et al. ("Reevaluating Policy Gradient Methods for Imperfect-Information Games," arXiv:2502.08938, Feb 2025) ran 7,000+ training runs across five imperfect-info benchmarks and found **PPO matches or beats Deep CFR / NFSP / PSRO on exploitability**. The specialized imperfect-info machinery isn't paying for itself anymore even on the games it was designed for.

Together these two results validate the current battleboats direction strongly and undercut most temptations toward CFR-family or hierarchical detours.

---

## 1. Network architectures

### What every recent SOTA agent uses

The dominant pattern across AlphaStar, OpenAI Five, DeepNash, Cicero, MuZero family:

- **Transformer / self-attention over a variable-length set of entity tokens** (units, pieces, hands).
- **CNN over the 2D spatial state** (board / map).
- **Scatter–gather connections** between the entity stream and the spatial stream.
- **LSTM or GRU recurrent core** over time for belief tracking under fog.
- **Autoregressive policy head** with **pointer attention** for variable-cardinality outputs (entity selection, target selection).
- **Centralized (asymmetric) critic** with privileged info during training only.

### Specific techniques worth knowing

| Technique | Paper | Relevance |
|---|---|---|
| **Deep Sets** | Zaheer et al. 2017 | Baseline permutation-invariant set encoder. Useful ablation target. |
| **Set Transformer (ISAB, PMA)** | Lee et al. 2019 | O(n²) overhead irrelevant at our entity counts; PMA pooling (learned-query attention) is the right way to summarize the entity set for the value head. |
| **AlphaStar entity transformer** | Vinyals et al., *Nature* 2019 | The canonical pattern for our exact problem class. Scale it down ~100×. |
| **Scatter Connection** (AlphaStar) | Vinyals et al. 2019 | Project entity embeddings → scatter onto grid at entity coordinates → convolve. Fuses grid and entity streams with zero added params. Strongly preferred over flat-concat. |
| **Pointer Networks** | Vinyals, Fortunato, Jaitly 2015 | Load-bearing primitive for "pick entity" and "pick target" heads. Variable-length output for free. |
| **Autoregressive factored policy head** | AlphaStar 2019; "Factored Action Spaces in Deep RL," OpenReview 2022 | Autoregressive consistently beats independent-factor heads when factors are dependent. They always are in `entity → action → target` schemas. |
| **Action masking at every autoregressive step** | AlphaStar 2019 | Set invalid logits to `-inf` at every head. Easy to forget, load-bearing. |
| **Rethinking Transformers in Solving POMDPs** | Lu et al., ICML 2024 | Pure causal transformers *underperform* on POMDPs vs recurrent cores. Keep an LSTM/GRU even if you transformerize the spatial/entity axis. |
| **Bi-directional recurrence improves transformer in POMDPs** | 2025, arXiv:2505.11153 | Reports 87–482% improvements on POMDP benchmarks vs pure transformers. |
| **Centralized asymmetric critic (CTDE)** | Lowe et al. (MADDPG) 2017; Pinto et al. 2017 | Critic sees both players' state, actor only sees own. Discard critic at deployment. Single biggest variance reducer for self-play. |
| **AlphaStar pseudo-reward value heads** | Vinyals et al. 2019 | Add auxiliary value heads (ships sunk, ports captured) alongside terminal +1/-1. Likely more impactful for sparse-reward learning than which architecture you pick. |
| **DreamerV3 engineering tricks** | Hafner et al., *Nature* 2025 | Symlog transforms, two-hot value head, RMSNorm. Cheap, model-free–compatible wins. |

### Deliberate omissions

- **GNNs.** Attention dominates on complete graphs; naval entities form a fully-connected graph constrained only by distance, which is the regime where attention beats GNNs.
- **Perceiver IO, Mamba, Transformer-XL.** Worthwhile bets at much larger scale; LSTM is sufficient here.
- **MuZero / EfficientZero.** Model-based RL is overkill when we have a perfect simulator and want on-policy data for self-play.
- **GFlowNets.** Optimize for diverse high-reward samples; zero-sum games want value maximization.
- **Decision Transformer.** Offline-RL paradigm; conditioning on return-to-go doesn't extend cleanly to adversarial self-play.

---

## 2. Hierarchical RL / orchestration

### The empirical verdict: do not bother

Every major recent game-playing result was **flat**: AlphaStar, MuZero, DeepNash, Cicero. The "hierarchy" in AlphaStar is **architectural and action-decomposition** (autoregressive pointer heads), not temporal (manager/worker). The action-decomposition piece is what we're already planning; that is the only "hierarchy" that has demonstrably worked on competitive game-playing at scale.

### What was investigated and why we're skipping it

| Technique | Paper | Why skip |
|---|---|---|
| **FeUdal Networks (FuN)** | Vezhnevets et al. 2017 | Manager–worker gradient separation, brittle goal-space dimensioning, intrinsic-reward shaping that often dominates extrinsic signal. Single-agent, not designed for variable entity counts. |
| **Option-Critic** | Bacon, Harb, Precup 2017 | Notorious "option collapse" and "option-as-primitive degeneration" pathologies. Optimizer-fighting time exceeds modeling-time. |
| **HIRO** | Nachum et al. 2018 | Off-policy goal-conditioned hierarchy assuming continuous state-space goals; doesn't fit a discrete strategy game. |
| **Director** | Hafner et al., NeurIPS 2022 | Solid hierarchy formulation but requires a learned world model — unjustified when we have an exact simulator. |
| **HiPPO** | Li & Florensa 2019 | Most PPO-compatible hierarchy variant. Holdout option only if a concrete failure mode demands hierarchy. |
| **DIAYN, VISR, EDL (skill discovery)** | Eysenbach et al. 2018; Hansen et al. 2020; Campos et al. 2020 | Designed for environments with no reward signal where exploration is the bottleneck. Battleboats has reward; bottleneck is credit assignment, not skill discovery. Never used in competitive game-playing at scale. |
| **HER (Hindsight Experience Replay)** | Andrychowicz et al. 2017 | Requires explicit (state, goal) framing; "win the game" isn't a state. |

### Lessons that *do* transfer from AlphaStar

1. **Action-decomposition hierarchy** (action-type → unit → target via autoregressive pointer heads) — already in our plan, keep doing it.
2. **Strategic diversity via a latent style variable** sampled per episode — small, cheap change vs a real manager/worker split. Use this *first* if monoculture appears in self-play.
3. AlphaStar had **human replays** for supervised init; we don't. This affects exploration budgets, not architecture choice.

---

## 3. Training methods (self-play, league, QDRL)

### Self-play variants — concept ladder

Each rung relaxes the assumption that "stronger" is a totally ordered scalar:

1. **Naive self-play** (latest vs latest). Cycles in non-transitive games; catastrophic forgetting. Useful diagnostic baseline — *deliberately* hit the failure mode to understand what subsequent methods are solving.
2. **Fictitious Self-Play (FSP)** — Brown 1951; Heinrich, Lanctot, Silver, ICML 2015. Best-respond to **time-averaged** opponent strategy. Resolves cycling.
3. **Neural FSP (NFSP)** — Heinrich & Silver 2016. Two networks per agent: RL best-response + SL average-policy net. Over-engineered for our scale.
4. **Prioritized Fictitious Self-Play (PFSP)** — AlphaStar 2019. Sample opponents weighted by `f(P[win])` (e.g., `(1−p)²`). Cheap, effective, well-understood. **The workhorse to reach for first.**
5. **Double Oracle / PSRO** — McMahan et al. 2003; Lanctot et al., NeurIPS 2017. Unification: PSRO = Double Oracle with learned policies; meta-Nash subsumes uniform-history (FSP) and put-all-weight-on-newest (naive self-play). Variants: Rectified PSRO, Pipeline PSRO (P2SRO), Joint PSRO, α-PSRO, PSD-PSRO.

### League training

AlphaStar's three roles:

- **Main agents** — PFSP against the whole league. Production output, must be robust.
- **Main exploiters** — train only vs current main agents; reset weights periodically. Adversarial probes that find new weaknesses rather than refining old ones.
- **League exploiters** — PFSP against whole league; reset periodically. Find systemic weaknesses across the population.

Three roles exist because PFSP alone leaves stable blind spots; exploiters are diversity drivers, not Nash-optimal agents.

**Connection to PSRO:** League training is *approximately* PSRO with a hand-engineered meta-solver and best-response procedure. ROA-Star (Huang et al., NeurIPS 2023) refined this with goal-conditioned exploiters and explicit opponent modeling, achieving superhuman SC2 at much lower compute.

**Minimum viable league for battleboats:** 1 main (PFSP) + 1 exploiter (resets, trains only vs main). Add a second exploiter only if observed blind spots persist.

### Quality-Diversity (QDRL) — skip for now

| Technique | Paper | Verdict |
|---|---|---|
| **MAP-Elites** | Mouret & Clune 2015 | Single-agent open-ended search foundation. |
| **PGA-MAP-Elites** | Nilsson & Cully, GECCO 2021 | Adds policy-gradient operator alongside GA mutation. |
| **Diversity via Determinants (DvD)** | Parker-Holder et al., NeurIPS 2020 | Determinant of behavioral-embedding Gram matrix as diversity metric. Usable as a regularizer. |
| **GAME (Generational Adversarial MAP-Elites)** | Faldor et al. 2025 | Adapts QD to adversarial coevolution. |
| **PSD-PSRO** | 2023 | Bridges QD and PSRO via diversity regularization in best-response objective. |

**Verdict:** Behavior-characterization design is itself a research project for a strategy game. PFSP's snapshot pool naturally provides behavior diversity for free. Revisit only if Phase 5 plateaus from monoculture. The diversity-as-regularizer angle (DvD, PSD-PSRO) is more directly applicable than a full MAP-Elites archive.

### PPO improvements

| Technique | Paper | Verdict |
|---|---|---|
| **PPG (Phasic Policy Gradient)** | Cobbe et al., ICML 2021 | ~30-line addition to PPO; modest sample-efficiency win. Save for post-baseline. |
| **V-MPO** | Song et al., ICLR 2020 | Wins mainly in massive multi-task setups; not worth implementation complexity. |
| **RND (Random Network Distillation)** | Burda et al., ICLR 2019 | Risky under partial observability — "noisy TV" can reward novel sighting configurations without strategic value. |
| **PopArt** | Hessel et al., AAAI 2019 | Only relevant if mixing dense rewards of different scales. |

---

## 4. Imperfect information & opponent modeling

### Belief state

| Approach | Verdict |
|---|---|
| **Implicit via recurrent/transformer policy** | The dominant practical approach. AlphaStar and DeepNash both rely on this. Likely sufficient for battleboats. |
| **Particle filter / explicit Bayesian belief** | Cheap to implement (~50 lines). Use as a **debug/interpretability tool**, *not* as policy input. |
| **DeepStack-style public belief states (PBS)** | Mathematical machinery requires clean public/private factorization (poker-shaped). Battleboats sightings/movement entangle public and private. Not worth importing. |
| **I-POMDP / interactive POMDP** | Intractable in general; truncated variants (PR2) exist but aren't used in production game-playing. |

### Opponent modeling

| Technique | Paper | Verdict |
|---|---|---|
| **ToMnet** | Rabinowitz et al., ICML 2018 | Useful only if there's a population of distinct opponent styles to adapt to within-episode. Self-play handles this implicitly. |
| **LOLA** | Foerster et al., AAMAS 2018 | General-sum tool (cooperative-equilibrium shaping). Inapplicable to zero-sum. |
| **M-FOS / opponent shaping** | Lu et al., ICML 2022 | Same — general-sum, inapplicable here. |
| **Recursive reasoning (PR2)** | Wen et al. 2019 | Redundant against a self-play partner converging to equilibrium. |

### Search + learning for imperfect info

| Technique | Paper | Verdict |
|---|---|---|
| **ReBeL** | Brown et al., NeurIPS 2020 | Provably convergent to Nash in 2p zero-sum. PBS factorization requires clean public/private structure — partially available here. **Most principled level-up target** if we ever want competition-grade play. |
| **Player of Games / Student of Games** | Schmid et al. 2021/2023 | Growing-Tree CFR + AlphaZero-style search; works on Scotland Yard (structurally similar to battleboats). **Closest published technique by problem structure.** Heavy lift. |
| **Information Set MCTS (IS-MCTS)** | Cowling et al. 2012 | Lighter-weight upgrade than ReBeL; search over information sets atop a learned value net. |
| **PIMC** (Perfect-Information Monte Carlo with determinizations) | GIB / bridge literature | Cannot bluff (strategy fusion). Fine as an early benchmark, not a final agent. |

### CFR family

Tabular CFR / CFR+ / MCCFR are infeasible at battleboats' state-space scale. Deep CFR / DREAM / ARMAC are technically applicable but engineering-heavy. The Rudolph 2025 finding undercuts the case for any of them at our scale.

### Exploration under partial observability

- **Curiosity / RND under fog:** noisy-TV pathology — agent rewarded for novel *observations* that don't correspond to genuinely novel underlying state. Use weakly or skip.
- **Active perception:** information-gain is already implicit in the optimal Q-value; a well-trained value function learns scouting automatically. Adding an explicit info-gain reward risks shaping bias.
- **Strategy-space exploration is the real exploration problem**, addressed by PFSP/league, not curiosity.

---

## 5. Evaluation hygiene

Already planned: **ELO + pairwise win-rate matrix**. Add:

- **Nash averaging** (Balduzzi et al., NeurIPS 2018) once snapshot pool ≥5 agents. One-line `nashpy` call given the win-rate matrix. **Invariant to redundant agents** — adding clones of weak agents doesn't change rankings. ELO alone hides cycling.
- **α-Rank** (Omidshafiei et al., *Scientific Reports* 2019) — Markov-Conley chains over the population; unique solution, scales to multi-player, handles non-transitivity. Worth adding if we ever go asymmetric.

Never rely on a single scalar to certify "agent improved." That's how cycling agents ship looking great on ELO and lose to last month's checkpoint.

---

## 6. Recommended progression for battleboats

| Phase | What | Why |
|---|---|---|
| 4a | **Naive PPO self-play** (latest vs latest), ~1M frames, track win rate vs heuristic | Expect cycling: win rate climbs, plateaus, drops. *Pedagogically critical* — see the failure mode in own runs. |
| 4b | **Snapshot self-play, uniform sampling** (FSP-lite) | Cycling resolves. Strictly simpler than NFSP. |
| 4c | **PFSP** (`f(p) = (1−p)²`) | AlphaStar's exact mechanism. Cleaner improvement, better pairwise matrix. |
| 5 | **Mini-league**: 1 main + 1 exploiter (resets periodically, trains only vs main) | Adversarial probing. Add second exploiter only if blind spots persist. |
| 6+ (optional) | **GT-CFR / Player of Games** or **ReBeL** | Only if we want exploitability bounds or superhuman play. |

---

## 7. Concrete ideas to try

1. **R-NaD-style regularization in PPO.** DeepNash's whole insight: self-play in zero-sum imperfect-info games *cycles* without explicit regularization. PPO's entropy + KL-to-previous-policy partially substitutes; R-NaD's reward-transformation is the principled version. Worth a read.
2. **Latent style variable per episode** (AlphaStar). Sample a categorical style at episode start; condition the policy on it. Strategic diversity without a full league.
3. **DreamerV3 engineering tricks on top of PPO** — symlog transforms, two-hot value head, RMSNorm. Independent of model-free vs model-based; just better PPO.
4. **Pseudo-reward value heads** (ships sunk, ports captured) alongside terminal +1/-1. AlphaStar's sparse-reward trick.
5. **Particle-filter belief tracker as debug tool**, not policy input. Inspect whether the network is learning sensible belief over hidden ship positions.
6. **Nash averaging** added to eval once snapshot pool ≥5 agents.
7. **Scatter–gather** between the grid CNN and entity transformer (not flat-concat).
8. **Asymmetric centralized critic** — same encoder, but fed both players' state. Discard at deployment.

---

## 8. Sources (selected)

- AlphaStar — [Vinyals et al., *Nature* 2019](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
- DeepNash / R-NaD — [Perolat et al., *Science* 2022](https://arxiv.org/abs/2206.15378)
- Player of Games / Student of Games — [Schmid et al. 2021/2023](https://arxiv.org/abs/2112.03178)
- ReBeL — [Brown et al., NeurIPS 2020](https://arxiv.org/abs/2007.13544)
- Pointer Networks — [Vinyals, Fortunato, Jaitly 2015](https://arxiv.org/abs/1506.03134)
- Rethinking Transformers in POMDPs — [Lu et al., ICML 2024](https://arxiv.org/abs/2405.17358)
- DreamerV3 — [Hafner et al., *Nature* 2025](https://arxiv.org/abs/2301.04104)
- PSRO — [Lanctot et al., NeurIPS 2017](https://arxiv.org/abs/1711.00832)
- PSRO survey — [Bighashdel et al. 2024](https://arxiv.org/html/2403.02227v1)
- ROA-Star — [Huang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
- Reevaluating Policy Gradient for Imperfect-Info Games — [Rudolph et al. 2025](https://arxiv.org/abs/2502.08938)
- MAP-Elites — [Mouret & Clune 2015](https://arxiv.org/abs/1504.04909)
- PGA-MAP-Elites — [Nilsson & Cully, GECCO 2021](http://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2021/GECCO/proceedings/proceedings_files/p866-nilsson.pdf)
- FeUdal Networks — [Vezhnevets et al. 2017](https://arxiv.org/abs/1703.01161)
- Director — [Hafner et al., NeurIPS 2022](https://arxiv.org/abs/2206.04114)
- Nash averaging — [Balduzzi et al., NeurIPS 2018](https://arxiv.org/abs/1806.02643)
- α-Rank — [Omidshafiei et al., *Sci Rep* 2019](https://www.nature.com/articles/s41598-019-45619-9)
