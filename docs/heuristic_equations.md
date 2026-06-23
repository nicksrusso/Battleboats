# Battleboats — Original Heuristic Evaluation

The hand-designed state evaluator `heuristic_eval(s, me)` used by the MCTS leaf
evaluator and as the linear-regression baseline. It scores a non-terminal state
in $[-1, +1]$ from player `me`'s perspective as a **bounded linear model over
features**: a weighted sum of interpretable features squashed through $\tanh$.

> These are the **original six features** (iteration 1), plus the
> worst-case-matchup term (§7) that the regression pass later added. The
> remaining regression-tuned additions — per-type ship counts and the other
> matchup-matrix terms — are omitted here.

---

## Notation

| Symbol | Meaning |
|---|---|
| $F,\ E$ | my ships, enemy ships (subscript $c$ = combat ships only) |
| $M$ | my merchant ships |
| $P_{\text{me}},\ P_{\text{opp}}$ | owned port positions |
| $t_{\text{me}},\ t_{\text{opp}}$ | my / enemy **home** port position |
| $d(a,b)$ | Manhattan distance between two cells |
| $D = W + H$ | map diagonal (width + height) |
| $v(\cdot),\ \mathrm{str}(\cdot)$ | ship material value, ship strength |
| $\mathrm{pk}(i\!\to\! j)$ | probability ship $i$ kills ship $j$ (kill curve) |
| $w_k,\ b$ | feature weight, intercept |

---

## Proximity kernel

Every spatial term reuses one normalized, scale-free linear proximity:

$$
\rho(a, b) \;=\; \max\!\left(0,\; 1 - \frac{d(a, b)}{D}\right)
$$

$\rho = 1$ when co-located, decaying linearly to $0$ at a map-diagonal away.

---

## State evaluation

$$
H(s) =
\begin{cases}
+1, & \text{$me$ has won} \\[2pt]
-1, & \text{$me$ has lost} \\[2pt]
\tanh\!\left( \displaystyle\sum_{k} w_k\, \phi_k(s) \;+\; b \right), & \text{otherwise}
\end{cases}
$$

The features $\phi_k$ follow. **Sign convention:** positive favors `me`.

---

## 1. Material differential

Zero-sum differential of ship value plus a per-port bonus $c$.

$$
\phi_{\text{mat}} =
\Big( \textstyle\sum_{i \in F} v(i) + c\,|P_{\text{me}}| \Big)
-
\Big( \textstyle\sum_{j \in E} v(j) + c\,|P_{\text{opp}}| \Big)
$$

## 2. Home-pressure differential

Threat on each home, summed over combat-capable attackers (Landings use a fixed
weight $W_{\text{land}}$; other combat ships use their strength).

$$
\Pi(t, S) = \sum_{\substack{s \in S \\ s\ \text{can threaten}}} w_s\,\rho(\mathrm{pos}_s, t),
\qquad
w_s =
\begin{cases}
W_{\text{land}}, & s\ \text{is a Landing} \\
\mathrm{str}(s), & \text{otherwise}
\end{cases}
$$

$$
\phi_{\text{home}} = \Pi(t_{\text{opp}}, F) \;-\; \Pi(t_{\text{me}}, E)
$$

## 3. Combat balance

Win-only, proximity-weighted, **averaged per enemy ship** (so building more ships
doesn't dilute existing balance). For each enemy combat ship, average the
favorable matchup margin across my combat ships that beat it.

$$
m(i, j) = \mathrm{pk}(i\!\to\! j)\,\mathrm{str}(i) - \mathrm{pk}(j\!\to\! i)\,\mathrm{str}(j)
$$

$$
A_j = \{\, i \in F_c : m(i, j) > 0 \,\}
$$

$$
\phi_{\text{cb}} = \sum_{j \in E_c}
\frac{1}{|A_j|} \sum_{i \in A_j} m(i, j)\,\rho(\mathrm{pos}_i, \mathrm{pos}_j)
\qquad (\text{term} = 0 \ \text{if}\ A_j = \varnothing)
$$

## 4. Economic value (one-sided)

Cash, port stockpiles, and merchant cargo in one consistent unit — cargo gets a
premium for nearing home, and spare capacity gets a small boost ($\beta$) for
being near a loading port.

$$
\phi_{\text{econ}} =
\underbrace{\mathrm{cash}}_{\text{liquid}}
+ \sum_{p \in P_{\text{me}}} \mathrm{stock}(p)
+ \sum_{m \in M} \Big[
\underbrace{\mathrm{cargo}_m\big(1 + \alpha\,\rho(\mathrm{pos}_m, t_{\text{me}})\big)}_{\text{loaded — premium for nearing home}}
+
\underbrace{(C - \mathrm{cargo}_m)\,\beta\,\rho\big(\mathrm{pos}_m, P^{-}_{\text{me}}\big)}_{\text{spare capacity near a loading port}}
\Big]
$$

where $C$ = merchant capacity, $\alpha = \texttt{ECON\_ALPHA}$, $\beta = \texttt{ECON\_BETA}$,
and proximity to the nearest owned **non-home** port $P^{-}_{\text{me}}$ is

$$
\rho\big(\mathrm{pos}_m, P^{-}_{\text{me}}\big) = \max\!\left(0,\; 1 - \frac{\min_{p \in P^{-}_{\text{me}}} d(\mathrm{pos}_m, p)}{D}\right).
$$

## 5. Landing capability (discrete)

A capability gate — without a Landing, `me` literally cannot capture the enemy home.

$$
\phi_{\text{hasLanding}} = \mathbf{1}\big[\, |\{\, s \in F : s\ \text{is a Landing} \,\}| \ge 1 \,\big]
$$

## 6. Landing pressure (one-sided)

Reward Landings for closing on the enemy home (independent of strength — Landings
have $\mathrm{str} = 0$).

$$
\phi_{\text{landPress}} = \sum_{\substack{s \in F \\ s\ \text{is a Landing}}} \rho(\mathrm{pos}_s, t_{\text{opp}})
$$

## 7. Worst-case combat matchup (coverage)

A min-over-max **defensive floor**: for every enemy combat ship, take my *best*
counter's matchup margin; the feature is the **worst** such value across all
enemy combat ships. It flags the single threat I am least equipped to answer —
a negative value means some enemy ship out-matches every ship I own.

Reusing the matchup margin $m(i, j)$ from §3 (here **without** the proximity
weight — it is a roster-capability term, not a positional one):

$$
\phi_{\text{cov}} = \min_{j \in E_c}\ \max_{i \in F_c}\ m(i, j)
\qquad (\phi_{\text{cov}} = 0 \ \text{if}\ F_c = \varnothing \ \text{or}\ E_c = \varnothing)
$$

Contrast with $\phi_{\text{cb}}$ (§3): combat balance **sums favorable** margins
across the whole fleet (offensive potential, proximity-weighted), whereas
$\phi_{\text{cov}}$ is the **min of best counters** (defensive worst case,
position-independent). A fleet can have strong aggregate balance yet a negative
coverage floor if one enemy ship type goes uncountered.

---

## Notes

- **Bounded linear model.** $H = \tanh(\sum_k w_k \phi_k + b)$ is linear-in-features
  before the squash — which is exactly why **linear regression** on harvested
  positions was the natural next step for tuning the weights $w_k$.
- **One spatial primitive.** Every "how close is X to Y" term reuses $\rho$,
  normalized by the map diagonal $D = W + H$, so the heuristic is scale-free
  across board sizes.
- **Sign convention.** Positive favors `me` throughout. Only $\phi_{\text{mat}}$
  and $\phi_{\text{home}}$ are zero-sum (literally me $-$ opponent). The rest —
  $\phi_{\text{cb}}$, $\phi_{\text{econ}}$, $\phi_{\text{hasLanding}}$, and
  $\phi_{\text{landPress}}$ — are one-sided ($\ge 0$). $\phi_{\text{cb}}$ *looks*
  two-sided because its per-matchup margin $m(i,j)$ subtracts the enemy's return
  damage, but the win-only filter $A_j$ keeps only my favorable matchups, so
  being out-matched contributes $0$, never a penalty — it is one-sided.
- **Weights.** The original $w_k$ were hand-picked first-cut values (home threat
  weighted $\approx 6\times$ economy); a later linear regression on harvest data
  replaced them.
