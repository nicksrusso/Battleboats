# Battleboats — Combat Resolution & Scouting

The two stochastic/geometric rules the engine uses to (a) resolve an attack and
(b) decide what each player can see under fog of war. They are driven by
per-ship-type stats in [`shipTypes.csv`](../battleboats/core/config/shipTypes.csv),
the matchup matrix in [`attackModifiers.csv`](../battleboats/core/config/attackModifiers.csv),
and the spotting-distance matrix in [`visibilityMatrix.csv`](../battleboats/core/config/visibilityMatrix.csv).

---

## Notation

| Symbol | Meaning |
|---|---|
| $S_a,\ S_d$ | attacker / defender **strength** |
| $t_a,\ t_d$ | attacker / defender ship **type** |
| $\mu(t_a, t_d)$ | type-matchup **attack modifier** (matrix; $1$ = neutral) |
| $k$ | $\texttt{kill\_curve\_k}$ — decisiveness exponent (default $2$) |
| $U$ | a draw from $\mathrm{Uniform}[0,1)$ (engine RNG; seeded → reproducible) |
| $d(a,b)$ | Manhattan distance $\lvert a_x-b_x\rvert + \lvert a_y-b_y\rvert$ |
| $R(s, T)$ | **spotting distance**: tiles within which unit $s$ sees unit $T$ (matrix) |
| $\mathcal{U}(o)$ | observer $o$'s units — its ships $\cup$ its ports |

---

## 1. Combat resolution

An attack is a single Bernoulli trial. The kill probability is a **Hill / logistic
curve in the strength ratio**, skewed by the attacker-vs-defender type matchup:

$$
P_{\text{kill}} \;=\; \frac{x^{k}}{1 + x^{k}},
\qquad
x \;=\; \frac{S_a \,\cdot\, \mu(t_a, t_d)}{S_d}.
$$

The defender is destroyed iff

$$
U < P_{\text{kill}}, \qquad U \sim \mathrm{Uniform}[0,1).
$$

**Degenerate case.** A strength-$0$ defender (Landing, Merchant, Builder) is always
destroyed: $S_d = 0 \Rightarrow P_{\text{kill}} \equiv 1$ (and avoids division by zero).

**Properties.**

- **Even matchup** $x = 1 \Rightarrow P_{\text{kill}} = \tfrac12$ — a coin flip when effective strengths are equal.
- **Monotone** in $x$: stronger (or favorably-typed) attackers win more often; $P \to 1$ as $x \to \infty$, $P \to 0$ as $x \to 0$.
- **$k$ controls decisiveness.** Larger $k$ steepens the curve around $x=1$ (toward a step function — small strength edges become near-certain kills); smaller $k$ flattens it (more upsets). Default $k = 2$.

$$
P_{\text{kill}}(x) \;=\; \frac{1}{1 + x^{-k}}
\quad\text{(equivalently, a logistic in } \log x\text{: }\ P = \sigma\!\big(k\ln x\big)).
$$

### Attack modifier $\mu(t_a, t_d)$

A rock-paper-scissors matrix over the five combat types (attacker = row, defender =
column); $1$ is neutral, $>1$ favors the attacker, $<1$ penalizes it. Excerpt from
[`attackModifiers.csv`](../battleboats/core/config/attackModifiers.csv):

| atk ↓ \ def → | Carrier | Battleship | Cruiser | Destroyer | Submarine |
|---|---|---|---|---|---|
| **Carrier** | 1.0 | 1.5 | 1.0 | 0.75 | 1.0 |
| **Battleship** | 1.0 | 1.0 | 1.5 | 1.0 | 0.75 |
| **Cruiser** | 1.25 | 1.0 | 1.0 | 1.5 | 1.0 |
| **Destroyer** | 1.25 | 1.0 | 1.0 | 1.0 | 4.0 |
| **Submarine** | 3.0 | 4.0 | 2.0 | 0.5 | 1.0 |

Non-combat types ($S=0$) need no modifier — they always die when attacked. Note
the Submarine line: strong multipliers against capital ships (Carrier, Battleship)
but $\times 0.5$ against a Destroyer, and the Destroyer's $\times 4$ against a
Submarine — the Destroyer is the dedicated sub-hunter.

---

## 2. Scouting (fog-of-war detection)

Detection is **range-thresholded and asymmetric**. Observer $o$ sees an enemy
target $T$ iff *any* of $o$'s units is within that pair's detection radius:

$$
\text{see}(o \to T) \;\iff\; \exists\, s \in \mathcal{U}(o)\ :\ d(s, T) \,\le\, R(s, T).
$$

The detection radius $R(s, T)$ is read directly from
[`visibilityMatrix.csv`](../battleboats/core/config/visibilityMatrix.csv)
(`ship_data.spot_distance(spotter, target)`): the **row** unit spots the
**column** unit within that many tiles. Ports have their own `Port` row and
column. Every pair over all ship types plus `Port` must be present; the loader
raises if one is missing.

**Asymmetry.** $R(s,T) \neq R(T,s)$ in general: a Destroyer spots a Submarine at
3 while the Submarine spots the Destroyer at 2; a Submarine spots a Carrier at 4
while the Carrier spots it at 2.

**Fog of war (freshness).** Each refresh tick, a sighting is **fresh** if $T$ is
currently within range of some unit in $\mathcal{U}(o)$; otherwise the last-known
record persists as **stale** (remembered position/type, flagged not-current). A
newly-visible enemy overwrites with a fresh record — which is also how a moving-
while-watched ship updates its tracked position and how a witnessed capture
updates a port's last-known owner.

---

## Ship stats

The stats these rules read, from [`shipTypes.csv`](../battleboats/core/config/shipTypes.csv):

| Type | Strength $S$ | Attack range | Speed |
|---|---|---|---|
| Carrier | 100 | 12 | 6 |
| Battleship | 125 | 4 | 3 |
| Cruiser | 75 | 3 | 5 |
| Destroyer | 25 | 3 | 5 |
| Submarine | 50 | 3 | 3 |
| Landing | 0 | 0 | 3 |
| Merchant | 0 | 0 | 3 |
| Builder | 0 | 0 | 3 |

Spotting distances $R(s, T)$, from [`visibilityMatrix.csv`](../battleboats/core/config/visibilityMatrix.csv)
(row spots column):

| spotter ↓ \ target → | Carrier | Battleship | Cruiser | Destroyer | Submarine | Landing | Merchant | Builder | Port |
|---|---|---|---|---|---|---|---|---|---|
| **Carrier** | 6 | 6 | 4 | 3 | 2 | 8 | 8 | 8 | 16 |
| **Battleship** | 4 | 4 | 3 | 3 | 1 | 4 | 4 | 4 | 8 |
| **Cruiser** | 4 | 4 | 3 | 3 | 1 | 6 | 6 | 6 | 12 |
| **Destroyer** | 4 | 4 | 3 | 3 | 3 | 8 | 8 | 8 | 16 |
| **Submarine** | 4 | 4 | 3 | 2 | 1 | 2 | 2 | 2 | 4 |
| **Landing** | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| **Merchant** | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| **Builder** | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 2 | 4 |
| **Port** | 16 | 16 | 12 | 12 | 1 | 8 | 8 | 8 | 16 |

The 5×5 warship block is hand-tuned. The Landing/Merchant/Builder and Port rows
and columns were carried over from the old `scouting × visibility` formula
(non-combat ships scouting 1 / visibility 2, ports 4 / 4, floor 1) and are
placeholders to tune.

---

## Notes

- **One RNG, seeded.** Combat is the only stochastic engine rule; it draws from
  the engine's seeded RNG, so a fixed seed makes attacks (and therefore MCTS
  rollouts) reproducible. `clone()` deep-copies the RNG, so search does not
  consume the real game's randomness.
- **Strength vs. range vs. detection are independent axes.** A unit's ability to
  *win* a fight ($S$, $\mu$), to *reach* one (attack range, speed), and to *see*
  or *hide* (the spotting matrix $R$) are separate — e.g. the Submarine is a
  stealthy brawler with poor reach that only a Destroyer spots before it is
  seen, while the Carrier is a far-seeing, long-reaching, but matchup-vulnerable
  capital ship.
