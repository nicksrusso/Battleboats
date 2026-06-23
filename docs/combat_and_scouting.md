# Battleboats — Combat Resolution & Scouting

The two stochastic/geometric rules the engine uses to (a) resolve an attack and
(b) decide what each player can see under fog of war. Both are driven by
per-ship-type coefficients in [`shipTypes.csv`](../battleboats/core/config/shipTypes.csv)
and the matchup matrix in [`attackModifiers.csv`](../battleboats/core/config/attackModifiers.csv).

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
| $\sigma(u)$ | **scouting** coefficient of unit $u$ (how far it sees) |
| $\nu(u)$ | **visibility** coefficient of unit $u$ (how easily it is seen) |
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
| **Destroyer** | 1.25 | 1.0 | 1.0 | 1.0 | 1.5 |
| **Submarine** | 2.0 | 2.0 | 1.0 | 0.5 | 1.0 |

Non-combat types ($S=0$) need no modifier — they always die when attacked. Note
the Submarine line: $\times 2$ against capital ships (Carrier, Battleship) but
$\times 0.5$ against a Destroyer — the Destroyer is the dedicated sub-hunter.

---

## 2. Scouting (fog-of-war detection)

Detection is **range-thresholded and asymmetric**. Observer $o$ sees an enemy
target $T$ iff *any* of $o$'s units is within that pair's detection radius:

$$
\text{see}(o \to T) \;\iff\; \exists\, s \in \mathcal{U}(o)\ :\ d(s, T) \,\le\, R(s, T).
$$

The detection radius factorizes into the **scout's** reach times the **target's**
conspicuousness:

$$
R(s, T) \;=\; \sigma(s)\,\cdot\,\nu(T).
$$

Ports carry no per-instance stats; they use module constants
$\sigma_{\text{port}} = \nu_{\text{port}} = 4$.

**Asymmetry.** Because $R$ mixes the scout's $\sigma$ with the target's $\nu$,
visibility is not mutual: in general $R(s,T) \neq R(T,s)$. A high-$\sigma$ Carrier
spots far; a low-$\nu$ Submarine ($\nu = 1$) stays hidden until something is nearly
on top of it.

**Fog of war (freshness).** Each refresh tick, a sighting is **fresh** if $T$ is
currently within range of some unit in $\mathcal{U}(o)$; otherwise the last-known
record persists as **stale** (remembered position/type, flagged not-current). A
newly-visible enemy overwrites with a fresh record — which is also how a moving-
while-watched ship updates its tracked position and how a witnessed capture
updates a port's last-known owner.

---

## Ship coefficients

The stats these rules read, from [`shipTypes.csv`](../battleboats/core/config/shipTypes.csv):

| Type | Strength $S$ | Attack range | Speed | Visibility $\nu$ | Scouting $\sigma$ |
|---|---|---|---|---|---|
| Carrier | 100 | 12 | 6 | 4 | 4 |
| Battleship | 125 | 4 | 3 | 4 | 2 |
| Cruiser | 75 | 3 | 5 | 3 | 3 |
| Destroyer | 25 | 3 | 5 | 3 | 3 |
| Submarine | 100 | 3 | 3 | 1 | 1 |
| Landing | 0 | 0 | 3 | 2 | 1 |
| Merchant | 0 | 0 | 3 | 2 | 1 |
| Builder | 0 | 0 | 3 | 2 | 1 |
| *(Port)* | — | — | — | 4 | 4 |

---

## Notes

- **One RNG, seeded.** Combat is the only stochastic engine rule; it draws from
  the engine's seeded RNG, so a fixed seed makes attacks (and therefore MCTS
  rollouts) reproducible. `clone()` deep-copies the RNG, so search does not
  consume the real game's randomness.
- **Strength vs. range vs. detection are independent axes.** A unit's ability to
  *win* a fight ($S$, $\mu$), to *reach* one (attack range, speed), and to *see*
  ($\sigma$) or *hide* ($\nu$) are separate coefficients — e.g. the Submarine is a
  strong, stealthy ($\nu=1$) brawler with poor reach, while the Carrier is a
  far-seeing, long-reaching, but matchup-vulnerable capital ship.
