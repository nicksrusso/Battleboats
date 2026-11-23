"""Epsilon exploration on the build hull choice (the build_ship target).

With probability ``eps`` a learner build decision takes its hull uniformly
from the legal hulls instead of from the policy's ship-type head. Asset and
verb are still sampled from the policy, so exploration only changes WHICH
hull gets built, never whether one is built. Opponent snapshots never explore.

Log-prob stored for PPO (``logprob``):

- ``policy`` (default): store log pi_old(hull), the policy's own probability
  of the hull that was built, as if pi had drawn it. The PPO ratio starts at 1
  and an explored hull gets the ordinary clipped policy-gradient step
  ``A * grad log pi``. This is biased (explored samples are not importance
  weighted, so they count as if on-policy) but the bias is limited to the eps
  fraction of build decisions and goes to 0 as eps anneals to 0.
- ``mixture``: store the behavior log-prob log((1 - eps) pi + eps / n_legal)
  for every build decision (explored or not). This is the textbook importance
  ratio pi_new / mu, but for a hull the policy gives ~1e-12 the ratio is
  ~1e-10, so the gradient on the explored sample is ~0: the policy cannot
  learn the hull it never picks. Kept as an option.

With ``logprob: policy`` an explored hull can be stored with log pi_old of
-100 or lower. If a later epoch moves that hull's probability, the PPO ratio
explodes (approx_kl hit 1e15 in runs/selfplay_12x6_smoke_v5), and for A < 0
the clipped surrogate is unbounded, which NaN'd the weights. Trainers turn on
dual-clip PPO (``ppo.dual_clip``, default 3.0 here) when this mode is used
(see ``ensure_dual_clip``).

``eps`` can anneal linearly from ``eps`` to ``eps_end`` over
``anneal_updates`` updates (0 = constant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

MODES = ("policy", "mixture")


@dataclass
class HullExplore:
    eps: float = 0.0
    eps_end: Optional[float] = None
    anneal_updates: int = 0
    logprob: str = "policy"

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "HullExplore":
        raw = dict(raw or {})
        out = cls(
            eps=float(raw.pop("eps", 0.0)),
            eps_end=None if raw.get("eps_end") is None else float(raw.pop("eps_end")),
            anneal_updates=int(raw.pop("anneal_updates", 0)),
            logprob=str(raw.pop("logprob", "policy")),
        )
        raw.pop("eps_end", None)
        if raw:
            raise ValueError(f"hull_explore: unknown keys {sorted(raw)}")
        out.validate()
        return out

    def validate(self) -> None:
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError("hull_explore.eps must be in [0, 1]")
        if self.eps_end is not None and not 0.0 <= self.eps_end <= 1.0:
            raise ValueError("hull_explore.eps_end must be in [0, 1]")
        if self.anneal_updates < 0:
            raise ValueError("hull_explore.anneal_updates must be >= 0")
        if self.logprob not in MODES:
            raise ValueError(f"hull_explore.logprob must be one of {MODES}")

    @property
    def enabled(self) -> bool:
        return self.eps > 0.0 or (self.eps_end or 0.0) > 0.0

    def eps_at(self, update: int) -> float:
        """eps for a 0-based update index."""
        end = self.eps if self.eps_end is None else self.eps_end
        if self.anneal_updates <= 0:
            return self.eps
        frac = min(1.0, max(0, int(update)) / self.anneal_updates)
        return self.eps + (end - self.eps) * frac

    def describe(self) -> str:
        end = self.eps if self.eps_end is None else self.eps_end
        return f"eps={self.eps} eps_end={end} anneal_updates={self.anneal_updates} logprob={self.logprob}"


def empty_explore_stats() -> Dict[str, Any]:
    """Learner build decisions, how many explored, and hulls by source."""
    return {"build": 0, "explored": 0, "explored_hulls": {}, "policy_hulls": {}}


def merge_explore_stats(into: Dict[str, Any], other: Dict[str, Any]) -> None:
    into["build"] += int(other.get("build", 0))
    into["explored"] += int(other.get("explored", 0))
    for key in ("explored_hulls", "policy_hulls"):
        for name, n in (other.get(key) or {}).items():
            into[key][name] = into[key].get(name, 0) + int(n)


def record_build(stats: Dict[str, Any], hull_name: str, explored: bool) -> None:
    stats["build"] += 1
    key = "explored_hulls" if explored else "policy_hulls"
    if explored:
        stats["explored"] += 1
    stats[key][hull_name] = stats[key].get(hull_name, 0) + 1


def set_policy_explore(policy, eps: float, mode: str = "policy") -> None:
    policy.hull_explore_eps = float(eps)
    policy.hull_explore_mode = str(mode)


def format_explore(stats: Dict[str, Any], eps: float) -> str:
    land_e = stats["explored_hulls"].get("landing", 0)
    land_p = stats["policy_hulls"].get("landing", 0)
    return f"explore=eps:{eps:.3f},{stats['explored']}/{stats['build']},landing:e{land_e}/p{land_p}"


DEFAULT_DUAL_CLIP = 3.0


def ensure_dual_clip(explore: "HullExplore", ppo) -> Optional[str]:
    """policy-mode exploration needs dual-clip PPO; set it if the config did not.
    Returns a log line when it changed ``ppo.dual_clip``."""
    if explore.enabled and explore.logprob == "policy" and getattr(ppo, "dual_clip", None) is None:
        ppo.dual_clip = DEFAULT_DUAL_CLIP
        return f"ppo.dual_clip unset; using {DEFAULT_DUAL_CLIP} for logprob=policy hull exploration"
    return None
