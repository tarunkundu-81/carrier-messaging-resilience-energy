"""Simultaneous-interval RETI selectors for the existing six-action simulator.

No operator data are used.  A certified winner is a *nominal* confidence
statement conditional on this simulator and its model assumptions:
  * Q: Bonferroni-adjusted, two-sided exact binomial (Clopper-Pearson).
  * E: Bonferroni-adjusted, two-sided Student t intervals for a sample mean.
    The t interval is approximate for the simulator's non-normal energy draws.
  * A candidate must be conservatively feasible (Q_lower >= tau).
  * It must beat EVERY possibly feasible competing action (Q_upper >= tau)
    on conservative Q/E bounds. Candidates with uncertain feasibility are NOT
    silently removed from the competitor set.
  * Otherwise abstain.  A lower-confidence-bound heuristic is supplied as a
    separate UNcertified policy so coverage can be meaningfully compared.

Simultaneous nominal 95% coverage assumes independent message events within
one configuration; actions are paired on common events and NOT independent.
Bonferroni does not require across-action independence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from scipy.stats import beta, t
from reti_sim_epoch_corrected import ACTIONS

@dataclass(frozen=True)
class Interval:
    qlo: float
    qhi: float
    elo: float
    ehi: float
    qhat: float
    ehat: float


def joint_intervals(raw: Mapping, confidence: float = 0.95, actions=ACTIONS):
    """Make adjusted intervals using paired message-level samples, not rounded means."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    m = len(actions)
    if m < 2: raise ValueError("Need at least two candidate actions")
    tail = (1-confidence)/(4*m)   # 2-sided intervals for each of Q,E and m actions
    ans = {}
    n0 = None
    for a in actions:
        st, energy, _ = raw[a]
        e = np.asarray(energy, float)
        if np.any(~np.isfinite(e)) or np.any(e <= 0):
            raise ValueError(f"Nonpositive/nonfinite realized energy for {a}")
        n = len(e)
        if n < 3 or len(st) != n or (n0 is not None and n != n0):
            raise ValueError("Need at least three common event trials per action")
        n0 = n
        k = int(np.isfinite(st).sum())
        qlo = float(beta.ppf(tail, k, n-k+1)) if k else 0.0
        qhi = float(beta.ppf(1-tail, k+1, n-k)) if k < n else 1.0
        ehat = float(e.mean())
        radius = float(t.ppf(1-tail,n-1)*e.std(ddof=1)/np.sqrt(n))
        # Positive energy from simulator; cap lower end for safe ratios.
        ans[a] = Interval(qlo=qlo,qhi=qhi,
                          elo=max(np.finfo(float).tiny,ehat-radius),
                          ehi=ehat+radius,qhat=k/n,ehat=ehat)
    return ans


def decide(intervals: Mapping[str, Interval], tau: float, policy="Certified-RETI"):
    """Return (action or None, status, details) for certified/heuristic policies."""
    if not 0 <= tau <= 1: raise ValueError("tau must be between zero and one")
    safe = [a for a,v in intervals.items() if v.qlo >= tau]
    possible = [a for a,v in intervals.items() if v.qhi >= tau]
    meta = {"safe_count":len(safe),"possible_count":len(possible)}
    if not safe:
        return None,"no_verified_feasible_action",meta
    if policy == "Robust-LCB":
        a=max(safe,key=lambda k:intervals[k].qlo/intervals[k].ehi)
        return a,"heuristic_uncertified",meta
    if policy != "Certified-RETI":raise ValueError(policy)
    for a in safe:
        ra = intervals[a].qlo/intervals[a].ehi
        if all(ra > intervals[b].qhi/intervals[b].elo for b in possible if b!=a):
            return a,"ratio_order_certified",meta
    return None,"uncertain_ranking",meta


def disagreement_margin(summ, higher_energy, lower_energy):
    """Positive => RETI prefers the higher-energy policy action (point estimate).
    Identity: ((Q_a-Q_b)/Q_b) - ((E_a-E_b)/E_b).
    """
    qa,ea=summ[higher_energy]['Q'],summ[higher_energy]['E']
    qb,eb=summ[lower_energy]['Q'],summ[lower_energy]['E']
    if ea <= eb:raise ValueError("higher_energy must truly be more energy costly")
    if qb<=0 or eb<=0:raise ValueError("Q_b and E_b must be positive")
    return (qa-qb)/qb-(ea-eb)/eb
