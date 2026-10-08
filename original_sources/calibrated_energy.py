"""
calibrated_energy.py: replaces the illustrative relative-energy costs with costs built from
published measurements listed in data/parameter_sources.csv (Chen et al. 2017, Carroll and
Heiser 2010). Success probabilities are NOT calibrated (no public measurement exists) and are
held at the paper's illustrative values, so only the energy axis changes.

Unit convention: charge in mAs for LTE data-path terms (Chen), joules for the SMS term
(Carroll), joined through a battery-voltage ASSUMPTION swept over a stated range.
Modeling assumptions (not measured, swept below): tail power fraction, tail amortization,
SMS marginal-cost fraction of the Carroll upper bound, MMS size, signal quality, payload size.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data")); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from generate_synthetic_data import (PATH_CLASSES, BASE_RESILIENCE_CONTRIB, generate)
from reti_metric import reti_table

# verified (Chen et al. 2017, Table 1 / Sec 5.3.1)
SETUP, RELEASE, FAIL, BASE_MA = 7.26, 16.53, 108.53, 95.65
TX = {"good": 0.42, "medium": 0.57, "poor": 1.26}; RX = {"good": 0.22, "medium": 0.36, "poor": 0.48}
TAIL_S = (4.6, 10.3)
# verified (Carroll and Heiser 2010): 7.9 mW over idle for 82 s => <= 0.648 J
SMS_J_MAX = 7.9e-3 * 82

def path_costs(V=3.8, tail_s=10.3, tail_frac=0.5, amort=1.0, sms_frac=1.0,
               mms_kb=100, sig="medium", payload_kb=2.0, hold_s=0.15):
    to_mas = lambda joules: joules / V * 1000.0
    tail = tail_s * BASE_MA * tail_frac * amort
    data = payload_kb * TX[sig] + 1.0 * RX[sig]
    rcs = SETUP + data + RELEASE + tail
    sms = to_mas(SMS_J_MAX * sms_frac)
    mms = SETUP + mms_kb * TX[sig] + RELEASE + tail
    c = {
        "RCS Direct Delivery": rcs,
        "SMS Fallback (SS7)": FAIL + sms,
        "MMS Fallback": FAIL + mms,
        "Retry-Then-Fallback": 2 * FAIL + sms,
        "Multi-Hop Interworking": rcs + hold_s * BASE_MA,
        "Redundant Multi-Path Delivery": rcs + sms,
    }
    return c

def rel(c):
    b = c["RCS Direct Delivery"]
    return {k: v / b for k, v in c.items()}

def pbar():
    # success probability averaged over region mix (46/34/20, rural x1.15 cap 0.97)
    out = {}
    for k, p in BASE_RESILIENCE_CONTRIB.items():
        out[k] = 0.8 * p + 0.2 * min(p * 1.15, 0.97)
    return out

def reti_analytic(c):
    r = rel(c); pb = pbar()
    return pd.Series({k: pb[k] / r[k] for k in PATH_CLASSES}).sort_values(ascending=False)

if __name__ == "__main__":
    # 1) central scenario through the full pipeline
    central = path_costs()
    print("Central absolute cost (mAs):"); print(pd.Series(central).round(1))
    r = rel(central); print("\nRelative to RCS Direct:"); print(pd.Series(r).round(2))
    df = generate(seed=42, energy_cost=r)
    t = reti_table(df); print("\nCalibrated RETI (full pipeline, seed 42):"); print(t.round(3))
    t.round(3).to_csv(os.path.join(ROOT, "data", "reti_table_calibrated.csv"))
    base = pd.read_csv(os.path.join(ROOT, "data", "reti_table.csv"), index_col=0)
    print("\nOriginal (illustrative) RETI ranking:", list(base.index))
    print("Calibrated RETI ranking:             ", list(t.index))

    # 2) Monte Carlo over stated assumption ranges
    rng = np.random.default_rng(7); rows = []
    for _ in range(5000):
        c = path_costs(V=rng.uniform(3.6, 4.35), tail_s=rng.uniform(*TAIL_S),
                       tail_frac=rng.uniform(0.2, 1.0), amort=rng.uniform(0.0, 1.0),
                       sms_frac=rng.uniform(0.25, 1.0), mms_kb=rng.uniform(50, 300),
                       sig=rng.choice(["good", "medium", "poor"]), payload_kb=rng.uniform(0.5, 5))
        s = reti_analytic(c); rel_c = rel(c)
        rows.append({"top": s.index[0], "rcs_rel_sms": rel_c["SMS Fallback (SS7)"],
                     **{f"rank_{k}": list(s.index).index(k) + 1 for k in PATH_CLASSES}})
    mc = pd.DataFrame(rows)
    print("\nTop-ranked path share over 5000 assumption draws:")
    print(mc["top"].value_counts(normalize=True).round(3))
    print("\nMean rank by path (1 = best):")
    print(mc[[c for c in mc if c.startswith("rank_")]].mean().round(2))
    print("\nShare of draws where SMS fallback costs LESS than RCS direct (rel<1):",
          round((mc["rcs_rel_sms"] < 1).mean(), 3))
    print("Share of draws where original paper ordering (RCS cheapest) holds:",
          round((mc["rcs_rel_sms"] > 1).mean(), 3))
    mc.to_csv(os.path.join(ROOT, "data", "calibrated_sensitivity_draws.csv"), index=False)
