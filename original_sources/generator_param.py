"""
generate_synthetic_data.py  (messaging-resilience-energy)

Fully SYNTHETIC delivery-path events for carrier messaging fallback analysis.
No production, subscriber, or carrier-identifying data is used. Relative
energy-cost units and resilience-contribution probabilities are illustrative
parameters, not measured power data.

Baseline: seed 42, N=25,000. Output: data/synthetic_path_events.csv
Generator body recovered verbatim from the 4S-Net 2026 build; only the
function wrapper and scale arguments were added so the sensitivity script
can reuse it. Baseline output is unchanged.
"""
import os
import numpy as np
import pandas as pd

N_EVENTS = 25000

PATH_CLASSES = [
    "RCS Direct Delivery", "SMS Fallback (SS7)", "MMS Fallback",
    "Retry-Then-Fallback", "Multi-Hop Interworking", "Redundant Multi-Path Delivery",
]
BASE_WEIGHTS = np.array([0.34, 0.24, 0.10, 0.14, 0.12, 0.06])
BASE_WEIGHTS = BASE_WEIGHTS / BASE_WEIGHTS.sum()

BASE_ENERGY_COST = {
    "RCS Direct Delivery": 1.0, "SMS Fallback (SS7)": 1.6, "MMS Fallback": 2.4,
    "Retry-Then-Fallback": 2.1, "Multi-Hop Interworking": 2.8,
    "Redundant Multi-Path Delivery": 3.6,
}
BASE_RESILIENCE_CONTRIB = {
    "RCS Direct Delivery": 0.05, "SMS Fallback (SS7)": 0.78, "MMS Fallback": 0.55,
    "Retry-Then-Fallback": 0.62, "Multi-Hop Interworking": 0.40,
    "Redundant Multi-Path Delivery": 0.88,
}
NETWORK_CONDITION = ["Normal", "Congested", "Partial Outage", "Severe Disruption"]
CONDITION_WEIGHTS = [0.55, 0.27, 0.13, 0.05]
CONDITION_FALLBACK_MULTIPLIER = {"Normal": 0.4, "Congested": 1.3, "Partial Outage": 2.6, "Severe Disruption": 4.2}
REGION_TYPES = ["Urban", "Suburban", "Rural"]
REGION_WEIGHTS = [0.46, 0.34, 0.20]


def generate(seed=42, n=N_EVENTS, energy_cost=None, resilience=None):
    rng = np.random.default_rng(seed)
    energy_cost = energy_cost or BASE_ENERGY_COST
    resilience = resilience or BASE_RESILIENCE_CONTRIB
    rows = []
    for _ in range(n):
        condition = rng.choice(NETWORK_CONDITION, p=CONDITION_WEIGHTS)
        region = rng.choice(REGION_TYPES, p=REGION_WEIGHTS)

        weights = BASE_WEIGHTS.copy()
        rcs_idx = PATH_CLASSES.index("RCS Direct Delivery")
        red_idx = PATH_CLASSES.index("Redundant Multi-Path Delivery")
        mult = CONDITION_FALLBACK_MULTIPLIER[condition]
        weights[rcs_idx] /= mult
        weights[red_idx] *= mult
        weights = weights / weights.sum()

        path = rng.choice(PATH_CLASSES, p=weights)

        energy = max(energy_cost[path] * (1 + rng.normal(0, 0.08)), 0.2)

        p = resilience[path]
        if region == "Rural":
            p = min(p * 1.15, 0.97)
        ok = rng.random() < p

        latency = float(np.clip(
            rng.gamma(shape=2.0, scale=90)
            + (150 if path == "Multi-Hop Interworking" else 0)
            + (200 if path == "Redundant Multi-Path Delivery" else 0), 20, 3000))

        rows.append({"event_id": None, "region_type": region, "network_condition": condition,
                     "path_class": path, "relative_energy_cost": round(energy, 3),
                     "succeeded_under_stress": ok, "delivery_latency_ms": round(latency, 1)})
    df = pd.DataFrame(rows)
    df["event_id"] = [f"MSG-{i:06d}" for i in range(1, len(df) + 1)]
    return df[["event_id", "region_type", "network_condition", "path_class",
               "relative_energy_cost", "succeeded_under_stress", "delivery_latency_ms"]]


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = generate()
    df.to_csv(os.path.join(root, "data", "synthetic_path_events.csv"), index=False)
    print(f"Wrote {len(df)} synthetic path events.")
    print(df["path_class"].value_counts(normalize=True).round(3))
