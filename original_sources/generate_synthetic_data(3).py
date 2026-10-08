"""
generate_synthetic_data.py

Generates a fully SYNTHETIC dataset illustrating the resilience-energy
trade-off across carrier messaging delivery paths (RCS direct, SMS/MMS
fallback, retry, multi-hop interworking, redundant multi-path delivery).
No production, subscriber, or carrier-identifying data is used anywhere in
this repository. Relative energy-cost units and resilience-contribution
scores are illustrative parameters set by the author, informed by general
publicly documented protocol overhead characteristics (IP session
signaling vs. SS7 circuit-switched signaling vs. multi-hop interworking),
NOT measured production or lab power data.

Output: data/synthetic_path_events.csv
"""

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

N_EVENTS = 25000

# --- Delivery-path taxonomy -------------------------------------------
PATH_CLASSES = [
    "RCS Direct Delivery",
    "SMS Fallback (SS7)",
    "MMS Fallback",
    "Retry-Then-Fallback",
    "Multi-Hop Interworking",
    "Redundant Multi-Path Delivery",
]

BASE_WEIGHTS = np.array([0.34, 0.24, 0.10, 0.14, 0.12, 0.06])
BASE_WEIGHTS = BASE_WEIGHTS / BASE_WEIGHTS.sum()

# Relative energy cost per successful delivery (illustrative units, not
# measured joules) - reflects that circuit-switched signaling, multi-hop
# interworking, and redundant delivery carry structurally higher
# network-side signaling/backhaul overhead than a single direct IP session.
BASE_ENERGY_COST = {
    "RCS Direct Delivery": 1.0,
    "SMS Fallback (SS7)": 1.6,
    "MMS Fallback": 2.4,
    "Retry-Then-Fallback": 2.1,
    "Multi-Hop Interworking": 2.8,
    "Redundant Multi-Path Delivery": 3.6,
}

# Resilience contribution: probability this path succeeds in a network
# condition (congestion / partial outage) where the primary RCS path fails.
BASE_RESILIENCE_CONTRIB = {
    "RCS Direct Delivery": 0.05,
    "SMS Fallback (SS7)": 0.78,
    "MMS Fallback": 0.55,
    "Retry-Then-Fallback": 0.62,
    "Multi-Hop Interworking": 0.40,
    "Redundant Multi-Path Delivery": 0.88,
}

NETWORK_CONDITION = ["Normal", "Congested", "Partial Outage", "Severe Disruption"]
CONDITION_WEIGHTS = [0.55, 0.27, 0.13, 0.05]
# Under stress, weight shifts away from RCS Direct toward fallback/redundant paths
CONDITION_FALLBACK_MULTIPLIER = {"Normal": 0.4, "Congested": 1.3, "Partial Outage": 2.6, "Severe Disruption": 4.2}

REGION_TYPES = ["Urban", "Suburban", "Rural"]
REGION_WEIGHTS = [0.46, 0.34, 0.20]


def sample_events(n):
    rows = []
    for _ in range(n):
        condition = rng.choice(NETWORK_CONDITION, p=CONDITION_WEIGHTS)
        region = rng.choice(REGION_TYPES, p=REGION_WEIGHTS)

        weights = BASE_WEIGHTS.copy()
        rcs_idx = PATH_CLASSES.index("RCS Direct Delivery")
        redundant_idx = PATH_CLASSES.index("Redundant Multi-Path Delivery")
        mult = CONDITION_FALLBACK_MULTIPLIER[condition]
        weights[rcs_idx] /= mult
        weights[redundant_idx] *= mult
        weights = weights / weights.sum()

        path = rng.choice(PATH_CLASSES, p=weights)

        energy = BASE_ENERGY_COST[path] * (1 + rng.normal(0, 0.08))
        energy = max(energy, 0.2)

        resilience_p = BASE_RESILIENCE_CONTRIB[path]
        if region == "Rural":
            resilience_p = min(resilience_p * 1.15, 0.97)
        succeeded_under_stress = rng.random() < resilience_p

        delivery_latency_ms = float(
            np.clip(rng.gamma(shape=2.0, scale=90) +
                    (150 if path == "Multi-Hop Interworking" else 0) +
                    (200 if path == "Redundant Multi-Path Delivery" else 0), 20, 3000)
        )

        rows.append({
            "event_id": None,
            "region_type": region,
            "network_condition": condition,
            "path_class": path,
            "relative_energy_cost": round(energy, 3),
            "succeeded_under_stress": succeeded_under_stress,
            "delivery_latency_ms": round(delivery_latency_ms, 1),
        })
    return rows


rows = sample_events(N_EVENTS)
df = pd.DataFrame(rows)
df["event_id"] = [f"MSG-{i:06d}" for i in range(1, len(df) + 1)]
df = df[["event_id", "region_type", "network_condition", "path_class",
         "relative_energy_cost", "succeeded_under_stress", "delivery_latency_ms"]]

df.to_csv(os.path.join(ROOT, "data", "synthetic_path_events.csv"), index=False)
print(f"Wrote {len(df)} synthetic path events.")
print(df["path_class"].value_counts(normalize=True).round(3))
