# RETI extension: generated results and suggested manuscript integration

**Status:** Successfully executed on the author-provided original simulator (with previously verified corrected decision-epoch energy logic). Results are **synthetic only**. Original six-action and 1,200-configuration sampling behavior was preserved. No operator measurements were available and none are claimed.

## Verified reconstruction and decision boundary

The previously verified 1,200-configuration sweep was reanalyzed by deadline tertile, using the original sampler, seeds, parameters, and generated sweep decisions. The original counts are reproduced:

| ρ | U disagreement / feasible | Literature-informed L disagreement / feasible |
|---|---:|---:|
| 0.00 | 51 / 1194 (4.27%) | 61 / 1194 (5.11%) |
| 0.50 | 42 / 1120 (3.75%) | 23 / 1120 (2.05%) |
| 0.85 | 40 / 1030 (3.88%) | 13 / 1030 (1.26%) |

The inequality `ΔQ / Q_minE > ΔE / E_minE` is an **algebraic identity**, not a new optimization theorem. Across all feasible disagreements, the RETI action has higher Q and higher E, as required by this identity. Deadline tertiles (see CSV) expose where method disagreements are concentrated, e.g. at ρ=0 under L energy they rise from 2.02% in the shortest tertile to 10.03% in the longest.

## Holdout method

600 configurations each from independent-uniform and Gaussian-copula input distributions, 1,024 training events and 2,048 independently generated validation events, two success-latency couplings κ∈{0, 0.7} at ρ=0.5, both uniform and corrected literature-informed energy. Six actions. In each energy model there are **2,400 configuration×coupling evaluations**, of which 2,205 yield a conventional point-feasible route. Paired 2,000-replicate confidence intervals use **configuration clusters** (retain both κ settings together).

## Coverage and abstention (all 2,400 cases per model)

| Policy | U: served | U: validation-floor violations | L: served | L: validation-floor violations |
|---|---:|---:|---:|---:|
| Conventional RETI | 2,205 (91.88%) | 96 | 2,205 (91.88%) | 91 |
| Wilson minimum-energy | 2,131 (88.79%) | 12 | 2,131 (88.79%) | 9 |
| **Robust-LCB (heuristic)** | **2,061 (85.88%)** | **2** | **2,061 (85.88%)** | **1** |
| **Strict Certified-RETI** | **528 (22.00%)** | **1** | **898 (37.42%)** | **1** |

A reported violation is a **validation-sample estimate below the simulated floor**; it is not an observed production SLA violation. Served-cohort means/violation rates must not be interpreted as unconditional performance gains; **coverage changes between rules**. The certification is model-conditional and nominal (the mean-energy t confidence bounds are approximate).

## Matched, paired comparisons

Every difference below is evaluated only where **both** policies select a route. Thus abstention remains an essential separate metric.

**Robust-LCB minus Wilson-screened minimum energy:**

| Energy | Matched N | ΔQ (pp) [95% cluster bootstrap CI] | ΔE (relative units) [95% CI] | Δfloor-violation fraction [95% CI] |
|---|---:|---|---|---|
| U | 2,061 | +0.795 [0.669, 0.931] | +0.128 [0.104, 0.153] | −0.00437 [−0.00730, −0.00194] |
| L | 2,061 | +0.842 [0.716, 0.981] | +0.124 [0.093, 0.163] | −0.00340 [−0.00628, −0.00097] |

These results reveal the expected **reliability–energy–abstention trade-off**, not free policy dominance.

**Strict Certified-RETI minus Wilson-screened min-energy:** on their common decided subsets, the two policies selected the same action under both energy models (N=528 U, N=898 L). There is therefore **no measured policy advantage** on those subsets, despite the stronger ranking condition. The principal new result is *when confidence intervals permit a decisive recommendation*, not superiority in routing outcome.

## Suggested paper edits (not applied to Version 5 DOCX)

1. In Section V-A, insert a **parameter-provenance** paragraph/table based on `parameter_provenance.csv`; keep Table III explicitly author-constructed and describe which rows derive from heterogeneous handset measurements. Do not claim success probabilities or 112–250 ms medians are calibrated.
2. In Section III-C, add the relative-gain versus relative-cost identity and direct readers to the deadline-tertile experiment in the results. Do not present the identity as a novel theorem.
3. In the results, report **policy coverage before conditional reliability**. Include Wilson-minE, Robust-LCB, and Certified-RETI on both energy models with a separate matched comparison. Do not combine their different served cohorts.
4. Define Certified-RETI's simultaneous nominal intervals precisely: Bonferroni-adjusted exact binomial Q bounds; approximate t bounds for E; competitors with Q_upper≥τ cannot be discarded solely because Q_lower<τ.
5. State that **novelty is operational decision auditing under uncertain energy and deadline constraints**, not novel fractional optimization. The high abstention rates and negligible incremental benefit of the strict certificate are important *negative results*.
6. For empirical calibration, use `trace_calibration.py` only with **actual observed events** and an explicitly consistent decision epoch. Until such measurements exist, “empirically validated baseline parameters” remains unaddressed.

## Reproduction and limitations

Commands: `python -m unittest -v test_reti_uncertainty`, `python run_reti_extended.py --out results`, `python compare_uncertainty_baselines.py`. See `README.md` for all parameters and optional trace pathway. The extension never alters the six-path event model, its simulator seed, or the previous literature-informed failure proxy. Synthetic counterfactuals are not standardized rankings or field-calibrated performance claims.
