# RETI: uncertainty-aware extension to existing author simulator

**Nature of the experiments:** Synthetic, controlled counterfactuals. The numeric parameters are *not* calibrated to operator routing logs, nor are the two heterogeneous handset energy studies a common-hardware benchmark.

## What was changed

The **original source files are preserved unchanged** under `original_sources/`. The verified energy-corrected six-action model and the original-reproduction driver are preserved as `reti_sim_epoch_corrected.py` and `run_original_corrected.py`. No original Table III/IV/sweep results were rewritten.

Additions:

- `reti_uncertainty.py`: implementation of simultaneous nominal confidence intervals for deadline-success and mean incremental energy, robust lower-bound selection, and *strict* uncertainty-certified RETI selection.
- `run_reti_extended.py`: full-seed, same-configuration and same-event holdout comparison across ten policies, at ρ=0.5 with κ=0 and 0.7, using original 600-config Uniform + 600-config Copula samplers. Saves every decision, abstention, and independent-validation outcome, compares on matched cohorts, and adds cluster-paired bootstrap intervals.
- `parameter_provenance.csv`: which Table III / Section V model inputs are assumed, derived, or partially literature-informed, plus what real measurements would be needed.
- `policy_decision_boundary_by_deadline.csv`: original validated 1,200-config sweep stratified by deadline tertile; RETI/minimum-energy selection differences and their energy/success effects.
- `trace_calibration.py`: optional **real trace input** for estimating condition-specific success, energy, and observed delivered-message timing. It **does not create data** or automatically replace model parameters: these require independent measurement and correct path mapping.
- `test_reti_uncertainty.py`: unit tests for bounds, strict dominance, possible competitors, and no-feasible-action abstention.

## Install & run

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_reti_uncertainty
python run_reti_extended.py --out results
```

Defaults: 600 configurations **per** design, 1,024 training events, 2,048 independent validation events, 8,192 independent MC-reference events, ρ=0.5, common configuration generator seed 20261011, cost draws 20261022; 95% nominal simultaneous interval. Configuration-cluster paired bootstrap: 2,000 resamples of Uniform/Copula+configuration IDs, with the two κ settings held together. The two energy models use shared sampled innovations.

Smoke test:

```bash
python run_reti_extended.py --configs 8 --train 256 --validation 512 --reference 1024 --out smoke
```

Legacy checks/re-run are unchanged:

```bash
python run_original_corrected.py
python original_sources/reproduce_reti.py output_original
```

The `run_original_corrected.py` script writes results into its directory; this extension has its own `results/` subfolder.

## Policy definition and confidence interpretation

Six actions are RCS, SMS-first, MMS-first, sequential RCS-retry-then-SMS, multihop, and simultaneous SMS+MMS.

For each action `a`, confidence intervals `Qlo(a), Qhi(a), Elo(a), Ehi(a)` are built from its **raw matched training events**, not rounded summary tables. `Q` uses Bonferroni-adjusted exact Clopper–Pearson intervals; `E` uses a Bonferroni-adjusted Student-t CI for its mean (an asymptotic approximation, not a finite-sample guarantee for non-Gaussian energy). The adjustment allocates familywise alpha across **both** Q and E and the six actions. Joint pairing matters for policy comparison; individual action confidence bounds use the marginal distributions, and Bonferroni does not require across-action independence.

- **Robust-LCB**: among actions with `Qlo≥τ`, pick largest `Qlo/Ehi`. This choice is *not* certified to beat competitors; it is a lower-confidence-bound heuristic. Abstain if none passes.
- **Certified-RETI**: select action a only if `Qlo(a)≥τ` and its lower efficiency bound `Qlo(a)/Ehi(a)` exceeds `Qhi(b)/Elo(b)` for **every other possibly feasible** action b with `Qhi(b)≥τ`. Otherwise abstain, distinguishing 'no_verified_feasible_action' from 'uncertain_ranking'. This is a **model-conditional nominal certificate**, not a deployment guarantee.
- **Wilson-minE**: old one-sided Wilson screen followed by min E. It is an existing baseline with an appreciably less strict safety adjustment, not a simultaneous confidence certificate.
- **MC-reference**: finite 8,192-event independent high-budget reference, *not* an omniscient oracle. May itself abstain.

**Important:** A choice being certified within a simulation does not imply that the simulator assumptions or energy proxy are physically correct. It does not establish or validate a 95% safety guarantee for deployments. The Student-t energy CI is approximate and may not achieve exact nominal coverage for non-normal event energies. If strict finite-sample guarantees are required, replace this interval with bounded or distribution-free concentration bounds supported by a validated physical upper energy bound, or a fully specified sampling model.

## Output interpretation

- `policy_coverage_and_outcomes.csv`: **every** policy, 2,400 test settings per energy model, including cases in which *all* baseline rules find no route. Success, energy, violations are **conditional on non-abstention**; do not rank policies using those means alone.
- `matched_paired_bootstrap.csv`: pairwise policy comparisons **only where both select an action**, with 95% cluster bootstrap intervals for success (pp), mean energy (relative units), and validation floor-violation rate.
- `all_policy_common_decisions.csv` and `all_policy_matched_results.csv`: same matching subset for all ten policies, which may be selected by the strict certification rule. Do not treat this subset as representative of all cases.
- `holdout_interval_diagnostics.csv`: confidence bounds for checking the certification and selection conditions.
- `policy_decision_boundary_by_deadline.csv`: controlled sweep sensitivity, not observed carrier prevalence.

## Optional empirical-input pathway

Provide recipient-confirmed messaging *measurement* traces with columns shown in `external_trace_TEMPLATE.csv`. The independent traces need delivery timestamps, a declared deadline, and measured incremental energy per attempted message (failed messages **must be retained**). Run:

```bash
python trace_calibration.py YOUR_ACTUAL_DATA.csv --out empirical_calibration.csv
```

The tool returns path-by-condition success-before-deadline probabilities with exact binomial uncertainty, measured mean energy, and delivered-message latency statistics. It refuses incomplete or invalid input and **does not** transplant those summaries into a different component-action model automatically: doing so would require a consistent decision epoch and compatible attempt definitions.

## Supporting manuscript narrative

- The decision boundary between RETI and minimum-energy selection is driven by relative Q gain > relative energy premium, which was verified on disagreement cases. Deadline tertiles explain where disagreements occur without implying prevalence in real systems.
- The new robust and strict-certified policies formalize policy uncertainty/abstention, but **do not constitute a new optimization theorem**. The novelty claim should be the *joint modeling + reproducible audit + concrete coverage/risk trade-off*.
- The original Table III offsets and latency medians are still **assumptions**. They are now completely catalogued but **not empirically justified**. External validation remains an open reviewer concern until independent observations are supplied.

## Additional Wilson comparison

```bash
python compare_uncertainty_baselines.py
```

This writes `results/paired_against_Wilson.csv` using already saved paired validation outputs. The paper-oriented description and conservative interpretation are in `RESULTS_AND_PAPER_GUIDANCE.md`.

Publication-oriented visualization (optional):

```bash
python plot_coverage.py
```

This plots policy coverage against synthetic validation-floor violation frequency (conditional on service). Each energy assumption gets a separate PNG. Interpret these as coverage/risk curves, **not observed carrier deployment behavior**.
