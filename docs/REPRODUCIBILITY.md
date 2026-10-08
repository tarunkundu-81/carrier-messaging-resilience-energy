# Reproducibility and interpretation

## Scope
This repository provides **synthetic** policy experiments and an optional real-trace summarizer. The data-generating probabilities, delivery-time distributions, common failure dependence and decision thresholds were selected for modeling; none should be represented as calibrated carrier performance.

## Paths and input provenance
- `original_sources/`: author's historical simulation source files, preserved as supplied.
- `reti_sim_epoch_corrected.py` and `run_original_corrected.py`: corrected energy-decision-epoch extension.
- `reti_uncertainty.py` and `run_reti_extended.py`: uncertainty-aware comparisons.
- `results/run_metadata.json`: exact saved run settings.
- `results/`: generated numerical evidence, including per-policy decisions and paired comparisons.
- `data/`: generated synthetic events, provenance, and empirical trace template.
- `experiments/v5_legacy/`: earlier preserved code and results relevant to Version 5 tables.

## Statistical cautions
- Report coverage/abstention whenever comparing safety-screened policies.
- Selected-decision success and energy are conditional on a policy selecting an action; different policy coverages imply different subsets.
- The strict certificate's energy interval uses an approximate Student-t bound, so it is not a distribution-free or field-validated safety guarantee.
- The high-budget Monte Carlo reference is finite-sample, not a perfect oracle.
- A synthetic comparison can be statistically reproducible without demonstrating real-world transport performance.

## Before publishing the GitHub repository
1. Choose an explicit license after confirming rights to all included files; see `LICENSE_SELECTION.md`.
2. Review manuscript author names and public email addresses before publishing.
3. Create a GitHub repository and upload the **contents of this extracted folder**, not the ZIP as a single repository file.
4. Run smoke tests and verify your platform/dependencies.
5. Tag a release and optionally archive the tagged commit with a persistent research-archive service; only then add the verified permanent URL/DOI to the paper and citation metadata.
6. Check any external datasets' terms before redistributing them. Raw third-party outage data are deliberately omitted.
