# Carrier Messaging Resilience–Energy Trade-offs (RETI)

Reproducible experimental code and supporting material for the research manuscript **“Resilience-Energy Trade-offs in Carrier Messaging Fallback Paths”** by Tarun Kundu and Srinivasa Chary Maringanti.

**Status:** A research-code snapshot accompanying Version 6 of the manuscript. Results are from synthetic, author-chosen parameters and literature-informed device-energy scenarios, **not** carrier measurements. The code does not implement or validate a deployed carrier-routing optimizer.

## Research focus

The simulations compare six messaging actions (RCS, SMS-first, MMS-first, sequential retries/fallback, multi-hop, and parallel redundant delivery), accounting for deadline compliance, executed-attempt energy, shared failure dependence, and uncertainty. They compare RETI with minimum-energy, maximum-success, weighted, Pareto, Wilson-screened, robust lower-bound, and strict certificate rules. The finite Monte Carlo reference is **not** an oracle.

## Repository contents

| Path | Contents |
| --- | --- |
| `reti_sim_epoch_corrected.py`, `run_original_corrected.py` | Reproducible baseline and corrected energy accounting |
| `reti_uncertainty.py`, `run_reti_extended.py` | Uncertainty-aware policy comparison and holdout experiments |
| `test_reti_uncertainty.py` | Confidence-bound / policy unit tests |
| `compare_uncertainty_baselines.py`, `plot_coverage.py` | Postprocessing and plots |
| `trace_calibration.py` | Optional summarizer for **independently measured** input traces |
| `original_sources/` | Preserved earlier simulator scripts |
| `experiments/v5_legacy/` | Earlier matched-cohort and energy-ablation experiments with archived results |
| `data/` | Synthetic event dataset, parameter provenance, and real-trace schema template |
| `results/` | Version 6 policy records, statistics, settings, and figures |
| `manuscript/` | Latest corresponding editable IEEE-format manuscript |
| `docs/` | Original implementation notes, reproducibility guidance, publication checklist |

## Quick start

Python 3.10+ recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_reti_uncertainty
python run_reti_extended.py --configs 8 --train 256 --validation 512 --reference 1024 --out smoke
```

**Full synthetic holdout run** (significantly more computationally intensive):

```bash
python run_reti_extended.py --out results
python compare_uncertainty_baselines.py
python plot_coverage.py
```

**Original author experiment** (also computationally intensive):

```bash
python original_sources/reproduce_reti.py original_reproduction
```

**Generate the historical 25,000-event synthetic dataset:**

```bash
python 'original_sources/generate_synthetic_data(3).py'
```

See [`experiments/README.md`](experiments/README.md) for experiment details and [`docs/ORIGINAL_V6_README.md`](docs/ORIGINAL_V6_README.md) for the original implementation instructions.

## Reported results and limitations

- See `results/run_metadata.json` for holdout settings, and `results/policy_coverage_and_outcomes.csv` for coverage, policy-specific outcomes, and floor violations.
- `results/paired_against_Wilson.csv` reports *matched*, configuration-cluster paired comparisons.
- `results/policy_decision_boundary_by_deadline.csv` reports disagreement sensitivity for the original modeled sweep, **not** prevalence in deployed networks.
- A strict confidence certificate often abstains; results should always report selection coverage alongside success and energy.
- No real operator dataset is included. The original Table III success and energy inputs and event latency distributions are not calibrated carrier estimates.

## Real-measurement pathway (optional)

The file `data/external_trace_TEMPLATE.csv` documents the expected format for separately collected and authorized real delivery measurements. To summarize such a dataset:

```bash
python trace_calibration.py path/to/actual_measured_messages.csv --out empirical_calibration.csv
```

The script cannot generate empirical observations or substitute them automatically into a differently defined routing model.

## Citation and release

See [`CITATION.cff`](CITATION.cff). After publishing on GitHub, add the actual repository URL. For a stable scholarly citation, tag a release and optionally archive it to obtain a genuine DOI, then update the manuscript and citation metadata. **No DOI is supplied or implied here.**

## Licensing and data permissions

A license is **not** automatically assigned to this package; see [`docs/LICENSE_SELECTION.md`](docs/LICENSE_SELECTION.md) before making it publicly reusable. No third-party raw outage dataset is redistributed in this ZIP. Confirm authorship rights, source permissions, and privacy before uploading publicly.
