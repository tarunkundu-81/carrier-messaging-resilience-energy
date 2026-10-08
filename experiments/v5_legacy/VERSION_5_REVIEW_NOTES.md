# Version 5: matched-cohort + energy-proxy sensitivity audit

## Common comparison
At rho=0.5, 2,205 configurations-with-kappa decisions were eligible for RETI/minE; all eight policies made a selection on 2,128 (1,083 original parameter-configuration clusters) under both cost models. Wilson abstained on 74 of the 2,205 and finite-MC oracle on 17; 77 unique decisions excluded for consistent all-eight-policy comparison.

On this common 2,128-decision cohort (paired, 2,000 bootstrap resamples of configuration clusters): RETI minus minE deadline-success gain under U = +0.226 pp [0.138, 0.326], L = +0.126 pp [0.073, 0.191]; modeled per-decision mean energy premiums = +0.122% [0.066, 0.184], +0.076% [0.041, 0.120]. The oracle is an 8,192-event model-based Monte-Carlo reference, not perfect foreknowledge. Success and energy averages in Table XII have the *same* 2,128 denominator for every policy.

## Independent failure proxy ablation
Original simulation core unchanged apart from its 108.53 mAs (normalized) proxy applied to completed unsuccessful RCS attempts only. Ablation compares that proxy against Eq. (16) duration charging for the same literature-informed input samples. 1,200-draw sweep at rho=0.5: disagreement 26/1,120 (duration) vs 23/1,120 (establishment proxy). 2,205 matched holdout decisions at rho=0.5: 59 RETI route changes between the accounting assumptions (2.7%). RETI-minus-minE held-out success gain: +0.146 pp [0.094,0.208] for duration vs +0.121 pp [0.069,0.183] for the failure proxy. The sign is stable within the same synthetic model but both accounting choices remain assumptions.

## Caveats
These data do not demonstrate real carrier performance; the earlier paper's mixed-device literature inputs remain heterogeneous. A permanent public source/data archive identifier has not been supplied. This is an author-review manuscript, not camera-ready.

## How to reproduce
From the package directory: `python build_matched_cohort.py`; `python run_proxy_ablation.py` (holds out rho=0.5; the sweep data also provided). Prerequisites: Python, numpy, scipy, pandas, python-docx for manuscript updates. All scripts from the original recovered source remain in `original_sources/`.
