# Experiments

The runnable Version 6 experiments are kept at the repository root to preserve the verified import paths.

From the repository root:

```bash
python -m unittest -v test_reti_uncertainty
python run_reti_extended.py --configs 8 --train 256 --validation 512 --reference 1024 --out smoke
python run_reti_extended.py --out results
python compare_uncertainty_baselines.py
python plot_coverage.py
```

The full run uses 600 Uniform and 600 Copula input configurations, two latency-coupling settings, uniform and literature-informed energy assumptions, separate training/validation/reference draws and original author-selected parameters. It is computationally intensive. `results/` contains output from the previously completed run.

For the original author's simulation (also computationally intensive):

```bash
python original_sources/reproduce_reti.py original_reproduction
```

The independent 25,000-event dataset can be regenerated with the command in `data/README.md`.

Earlier Version 5 analyses that support existing manuscript tables are archived separately under `v5_legacy/`.
