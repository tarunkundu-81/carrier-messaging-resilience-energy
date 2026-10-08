# RETI original-script verification and corrected rerun

## Recovered source verification

The originally uploaded `reproduce_reti.py` executed successfully. Its 1,200-configuration sweep EXACTLY reproduced the manuscript's 51/1194, 42/1120, and 40/1030 RETI-vs-minimum-energy disagreements at rho 0, 0.5, and 0.85 respectively. All twelve uniform-energy holdout settings were cross-checked for feasible counts and RETI/minE point-estimate violation counts.

## Corrected Table XI

| rho | Cost | Feasible | RETI/minE different | Disagreement among feasible | 95% Wilson CI | Mean gain on differences (pp) | Mean energy premium on differences (%) | Changed RETI choice vs U |
|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|
| 0.00 | U | 1194 | 51 | 4.27% | 3.26–5.57 | 10.61 | 5.95 | — |
| 0.00 | L | 1194 | 61 | 5.11% | 4.00–6.51 | 8.27 | 6.76 | 428 |
| 0.50 | U | 1120 | 42 | 3.75% | 2.79–5.03 | 10.06 | 5.49 | — |
| 0.50 | L | 1120 | 23 | 2.05% | 1.37–3.06 | 7.69 | 5.56 | 410 |
| 0.85 | U | 1030 | 40 | 3.88% | 2.86–5.24 | 11.17 | 6.40 | — |
| 0.85 | L | 1030 | 13 | 1.26% | 0.74–2.15 | 11.50 | 8.26 | 454 |

Disagreement-only improvements are not overall improvements; do not extrapolate these gains to all messages.

## Corrected Table XII and overall paired comparison (rho=0.5)

There are **2,205** matched decisions for RETI and minE. The corrected literature-informed energy model yields RETI average deadline success **89.647%**, versus minE **89.526%**; average modeled energy **1.233** versus **1.232** relative units, respectively. The uniform-energy corresponding success rates are 89.371% and 89.153%.

| Paired metric | Uniform U | Corrected literature-informed L |
|---|---:|---:|
| Mean deadline success difference, RETI minus minE (percentage points) | +0.218 [0.141, 0.317] | +0.121 [0.070, 0.180] |
| Mean relative energy premium (%) | +0.118 [0.067, 0.183] | +0.074 [0.040, 0.112] |
| Matched decisions / configuration clusters | 2205 / 1112 | 2205 / 1112 |
| RETI test-floor violations | 96 | 91 |
| minE test-floor violations | 105 | 92 |
| Wilson-screened minE test-floor violations and decision count | 12 / 2131 | 9 / 2131 |
| Finite-MC oracle test-floor violations and decision count | 63 / 2188 | 55 / 2188 |

The 95% confidence intervals use 2,000 paired cluster-bootstrap draws; `Table_XII_paired_confidence_original_sim.csv` contains the absolute energy and violation differences too. Their lower bounds are positive within these author-selected synthetic distributions, but expected effect sizes remain small.

### Independent validation and oracle caveats

For every configuration the 1,024 training messages, 2,048 evaluation messages, and 8,192 oracle-reference events come from distinct simulated random innovations. The Monte-Carlo oracle is **not perfect**: it uses independent estimates of the true constrained minimum-energy route and is itself subject to simulation noise and near-floor infeasibility. Its validation violations do not imply the route-choice algorithm was flawed, nor do they represent operational SLAs. Oracle regret is reported on a matched approximately oracle-feasible cohort in `oracle_reference_regret.csv`.

### Important limitations

1. No carrier traces, true RCS/SMS delivered-to-recipient observations, or end-to-end device/network energy measurements were provided.
2. The Chen LTE failed-establishment proxy is applied to unsuccessful **completed synthetic RCS attempts** as a new explicit modeling assumption; a generic delivery failure need not represent that modem event in a deployed device.
3. Heterogeneous historic RCS modem and SMS power ingredients make cross-protocol rankings assumption-sensitive. Numbers are not direct battery-drain estimates.
4. Reproducing the original author synthetic results does **not** validate its generated conditions against carrier deployments.
5. The author-review Word draft still needs independently checked references, archival link, and conference page-limit/camera-ready validation.
