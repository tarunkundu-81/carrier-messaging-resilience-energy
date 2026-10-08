# Data inventory and provenance

- `synthetic_path_events.csv`: generated locally from unchanged author script `original_sources/generate_synthetic_data(3).py` (25,000 synthetic events; seed 42). **Not observations of carrier traffic.**
- `parameter_provenance.csv`: input origin, units, limitations, and suggested empirical replacements.
- `external_trace_TEMPLATE.csv`: empty **schema/template** for future real-world measurements. It is not evidence of collected traces.

To recreate the synthetic event dataset (from the repository root):

```bash
python 'original_sources/generate_synthetic_data(3).py'
```

Third-party raw outage exports have **not** been republished in this repository ZIP. If such data are needed, obtain them from the original provider, follow its use and attribution conditions, and document the export period and filters. References to outage-duration analyses in the manuscript do not imply that those raw data are redistributable.
