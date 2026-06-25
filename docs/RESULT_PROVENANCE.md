# Result provenance

The final manuscript SDSS table is reconstructed from archived profile-grid experiment summaries.

Workflow:

```text
Archived profile-grid summaries
    -> fixed task-specific final profiles
    -> build_final_sdss32_from_archived_grid.py
    -> final_sdss32_provenance.xlsx and manuscript-ready CSV tables
```

Final profiles:

| Dataset | Strata | hard_ratio | mid_ratio | easy_ratio |
|---|---:|---:|---:|---:|
| DN | 10 | 0.55 | 0.25 | 0.20 |
| DF | 10 | 0.60 | 0.20 | 0.20 |
| DPN | 8 | 0.60 | 0.20 | 0.20 |
| DKA | 8 | 0.60 | 0.20 | 0.20 |

The reconstruction step should not be described as fresh retraining. It is an audit step for rebuilding the manuscript table from archived experimental outputs.

Seed-level detail is included when the corresponding archived detail files are available; the summary CSV files are the authoritative source for the manuscript mean ± SD table.
