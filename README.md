# SDSS: Similarity Distance Stratified Sampling for Diabetic Complication Prediction

This repository is a reviewer-facing release for the manuscript **"SDSS: a similarity-distance stratified negative sampling framework for diabetic complication prediction"**.

The release contains two complementary parts:

1. **Core SDSS code** under `sdsskg/`, including cosine-distance representation and SDSS/SDSS3.2 negative samplers.
2. **Archived result provenance** under `results/archived_profile_grid/`, containing the profile-grid summary files used to reconstruct the manuscript SDSS3.2-Final table.

The manuscript method defines SDSS as a training-only framework that builds a positive centroid from training positives, computes cosine distance from training negatives to that centroid, partitions negatives into distance strata, and samples hard/middle/easy negatives according to a fixed profile; Test1 and Test2 are used only for final external evaluation, not profile selection.

## Quick start: reproduce the manuscript final table

```bash
pip install -r requirements.txt
python scripts/build_final_sdss32_from_archived_grid.py --root results/archived_profile_grid --out results/final_sdss32_provenance
python scripts/check_final_table.py
```

Expected F1 table:

| Dataset | Strata | hard/mid/easy | Test1 F1 | Test2 F1 |
|---|---:|---:|---:|---:|
| DN | 10 | 0.55/0.25/0.20 | 0.8491 ± 0.0341 | 0.8607 ± 0.0136 |
| DF | 10 | 0.60/0.20/0.20 | 0.9815 ± 0.0124 | 0.9723 ± 0.0199 |
| DPN | 8 | 0.60/0.20/0.20 | 0.9918 ± 0.0075 | 0.9878 ± 0.0088 |
| DKA | 8 | 0.60/0.20/0.20 | 0.9360 ± 0.0171 | 0.9191 ± 0.0120 |

## Important reproducibility note

The manuscript table is reconstructed from archived profile-grid experiment outputs. The reconstruction script **does not retrain models and does not edit scores**. It extracts the fixed final profile for each task from archived `*_sdss32_final_profile_summary_*.csv` files and writes the paper-ready mean ± SD tables.

Current-code retraining is also possible if private de-identified clinical data are supplied, but current-code reruns may not exactly match the archived manuscript table if preprocessing, leakage-column exclusion, threshold selection, or model code has been updated after the archived experiments.

## Repository structure

```text
sdsskg/                         # Core SDSS code
  sampling.py                   # random, all-easy, all-hard, SDSS, SDSS3.2 samplers
  similarity.py                 # HF/TF-IDF text encoder and cosine distance to positive centroid
scripts/
  build_final_sdss32_from_archived_grid.py
  summarize_final_profile_rerun.py
  check_final_table.py
  reference_experiment_runners/ # historical runner scripts, for audit/reference
configs/                        # YAML templates; paths should be adjusted for private data
results/
  archived_profile_grid/        # archived profile-grid summary/detail files
  final_sdss32_provenance/      # generated final tables and Excel provenance workbook
  current_code_rerun/           # place current-code rerun outputs here if needed
data/                           # raw clinical data are not included
examples/                       # toy SDSS sampling demo with synthetic text
```

## Toy SDSS sampling demo

```bash
python examples/demo_sdss_sampling.py
```

This toy example only demonstrates the SDSS sampling mechanism and does not reproduce manuscript results.

## Data availability

Raw electronic medical records are not included because they contain sensitive clinical information. De-identified or synthetic data can be placed under `data/`, and YAML paths under `configs/` can be edited accordingly.
