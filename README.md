# SDSS: Similarity-Distance Stratified Sampling for Diabetic Complication Prediction

This repository is a reviewer-facing release for the manuscript **"SDSS: a similarity-distance stratified negative sampling framework for diabetic complication prediction"**.

The release contains three complementary parts:

1. **Core SDSS code** under `sdsskg/`, including positive-centroid cosine-distance representation, distance-stratified negative sampling, and the MLP-based prediction model.
2. **Training and reproducibility scripts** under `scripts/`, including a reviewer-visible SDSS + MLP training entry point, reference experiment runners, and result-reconstruction utilities.
3. **Archived result provenance** under `results/archived_profile_grid/`, containing the profile-grid summary files used to reconstruct the final SDSS table reported in the manuscript.

The manuscript method defines SDSS as a training-only framework that builds a positive centroid from training positives, computes cosine distance from training negatives to that centroid, partitions negatives into distance strata, and samples hard/middle/easy negatives according to a fixed profile. Test1 and Test2 are used only for final external evaluation, not for profile selection.

## Quick start: reproduce the manuscript final table

Linux/macOS:

```bash
pip install -r requirements.txt

python scripts/build_final_sdss32_from_archived_grid.py \
  --root results/archived_profile_grid \
  --out results/final_sdss32_provenance

python scripts/check_final_table.py
```

Windows PowerShell:

```powershell
pip install -r requirements.txt

python scripts/build_final_sdss32_from_archived_grid.py `
  --root results/archived_profile_grid `
  --out results/final_sdss32_provenance

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

Full retraining scripts are provided for experiments using private de-identified clinical data. Minor numerical differences may occur because of stochastic optimization, software versions, and local runtime environments. The archived result files are provided to reproduce the exact manuscript tables.

For full retraining with local HuggingFace transformer embeddings and the PyTorch MLP model, install the optional full environment:

```bash
pip install -r requirements-full.txt
```

A typical private-data training command is:

```bash
python scripts/run_sdss_mlp_training.py --config configs/dn_two_external_wokg_sdss32.yaml
```

The command above requires private de-identified Train/Test1/Test2 files whose local paths are specified in the YAML configuration. The raw clinical files are not included in this public repository.

## Repository structure

```text
sdsskg/                         # Core SDSS and prediction-model code
  sampling.py                   # random, all-easy, all-hard, and SDSS samplers
  similarity.py                 # text encoder and cosine distance to positive centroid
  model.py                      # MLP prediction model with optional KG-feature input
  train_eval.py                 # PyTorch training, threshold selection, and evaluation
  kge.py                        # optional KG/DCKG feature encoder interface
  run_experiment.py             # single SDSS + MLP experiment runner
  run_experiment_clinical_ablation.py
  run_experiment_new_sampling.py
  run_asdss_profile_suite.py
  reselect_sdss32_final_profile_v3.py
scripts/
  run_sdss_mlp_training.py      # reviewer-visible SDSS + MLP training entry point
  build_final_sdss32_from_archived_grid.py
  summarize_final_profile_rerun.py
  check_final_table.py
  figure1_distance_distribution.py
  reference_experiment_runners/ # historical runner scripts, for audit/reference
configs/                        # YAML templates; paths should be adjusted for private data
results/
  archived_profile_grid/        # archived profile-grid summary/detail files
  final_sdss32_provenance/      # generated final tables and Excel provenance workbook
  all_datasets_sdss_final_table.xlsx # compact final F1 table
  current_code_rerun/           # place current-code rerun outputs here if needed
docs/
  REVIEWER_GUIDE.md
  MODEL_CODE_README.md
  DCKG_CONSTRUCTION.md          # pseudocode for optional KG/DCKG feature construction
  DATA_ACCESS_AND_PRIVACY.md
data/                           # raw clinical data are not included
examples/                       # toy SDSS sampling demo with synthetic text
```

The file names that contain `sdss32` are retained for traceability to the archived experiment snapshot. In the manuscript and result tables, the final method is reported as **SDSS**.

## Toy SDSS sampling demo

```bash
python examples/demo_sdss_sampling.py
```

This toy example only demonstrates the SDSS sampling mechanism and does not reproduce manuscript results.

## Data availability

The raw electronic medical record data used in this study are not publicly included in this repository because they contain sensitive patient-level clinical information and are subject to institutional ethics approval and hospital data-governance restrictions.

Qualified researchers may request access to de-identified data from the corresponding author. Access will be considered after approval by the relevant ethics committee and data-governance authority and may require a data use agreement. To support reproducibility without exposing patient-level records, this repository provides configuration templates, data-field examples, synthetic demonstration data, SDSS implementation code, MLP training scripts, and archived result files for reconstructing the manuscript tables.
