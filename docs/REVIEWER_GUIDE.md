# Reviewer guide

## 1. Verify final reported table

```bash
python scripts/build_final_sdss32_from_archived_grid.py --root results/archived_profile_grid --out results/final_sdss32_provenance
python scripts/check_final_table.py
```

## 2. Inspect SDSS implementation

Core files:

- `sdsskg/similarity.py`: text embedding and cosine distance to the positive centroid.
- `sdsskg/sampling.py`: distance-stratified hard/middle/easy negative sampling.

Historical class and file names that contain `sdss32` are retained only for traceability to the archived experiment snapshot. The final manuscript method is reported as **SDSS**.

## 3. Run toy demo

```bash
python examples/demo_sdss_sampling.py
```

## 4. Rerun with private data

The repository does not include raw EMR data. To rerun model training, place de-identified Train/Test1/Test2 files under `data/`, edit YAML paths in `configs/`, and use the reference runners under `scripts/reference_experiment_runners/` as templates.

Because the archived manuscript results were generated from a fixed historical experiment snapshot, a current-code rerun may show small numerical differences. The archived profile-grid outputs are therefore included for transparent result provenance.
